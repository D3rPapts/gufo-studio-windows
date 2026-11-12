// TRELLIS.2: sparse decoders on HIP, built on the qwen_image_21 runtime (BF16 hipBLASLt
// GEMMs for conv-as-GEMM and the MLPs, runtime LayerNorm/SiLU) plus the sparse voxel
// engine: device hashmap (atomicCAS, 30-bit packed keys), 27-tap neighbour table, gather
// kernel (im2col rows in blocks of <= 512 MB), Channel2Spatial gather and the repeated
// skip. Subdivision masks and child coordinates are computed on the host (one download
// of the [N, 8] logits per up-block). Same maths as sparse_dec_cpu.cpp.
// AI-assisted (Claude Code), 2026-11-12.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "src/models/qwen_image_21/hip/runtime.hpp"
#include "src/models/qwen_image_21/weights.hpp"
#include "src/models/trellis2/sparse_dec.hpp"

namespace gufo::trellis2 {
namespace {

using models::qwen_image_21::SafetensorsSource;
using models::qwen_image_21::Weights;
using models::qwen_image_21::hip::Activation;
using models::qwen_image_21::hip::Matrix;
using models::qwen_image_21::hip::Norm;
using models::qwen_image_21::hip::Runtime;

constexpr std::uint32_t kEmpty = 0xFFFFFFFFu;

__device__ __forceinline__ std::uint32_t HashKey(std::uint32_t key) {
    key = (key ^ 61u) ^ (key >> 16);
    key += key << 3;
    key ^= key >> 4;
    key *= 0x27d4eb2du;
    key ^= key >> 15;
    return key;
}
__device__ __forceinline__ std::uint32_t Pack(int x, int y, int z) {
    return (static_cast<std::uint32_t>(x) << 20) | (static_cast<std::uint32_t>(y) << 10) | static_cast<std::uint32_t>(z);
}

__global__ void HashInsertKernel(const std::int32_t* coords, int n, std::uint32_t* keys, std::int32_t* values, std::uint32_t mask) {
    const int i = static_cast<int>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const std::uint32_t key = Pack(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
    std::uint32_t slot = HashKey(key) & mask;
    while (true) {
        const std::uint32_t prev = atomicCAS(&keys[slot], kEmpty, key);
        if (prev == kEmpty || prev == key) {
            values[slot] = i;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

__device__ __forceinline__ std::int32_t HashFind(const std::uint32_t* keys, const std::int32_t* values, std::uint32_t mask, int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= 1024 || y >= 1024 || z >= 1024) return -1;
    const std::uint32_t key = Pack(x, y, z);
    std::uint32_t slot = HashKey(key) & mask;
    while (true) {
        const std::uint32_t k = keys[slot];
        if (k == kEmpty) return -1;
        if (k == key) return values[slot];
        slot = (slot + 1) & mask;
    }
}

__global__ void NeighbourTableKernel(const std::int32_t* coords, int n, const std::uint32_t* keys, const std::int32_t* values,
                                     std::uint32_t mask, std::int32_t* table) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(n) * 27) return;
    const int v = static_cast<int>(i / 27), t = static_cast<int>(i % 27);
    const int dx = t / 9 - 1, dy = (t / 3) % 3 - 1, dz = t % 3 - 1;
    table[i] = HashFind(keys, values, mask, coords[v * 3] + dx, coords[v * 3 + 1] + dy, coords[v * 3 + 2] + dz);
}

// im2col[r, t*Ci + c] = x[table[(start + r)*27 + t], c] or 0 (BF16 bit copies).
__global__ void GatherKernel(const std::uint16_t* x, const std::int32_t* table, int start, int rows, int ci, std::uint16_t* out) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * 27 * ci;
    if (i >= total) return;
    const int c = static_cast<int>(i % ci);
    const int t = static_cast<int>((i / ci) % 27);
    const int r = static_cast<int>(i / (static_cast<std::size_t>(ci) * 27));
    const std::int32_t src = table[static_cast<std::size_t>(start + r) * 27 + t];
    out[i] = src < 0 ? static_cast<std::uint16_t>(0) : x[static_cast<std::size_t>(src) * ci + c];
}

// child m: out[m, c] = x[parent[m], k[m]*C + c]
__global__ void C2SKernel(const std::uint16_t* x, const std::int32_t* parent, const std::int32_t* child, int m, int channels, std::uint16_t* out) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(m) * channels) return;
    const int c = static_cast<int>(i % channels), r = static_cast<int>(i / channels);
    out[i] = x[(static_cast<std::size_t>(parent[r]) * 8 + static_cast<std::size_t>(child[r])) * channels + c];
}

// skip: out[m, cs*repeat + rep] = x[parent[m], k[m]*cs + cs_index]  (channels_in = 8*cs)
__global__ void SkipKernel(const std::uint16_t* x, const std::int32_t* parent, const std::int32_t* child, int m, int channels_in, int repeat, std::uint16_t* out) {
    const int cs = channels_in / 8;
    const int cols = cs * repeat;
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(m) * cols) return;
    const int col = static_cast<int>(i % cols), r = static_cast<int>(i / cols);
    out[i] = x[static_cast<std::size_t>(parent[r]) * channels_in + static_cast<std::size_t>(child[r]) * cs + col / repeat];
}

unsigned Blocks(std::size_t n) { return static_cast<unsigned>((n + 255) / 256); }
void CheckLaunch(const char* what) {
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}
void CheckHip(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

class DevInts {
public:
    DevInts() = default;
    ~DevInts() { Release(); }
    DevInts(const DevInts&) = delete;
    DevInts& operator=(const DevInts&) = delete;
    DevInts(DevInts&& o) noexcept : data_(std::exchange(o.data_, nullptr)), count_(o.count_) {}
    DevInts& operator=(DevInts&& o) noexcept {
        if (this != &o) { Release(); data_ = std::exchange(o.data_, nullptr); count_ = o.count_; }
        return *this;
    }
    void Allocate(std::size_t count) {
        Release();
        count_ = count;
        CheckHip(hipMalloc(&data_, std::max<std::size_t>(count * 4, 256)), "hipMalloc");
    }
    void Upload(const std::vector<std::int32_t>& host, hipStream_t stream) {
        Allocate(host.size());
        CheckHip(hipMemcpyAsync(data_, host.data(), host.size() * 4, hipMemcpyHostToDevice, stream), "upload ints");
    }
    void Fill(std::uint32_t value, hipStream_t stream) { CheckHip(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(data_), value, count_, stream), "memset"); }
    std::int32_t* i32() const { return static_cast<std::int32_t*>(data_); }
    std::uint32_t* u32() const { return static_cast<std::uint32_t*>(data_); }
    std::size_t count() const { return count_; }

private:
    void Release() {
        if (data_) (void)hipFree(data_);
        data_ = nullptr;
        count_ = 0;
    }
    void* data_{nullptr};
    std::size_t count_{0};
};

struct DevLevel {
    std::vector<Coord> coords;
    DevInts coords_dev, keys, values, table;
    std::uint32_t mask = 0;
    int n() const { return static_cast<int>(coords.size()); }
    void Build(std::vector<Coord> c, hipStream_t stream) {
        coords = std::move(c);
        std::vector<std::int32_t> flat(coords.size() * 3);
        for (std::size_t i = 0; i < coords.size(); ++i) { flat[i * 3] = coords[i].x; flat[i * 3 + 1] = coords[i].y; flat[i * 3 + 2] = coords[i].z; }
        coords_dev.Upload(flat, stream);
        std::size_t cap = 16;
        while (cap < 2 * coords.size()) cap <<= 1;
        mask = static_cast<std::uint32_t>(cap - 1);
        keys.Allocate(cap);
        values.Allocate(cap);
        keys.Fill(kEmpty, stream);
        HashInsertKernel<<<Blocks(coords.size()), 256, 0, stream>>>(coords_dev.i32(), n(), keys.u32(), values.i32(), mask);
        CheckLaunch("hash insert");
        table.Allocate(coords.size() * 27);
        NeighbourTableKernel<<<Blocks(coords.size() * 27), 256, 0, stream>>>(coords_dev.i32(), n(), keys.u32(), values.i32(), mask, table.i32());
        CheckLaunch("neighbour table");
    }
};

}  // namespace

struct SparseDecHip::Impl {
    explicit Impl(const std::filesystem::path& file)
        : weights(Weights::FromFiles(std::vector<SafetensorsSource>{{file, ""}})), rt(weights) {}
    Weights weights;
    Runtime rt;
    SparseDecShape shape;

    // Submanifold conv as blocked gather + GEMM through the runtime's Linear (weight [Co, 27*Ci]).
    Matrix Conv(const Matrix& x, const DevLevel& level, const std::string& name, int co) {
        const int n = x.rows, ci = x.cols;
        const std::size_t k = static_cast<std::size_t>(27) * ci;
        const int block = static_cast<int>(std::max<std::size_t>(1024, (std::size_t{1} << 28) / k));   // <= 512 MB im2col
        Matrix out = rt.New(n, co);
        for (int start = 0; start < n; start += block) {
            const int rows = std::min(block, n - start);
            Matrix col = rt.New(rows, static_cast<int>(k));
            GatherKernel<<<Blocks(static_cast<std::size_t>(rows) * k), 256, 0, rt.stream()>>>(x.data, level.table.i32(), start, rows, ci, col.data);
            CheckLaunch("gather");
            Matrix part = rt.Linear(col, name);
            rt.CopyRows(out, start, part);
        }
        return out;
    }
};

SparseDecHip::SparseDecHip(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SparseDecHip::~SparseDecHip() = default;
std::size_t SparseDecHip::weight_bytes() const { return impl_->rt.WeightBytes(); }

std::unique_ptr<SparseDecHip> SparseDecHip::Create(const std::filesystem::path& file, const SparseDecShape& shape, std::string* error) {
    try {
        auto impl = std::make_unique<Impl>(file);
        impl->shape = shape;
        if (!impl->weights.Contains("from_latent.weight") || !impl->weights.Contains("output_layer.weight")) {
            if (error) *error = "not a converted TRELLIS.2 sparse decoder: " + file.string();
            return nullptr;
        }
        impl->shape.latent_channels = impl->weights.Get("from_latent.weight").shape[1];
        impl->shape.outputs = impl->weights.Get("output_layer.weight").shape[0];
        impl->shape.pred_subdiv = impl->weights.Contains("blocks.0.4.to_subdiv.weight");
        return std::unique_ptr<SparseDecHip>(new SparseDecHip(std::move(impl)));
    } catch (const std::exception& e) {
        if (error) *error = std::string("TRELLIS.2 sparse decoder: ") + e.what();
        return nullptr;
    }
}

bool SparseDecHip::Forward(const float* latent, const std::vector<Coord>& coords_in, SparseDecResult* result, std::string* error,
                           const SparseDecOptions& options) {
    Impl& m = *impl_;
    Runtime& rt = m.rt;
    const SparseDecShape& s = m.shape;
    if (!s.pred_subdiv && (!options.guide_masks || options.guide_masks->size() < static_cast<std::size_t>(s.levels - 1))) {
        if (error) *error = "sparse decoder without pred_subdiv needs guide masks";
        return false;
    }
    try {
        DevLevel level;
        level.Build(coords_in, rt.stream());
        Matrix x = rt.Linear(rt.Upload(std::span<const float>(latent, coords_in.size() * static_cast<std::size_t>(s.latent_channels)),
                                       level.n(), s.latent_channels), "from_latent");
        result->masks.clear();
        result->level_feats.clear();
        result->level_coords.clear();
        const int levels = options.level_limit > 0 ? std::min(options.level_limit, s.levels) : s.levels;
        for (int l = 0; l < levels; ++l) {
            const int C = s.channels[l];
            const int N = level.n();
            for (int j = 0; j < s.blocks[l]; ++j) {
                const std::string p = "blocks." + std::to_string(l) + "." + std::to_string(j) + ".";
                Matrix h = rt.Normalize(m.Conv(x, level, p + "conv", C), Norm::kLayer, p + "norm", SparseDecConfig::norm_eps);
                h = rt.Linear(rt.Activate(rt.Linear(h, p + "mlp.0"), Activation::kSilu), p + "mlp.2");
                x = rt.Add(x, h);
            }
            if (options.capture_levels) {
                result->level_feats.push_back(rt.Download(x));
                result->level_coords.push_back(level.coords);
            }
            if (l + 1 >= s.levels || l + 1 >= levels) break;
            const std::string p = "blocks." + std::to_string(l) + "." + std::to_string(s.blocks[l]) + ".";
            const int C2 = s.channels[l + 1];
            if (C < 8 || C2 % (C / 8) != 0) throw std::logic_error("channels not supported by the skip path");
            std::vector<std::uint8_t> mask(static_cast<std::size_t>(N) * 8);
            if (s.pred_subdiv) {
                const std::vector<float> logits = rt.Download(rt.Linear(x, p + "to_subdiv"));
                for (std::size_t i = 0; i < mask.size(); ++i) mask[i] = logits[i] > 0.f ? 1 : 0;
            } else {
                mask = (*options.guide_masks)[static_cast<std::size_t>(l)];
                if (mask.size() != static_cast<std::size_t>(N) * 8) throw std::logic_error("guide mask size mismatch");
            }
            result->masks.push_back(mask);
            Matrix h = rt.Normalize(x, Norm::kLayer, p + "norm1", SparseDecConfig::norm_eps, true);   // LN + SiLU
            h = m.Conv(h, level, p + "conv1", 8 * C2);                                                   // [N, 8*C2]
            const Children ch = channel2spatial_coords(level.coords, mask);
            const int M = static_cast<int>(ch.coords.size());
            if (M == 0) throw std::logic_error("no voxels subdivided at level " + std::to_string(l));
            std::vector<std::int32_t> parent(ch.parent.begin(), ch.parent.end()), child(ch.child_index.begin(), ch.child_index.end());
            DevInts parent_dev, child_dev;
            parent_dev.Upload(parent, rt.stream());
            child_dev.Upload(child, rt.stream());
            Matrix hc = rt.New(M, C2);
            C2SKernel<<<Blocks(static_cast<std::size_t>(M) * C2), 256, 0, rt.stream()>>>(h.data, parent_dev.i32(), child_dev.i32(), M, C2, hc.data);
            CheckLaunch("channel2spatial");
            hc = rt.Normalize(hc, Norm::kLayer, "", SparseDecConfig::norm_eps, true);
            DevLevel next;
            next.Build(ch.coords, rt.stream());
            Matrix conv2 = m.Conv(hc, next, p + "conv2", C2);
            Matrix skip = rt.New(M, C2);
            SkipKernel<<<Blocks(static_cast<std::size_t>(M) * C2), 256, 0, rt.stream()>>>(x.data, parent_dev.i32(), child_dev.i32(), M, C, C2 / (C / 8), skip.data);
            CheckLaunch("skip");
            rt.Synchronize();   // parent/child buffers are freed at the end of the scope
            x = rt.Add(conv2, skip);
            level = std::move(next);
        }
        result->coords = level.coords;
        if (levels < s.levels) {
            result->outputs.clear();
            rt.Synchronize();
            return true;
        }
        Matrix out = rt.Linear(rt.Normalize(x, Norm::kLayer, "", SparseDecConfig::final_norm_eps), "output_layer");
        result->outputs = rt.Download(out);
        rt.Synchronize();
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("TRELLIS.2 sparse decoder forward: ") + e.what();
        return false;
    }
}

}  // namespace gufo::trellis2
