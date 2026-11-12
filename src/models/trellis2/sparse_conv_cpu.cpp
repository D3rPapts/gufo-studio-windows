// TRELLIS.2 sparse voxel engine, CPU reference (see sparse_conv.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/sparse_conv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace gufo::trellis2 {

std::uint32_t VoxelHashMap::Hash(std::uint32_t key) {
    // 32-bit mix (Wang); the key is 30 bits of packed coordinates.
    key = (key ^ 61u) ^ (key >> 16);
    key += key << 3;
    key ^= key >> 4;
    key *= 0x27d4eb2du;
    key ^= key >> 15;
    return key;
}

void VoxelHashMap::Build(const std::vector<Coord>& coords) {
    std::size_t cap = 16;
    while (cap < 2 * coords.size()) cap <<= 1;
    keys_.assign(cap, kEmpty);
    values_.assign(cap, -1);
    count_ = 0;
    const std::uint32_t mask = static_cast<std::uint32_t>(cap - 1);
    for (std::size_t i = 0; i < coords.size(); ++i) {
        const Coord& c = coords[i];
        if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= 1024 || c.y >= 1024 || c.z >= 1024)
            throw std::out_of_range("VoxelHashMap: coordinate outside [0, 1024)");
        const std::uint32_t key = Pack(c.x, c.y, c.z);
        std::uint32_t slot = Hash(key) & mask;
        while (keys_[slot] != kEmpty) {
            if (keys_[slot] == key) throw std::invalid_argument("VoxelHashMap: duplicate coordinate");
            slot = (slot + 1) & mask;
        }
        keys_[slot] = key;
        values_[slot] = static_cast<std::int32_t>(i);
        ++count_;
    }
}

std::int64_t VoxelHashMap::Find(int x, int y, int z) const {
    if (keys_.empty() || x < 0 || y < 0 || z < 0 || x >= 1024 || y >= 1024 || z >= 1024) return -1;
    const std::uint32_t key = Pack(x, y, z);
    const std::uint32_t mask = static_cast<std::uint32_t>(keys_.size() - 1);
    std::uint32_t slot = Hash(key) & mask;
    while (keys_[slot] != kEmpty) {
        if (keys_[slot] == key) return values_[slot];
        slot = (slot + 1) & mask;
    }
    return -1;
}

std::vector<std::int32_t> build_neighbour_table(const std::vector<Coord>& coords, const VoxelHashMap& map) {
    std::vector<std::int32_t> table(coords.size() * kTaps, -1);
#pragma omp parallel for
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(coords.size()); ++i) {
        const Coord& c = coords[static_cast<std::size_t>(i)];
        for (int t = 0; t < kTaps; ++t) {
            int dx, dy, dz;
            tap_offset(t, &dx, &dy, &dz);
            table[static_cast<std::size_t>(i) * kTaps + t] =
                static_cast<std::int32_t>(map.Find(c.x + dx, c.y + dy, c.z + dz));
        }
    }
    return table;
}

void submanifold_conv3(const float* in, int n, int ci, const std::int32_t* neighbours, const float* weight,
                       const float* bias, int co, float* out, int block_rows) {
    if (block_rows <= 0) block_rows = n;
    const std::size_t k = static_cast<std::size_t>(kTaps) * ci;
    std::vector<float> im2col(static_cast<std::size_t>(block_rows) * k);
    for (int start = 0; start < n; start += block_rows) {
        const int rows = std::min(block_rows, n - start);
        // gather
#pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            float* row = im2col.data() + static_cast<std::size_t>(r) * k;
            const std::int32_t* nb = neighbours + static_cast<std::size_t>(start + r) * kTaps;
            for (int t = 0; t < kTaps; ++t) {
                float* dst = row + static_cast<std::size_t>(t) * ci;
                if (nb[t] < 0) {
                    std::memset(dst, 0, sizeof(float) * static_cast<std::size_t>(ci));
                } else {
                    std::memcpy(dst, in + static_cast<std::size_t>(nb[t]) * ci, sizeof(float) * static_cast<std::size_t>(ci));
                }
            }
        }
        // GEMM: out[start + r, o] = row . weight[o]
#pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            const float* row = im2col.data() + static_cast<std::size_t>(r) * k;
            float* dst = out + static_cast<std::size_t>(start + r) * co;
            for (int o = 0; o < co; ++o) {
                const float* w = weight + static_cast<std::size_t>(o) * k;
                double acc = bias ? bias[o] : 0.0;
                for (std::size_t j = 0; j < k; ++j) acc += static_cast<double>(row[j]) * w[j];
                dst[o] = static_cast<float>(acc);
            }
        }
    }
}

void submanifold_conv3_dense_reference(const std::vector<Coord>& coords, const float* in, int ci, const float* weight,
                                       const float* bias, int co, float* out) {
    struct Key {
        std::size_t operator()(const Coord& c) const {
            return (static_cast<std::size_t>(c.x) * 73856093u) ^ (static_cast<std::size_t>(c.y) * 19349663u) ^
                   (static_cast<std::size_t>(c.z) * 83492791u);
        }
    };
    std::unordered_map<Coord, std::size_t, Key> index;
    for (std::size_t i = 0; i < coords.size(); ++i) index.emplace(coords[i], i);
    const std::size_t k = static_cast<std::size_t>(kTaps) * ci;
    for (std::size_t i = 0; i < coords.size(); ++i) {
        for (int o = 0; o < co; ++o) {
            double acc = bias ? bias[o] : 0.0;
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        auto it = index.find(Coord{coords[i].x + dx, coords[i].y + dy, coords[i].z + dz});
                        if (it == index.end()) continue;
                        const int t = tap_index(dx, dy, dz);
                        const float* src = in + it->second * ci;
                        const float* w = weight + static_cast<std::size_t>(o) * k + static_cast<std::size_t>(t) * ci;
                        for (int c = 0; c < ci; ++c) acc += static_cast<double>(src[c]) * w[c];
                    }
            out[i * co + o] = static_cast<float>(acc);
        }
    }
}

Children channel2spatial_coords(const std::vector<Coord>& parents, const std::vector<std::uint8_t>& mask) {
    Children ch;
    for (std::size_t i = 0; i < parents.size(); ++i) {
        for (int k = 0; k < 8; ++k) {
            if (!mask[i * 8 + static_cast<std::size_t>(k)]) continue;
            ch.coords.push_back(Coord{2 * parents[i].x + (k & 1), 2 * parents[i].y + ((k >> 1) & 1),
                                      2 * parents[i].z + ((k >> 2) & 1)});
            ch.parent.push_back(static_cast<std::int32_t>(i));
            ch.child_index.push_back(static_cast<std::int8_t>(k));
        }
    }
    return ch;
}

void channel2spatial_feats(const float* in, int channels, const Children& children, float* out) {
    for (std::size_t m = 0; m < children.coords.size(); ++m) {
        const float* src = in + (static_cast<std::size_t>(children.parent[m]) * 8 + static_cast<std::size_t>(children.child_index[m])) * channels;
        std::memcpy(out + m * channels, src, sizeof(float) * static_cast<std::size_t>(channels));
    }
}

void channel2spatial_skip(const float* in, int channels_in, int repeat, const Children& children, float* out) {
    const int cs = channels_in / 8;
    for (std::size_t m = 0; m < children.coords.size(); ++m) {
        const float* src = in + static_cast<std::size_t>(children.parent[m]) * channels_in + static_cast<std::size_t>(children.child_index[m]) * cs;
        float* dst = out + m * static_cast<std::size_t>(cs) * repeat;
        for (int c = 0; c < cs; ++c)
            for (int r = 0; r < repeat; ++r) dst[c * repeat + r] = src[c];
    }
}

void layer_norm_rows(float* x, int rows, int cols, const float* weight, const float* bias, float eps) {
#pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        float* row = x + static_cast<std::size_t>(r) * cols;
        double mean = 0;
        for (int c = 0; c < cols; ++c) mean += row[c];
        mean /= cols;
        double var = 0;
        for (int c = 0; c < cols; ++c) var += (row[c] - mean) * (row[c] - mean);
        var /= cols;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int c = 0; c < cols; ++c) {
            double v = (row[c] - mean) * inv;
            if (weight) v = v * weight[c] + (bias ? bias[c] : 0.0);
            row[c] = static_cast<float>(v);
        }
    }
}

void silu_rows(float* x, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) x[i] = x[i] / (1.f + std::exp(-x[i]));
}

}  // namespace gufo::trellis2
