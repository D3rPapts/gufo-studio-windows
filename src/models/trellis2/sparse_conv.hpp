// TRELLIS.2: sparse voxel engine, CPU reference. The sparse decoders run on a list of
// active voxels (coordinates + feature rows). Submanifold 3^3 convolution reads only
// active neighbours and writes only active voxels, so one neighbour table [N, 27]
// (-1 = missing) serves every conv of a level, and the conv is a gather
// (im2col rows [rows, 27*Ci]) -> GEMM ([27*Ci] x [Co]) -> write. The HIP version
// (stage 5) does the same in blocks of rows with hipBLASLt (wan22/gemm.hpp); this file
// is the bit-for-bit reference and the test oracle for it.
// Coordinates are 0 <= c < 1024 (10 bits each); the hashmap packs them into 30 bits.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gufo::trellis2 {

struct Coord {
    std::int32_t x, y, z;
    bool operator==(const Coord& o) const { return x == o.x && y == o.y && z == o.z; }
};

constexpr int kTaps = 27;
constexpr int kCenterTap = 13;  // offset (0,0,0)

// Tap index of an offset (dx,dy,dz) in {-1,0,1}^3, matching the converter's weight
// column order t * Ci + ci (PyTorch Conv3d kD,kH,kW over the axes x,y,z).
inline int tap_index(int dx, int dy, int dz) { return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1); }
inline void tap_offset(int tap, int* dx, int* dy, int* dz) {
    *dx = tap / 9 - 1;
    *dy = (tap / 3) % 3 - 1;
    *dz = tap % 3 - 1;
}

// Open-addressing hashmap coordinate -> row index (capacity = next power of two >= 2N).
class VoxelHashMap {
public:
    void Build(const std::vector<Coord>& coords);
    // Row of the voxel or -1.
    std::int64_t Find(int x, int y, int z) const;
    std::size_t size() const { return count_; }
    std::size_t capacity() const { return keys_.size(); }

private:
    static constexpr std::uint32_t kEmpty = 0xFFFFFFFFu;
    static std::uint32_t Pack(int x, int y, int z) {
        return (static_cast<std::uint32_t>(x) << 20) | (static_cast<std::uint32_t>(y) << 10) | static_cast<std::uint32_t>(z);
    }
    static std::uint32_t Hash(std::uint32_t key);
    std::vector<std::uint32_t> keys_;
    std::vector<std::int32_t> values_;
    std::size_t count_{0};
};

// [N * 27] neighbour rows (-1 = no active voxel at that offset); table[i*27 + 13] = i.
std::vector<std::int32_t> build_neighbour_table(const std::vector<Coord>& coords, const VoxelHashMap& map);

// Submanifold conv: out[N, Co] = sum_t sum_ci in[nbr[i][t], ci] * w[co, t*Ci + ci] + bias[co].
// Processes `block_rows` rows at a time through an im2col buffer [block_rows, 27*Ci]
// (the HIP engine uses the same blocking); block_rows 0 = all rows at once.
void submanifold_conv3(const float* in, int n, int ci, const std::int32_t* neighbours, const float* weight,
                       const float* bias, int co, float* out, int block_rows = 0);

// Brute-force reference on a dense grid for tests: same semantics, no neighbour table.
void submanifold_conv3_dense_reference(const std::vector<Coord>& coords, const float* in, int ci, const float* weight,
                                       const float* bias, int co, float* out);

// Channel2Spatial (factor 2): parent i with mask bits m[i*8 + k] spawns child k at
// 2*p + (k & 1, (k >> 1) & 1, (k >> 2) & 1) with features in[i, k*C .. (k+1)*C). Children are
// emitted parent by parent, k ascending (= torch.repeat_interleave + nonzero order).
struct Children {
    std::vector<Coord> coords;
    std::vector<std::int32_t> parent;   // child -> parent row
    std::vector<std::int8_t> child_index;  // k in 0..7
};
Children channel2spatial_coords(const std::vector<Coord>& parents, const std::vector<std::uint8_t>& mask);
// Gathers the child features: out[m, C] from in[N, 8*C].
void channel2spatial_feats(const float* in, int channels, const Children& children, float* out);
// Skip path of SparseResBlockC2S3d: child features in[parent, k*Cs .. (k+1)*Cs) (Cs = C/8)
// with every channel repeated `repeat` times consecutively -> out[m, Cs * repeat].
void channel2spatial_skip(const float* in, int channels_in, int repeat, const Children& children, float* out);

// Row-wise LayerNorm (weight/bias optional) and SiLU, FP32.
void layer_norm_rows(float* x, int rows, int cols, const float* weight, const float* bias, float eps);
void silu_rows(float* x, std::size_t count);

}  // namespace gufo::trellis2
