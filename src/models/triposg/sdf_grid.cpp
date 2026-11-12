// TripoSG: hierarchical SDF grid extraction (see sdf_grid.hpp).
// AI-assisted (Claude Code), 2026-11-12.
#include "sdf_grid.hpp"

#include <algorithm>
#include <cmath>

#include "src/models/triposg/marching_cubes.hpp"

namespace gufo::triposg {
namespace {

inline std::size_t idx(std::size_t n, std::size_t x, std::size_t y, std::size_t z) { return (x * n + y) * n + z; }

// Dense query of all n^3 lattice points.
bool query_dense(const SdfQuery& query, const SdfGridOptions& o, int n, std::vector<float>* grid, std::size_t* count,
                 std::string* error) {
    const std::size_t N = static_cast<std::size_t>(n);
    grid->resize(N * N * N);
    const float b = o.bounds;
    std::vector<float> points, values;
    const std::size_t rows = std::max<std::size_t>(1, o.chunk / (N * N));
    for (std::size_t x0 = 0; x0 < N; x0 += rows) {
        const std::size_t x1 = std::min(N, x0 + rows);
        points.resize((x1 - x0) * N * N * 3);
        std::size_t k = 0;
        for (std::size_t x = x0; x < x1; ++x)
            for (std::size_t y = 0; y < N; ++y)
                for (std::size_t z = 0; z < N; ++z) {
                    points[k++] = -b + 2 * b * static_cast<float>(x) / static_cast<float>(N - 1);
                    points[k++] = -b + 2 * b * static_cast<float>(y) / static_cast<float>(N - 1);
                    points[k++] = -b + 2 * b * static_cast<float>(z) / static_cast<float>(N - 1);
                }
        if (!query(points.data(), static_cast<int>((x1 - x0) * N * N), &values, error)) return false;
        std::copy(values.begin(), values.end(), grid->begin() + static_cast<std::ptrdiff_t>(x0 * N * N));
        *count += (x1 - x0) * N * N;
        if (o.progress) o.progress("grid " + std::to_string(n) + ": rows " + std::to_string(x1) + "/" + std::to_string(N));
    }
    return true;
}

// Trilinear interpolation of the coarse grid (nc samples) at fine lattice point (fine n, index).
inline float interpolate(const std::vector<float>& coarse, std::size_t nc, std::size_t n, std::size_t x, std::size_t y, std::size_t z) {
    const double s = static_cast<double>(nc - 1) / static_cast<double>(n - 1);
    const double fx = x * s, fy = y * s, fz = z * s;
    const std::size_t x0 = std::min(nc - 2, static_cast<std::size_t>(fx)), y0 = std::min(nc - 2, static_cast<std::size_t>(fy)),
                      z0 = std::min(nc - 2, static_cast<std::size_t>(fz));
    const float tx = static_cast<float>(fx - x0), ty = static_cast<float>(fy - y0), tz = static_cast<float>(fz - z0);
    auto c = [&](std::size_t dx, std::size_t dy, std::size_t dz) { return coarse[idx(nc, x0 + dx, y0 + dy, z0 + dz)]; };
    const float c00 = c(0, 0, 0) * (1 - tx) + c(1, 0, 0) * tx, c10 = c(0, 1, 0) * (1 - tx) + c(1, 1, 0) * tx;
    const float c01 = c(0, 0, 1) * (1 - tx) + c(1, 0, 1) * tx, c11 = c(0, 1, 1) * (1 - tx) + c(1, 1, 1) * tx;
    const float c0 = c00 * (1 - ty) + c10 * ty, c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
}

}  // namespace

bool extract_sdf_grid(const SdfQuery& query, const SdfGridOptions& o, SdfGridResult* out, std::string* error) {
    if (o.levels.empty()) {
        if (error) *error = "no grid levels";
        return false;
    }
    for (std::size_t i = 1; i < o.levels.size(); ++i)
        if (o.levels[i] != 2 * o.levels[i - 1]) {
            if (error) *error = "grid levels must double";
            return false;
        }
    out->queried_per_level.assign(o.levels.size(), 0);
    std::vector<float> grid;
    if (!query_dense(query, o, o.levels[0], &grid, &out->queried_per_level[0], error)) return false;
    std::size_t nc = static_cast<std::size_t>(o.levels[0]);

    for (std::size_t level = 1; level < o.levels.size(); ++level) {
        const std::size_t n = static_cast<std::size_t>(o.levels[level]);
        // 1. Band voxels on the coarse grid: any corner |v| < band or a sign change.
        std::vector<std::uint8_t> band((nc - 1) * (nc - 1) * (nc - 1), 0);
        for (std::size_t x = 0; x + 1 < nc; ++x)
            for (std::size_t y = 0; y + 1 < nc; ++y)
                for (std::size_t z = 0; z + 1 < nc; ++z) {
                    bool pos = false, neg = false, near = false;
                    for (int c = 0; c < 8; ++c) {
                        const float v = grid[idx(nc, x + ((c >> 0) & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1))];
                        pos |= v > 0.f; neg |= v <= 0.f; near |= std::fabs(v) < o.band;
                    }
                    if ((pos && neg) || near) band[(x * (nc - 1) + y) * (nc - 1) + z] = 1;
                }
        // 2. Dilate by `dilate` voxels.
        for (int d = 0; d < o.dilate; ++d) {
            std::vector<std::uint8_t> grown = band;
            const std::size_t m = nc - 1;
            for (std::size_t x = 0; x < m; ++x)
                for (std::size_t y = 0; y < m; ++y)
                    for (std::size_t z = 0; z < m; ++z) {
                        if (!band[(x * m + y) * m + z]) continue;
                        for (int dx = -1; dx <= 1; ++dx)
                            for (int dy = -1; dy <= 1; ++dy)
                                for (int dz = -1; dz <= 1; ++dz) {
                                    const long xx = static_cast<long>(x) + dx, yy = static_cast<long>(y) + dy, zz = static_cast<long>(z) + dz;
                                    if (xx < 0 || yy < 0 || zz < 0 || xx >= static_cast<long>(m) || yy >= static_cast<long>(m) || zz >= static_cast<long>(m)) continue;
                                    grown[(static_cast<std::size_t>(xx) * m + static_cast<std::size_t>(yy)) * m + static_cast<std::size_t>(zz)] = 1;
                                }
                    }
            band.swap(grown);
        }
        // 3. Fine lattice points to query: all points of band voxels (2x2x2 -> 3x3x3 incl. shared edges).
        std::vector<bool> mark(n * n * n, false);
        std::size_t marked = 0;
        const std::size_t m = nc - 1;
        for (std::size_t x = 0; x < m; ++x)
            for (std::size_t y = 0; y < m; ++y)
                for (std::size_t z = 0; z < m; ++z) {
                    if (!band[(x * m + y) * m + z]) continue;
                    for (std::size_t dx = 0; dx <= 2; ++dx)
                        for (std::size_t dy = 0; dy <= 2; ++dy)
                            for (std::size_t dz = 0; dz <= 2; ++dz) {
                                const std::size_t fx = std::min(n - 1, 2 * x + dx), fy = std::min(n - 1, 2 * y + dy), fz = std::min(n - 1, 2 * z + dz);
                                const std::size_t fi = idx(n, fx, fy, fz);
                                if (!mark[fi]) { mark[fi] = true; ++marked; }
                            }
                }
        if (o.progress) o.progress("grid " + std::to_string(n) + ": band " + std::to_string(marked) + " points");
        // 4. Fine grid (half precision: 512^3 = 256 MB): interpolate everything, then
        // overwrite the band points with exact values.
        std::vector<std::uint16_t> fine(n * n * n);
        for (std::size_t x = 0; x < n; ++x)
            for (std::size_t y = 0; y < n; ++y)
                for (std::size_t z = 0; z < n; ++z) fine[idx(n, x, y, z)] = ScalarGrid::float_to_half(interpolate(grid, nc, n, x, y, z));
        grid.clear();
        grid.shrink_to_fit();
        std::vector<float> points, values;
        std::vector<std::size_t> where;
        const float b = o.bounds;
        auto flush = [&]() -> bool {
            if (where.empty()) return true;
            if (!query(points.data(), static_cast<int>(where.size()), &values, error)) return false;
            for (std::size_t i = 0; i < where.size(); ++i) fine[where[i]] = ScalarGrid::float_to_half(values[i]);
            out->queried_per_level[level] += where.size();
            if (o.progress) o.progress("grid " + std::to_string(n) + ": " + std::to_string(out->queried_per_level[level]) + "/" + std::to_string(marked));
            points.clear();
            where.clear();
            return true;
        };
        for (std::size_t x = 0; x < n; ++x)
            for (std::size_t y = 0; y < n; ++y)
                for (std::size_t z = 0; z < n; ++z) {
                    const std::size_t i = idx(n, x, y, z);
                    if (!mark[i]) continue;
                    points.push_back(-b + 2 * b * static_cast<float>(x) / static_cast<float>(n - 1));
                    points.push_back(-b + 2 * b * static_cast<float>(y) / static_cast<float>(n - 1));
                    points.push_back(-b + 2 * b * static_cast<float>(z) / static_cast<float>(n - 1));
                    where.push_back(i);
                    if (where.size() >= o.chunk && !flush()) return false;
                }
        if (!flush()) return false;
        nc = n;
        if (level + 1 < o.levels.size()) {
            // Another level follows: widen back to FP32 as the next coarse grid.
            grid.resize(fine.size());
            for (std::size_t i = 0; i < fine.size(); ++i) grid[i] = ScalarGrid::half_to_float(fine[i]);
        } else {
            out->half.swap(fine);
        }
    }
    if (out->half.empty()) {  // single level
        out->half.resize(grid.size());
        for (std::size_t i = 0; i < grid.size(); ++i) out->half[i] = ScalarGrid::float_to_half(grid[i]);
    }
    out->n = static_cast<int>(nc);
    return true;
}

}  // namespace gufo::triposg
