// TRELLIS.2: flexible dual grid -> mesh (see dual_grid_mesh.hpp).
//
// Closed construction. The original emits one quad per flagged primal edge in a fixed vertex
// order and leaves parity defects (a voxel face with one or three flagged edges) as holes,
// which CuMesh fill_holes then patches. Here the flags are first turned into an inside /
// outside sign per primal grid corner: a flood fill from the outside of the bounding box
// flips the sign whenever it crosses a flagged edge; enclosed regions are seeded from a
// visited neighbour across a flagged edge. The mesh is then the boundary between the two
// signs: one quad per edge whose end corners differ, wound by the sign (outside at the
// lower corner keeps the table order, which faces -axis). That boundary is closed by
// construction; parity defects become small seams (edges whose sign change and flag
// disagree) instead of holes: a gap in the predicted surface lets the outside leak in until it
// meets the inside front, so the seam is a cap over the gap. The exterior (reachable from the
// box boundary without touching the active shell, cavities included) is fixed first and the
// parity flood starts from the whole shell at once, so leaks stay local. Voxels that are not
// active but border a sign change get a vertex at their centre.
//
// Sheets: a voxel crossed by two surface sheets (a face with four sign changes) gets one
// vertex per sheet so that the dual edge to its neighbour carries two quads per sheet. The
// crossed edges of a voxel are paired per face (two changes: pair; four: the two edges
// around each inside corner), and the pairs are joined across faces. Both voxels at a
// face decide from the same signs, so the pairing agrees and the mesh stays closed; an
// edge keeps four quads only when the sheet folds back on both sides (rare).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/dual_grid_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "src/models/trellis2/config.hpp"

namespace gufo::trellis2 {

namespace {

// Slot s = axis * 4 + k of voxel u is the primal edge (u - off[axis][k], axis). In the unit
// cube of u it runs along `axis` from local corner start[s] (x_axis = 0, x_d = 1 - off[axis][k][d]).
// Cube faces: id = d * 2 + side (side 1: x_d = 1).
struct SlotTables {
    int start[12][3];        // local start corner of each slot
    int faces[12][2];        // the two cube faces each slot lies on
    int ring[6][4];          // the four slots of each face in cyclic order
    int ring_corner[6][4][3];  // local corner between ring[f][i] and ring[f][i+1]
};

SlotTables MakeSlotTables() {
    const auto& off = MeshConfig::edge_neighbour_offsets;
    SlotTables t{};
    for (int a = 0; a < 3; ++a)
        for (int k = 0; k < 4; ++k) {
            const int s = a * 4 + k;
            int n = 0;
            for (int d = 0; d < 3; ++d) {
                t.start[s][d] = d == a ? 0 : 1 - off[a][k][d];
                if (d != a) t.faces[s][n++] = d * 2 + (1 - off[a][k][d]);
            }
        }
    for (int f = 0; f < 6; ++f) {
        const int d = f / 2, b = (d + 1) % 3, c = (d + 2) % 3;
        struct Entry { int slot; double angle; } entries[4];
        int n = 0;
        for (int slot = 0; slot < 12; ++slot) {
            if (t.faces[slot][0] != f && t.faces[slot][1] != f) continue;
            const int a = slot / 4;
            double p[3];
            for (int e = 0; e < 3; ++e) p[e] = e == a ? 0.5 : t.start[slot][e];
            entries[n++] = {slot, std::atan2(p[c] - 0.5, p[b] - 0.5)};
        }
        std::sort(entries, entries + n, [](const Entry& l, const Entry& r) { return l.angle < r.angle; });
        for (int i = 0; i < 4; ++i) t.ring[f][i] = entries[i].slot;
        // Corner shared by consecutive ring slots: the end of one that lies on the other.
        for (int i = 0; i < 4; ++i) {
            const int s0 = t.ring[f][i], s1 = t.ring[f][(i + 1) % 4];
            const int a0 = s0 / 4, a1 = s1 / 4;
            // s0 runs along a0; its corner on s1 has x_a0 = start[s1][a0], x_a1 = start[s0][a1], x_d = f side.
            for (int e = 0; e < 3; ++e) t.ring_corner[f][i][e] = e == a0 ? t.start[s1][a0] : (e == a1 ? t.start[s0][a1] : f % 2);
        }
    }
    return t;
}

int Root(int* parent, int i) {
    while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
    return i;
}

// Dense bit field over the corner box.
class Bits {
public:
    explicit Bits(std::size_t n) : words_((n + 63) / 64, 0) {}
    bool get(std::size_t i) const { return (words_[i >> 6] >> (i & 63)) & 1u; }
    void set(std::size_t i, bool v) {
        if (v) words_[i >> 6] |= std::uint64_t{1} << (i & 63);
        else words_[i >> 6] &= ~(std::uint64_t{1} << (i & 63));
    }
private:
    std::vector<std::uint64_t> words_;
};

}  // namespace

void decode_shape_outputs(const float* out7, std::size_t n, std::vector<float>* dual, std::vector<std::uint8_t>* intersected,
                          std::vector<float>* split_weight, float margin) {
    dual->resize(n * 3);
    intersected->resize(n * 3);
    split_weight->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float* o = out7 + i * 7;
        for (int a = 0; a < 3; ++a) {
            const float s = 1.f / (1.f + std::exp(-o[a]));
            (*dual)[i * 3 + static_cast<std::size_t>(a)] = (1.f + 2.f * margin) * s - margin;
            (*intersected)[i * 3 + static_cast<std::size_t>(a)] = o[3 + a] > 0.f ? 1 : 0;
        }
        const float w = o[6];
        (*split_weight)[i] = w > 20.f ? w : std::log1p(std::exp(w));   // softplus
    }
}

Mesh dual_grid_to_mesh(const DualGridInput& in, DualGridStats* stats) {
    static const SlotTables T = MakeSlotTables();
    Mesh mesh;
    const auto& coords = *in.coords;
    const std::size_t n = coords.size();
    const float voxel = (in.aabb_max - in.aabb_min) / static_cast<float>(in.resolution);
    DualGridStats st;
    if (n == 0) { if (stats) *stats = st; return mesh; }
    const auto& off = MeshConfig::edge_neighbour_offsets;
    VoxelHashMap map;
    map.Build(coords);

    // Corner box: voxels vmin-1 .. vmax+1 (one ring of virtual voxels), corners vmin-1 .. vmax+2.
    int vmin[3] = {coords[0].x, coords[0].y, coords[0].z}, vmax[3] = {coords[0].x, coords[0].y, coords[0].z};
    for (const Coord& c : coords) {
        const int v[3] = {c.x, c.y, c.z};
        for (int d = 0; d < 3; ++d) { vmin[d] = std::min(vmin[d], v[d]); vmax[d] = std::max(vmax[d], v[d]); }
    }
    const int o[3] = {vmin[0] - 1, vmin[1] - 1, vmin[2] - 1};                 // corner (0,0,0) of the box
    const int D[3] = {vmax[0] - o[0] + 3, vmax[1] - o[1] + 3, vmax[2] - o[2] + 3};   // corners per axis
    const std::size_t total = static_cast<std::size_t>(D[0]) * static_cast<std::size_t>(D[1]) * static_cast<std::size_t>(D[2]);
    auto index = [&](int x, int y, int z) { return (static_cast<std::size_t>(x) * static_cast<std::size_t>(D[1]) + static_cast<std::size_t>(y)) * static_cast<std::size_t>(D[2]) + static_cast<std::size_t>(z); };
    const std::size_t stride[3] = {static_cast<std::size_t>(D[1]) * static_cast<std::size_t>(D[2]), static_cast<std::size_t>(D[2]), 1};

    // Flag of the edge leaving corner (x,y,z) along axis a: owner voxel = corner minus one in the other axes.
    auto flagged = [&](int x, int y, int z, int a) -> bool {
        const int vx = o[0] + x - (a != 0), vy = o[1] + y - (a != 1), vz = o[2] + z - (a != 2);
        if (vx < 0 || vy < 0 || vz < 0) return false;
        const std::int64_t row = map.Find(vx, vy, vz);
        return row >= 0 && in.intersected[static_cast<std::size_t>(row) * 3 + static_cast<std::size_t>(a)];
    };

    // Corner signs (true = inside) in two floods. First the exterior: everything reachable from the
    // box boundary without touching a corner of an active voxel is outside, flags ignored (this
    // reaches cavities such as the inside of a cup through their opening). Then the parity flood
    // starts from every exterior corner next to the active shell at once, so a gap in the
    // predicted surface can only leak as far as the nearest intact surface, never across a cavity.
    Bits near(total), inside(total), visited(total);
    for (const Coord& c : coords)
        for (int k = 0; k < 8; ++k) near.set(index(c.x - o[0] + (k & 1), c.y - o[1] + ((k >> 1) & 1), c.z - o[2] + (k >> 2)), true);
    std::vector<std::uint32_t> queue;
    queue.reserve(1u << 20);
    auto flood = [&]() {
        std::size_t head = 0;
        while (head < queue.size()) {
            const std::size_t i = queue[head++];
            const int z = static_cast<int>(i % static_cast<std::size_t>(D[2]));
            const int y = static_cast<int>((i / static_cast<std::size_t>(D[2])) % static_cast<std::size_t>(D[1]));
            const int x = static_cast<int>(i / stride[0]);
            const bool s = inside.get(i);
            const int c[3] = {x, y, z};
            for (int a = 0; a < 3; ++a)
                for (int dir = -1; dir <= 1; dir += 2) {
                    int nb[3] = {x, y, z};
                    nb[a] += dir;
                    if (nb[a] < 0 || nb[a] >= D[a]) continue;
                    const std::size_t j = i + (dir > 0 ? stride[a] : static_cast<std::size_t>(0) - stride[a]);
                    if (visited.get(j)) continue;
                    // Edge between i and j leaves the lower corner along a.
                    const bool f = dir > 0 ? flagged(c[0], c[1], c[2], a) : flagged(nb[0], nb[1], nb[2], a);
                    visited.set(j, true);
                    inside.set(j, s != f);
                    queue.push_back(static_cast<std::uint32_t>(j));
                }
            if (head > (1u << 22) && head * 2 > queue.size()) { queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(head)); head = 0; }
        }
        queue.clear();
    };
    std::vector<std::uint32_t> shell_seeds;   // exterior corners with a neighbour on the active shell
    {
        std::vector<std::uint32_t> ext;
        ext.reserve(1u << 20);
        ext.push_back(0);
        visited.set(0, true);
        std::size_t head = 0;
        while (head < ext.size()) {
            const std::size_t i = ext[head++];
            const int z = static_cast<int>(i % static_cast<std::size_t>(D[2]));
            const int y = static_cast<int>((i / static_cast<std::size_t>(D[2])) % static_cast<std::size_t>(D[1]));
            const int x = static_cast<int>(i / stride[0]);
            const int c[3] = {x, y, z};
            bool touches_shell = false;
            for (int a = 0; a < 3; ++a)
                for (int dir = -1; dir <= 1; dir += 2) {
                    if (c[a] + dir < 0 || c[a] + dir >= D[a]) continue;
                    const std::size_t j = i + (dir > 0 ? stride[a] : static_cast<std::size_t>(0) - stride[a]);
                    if (near.get(j)) { touches_shell = true; continue; }
                    if (visited.get(j)) continue;
                    visited.set(j, true);
                    ext.push_back(static_cast<std::uint32_t>(j));
                }
            if (touches_shell) shell_seeds.push_back(static_cast<std::uint32_t>(i));
            if (head > (1u << 22) && head * 2 > ext.size()) { ext.erase(ext.begin(), ext.begin() + static_cast<std::ptrdiff_t>(head)); head = 0; }
        }
    }
    queue = std::move(shell_seeds);   // already visited and outside (inside bits default to 0)
    flood();
    // Enclosed regions (not reached from the exterior, e.g. the interior of a wall sealed by the
    // active shell): the whole component is collected first with the parity of every member
    // relative to its start corner, then its sign is decided by a majority vote over all edges
    // into already visited corners. A single neighbour would be fragile: one leaked corner next
    // to a sealed wall interior would turn the whole wall inside out.
    {
        Bits member(total);
        std::vector<std::uint32_t> members;
        std::vector<std::uint8_t> rel;
        for (std::size_t start = 0; start < total; ++start) {
            if (visited.get(start)) continue;
            members.clear();
            rel.clear();
            members.push_back(static_cast<std::uint32_t>(start));
            rel.push_back(0);
            visited.set(start, true);
            member.set(start, true);
            std::size_t votes_inside = 0, votes_outside = 0, head = 0;
            while (head < members.size()) {
                const std::size_t i = members[head];
                const bool r = rel[head] != 0;
                ++head;
                const int z = static_cast<int>(i % static_cast<std::size_t>(D[2]));
                const int y = static_cast<int>((i / static_cast<std::size_t>(D[2])) % static_cast<std::size_t>(D[1]));
                const int x = static_cast<int>(i / stride[0]);
                const int c[3] = {x, y, z};
                for (int a = 0; a < 3; ++a)
                    for (int dir = -1; dir <= 1; dir += 2) {
                        int nb[3] = {x, y, z};
                        nb[a] += dir;
                        if (nb[a] < 0 || nb[a] >= D[a]) continue;
                        const std::size_t j = i + (dir > 0 ? stride[a] : static_cast<std::size_t>(0) - stride[a]);
                        const bool f = dir > 0 ? flagged(c[0], c[1], c[2], a) : flagged(nb[0], nb[1], nb[2], a);
                        if (!visited.get(j)) {
                            visited.set(j, true);
                            member.set(j, true);
                            members.push_back(static_cast<std::uint32_t>(j));
                            rel.push_back(static_cast<std::uint8_t>(r != f));
                        } else if (!member.get(j)) {
                            // start sign implied by this neighbour: sign(j) xor flag, undone by the member's parity
                            const bool start_inside = (inside.get(j) != f) != r;
                            if (start_inside) ++votes_inside; else ++votes_outside;
                        }
                    }
            }
            const bool start_sign = votes_inside > votes_outside;
            for (std::size_t k = 0; k < members.size(); ++k) {
                inside.set(members[k], start_sign != (rel[k] != 0));
                member.set(members[k], false);
            }
        }
    }
    std::vector<std::uint32_t>().swap(queue);

    auto sign_at = [&](int x, int y, int z) { return inside.get(index(x, y, z)); };   // box corner coordinates

    // Sheets per active voxel from the sign changes on its 12 edges.
    std::vector<std::uint8_t> sheet(n * 12, 0xFF), sheets(n, 1);
    std::vector<std::uint32_t> base(n);
    std::uint32_t next_vertex = 0;
    for (std::size_t u = 0; u < n; ++u) {
        const int bx = coords[u].x - o[0], by = coords[u].y - o[1], bz = coords[u].z - o[2];   // box corner of the cube
        bool crossed[12];
        int parent[12];
        for (int slot = 0; slot < 12; ++slot) {
            const int a = slot / 4;
            const int c0[3] = {bx + T.start[slot][0], by + T.start[slot][1], bz + T.start[slot][2]};
            int c1[3] = {c0[0], c0[1], c0[2]};
            c1[a] += 1;
            crossed[slot] = sign_at(c0[0], c0[1], c0[2]) != sign_at(c1[0], c1[1], c1[2]);
            parent[slot] = slot;
        }
        auto unite = [&](int a, int b) { parent[Root(parent, a)] = Root(parent, b); };
        for (int f = 0; f < 6; ++f) {
            int m = 0;
            for (int i = 0; i < 4; ++i) m += crossed[T.ring[f][i]];
            if (m == 2) {
                int r[2], k = 0;
                for (int i = 0; i < 4; ++i)
                    if (crossed[T.ring[f][i]]) r[k++] = T.ring[f][i];
                unite(r[0], r[1]);
            } else if (m == 4) {
                // Alternating face: keep the two inside corners apart (pair around each of them).
                const int* cc = T.ring_corner[f][0];
                const int i0 = sign_at(bx + cc[0], by + cc[1], bz + cc[2]) ? 0 : 1;
                unite(T.ring[f][i0], T.ring[f][i0 + 1]);
                unite(T.ring[f][(i0 + 2) % 4], T.ring[f][(i0 + 3) % 4]);
            }
        }
        int id_of_root[12];
        std::fill(id_of_root, id_of_root + 12, -1);
        int count = 0;
        for (int slot = 0; slot < 12; ++slot) {
            if (!crossed[slot]) continue;
            const int root = Root(parent, slot);
            if (id_of_root[root] < 0) id_of_root[root] = count++;
            sheet[u * 12 + static_cast<std::size_t>(slot)] = static_cast<std::uint8_t>(id_of_root[root]);
        }
        sheets[u] = static_cast<std::uint8_t>(std::max(count, 1));
        base[u] = next_vertex;
        next_vertex += sheets[u];
    }
    st.extra_vertices = next_vertex - static_cast<std::uint32_t>(n);
    mesh.vertices.resize(next_vertex);
    for (std::size_t u = 0; u < n; ++u) {
        const MeshVertex p{(static_cast<float>(coords[u].x) + in.dual[u * 3]) * voxel + in.aabb_min,
                           (static_cast<float>(coords[u].y) + in.dual[u * 3 + 1]) * voxel + in.aabb_min,
                           (static_cast<float>(coords[u].z) + in.dual[u * 3 + 2]) * voxel + in.aabb_min};
        for (std::uint8_t s = 0; s < sheets[u]; ++s) mesh.vertices[base[u] + s] = p;
    }

    // Virtual voxels (not active, but a corner of a sign-change edge) sit at their centre.
    std::unordered_map<std::uint64_t, std::uint32_t> virtual_ids;
    auto virtual_vertex = [&](int vx, int vy, int vz) {
        const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(vx + 4)) << 42) |
                                  (static_cast<std::uint64_t>(static_cast<std::uint32_t>(vy + 4)) << 21) | static_cast<std::uint32_t>(vz + 4);
        auto it = virtual_ids.find(key);
        if (it != virtual_ids.end()) return it->second;
        const auto id = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({(static_cast<float>(vx) + 0.5f) * voxel + in.aabb_min, (static_cast<float>(vy) + 0.5f) * voxel + in.aabb_min,
                                 (static_cast<float>(vz) + 0.5f) * voxel + in.aabb_min});
        virtual_ids.emplace(key, id);
        return id;
    };

    // Quads: one per edge with a sign change, in the order of the owner voxel's box position.
    for (int x = 0; x < D[0]; ++x)
        for (int y = 0; y < D[1]; ++y)
            for (int z = 0; z < D[2]; ++z) {
                const int c[3] = {x, y, z};
                const bool s0 = sign_at(x, y, z);
                for (int a = 0; a < 3; ++a) {
                    if (c[a] + 1 >= D[a]) continue;
                    int c1[3] = {x, y, z};
                    c1[a] += 1;
                    const bool s1 = sign_at(c1[0], c1[1], c1[2]);
                    const bool f = flagged(x, y, z, a);
                    if ((s0 != s1) != f) ++st.seam_edges;
                    if (s0 == s1) continue;
                    ++st.quads;
                    // Owner voxel (absolute) = corner minus one in the other axes.
                    const int vx = o[0] + x - (a != 0), vy = o[1] + y - (a != 1), vz = o[2] + z - (a != 2);
                    std::uint32_t q[4];
                    float w[4];
                    for (int k = 0; k < 4; ++k) {
                        const int ux = vx + off[a][k][0], uy = vy + off[a][k][1], uz = vz + off[a][k][2];
                        const std::int64_t row = (ux < 0 || uy < 0 || uz < 0) ? -1 : map.Find(ux, uy, uz);
                        if (row >= 0) {
                            const auto u = static_cast<std::size_t>(row);
                            q[k] = base[u] + sheet[u * 12 + static_cast<std::size_t>(a * 4 + k)];
                            w[k] = in.split_weight ? in.split_weight[u] : 1.f;
                        } else {
                            q[k] = virtual_vertex(ux, uy, uz);
                            w[k] = 1.f;
                        }
                    }
                    // Table order faces -axis: right when the outside is at the lower corner.
                    if (s0) std::swap(q[1], q[3]);
                    bool split_02;
                    if (in.split_weight) {
                        split_02 = w[0] * w[2] > w[1] * w[3];
                    } else {
                        auto normal = [&](std::uint32_t pa, std::uint32_t pb, std::uint32_t pc, float* out) {
                            const MeshVertex& A = mesh.vertices[pa];
                            const MeshVertex& B = mesh.vertices[pb];
                            const MeshVertex& C = mesh.vertices[pc];
                            const float uu[3] = {B.x - A.x, B.y - A.y, B.z - A.z}, vv[3] = {C.x - A.x, C.y - A.y, C.z - A.z};
                            out[0] = uu[1] * vv[2] - uu[2] * vv[1]; out[1] = uu[2] * vv[0] - uu[0] * vv[2]; out[2] = uu[0] * vv[1] - uu[1] * vv[0];
                        };
                        float n0[3], n1[3], m0[3], m1[3];
                        normal(q[0], q[1], q[2], n0); normal(q[1], q[2], q[3], n1);
                        normal(q[0], q[1], q[3], m0); normal(q[1], q[2], q[3], m1);
                        const float align0 = std::fabs(n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]);
                        const float align1 = std::fabs(m0[0] * m1[0] + m0[1] * m1[1] + m0[2] * m1[2]);
                        split_02 = align0 > align1;
                    }
                    if (split_02) {
                        mesh.faces.push_back({q[0], q[1], q[2]});
                        mesh.faces.push_back({q[0], q[2], q[3]});
                    } else {
                        mesh.faces.push_back({q[0], q[1], q[3]});
                        mesh.faces.push_back({q[3], q[1], q[2]});
                    }
                }
            }
    st.virtual_vertices = virtual_ids.size();
    if (stats) {
        st.open_edges = count_boundary_edges(mesh);
        *stats = st;
    }
    return mesh;
}

namespace {
// Sorted undirected edge keys of all triangles.
std::vector<std::uint64_t> SortedEdges(const Mesh& mesh) {
    std::vector<std::uint64_t> edges;
    edges.reserve(mesh.faces.size() * 3);
    for (const auto& f : mesh.faces)
        for (int e = 0; e < 3; ++e) {
            const std::uint32_t a = f[static_cast<std::size_t>(e)], b = f[static_cast<std::size_t>((e + 1) % 3)];
            edges.push_back((static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b));
        }
    std::sort(edges.begin(), edges.end());
    return edges;
}
}  // namespace

std::size_t count_boundary_edges(const Mesh& mesh) {
    // A boundary edge appears exactly once.
    const std::vector<std::uint64_t> edges = SortedEdges(mesh);
    std::size_t open = 0;
    for (std::size_t i = 0; i < edges.size();) {
        std::size_t j = i + 1;
        while (j < edges.size() && edges[j] == edges[i]) ++j;
        open += (j - i) == 1;
        i = j;
    }
    return open;
}

std::size_t max_edge_multiplicity(const Mesh& mesh) {
    const std::vector<std::uint64_t> edges = SortedEdges(mesh);
    std::size_t best = 0;
    for (std::size_t i = 0; i < edges.size();) {
        std::size_t j = i + 1;
        while (j < edges.size() && edges[j] == edges[i]) ++j;
        best = std::max(best, j - i);
        i = j;
    }
    return best;
}

}  // namespace gufo::trellis2
