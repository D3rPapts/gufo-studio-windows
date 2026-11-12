// TRELLIS.2 (from the TripoSG port): minimal glTF 2.0 binary (GLB) and OBJ writers plus a GLB reader.
// glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/glb_writer.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "src/core/json.hpp"

namespace gufo::trellis2 {
namespace {

constexpr std::uint32_t kGlbMagic = 0x46546C67;  // "glTF"
constexpr std::uint32_t kChunkJson = 0x4E4F534A;  // "JSON"
constexpr std::uint32_t kChunkBin = 0x004E4942;   // "BIN\0"
constexpr int kComponentFloat = 5126;
constexpr int kComponentUnsignedInt = 5125;
constexpr int kComponentUnsignedShort = 5123;
constexpr int kTargetArrayBuffer = 34962;
constexpr int kTargetElementArrayBuffer = 34963;

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
}

void put_f32(std::vector<std::uint8_t>& out, float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    put_u32(out, bits);
}

std::uint32_t get_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

float get_f32(const std::uint8_t* p) {
    const std::uint32_t bits = get_u32(p);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void pad_to(std::vector<std::uint8_t>& out, std::size_t align, std::uint8_t fill) {
    while (out.size() % align != 0) out.push_back(fill);
}

}  // namespace

std::vector<MeshVertex> vertex_normals(const Mesh& mesh) {
    std::vector<MeshVertex> normals(mesh.vertices.size(), MeshVertex{0.f, 0.f, 0.f});
    for (const auto& f : mesh.faces) {
        const MeshVertex& a = mesh.vertices[f[0]];
        const MeshVertex& b = mesh.vertices[f[1]];
        const MeshVertex& c = mesh.vertices[f[2]];
        const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
        const float vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        // Cross product length is twice the triangle area: area weighting for free.
        const float nx = uy * vz - uz * vy;
        const float ny = uz * vx - ux * vz;
        const float nz = ux * vy - uy * vx;
        for (int k = 0; k < 3; ++k) {
            MeshVertex& n = normals[f[static_cast<std::size_t>(k)]];
            n.x += nx; n.y += ny; n.z += nz;
        }
    }
    for (auto& n : normals) {
        const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (len > 0.f) { n.x /= len; n.y /= len; n.z /= len; }
    }
    return normals;
}

std::vector<std::uint8_t> write_glb(const Mesh& mesh, const GlbWriteOptions& options) {
    using gufo::json::Value;

    // --- BIN chunk -------------------------------------------------------
    std::vector<std::uint8_t> bin;
    bin.reserve(mesh.vertices.size() * 24 + mesh.faces.size() * 12);

    float pmin[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max()};
    float pmax[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                     std::numeric_limits<float>::lowest()};
    const std::size_t positions_offset = bin.size();
    for (const auto& v : mesh.vertices) {
        put_f32(bin, v.x); put_f32(bin, v.y); put_f32(bin, v.z);
        const float c[3] = {v.x, v.y, v.z};
        for (int k = 0; k < 3; ++k) {
            pmin[k] = std::min(pmin[k], c[k]);
            pmax[k] = std::max(pmax[k], c[k]);
        }
    }
    const std::size_t positions_bytes = bin.size() - positions_offset;
    if (mesh.vertices.empty()) pmin[0] = pmin[1] = pmin[2] = pmax[0] = pmax[1] = pmax[2] = 0.f;

    std::size_t normals_offset = 0, normals_bytes = 0;
    if (options.write_normals) {
        normals_offset = bin.size();
        for (const auto& n : vertex_normals(mesh)) { put_f32(bin, n.x); put_f32(bin, n.y); put_f32(bin, n.z); }
        normals_bytes = bin.size() - normals_offset;
    }

    const std::size_t indices_offset = bin.size();
    for (const auto& f : mesh.faces) { put_u32(bin, f[0]); put_u32(bin, f[1]); put_u32(bin, f[2]); }
    const std::size_t indices_bytes = bin.size() - indices_offset;
    pad_to(bin, 4, 0x00);

    // --- JSON chunk ------------------------------------------------------
    Value root = Value::object();
    Value asset = Value::object();
    asset["version"] = Value("2.0");
    asset["generator"] = Value(options.generator);
    root["asset"] = asset;

    Value buffers = Value::array();
    Value buffer = Value::object();
    buffer["byteLength"] = Value(static_cast<unsigned long long>(bin.size()));
    buffers.push_back(buffer);
    root["buffers"] = buffers;

    Value views = Value::array();
    Value accessors = Value::array();
    auto add_view = [&](std::size_t offset, std::size_t length, int target) {
        Value v = Value::object();
        v["buffer"] = Value(0);
        v["byteOffset"] = Value(static_cast<unsigned long long>(offset));
        v["byteLength"] = Value(static_cast<unsigned long long>(length));
        v["target"] = Value(target);
        views.push_back(v);
        return static_cast<int>(views.size() - 1);
    };
    auto add_accessor = [&](int view, int component, std::size_t count, const char* type) {
        Value a = Value::object();
        a["bufferView"] = Value(view);
        a["byteOffset"] = Value(0);
        a["componentType"] = Value(component);
        a["count"] = Value(static_cast<unsigned long long>(count));
        a["type"] = Value(type);
        accessors.push_back(a);
        return static_cast<int>(accessors.size() - 1);
    };

    const int pos_view = add_view(positions_offset, positions_bytes, kTargetArrayBuffer);
    const int pos_acc = add_accessor(pos_view, kComponentFloat, mesh.vertices.size(), "VEC3");
    {
        // min/max are required for POSITION accessors.
        Value& acc = const_cast<Value&>(accessors.items()[static_cast<std::size_t>(pos_acc)]);
        Value vmin = Value::array(), vmax = Value::array();
        for (int k = 0; k < 3; ++k) { vmin.push_back(Value(static_cast<double>(pmin[k]))); vmax.push_back(Value(static_cast<double>(pmax[k]))); }
        acc["min"] = vmin;
        acc["max"] = vmax;
    }
    int nrm_acc = -1;
    if (options.write_normals) {
        const int nrm_view = add_view(normals_offset, normals_bytes, kTargetArrayBuffer);
        nrm_acc = add_accessor(nrm_view, kComponentFloat, mesh.vertices.size(), "VEC3");
    }
    const int idx_view = add_view(indices_offset, indices_bytes, kTargetElementArrayBuffer);
    const int idx_acc = add_accessor(idx_view, kComponentUnsignedInt, mesh.faces.size() * 3, "SCALAR");

    root["bufferViews"] = views;
    root["accessors"] = accessors;

    Value attributes = Value::object();
    attributes["POSITION"] = Value(pos_acc);
    if (nrm_acc >= 0) attributes["NORMAL"] = Value(nrm_acc);
    Value primitive = Value::object();
    primitive["attributes"] = attributes;
    primitive["indices"] = Value(idx_acc);
    primitive["mode"] = Value(4);  // TRIANGLES
    Value primitives = Value::array();
    primitives.push_back(primitive);
    Value mesh_obj = Value::object();
    mesh_obj["name"] = Value(options.mesh_name);
    mesh_obj["primitives"] = primitives;
    Value meshes = Value::array();
    meshes.push_back(mesh_obj);
    root["meshes"] = meshes;

    Value node = Value::object();
    node["mesh"] = Value(0);
    node["name"] = Value(options.mesh_name);
    Value nodes = Value::array();
    nodes.push_back(node);
    root["nodes"] = nodes;

    Value scene = Value::object();
    Value scene_nodes = Value::array();
    scene_nodes.push_back(Value(0));
    scene["nodes"] = scene_nodes;
    Value scenes = Value::array();
    scenes.push_back(scene);
    root["scenes"] = scenes;
    root["scene"] = Value(0);

    std::string json_text = root.dump();
    while (json_text.size() % 4 != 0) json_text.push_back(' ');

    // --- container -------------------------------------------------------
    std::vector<std::uint8_t> out;
    const std::size_t total = 12 + 8 + json_text.size() + 8 + bin.size();
    out.reserve(total);
    put_u32(out, kGlbMagic);
    put_u32(out, 2);
    put_u32(out, static_cast<std::uint32_t>(total));
    put_u32(out, static_cast<std::uint32_t>(json_text.size()));
    put_u32(out, kChunkJson);
    out.insert(out.end(), json_text.begin(), json_text.end());
    put_u32(out, static_cast<std::uint32_t>(bin.size()));
    put_u32(out, kChunkBin);
    out.insert(out.end(), bin.begin(), bin.end());
    return out;
}

bool write_glb_file(const std::filesystem::path& path, const Mesh& mesh, std::string* error,
                    const GlbWriteOptions& options) {
    const std::vector<std::uint8_t> bytes = write_glb(mesh, options);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) *error = "cannot open " + path.string() + " for writing";
        return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        if (error) *error = "write failed for " + path.string();
        return false;
    }
    return true;
}

std::string write_obj(const Mesh& mesh) {
    std::ostringstream o;
    o.precision(7);
    o << "# gufo triposg\n";
    for (const auto& v : mesh.vertices) o << "v " << v.x << ' ' << v.y << ' ' << v.z << '\n';
    for (const auto& f : mesh.faces) o << "f " << f[0] + 1 << ' ' << f[1] + 1 << ' ' << f[2] + 1 << '\n';
    return o.str();
}

bool read_glb(std::span<const std::uint8_t> bytes, Mesh* mesh, std::string* error,
              std::string* json_text_out) {
    auto fail = [&](const std::string& what) {
        if (error) *error = what;
        return false;
    };
    if (bytes.size() < 20) return fail("GLB too short");
    if (get_u32(bytes.data()) != kGlbMagic) return fail("not a GLB (bad magic)");
    if (get_u32(bytes.data() + 4) != 2) return fail("unsupported GLB version");
    const std::uint32_t total = get_u32(bytes.data() + 8);
    if (total > bytes.size()) return fail("GLB length exceeds data");

    std::string json_text;
    std::span<const std::uint8_t> bin;
    std::size_t pos = 12;
    while (pos + 8 <= total) {
        const std::uint32_t len = get_u32(bytes.data() + pos);
        const std::uint32_t type = get_u32(bytes.data() + pos + 4);
        pos += 8;
        if (pos + len > total) return fail("GLB chunk exceeds data");
        if (type == kChunkJson) json_text.assign(reinterpret_cast<const char*>(bytes.data() + pos), len);
        else if (type == kChunkBin) bin = bytes.subspan(pos, len);
        pos += len;
    }
    if (json_text.empty()) return fail("GLB has no JSON chunk");
    if (json_text_out) *json_text_out = json_text;

    gufo::json::Value root;
    try {
        root = gufo::json::parse(json_text);
    } catch (const std::exception& e) {
        return fail(std::string("GLB JSON: ") + e.what());
    }
    const gufo::json::Value* meshes = root.find("meshes");
    if (!meshes || !meshes->is_array() || meshes->empty()) return fail("GLB has no meshes");
    const gufo::json::Value* prims = meshes->items()[0].find("primitives");
    if (!prims || !prims->is_array() || prims->empty()) return fail("mesh has no primitives");
    const gufo::json::Value& prim = prims->items()[0];
    if (prim.find("mode") && prim.find("mode")->as_size(4) != 4) return fail("primitive is not TRIANGLES");
    const gufo::json::Value* attrs = prim.find("attributes");
    if (!attrs || !attrs->find("POSITION")) return fail("primitive has no POSITION");
    const gufo::json::Value* accessors = root.find("accessors");
    const gufo::json::Value* views = root.find("bufferViews");
    if (!accessors || !views) return fail("GLB has no accessors/bufferViews");

    struct Region { const std::uint8_t* data; std::size_t count; int component; std::size_t stride; };
    auto resolve = [&](std::size_t acc_index, Region* r) -> bool {
        if (acc_index >= accessors->size()) return false;
        const gufo::json::Value& acc = accessors->items()[acc_index];
        const std::size_t view_index = acc.member_size("bufferView", static_cast<std::size_t>(-1));
        if (view_index >= views->size()) return false;
        const gufo::json::Value& view = views->items()[view_index];
        if (view.member_size("buffer", 0) != 0) return false;  // only the embedded buffer
        const std::size_t offset = view.member_size("byteOffset", 0) + acc.member_size("byteOffset", 0);
        const std::size_t length = view.member_size("byteLength", 0);
        if (offset + length > bin.size() + view.member_size("byteOffset", 0)) return false;
        if (offset > bin.size()) return false;
        r->data = bin.data() + offset;
        r->count = acc.member_size("count", 0);
        r->component = static_cast<int>(acc.member_size("componentType", 0));
        r->stride = view.member_size("byteStride", 0);
        return true;
    };

    Region pos_region{};
    if (!resolve(attrs->find("POSITION")->as_size(), &pos_region)) return fail("bad POSITION accessor");
    if (pos_region.component != kComponentFloat) return fail("POSITION is not FLOAT");
    const std::size_t pos_stride = pos_region.stride ? pos_region.stride : 12;
    if (pos_region.data + pos_region.count * pos_stride > bin.data() + bin.size() && pos_region.count)
        return fail("POSITION data exceeds BIN chunk");

    Mesh out;
    out.vertices.reserve(pos_region.count);
    for (std::size_t i = 0; i < pos_region.count; ++i) {
        const std::uint8_t* p = pos_region.data + i * pos_stride;
        out.vertices.push_back(MeshVertex{get_f32(p), get_f32(p + 4), get_f32(p + 8)});
    }

    const gufo::json::Value* indices = prim.find("indices");
    if (indices) {
        Region idx{};
        if (!resolve(indices->as_size(), &idx)) return fail("bad indices accessor");
        const std::size_t elem = idx.component == kComponentUnsignedInt ? 4 : idx.component == kComponentUnsignedShort ? 2 : 0;
        if (elem == 0) return fail("indices are neither UNSIGNED_INT nor UNSIGNED_SHORT");
        if (idx.count % 3 != 0) return fail("index count is not a multiple of 3");
        if (idx.data + idx.count * elem > bin.data() + bin.size()) return fail("index data exceeds BIN chunk");
        out.faces.reserve(idx.count / 3);
        for (std::size_t i = 0; i + 2 < idx.count; i += 3) {
            std::array<std::uint32_t, 3> f{};
            for (std::size_t k = 0; k < 3; ++k) {
                const std::uint8_t* p = idx.data + (i + k) * elem;
                f[k] = elem == 4 ? get_u32(p) : static_cast<std::uint32_t>(p[0] | (p[1] << 8));
                if (f[k] >= out.vertices.size()) return fail("index out of range");
            }
            out.faces.push_back(f);
        }
    } else {
        if (out.vertices.size() % 3 != 0) return fail("non-indexed vertex count is not a multiple of 3");
        for (std::uint32_t i = 0; i + 2 < out.vertices.size(); i += 3) out.faces.push_back({i, i + 1, i + 2});
    }
    *mesh = std::move(out);
    return true;
}

}  // namespace gufo::trellis2
