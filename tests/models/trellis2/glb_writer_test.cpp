// CPU tests for src/models/trellis2/glb_writer.*: a tetrahedron is written as GLB, the container is checked byte-wise (magic, version,
// 4-byte alignment, chunk types/lengths), the JSON chunk is parsed and checked
// against the glTF 2.0 core requirements we rely on (asset.version, accessor
// min/max, bufferView targets, triangle mode), and read_glb() restores the same
// vertices and faces. OBJ output is checked line-wise.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/trellis2/glb_writer.hpp"

namespace tp = gufo::trellis2;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::uint32_t U32(const std::vector<std::uint8_t>& b, std::size_t at) {
    return static_cast<std::uint32_t>(b[at]) | (static_cast<std::uint32_t>(b[at + 1]) << 8) |
           (static_cast<std::uint32_t>(b[at + 2]) << 16) | (static_cast<std::uint32_t>(b[at + 3]) << 24);
}

tp::Mesh Tetrahedron() {
    tp::Mesh m;
    m.vertices = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};
    // Outward-facing winding (positive signed volume 1/6).
    m.faces = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    return m;
}

bool SameMesh(const tp::Mesh& a, const tp::Mesh& b) {
    if (a.vertices.size() != b.vertices.size() || a.faces.size() != b.faces.size()) return false;
    for (std::size_t i = 0; i < a.vertices.size(); ++i)
        if (a.vertices[i].x != b.vertices[i].x || a.vertices[i].y != b.vertices[i].y ||
            a.vertices[i].z != b.vertices[i].z)
            return false;
    for (std::size_t i = 0; i < a.faces.size(); ++i)
        if (a.faces[i] != b.faces[i]) return false;
    return true;
}

void ContainerTest(const tp::Mesh& mesh, bool normals, const std::string& tag) {
    tp::GlbWriteOptions opt;
    opt.write_normals = normals;
    const std::vector<std::uint8_t> glb = tp::write_glb(mesh, opt);
    Expect(glb.size() >= 28, tag + ": size");
    Expect(U32(glb, 0) == 0x46546C67u, tag + ": magic glTF");
    Expect(U32(glb, 4) == 2u, tag + ": version 2");
    Expect(U32(glb, 8) == glb.size(), tag + ": total length");
    Expect(glb.size() % 4 == 0, tag + ": 4-byte aligned file");
    const std::uint32_t json_len = U32(glb, 12);
    Expect(U32(glb, 16) == 0x4E4F534Au, tag + ": first chunk JSON");
    Expect(json_len % 4 == 0, tag + ": JSON chunk padded");
    const std::size_t bin_at = 20 + json_len;
    Expect(bin_at + 8 <= glb.size(), tag + ": BIN header present");
    const std::uint32_t bin_len = U32(glb, bin_at);
    Expect(U32(glb, bin_at + 4) == 0x004E4942u, tag + ": second chunk BIN");
    Expect(bin_len % 4 == 0 && bin_at + 8 + bin_len == glb.size(), tag + ": BIN chunk ends the file");
    for (std::size_t i = 20 + json_len; i-- > 20;) {
        if (glb[i] != ' ') { Expect(glb[i] == '}', tag + ": JSON ends with } then spaces"); break; }
    }

    const std::string json_text(reinterpret_cast<const char*>(glb.data()) + 20, json_len);
    gufo::json::Value root;
    bool parsed = true;
    try { root = gufo::json::parse(json_text); } catch (...) { parsed = false; }
    Expect(parsed, tag + ": JSON parses");
    if (!parsed) return;
    Expect(root.find("asset") && root.find("asset")->member_str("version") == "2.0", tag + ": asset.version 2.0");
    Expect(root.find("asset")->member_str("generator").find("gufo") != std::string::npos, tag + ": generator");
    Expect(root.find("buffers") && root.find("buffers")->size() == 1 &&
               root.find("buffers")->items()[0].member_size("byteLength") == bin_len,
           tag + ": buffer length equals BIN chunk");
    Expect(root.find("scene") && root.find("scenes") && root.find("nodes") && root.find("meshes"),
           tag + ": scene/nodes/meshes present");
    const gufo::json::Value* accessors = root.find("accessors");
    const gufo::json::Value* views = root.find("bufferViews");
    Expect(accessors && views, tag + ": accessors + bufferViews");
    const std::size_t expected_accessors = normals ? 3 : 2;
    Expect(accessors->size() == expected_accessors && views->size() == expected_accessors, tag + ": accessor count");
    const gufo::json::Value& prim = root.find("meshes")->items()[0].find("primitives")->items()[0];
    Expect(prim.member_size("mode", 4) == 4, tag + ": TRIANGLES");
    const gufo::json::Value* attrs = prim.find("attributes");
    Expect(attrs && attrs->find("POSITION"), tag + ": POSITION attribute");
    Expect((attrs->find("NORMAL") != nullptr) == normals, tag + ": NORMAL attribute iff requested");
    const gufo::json::Value& pos = accessors->items()[attrs->find("POSITION")->as_size()];
    Expect(pos.member_size("componentType") == 5126 && pos.member_str("type") == "VEC3" &&
               pos.member_size("count") == mesh.vertices.size(),
           tag + ": POSITION accessor FLOAT VEC3 count");
    Expect(pos.find("min") && pos.find("max") && pos.find("min")->size() == 3 && pos.find("max")->size() == 3,
           tag + ": POSITION min/max");
    float bmin[3], bmax[3];
    if (tp::bounding_box(mesh, bmin, bmax)) {
        for (std::size_t k = 0; k < 3; ++k) {
            Expect(std::fabs(pos.find("min")->items()[k].as_double() - bmin[k]) < 1e-6 &&
                       std::fabs(pos.find("max")->items()[k].as_double() - bmax[k]) < 1e-6,
                   tag + ": min/max axis " + std::to_string(k));
        }
    }
    const gufo::json::Value& idx = accessors->items()[prim.find("indices")->as_size()];
    Expect(idx.member_size("componentType") == 5125 && idx.member_str("type") == "SCALAR" &&
               idx.member_size("count") == mesh.faces.size() * 3,
           tag + ": index accessor UNSIGNED_INT SCALAR");
    const gufo::json::Value& idx_view = views->items()[idx.member_size("bufferView")];
    Expect(idx_view.member_size("target") == 34963, tag + ": ELEMENT_ARRAY_BUFFER target");
    const gufo::json::Value& pos_view = views->items()[pos.member_size("bufferView")];
    Expect(pos_view.member_size("target") == 34962, tag + ": ARRAY_BUFFER target");
    Expect(pos_view.member_size("byteOffset") % 4 == 0 && idx_view.member_size("byteOffset") % 4 == 0,
           tag + ": bufferView offsets aligned");
    Expect(pos_view.member_size("byteLength") == mesh.vertices.size() * 12, tag + ": position bytes");
    Expect(idx_view.member_size("byteLength") == mesh.faces.size() * 12, tag + ": index bytes");

    // Round trip through our reader.
    tp::Mesh back;
    std::string error;
    std::string json_seen;
    Expect(tp::read_glb(glb, &back, &error, &json_seen), tag + ": read_glb ok (" + error + ")");
    Expect(json_seen == json_text, tag + ": read_glb returns the JSON chunk");
    Expect(SameMesh(mesh, back), tag + ": round trip restores vertices and faces");

    if (normals) {
        const auto n = tp::vertex_normals(mesh);
        Expect(n.size() == mesh.vertices.size(), tag + ": one normal per vertex");
        bool unit = true;
        for (const auto& v : n) {
            const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            if (std::fabs(len - 1.f) > 1e-4f) unit = false;
        }
        Expect(unit, tag + ": normals unit length");
    }
}

void ObjTest() {
    const std::string obj = tp::write_obj(Tetrahedron());
    std::istringstream in(obj);
    std::string line;
    int v = 0, f = 0;
    std::string first_v, first_f;
    while (std::getline(in, line)) {
        if (line.rfind("v ", 0) == 0) { if (!v) first_v = line; ++v; }
        if (line.rfind("f ", 0) == 0) { if (!f) first_f = line; ++f; }
    }
    Expect(v == 4 && f == 4, "obj: 4 vertices, 4 faces");
    Expect(first_v == "v 0 0 0", "obj: first vertex line (" + first_v + ")");
    Expect(first_f == "f 1 3 2", "obj: 1-based face indices (" + first_f + ")");
}

void MalformedTest() {
    tp::Mesh m;
    std::string err;
    const std::vector<std::uint8_t> junk = {1, 2, 3};
    Expect(!tp::read_glb(junk, &m, &err) && !err.empty(), "malformed: too short");
    std::vector<std::uint8_t> glb = tp::write_glb(Tetrahedron());
    glb[0] = 'x';
    Expect(!tp::read_glb(glb, &m, &err), "malformed: bad magic");
    glb = tp::write_glb(Tetrahedron());
    glb.resize(glb.size() - 8);  // truncated BIN chunk
    Expect(!tp::read_glb(glb, &m, &err), "malformed: truncated");
}

}  // namespace

int main() {
    ContainerTest(Tetrahedron(), true, "tetra");
    ContainerTest(Tetrahedron(), false, "tetra no normals");
    ObjTest();
    MalformedTest();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "glb_writer_test: ok\n";
    return 0;
}
