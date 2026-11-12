// CPU tests for src/models/trellis2/texture_bake.cpp: the unwrap of a cube gives UVs in [0, 1]
// for every face and source indices into the input; the voxel sampler interpolates between two
// voxel centres and falls back to the nearest voxel for far positions; baking one large
// triangle with a constant voxel colour fills its texels (and a dilated rim) with that colour
// and the metallic/roughness channels; the GLB writer embeds TEXCOORD_0, images, textures and a
// textured material that the reader accepts.
// AI-assisted (Claude Code), 2026-11-12.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/trellis2/glb_writer.hpp"
#include "src/models/trellis2/png.hpp"
#include "src/models/trellis2/texture_bake.hpp"

namespace tr = gufo::trellis2;

namespace {
int failures = 0;
void Expect(bool c, const std::string& what) {
    if (!c) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}
}  // namespace

int main() {
    // --- unwrap a cube
    tr::Mesh cube;
    cube.vertices = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    cube.faces = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5}, {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7}};
    tr::UvMesh uvm;
    std::string error;
    const bool ok = tr::unwrap_uv(cube, 256, &uvm, &error);
    Expect(ok, "unwrap succeeds: " + error);
    if (ok) {
        Expect(uvm.mesh.faces.size() == 12 && uvm.mesh.vertices.size() >= 8 && uvm.charts >= 1,
               "unwrap keeps 12 faces, splits seams (" + std::to_string(uvm.mesh.vertices.size()) + " vertices, " + std::to_string(uvm.charts) + " charts)");
        bool in_range = true, src_ok = true;
        for (std::size_t i = 0; i < uvm.mesh.vertices.size(); ++i) {
            in_range = in_range && uvm.uv[i * 2] >= 0.f && uvm.uv[i * 2] <= 1.f && uvm.uv[i * 2 + 1] >= 0.f && uvm.uv[i * 2 + 1] <= 1.f;
            src_ok = src_ok && uvm.source[i] < 8 && uvm.mesh.vertices[i].x == cube.vertices[uvm.source[i]].x;
        }
        Expect(in_range, "uv in [0, 1]");
        Expect(src_ok, "source indices map to the input vertices");
        Expect(uvm.atlas_width > 0 && uvm.atlas_height > 0, "atlas has a size (" + std::to_string(uvm.atlas_width) + "x" + std::to_string(uvm.atlas_height) + ")");
        std::cout << "atlas " << uvm.atlas_width << "x" << uvm.atlas_height << ", " << uvm.charts << " charts\n";
    }

    // --- sampler: two voxels at resolution 2, red and blue
    std::vector<tr::Coord> coords = {{0, 0, 0}, {1, 0, 0}};
    std::vector<float> attrs = {1, 0, 0, 0.2f, 0.4f, 1, 0, 0, 1, 0.6f, 0.8f, 1};
    tr::VoxelAttrSampler sampler(coords, attrs, 6, 2, 8);
    float a[6];
    Expect(sampler.Sample(-0.25f, -0.25f, -0.25f, a) && std::fabs(a[0] - 1.f) < 1e-5f && std::fabs(a[2]) < 1e-5f, "centre of voxel 0 is red");
    Expect(sampler.Sample(0.f, -0.25f, -0.25f, a) && std::fabs(a[0] - 0.5f) < 1e-5f && std::fabs(a[2] - 0.5f) < 1e-5f && std::fabs(a[3] - 0.4f) < 1e-5f,
           "midpoint between the centres mixes half/half (" + std::to_string(a[0]) + ")");
    Expect(sampler.Sample(0.49f, 0.49f, 0.49f, a) && std::fabs(a[2] - 1.f) < 1e-5f, "far position falls back to the nearest voxel (blue)");
    tr::VoxelAttrSampler tight(coords, attrs, 6, 2, 0);
    a[0] = -1.f;
    Expect(!tight.Sample(0.49f, 0.49f, 0.49f, a) && a[0] == -1.f, "no fallback radius -> no sample, output untouched");

    // --- bake: one triangle over the lower-left half of a 64x64 atlas, every position samples voxel 0
    tr::UvMesh tri;
    // all three corners at the centre of voxel 0, so every texel samples exactly that voxel
    tri.mesh.vertices = {{-0.25f, -0.25f, -0.25f}, {-0.25f, -0.25f, -0.25f}, {-0.25f, -0.25f, -0.25f}};
    tri.mesh.faces = {{0, 1, 2}};
    tri.uv = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f};
    tri.source = {0, 1, 2};
    const tr::BakedTextures bt = tr::bake_textures(tri, sampler, 64, 2);
    Expect(bt.size == 64 && bt.covered_texels > 64 * 64 * 0.45 && bt.covered_texels < 64 * 64 * 0.56 && bt.unsampled_texels == 0,
           "half the atlas is covered (" + std::to_string(bt.covered_texels) + ")");
    // inside texel (2, 2): red, alpha 255, roughness 0.4 -> G 102, metallic 0.2 -> B 51
    const std::size_t t = 2 * 64 + 2;
    Expect(bt.base_color[t * 4] == 255 && bt.base_color[t * 4 + 1] == 0 && bt.base_color[t * 4 + 3] == 255, "covered texel is red and opaque");
    Expect(bt.metallic_roughness[t * 3] == 0 && bt.metallic_roughness[t * 3 + 1] == 102 && bt.metallic_roughness[t * 3 + 2] == 51,
           "metallic/roughness texel (G roughness, B metallic) " + std::to_string(bt.metallic_roughness[t * 3 + 1]) + "/" + std::to_string(bt.metallic_roughness[t * 3 + 2]));
    // a texel just outside the diagonal is filled by the dilation, one far away stays black
    const std::size_t near = 33 * 64 + 32, far = 60 * 64 + 60;
    Expect(bt.base_color[near * 4] == 255, "dilation fills the rim");
    Expect(bt.base_color[far * 4] == 0 && bt.base_color[far * 4 + 3] == 255, "far texel stays empty");

    // --- GLB with textures
    if (ok) {
        const std::vector<std::uint8_t> base = tr::encode_png_rgba(bt.base_color.data(), bt.size, bt.size);
        const std::vector<std::uint8_t> mr = tr::encode_png_rgb(bt.metallic_roughness.data(), bt.size, bt.size);
        Expect(base.size() > 100 && base.size() < bt.base_color.size(), "RGBA PNG is compressed (" + std::to_string(base.size()) + " bytes)");
        tr::GlbWriteOptions go;
        go.texcoords = &uvm.uv;
        go.base_color_png = &base;
        go.metallic_roughness_png = &mr;
        const std::vector<std::uint8_t> glb = tr::write_glb(uvm.mesh, go);
        tr::Mesh back;
        std::string json;
        const bool read = tr::read_glb(glb, &back, &error, &json);
        Expect(read && back.faces.size() == 12 && back.vertices.size() == uvm.mesh.vertices.size(), "textured GLB reads back: " + error);
        Expect(json.find("TEXCOORD_0") != std::string::npos && json.find("\"images\"") != std::string::npos && json.find("baseColorTexture") != std::string::npos &&
               json.find("metallicRoughnessTexture") != std::string::npos && json.find("image/png") != std::string::npos && json.find("\"doubleSided\":true") != std::string::npos,
               "GLB JSON carries texcoords, images, textures and the PBR material");
        Expect(json.find("COLOR_0") == std::string::npos, "no vertex colours next to textures");
    }
    if (failures == 0) std::cout << "trellis2_texture_bake_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
