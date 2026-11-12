// TRELLIS.2 pipeline (see pipeline.hpp). HIP build only.
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

#include "src/models/trellis2/config.hpp"
#include "src/models/trellis2/dinov3.hpp"
#include "src/models/trellis2/dit.hpp"
#include "src/models/trellis2/dual_grid_mesh.hpp"
#include "src/models/trellis2/glb_writer.hpp"
#include "src/models/trellis2/mesh_post.hpp"
#include "src/models/trellis2/mesh_preview.hpp"
#include "src/models/trellis2/png.hpp"
#include "src/models/trellis2/preprocess.hpp"
#include "src/models/trellis2/scheduler.hpp"
#include "src/models/trellis2/sparse_conv.hpp"
#include "src/models/trellis2/sparse_dec.hpp"
#include "src/models/trellis2/ss_dec.hpp"

namespace gufo::trellis2 {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
bool cancelled(const std::atomic<bool>* cancel) { return cancel != nullptr && cancel->load(); }

std::vector<float> gaussian(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// Euler flow with interval CFG and x0 rescale; x in place. `extra` columns (texture flow:
// the normalised shape latent) are appended to x for every forward and never updated.
bool run_flow(DitHip& dit, std::vector<float>& x, int channels, const std::vector<float>* extra, int extra_channels,
              const std::vector<Coord>& coords, const float* cond, const float* zero_cond, int cond_tokens,
              const SamplerParams& p, int steps, const std::atomic<bool>* cancel, const std::function<void(int, int)>& tick,
              std::string* error) {
    const int N = static_cast<int>(coords.size());
    const std::vector<float> times = flow_times(steps, p.rescale_t);
    std::vector<float> input;
    auto make_input = [&]() -> const float* {
        if (!extra) return x.data();
        input.resize(static_cast<std::size_t>(N) * (channels + extra_channels));
        for (int i = 0; i < N; ++i) {
            std::copy(x.begin() + static_cast<std::ptrdiff_t>(i) * channels, x.begin() + static_cast<std::ptrdiff_t>(i + 1) * channels,
                      input.begin() + static_cast<std::ptrdiff_t>(i) * (channels + extra_channels));
            std::copy(extra->begin() + static_cast<std::ptrdiff_t>(i) * extra_channels, extra->begin() + static_cast<std::ptrdiff_t>(i + 1) * extra_channels,
                      input.begin() + static_cast<std::ptrdiff_t>(i) * (channels + extra_channels) + channels);
        }
        return input.data();
    };
    for (int i = 0; i < steps; ++i) {
        if (cancelled(cancel)) { if (error) *error = "cancelled"; return false; }
        const float t = times[static_cast<std::size_t>(i)], t_prev = times[static_cast<std::size_t>(i) + 1];
        std::vector<float> pos, neg;
        const float* in = make_input();
        if (!dit.Forward(in, coords, cond, cond_tokens, model_timestep(t), &pos, nullptr, error)) return false;
        if (guidance_active(t, p)) {
            if (!dit.Forward(in, coords, zero_cond, cond_tokens, model_timestep(t), &neg, nullptr, error)) return false;
            apply_guidance(&pos, neg, x, t, p);
        }
        euler_step(x, pos, t, t_prev);
        tick(i + 1, steps);
    }
    return true;
}


// Colours per mesh vertex from the per-voxel attributes [N, 6] (base colour 3, metallic,
// roughness, alpha) by voxel lookup; returns RGBA [vertices * 4] and the mean metallic /
// roughness of the used voxels.
std::vector<float> voxel_colours(const Mesh& mesh, const std::vector<Coord>& coords, const std::vector<float>& attrs,
                                 int resolution, float* metallic, float* roughness) {
    VoxelHashMap map;
    map.Build(coords);
    std::vector<float> out(mesh.vertices.size() * 4, 1.f);
    double m_sum = 0, r_sum = 0;
    std::size_t used = 0;
    const float voxel = 1.f / static_cast<float>(resolution);
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const MeshVertex& v = mesh.vertices[i];
        const int cx = static_cast<int>(std::floor((v.x - MeshConfig::aabb_min) / voxel));
        const int cy = static_cast<int>(std::floor((v.y - MeshConfig::aabb_min) / voxel));
        const int cz = static_cast<int>(std::floor((v.z - MeshConfig::aabb_min) / voxel));
        std::int64_t idx = map.Find(cx, cy, cz);
        for (int r = 1; r <= 2 && idx < 0; ++r)
            for (int dx = -r; dx <= r && idx < 0; ++dx)
                for (int dy = -r; dy <= r && idx < 0; ++dy)
                    for (int dz = -r; dz <= r && idx < 0; ++dz) idx = map.Find(cx + dx, cy + dy, cz + dz);
        if (idx < 0) { out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = 0.5f; continue; }
        const float* a = attrs.data() + static_cast<std::size_t>(idx) * 6;
        out[i * 4] = a[0]; out[i * 4 + 1] = a[1]; out[i * 4 + 2] = a[2]; out[i * 4 + 3] = 1.f;
        m_sum += a[3]; r_sum += a[4];
        ++used;
    }
    if (used) { *metallic = static_cast<float>(m_sum / used); *roughness = static_cast<float>(r_sum / used); }
    return out;
}

}  // namespace

struct Pipeline::Impl {
    Dinov3Weights dino_weights;
    std::unique_ptr<Dinov3Hip> dino;
    std::unique_ptr<DitHip> ss_flow, shape_flow, tex_flow;
    SsDecWeights ss_dec_weights;
    std::unique_ptr<SsDecHip> ss_dec;
    std::unique_ptr<SparseDecHip> shape_dec, tex_dec;
    std::size_t weight_bytes = 0;
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Pipeline::~Pipeline() = default;
std::size_t Pipeline::weight_bytes() const { return impl_->weight_bytes; }
bool Pipeline::has_texture() const { return impl_->tex_flow != nullptr && impl_->tex_dec != nullptr; }

std::unique_ptr<Pipeline> Pipeline::Load(const std::filesystem::path& dir, bool with_texture, std::string* error) {
    auto impl = std::make_unique<Impl>();
    Impl& m = *impl;
    auto file = [&](const char* part) { return dir / (std::string("trellis2-") + part + ".safetensors"); };
    if (!Dinov3Weights::Load(file("dinov3"), &m.dino_weights, error)) return nullptr;
    m.dino = Dinov3Hip::Create(m.dino_weights, PipelineConfig::image_cond_512, error);
    if (!m.dino) return nullptr;
    m.weight_bytes += m.dino->device_bytes();
    m.dino_weights = Dinov3Weights();   // host copy no longer needed
    DitShape ss_shape;
    ss_shape.in_channels = 8; ss_shape.out_channels = 8;
    m.ss_flow = DitHip::Create(file("ss_flow"), ss_shape, error);
    if (!m.ss_flow) return nullptr;
    if (!SsDecWeights::Load(file("ss_dec"), &m.ss_dec_weights, error)) return nullptr;
    m.ss_dec = SsDecHip::Create(m.ss_dec_weights, error);
    if (!m.ss_dec) return nullptr;
    m.weight_bytes += m.ss_dec->device_bytes();
    m.ss_dec_weights = SsDecWeights();
    DitShape shape_shape;
    shape_shape.in_channels = 32; shape_shape.out_channels = 32;
    m.shape_flow = DitHip::Create(file("shape_flow_512"), shape_shape, error);
    if (!m.shape_flow) return nullptr;
    SparseDecShape dec_shape;
    m.shape_dec = SparseDecHip::Create(file("shape_dec"), dec_shape, error);
    if (!m.shape_dec) return nullptr;
    if (with_texture) {
        DitShape tex_shape;
        tex_shape.in_channels = 64; tex_shape.out_channels = 32;
        m.tex_flow = DitHip::Create(file("tex_flow_512"), tex_shape, error);
        if (!m.tex_flow) return nullptr;
        SparseDecShape tex_dec_shape;
        tex_dec_shape.outputs = SparseDecConfig::tex_outputs;
        tex_dec_shape.pred_subdiv = false;
        m.tex_dec = SparseDecHip::Create(file("tex_dec"), tex_dec_shape, error);
        if (!m.tex_dec) return nullptr;
    }
    return std::unique_ptr<Pipeline>(new Pipeline(std::move(impl)));
}

bool Pipeline::Generate(const std::uint8_t* rgba, int width, int height, const GenerateOptions& o, const GenerateProgress& progress,
                        const std::atomic<bool>* cancel, GenerateResult* result, std::string* error) {
    Impl& m = *impl_;
    auto report = [&](const std::string& phase, int done, int total) { if (progress) progress(phase, done, total); };
    auto fail_cancel = [&]() { if (error) *error = "cancelled"; return false; };
    const int steps = o.steps > 0 ? o.steps : 12;
    if (o.resolution != 512) {
        if (error) *error = "TRELLIS.2: only resolution 512 is supported in this stage";
        return false;
    }
    // 1. encode
    auto t0 = Clock::now();
    report("encode", 0, 2);
    Preprocessed pre;
    if (!preprocess_rgba(rgba, width, height, PipelineConfig::image_cond_512, &pre, error)) return false;
    std::vector<float> cond;
    if (!m.dino->Encode(pre.pixels.data(), &cond, error)) return false;
    const int M = DinoV3Config::tokens_512;
    const std::vector<float> zero_cond(cond.size(), 0.f);
    result->seconds_encode = seconds_since(t0);
    report("encode", 2, 2);
    if (cancelled(cancel)) return fail_cancel();

    // 2. sparse structure
    t0 = Clock::now();
    const int R = SsDecConfig::latent_res, Cs = 8, Ns = R * R * R;
    std::vector<float> noise_cm = gaussian(static_cast<std::size_t>(Cs) * Ns, o.seed);
    std::vector<float> x(static_cast<std::size_t>(Ns) * Cs);
    for (int c = 0; c < Cs; ++c)
        for (int v = 0; v < Ns; ++v) x[static_cast<std::size_t>(v) * Cs + c] = noise_cm[static_cast<std::size_t>(c) * Ns + v];
    const std::vector<Coord> grid = dense_grid_coords(R);
    if (!run_flow(*m.ss_flow, x, Cs, nullptr, 0, grid, cond.data(), zero_cond.data(), M, kSsSampler, steps, cancel,
                  [&](int d, int t) { report("structure", d, t); }, error))
        return false;
    std::vector<float> logits;
    if (!m.ss_dec->Forward(x.data(), &logits, error)) return false;
    const std::vector<Coord> coords = occupancy_to_coords(logits.data(), 64, 32);
    result->structure_voxels = coords.size();
    result->seconds_structure = seconds_since(t0);
    if (coords.empty()) { if (error) *error = "no active voxels in the sparse structure"; return false; }
    if (cancelled(cancel)) return fail_cancel();

    // 3. shape flow
    t0 = Clock::now();
    const int N = static_cast<int>(coords.size()), Cl = 32;
    std::vector<float> slat = gaussian(static_cast<std::size_t>(N) * Cl, o.seed + 1);
    if (!run_flow(*m.shape_flow, slat, Cl, nullptr, 0, coords, cond.data(), zero_cond.data(), M, kShapeSampler, steps, cancel,
                  [&](int d, int t) { report("shape", d, t); }, error))
        return false;
    std::vector<float> slat_norm = slat;   // normalised (for the texture flow), slat -> denormalised
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < Cl; ++c) slat[static_cast<std::size_t>(i) * Cl + c] = slat[static_cast<std::size_t>(i) * Cl + c] * kShapeStd[c] + kShapeMean[c];
    result->seconds_shape = seconds_since(t0);
    if (cancelled(cancel)) return fail_cancel();

    // 4. shape decoder
    t0 = Clock::now();
    report("decode", 0, 1);
    SparseDecResult dec;
    if (!m.shape_dec->Forward(slat.data(), coords, &dec, error)) return false;
    result->shape_voxels = dec.coords.size();
    result->seconds_decode = seconds_since(t0);
    report("decode", 1, 1);
    if (cancelled(cancel)) return fail_cancel();

    // 5. texture (optional): attributes per voxel of the final level
    std::vector<float> attrs;   // [M, 6] base colour, metallic, roughness, alpha
    if (o.texture && has_texture()) {
        t0 = Clock::now();
        std::vector<float> tex = gaussian(static_cast<std::size_t>(N) * Cl, o.seed + 2);
        if (!run_flow(*m.tex_flow, tex, Cl, &slat_norm, Cl, coords, cond.data(), zero_cond.data(), M, kTexSampler, steps, cancel,
                      [&](int d, int t) { report("texture", d, t); }, error))
            return false;
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < Cl; ++c) tex[static_cast<std::size_t>(i) * Cl + c] = tex[static_cast<std::size_t>(i) * Cl + c] * kTexStd[c] + kTexMean[c];
        SparseDecOptions guided;
        guided.guide_masks = &dec.masks;
        SparseDecResult tdec;
        if (!m.tex_dec->Forward(tex.data(), coords, &tdec, error, guided)) return false;
        if (tdec.coords != dec.coords) { if (error) *error = "texture decoder voxels differ from the shape decoder"; return false; }
        attrs = std::move(tdec.outputs);
        for (auto& v : attrs) v = std::clamp(v * 0.5f + 0.5f, 0.f, 1.f);
        result->seconds_texture = seconds_since(t0);
        if (cancelled(cancel)) return fail_cancel();
    }

    // 6. mesh
    t0 = Clock::now();
    report("mesh", 0, 3);
    std::vector<float> dual, wgt;
    std::vector<std::uint8_t> inter;
    decode_shape_outputs(dec.outputs.data(), dec.coords.size(), &dual, &inter, &wgt);
    DualGridInput in;
    in.coords = &dec.coords; in.dual = dual.data(); in.intersected = inter.data(); in.split_weight = wgt.data();
    in.resolution = o.resolution;
    DualGridStats st;
    Mesh mesh = dual_grid_to_mesh(in, &st);
    result->faces_before_decimation = mesh.faces.size();
    report("mesh", 1, 3);
    const HoleFillStats hf = fill_holes(&mesh, o.hole_perimeter);
    result->holes_filled = hf.filled;
    report("mesh", 2, 3);
    if (o.max_faces > 0 && mesh.faces.size() > static_cast<std::size_t>(o.max_faces))
        mesh = decimate_mesh(mesh, static_cast<std::size_t>(o.max_faces));
    // Vertex colours after decimation: every vertex takes the attributes of the active voxel
    // it lies in (or the nearest of the 26 neighbours) in the z-up voxel frame.
    std::vector<float> colours;
    float metallic = 0.f, roughness = 0.9f;
    if (!attrs.empty()) {
        colours = voxel_colours(mesh, dec.coords, attrs, o.resolution, &metallic, &roughness);
    }
    mesh = to_y_up(mesh);
    const MeshStats ms = mesh_stats(mesh);
    result->vertices = mesh.vertices.size();
    result->faces = mesh.faces.size();
    result->open_edges = ms.open_edges;
    result->closed = ms.closed;
    result->volume = ms.volume;
    for (int a = 0; a < 3; ++a) { result->bbox_min[a] = ms.bbox_min[a]; result->bbox_max[a] = ms.bbox_max[a]; }
    result->seconds_mesh = seconds_since(t0);
    report("mesh", 3, 3);
    if (cancelled(cancel)) return fail_cancel();

    // 7. export
    report("export", 0, 1);
    GlbWriteOptions go;
    go.generator = "gufo trellis2";
    go.mesh_name = "trellis2";
    if (!colours.empty()) {
        go.vertex_colors = &colours;
        go.metallic_factor = metallic;
        go.roughness_factor = roughness;
    }
    result->glb = write_glb(mesh, go);
    if (o.preview) {
        PreviewOptions po;
        po.size = o.preview_size;
        int pw, ph;
        const auto rgb = render_preview(mesh, po, &pw, &ph);
        result->preview_png = encode_png_rgb(rgb.data(), pw, ph);
    }
    report("export", 1, 1);
    return true;
}

}  // namespace gufo::trellis2
