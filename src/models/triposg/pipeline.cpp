// TripoSG: image -> GLB pipeline (see pipeline.hpp). Compiled into the HIP library.
// AI-assisted (Claude Code), 2026-11-12.
#include "pipeline.hpp"

#include <chrono>
#include <cmath>
#include <map>

#include "src/models/triposg/dinov2.hpp"
#include "src/models/triposg/dit.hpp"
#include "src/models/triposg/glb_writer.hpp"
#include "src/models/triposg/marching_cubes.hpp"
#include "src/models/triposg/mesh_decimate.hpp"
#include "src/models/triposg/mesh_preview.hpp"
#include "src/models/triposg/preprocess.hpp"
#include "src/models/triposg/scheduler.hpp"
#include "src/models/triposg/sdf_grid.hpp"
#include "src/models/triposg/vae.hpp"

namespace gufo::triposg {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::vector<float> gaussian_noise(std::size_t count, std::uint64_t seed) {
    std::vector<float> out(count);
    std::uint64_t state = seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL;
    auto uniform = [&]() {
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        return (static_cast<double>((state * 0x2545F4914F6CDD1DULL) >> 11) + 0.5) / 9007199254740992.0;
    };
    for (std::size_t i = 0; i + 1 < count; i += 2) {
        const double u1 = uniform(), u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = static_cast<float>(r * std::cos(6.283185307179586 * u2));
        out[i + 1] = static_cast<float>(r * std::sin(6.283185307179586 * u2));
    }
    if (count % 2) out[count - 1] = static_cast<float>(std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307179586 * uniform()));
    return out;
}

bool cancelled(const std::atomic<bool>* cancel) { return cancel != nullptr && cancel->load(); }

}  // namespace

struct Pipeline::Impl {
    std::unique_ptr<Dinov2Hip> dino;
    std::unique_ptr<DitHip> dit;
    std::unique_ptr<VaeHip> vae;
    std::size_t weight_bytes = 0;
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Pipeline::~Pipeline() = default;
std::size_t Pipeline::weight_bytes() const { return impl_->weight_bytes; }

std::unique_ptr<Pipeline> Pipeline::Load(const std::filesystem::path& weights, std::string* error) {
    auto impl = std::make_unique<Impl>();
    Dinov2Weights dw;
    if (!Dinov2Weights::Load(weights, &dw, error)) return nullptr;
    impl->dino = Dinov2Hip::Create(dw, error);
    if (!impl->dino) return nullptr;
    impl->weight_bytes += impl->dino->device_bytes();
    impl->dit = DitHip::Create(weights, error);
    if (!impl->dit) return nullptr;
    impl->vae = VaeHip::Create(weights, error);
    if (!impl->vae) return nullptr;
    return std::unique_ptr<Pipeline>(new Pipeline(std::move(impl)));
}

bool Pipeline::Generate(const std::uint8_t* rgba, int width, int height, const GenerateOptions& o,
                        const GenerateProgress& progress, const std::atomic<bool>* cancel, GenerateResult* result,
                        std::string* error) {
    Impl& m = *impl_;
    auto report = [&](const std::string& phase, int done, int total) {
        if (progress) progress(phase, done, total);
    };
    auto fail_cancel = [&]() {
        if (error) *error = "cancelled";
        return false;
    };
    // 1. Preprocess + DINOv2.
    auto t0 = Clock::now();
    report("encode", 0, 2);
    Preprocessed pre;
    if (!preprocess_rgba(rgba, width, height, &pre, error)) return false;
    report("encode", 1, 2);
    std::vector<float> tokens;
    if (!m.dino->Encode(pre.pixels.data(), &tokens, error)) return false;
    report("encode", 2, 2);
    result->seconds_encode = seconds_since(t0);
    if (cancelled(cancel)) return fail_cancel();

    // 2. Denoising with CFG (Euler on the host in FP32).
    t0 = Clock::now();
    const int steps = std::max(1, o.steps);
    const std::vector<float> sigmas = flow_sigmas(steps);
    std::vector<float> x = gaussian_noise(static_cast<std::size_t>(DitConfig::latent_tokens) * DitConfig::in_channels, o.seed);
    const std::vector<float> zeros(tokens.size(), 0.f);
    std::vector<float> v_cond, v_uncond;
    const bool cfg = o.guidance > 1.f;
    for (int i = 0; i < steps; ++i) {
        if (cancelled(cancel)) return fail_cancel();
        const float t = flow_timestep(sigmas, i);
        if (!m.dit->Forward(x.data(), DitConfig::latent_tokens, tokens.data(), DinoConfig::tokens, t, &v_cond, nullptr, error)) return false;
        if (cfg) {
            if (!m.dit->Forward(x.data(), DitConfig::latent_tokens, zeros.data(), DinoConfig::tokens, t, &v_uncond, nullptr, error)) return false;
            apply_guidance(&v_cond, v_uncond, o.guidance);
        }
        euler_step(x, v_cond, sigmas, i);
        report("denoise", i + 1, steps);
    }
    result->seconds_denoise = seconds_since(t0);

    // 3. VAE decode on the hierarchical grid.
    t0 = Clock::now();
    if (!m.vae->Prepare(x.data(), DitConfig::latent_tokens, error)) return false;
    SdfGridOptions go;
    go.levels = o.resolution >= 512 ? std::vector<int>{256, 512} : std::vector<int>{256};
    go.chunk = 32768;
    const std::size_t coarse_points = 256u * 256u * 256u;
    std::size_t done_points = 0;
    go.progress = [&](const std::string&) {};
    SdfGridResult grid;
    const SdfQuery query = [&](const float* p, int n, std::vector<float>* values, std::string* e) {
        if (cancelled(cancel)) {
            if (e) *e = "cancelled";
            return false;
        }
        if (!m.vae->Query(p, n, values, e, 32768)) return false;
        done_points += static_cast<std::size_t>(n);
        // The coarse grid is ~87 % of the work when refining to 512.
        const std::size_t total = o.resolution >= 512 ? coarse_points * 115 / 100 : coarse_points;
        report("decode", static_cast<int>(std::min<std::size_t>(done_points, total) / 1024), static_cast<int>(total / 1024));
        return true;
    };
    if (!extract_sdf_grid(query, go, &grid, error)) return false;
    result->seconds_decode = seconds_since(t0);

    // 4. Marching cubes, decimation, export.
    t0 = Clock::now();
    report("mesh", 0, 3);
    ScalarGrid sg;
    sg.half_values = grid.half.data();
    sg.nx = sg.ny = sg.nz = static_cast<std::size_t>(grid.n);
    for (int a = 0; a < 3; ++a) { sg.min[a] = -GeometryConfig::bounds; sg.max[a] = GeometryConfig::bounds; }
    MarchingCubesOptions mo;
    mo.inside_positive = false;
    Mesh mesh = marching_cubes(sg, mo);
    grid.half.clear();
    grid.half.shrink_to_fit();
    result->faces_before_decimation = mesh.faces.size();
    report("mesh", 1, 3);
    if (cancelled(cancel)) return fail_cancel();
    if (o.max_faces > 0 && mesh.faces.size() > static_cast<std::size_t>(o.max_faces)) {
        DecimateOptions d;
        d.target_faces = static_cast<std::size_t>(o.max_faces);
        mesh = decimate(mesh, d);
    }
    report("mesh", 2, 3);
    result->vertices = mesh.vertices.size();
    result->faces = mesh.faces.size();
    result->closed = count_open_edges(mesh) == 0;
    result->volume = signed_volume(mesh);
    bounding_box(mesh, result->bbox_min, result->bbox_max);
    result->seconds_mesh = seconds_since(t0);
    report("export", 0, 1);
    result->glb = write_glb(mesh);
    if (o.preview) {
        PreviewOptions po;
        po.size = o.preview_size;
        int pw = 0, ph = 0;
        const auto rgb = render_preview(mesh, po, &pw, &ph);
        result->preview_png = encode_png_rgb(rgb.data(), pw, ph);
    }
    report("export", 1, 1);
    return true;
}

}  // namespace gufo::triposg
