// TRELLIS.2: ObjektRunner on top of Pipeline (HIP library).
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/runner.hpp"

#include "src/models/trellis2/pipeline.hpp"

namespace gufo::trellis2 {
namespace {

class PipelineRunner final : public triposg::ObjektRunner {
public:
    explicit PipelineRunner(std::unique_ptr<Pipeline> pipeline) : pipeline_(std::move(pipeline)) {}
    bool Run(const triposg::ObjektRequest& r, const triposg::ObjektProgress& progress, const std::atomic<bool>* cancel,
             triposg::ObjektResult* out, std::string* error) override {
        GenerateOptions o;
        o.steps = r.steps;
        o.seed = r.seed;
        o.resolution = 512;   // TRELLIS.2 stage A: 512 only (256 requests are served at 512)
        o.max_faces = r.max_faces;
        o.preview = r.preview;
        o.texture = r.texture && pipeline_->has_texture();
        GenerateResult g;
        if (!pipeline_->Generate(r.rgba.data(), r.width, r.height, o, progress, cancel, &g, error)) return false;
        out->glb = std::move(g.glb);
        out->preview_png = std::move(g.preview_png);
        out->vertices = g.vertices;
        out->faces = g.faces;
        out->faces_before_decimation = g.faces_before_decimation;
        for (int a = 0; a < 3; ++a) { out->bbox_min[a] = g.bbox_min[a]; out->bbox_max[a] = g.bbox_max[a]; }
        out->volume = g.volume;
        out->closed = g.closed;
        out->seconds_encode = g.seconds_encode;
        out->seconds_denoise = g.seconds_structure + g.seconds_shape + g.seconds_texture;
        out->seconds_decode = g.seconds_decode;
        out->seconds_mesh = g.seconds_mesh;
        return true;
    }
    std::size_t weight_bytes() const override { return pipeline_->weight_bytes(); }
    int default_steps() const override { return 12; }
    std::string capability_note() const override { return pipeline_->has_texture() ? "shape+pbr-vertex-colours" : "shape"; }
    // TRELLIS.2 encodes the image with DINOv3 (Meta, DINOv3 License).
    std::string attribution() const override { return "TRELLIS.2 (Microsoft, MIT). Built with DINOv3 (Meta, DINOv3 License)."; }

private:
    std::unique_ptr<Pipeline> pipeline_;
};

}  // namespace

std::unique_ptr<triposg::ObjektRunner> CreateObjektRunner(const std::filesystem::path& dir, bool with_texture, std::string* error) {
    auto pipeline = Pipeline::Load(dir, with_texture, error);
    if (!pipeline) return nullptr;
    return std::make_unique<PipelineRunner>(std::move(pipeline));
}

}  // namespace gufo::trellis2
