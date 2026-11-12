// TripoSG: ObjektRunner on top of Pipeline (HIP library).
// AI-assisted (Claude Code), 2026-11-12.
#include "runner.hpp"

#include "src/models/triposg/pipeline.hpp"

namespace gufo::triposg {
namespace {

class PipelineRunner final : public ObjektRunner {
public:
    explicit PipelineRunner(std::unique_ptr<Pipeline> pipeline) : pipeline_(std::move(pipeline)) {}
    bool Run(const ObjektRequest& r, const ObjektProgress& progress, const std::atomic<bool>* cancel, ObjektResult* out,
             std::string* error) override {
        GenerateOptions o;
        o.steps = r.steps;
        o.guidance = r.guidance;
        o.seed = r.seed;
        o.resolution = r.resolution;
        o.max_faces = r.max_faces;
        o.preview = r.preview;
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
        out->seconds_denoise = g.seconds_denoise;
        out->seconds_decode = g.seconds_decode;
        out->seconds_mesh = g.seconds_mesh;
        return true;
    }
    std::size_t weight_bytes() const override { return pipeline_->weight_bytes(); }

private:
    std::unique_ptr<Pipeline> pipeline_;
};

}  // namespace

std::unique_ptr<ObjektRunner> CreateObjektRunner(const std::filesystem::path& weights, std::string* error) {
    auto pipeline = Pipeline::Load(weights, error);
    if (!pipeline) return nullptr;
    return std::make_unique<PipelineRunner>(std::move(pipeline));
}

}  // namespace gufo::triposg
