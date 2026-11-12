// Checks that the tensor inventories in model_description.hpp reproduce the parameter
// counts read from the real safetensors headers (microsoft/TRELLIS.2-4B and the
// TRELLIS v1 ss_dec, 2026-11-12) and that the converter's flattening keeps sizes.
// AI-assisted (Claude Code), 2026-11-12.
#include <iostream>
#include <set>
#include <string>

#include "src/models/trellis2/model_description.hpp"

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
    for (const auto& v : tr::kFlows) {
        const auto specs = tr::describe_flow(v);
        const std::size_t n = tr::total_params(specs);
        Expect(n == v.params, std::string("flow ") + v.file + " params " + std::to_string(n) + " != " + std::to_string(v.params));
        std::size_t tensors = 0;
        for (const auto& s : specs) tensors += s.repeat;
        Expect(tensors == 640, std::string("flow ") + v.file + " has 640 tensors, got " + std::to_string(tensors));
    }
    const auto ss = tr::describe_ss_dec();
    Expect(tr::total_params(ss) == tr::kParamsSsDec, "ss_dec params " + std::to_string(tr::total_params(ss)));
    Expect(ss.size() == 74, "ss_dec has 74 tensors, got " + std::to_string(ss.size()));

    const auto shape = tr::describe_sparse_dec(true);
    Expect(tr::total_params(shape) == tr::kParamsShapeDec, "shape_dec params " + std::to_string(tr::total_params(shape)));
    Expect(shape.size() == 292, "shape_dec has 292 tensors, got " + std::to_string(shape.size()));
    const auto tex = tr::describe_sparse_dec(false);
    Expect(tr::total_params(tex) == tr::kParamsTexDec, "tex_dec params " + std::to_string(tr::total_params(tex)));
    Expect(tex.size() == 284, "tex_dec has 284 tensors (no to_subdiv), got " + std::to_string(tex.size()));

    // names unique
    std::set<std::string> names;
    for (const auto& s : shape) names.insert(s.name);
    Expect(names.size() == shape.size(), "shape_dec names unique");

    // conv flattening: a C2S conv1 of level 0 is [8*512, 27*1024]
    bool found = false;
    for (const auto& s : shape)
        if (s.name == "blocks.0.4.conv1.weight") found = s.shape[0] == 4096 && s.shape[1] == 27 * 1024 && s.bf16;
    Expect(found, "blocks.0.4.conv1.weight -> [4096, 27648] BF16");
    if (failures == 0) std::cout << "trellis2_model_description_test: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
