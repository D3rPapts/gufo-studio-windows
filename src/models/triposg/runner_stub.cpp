// TripoSG: ObjektRunner stub for builds without HIP.
// AI-assisted (Claude Code), 2026-11-12.
#include "runner.hpp"

namespace gufo::triposg {

std::unique_ptr<ObjektRunner> CreateObjektRunner(const std::filesystem::path& weights, std::string* error) {
    if (error) *error = "TripoSG needs the HIP build (GPU): " + weights.string();
    return nullptr;
}

}  // namespace gufo::triposg
