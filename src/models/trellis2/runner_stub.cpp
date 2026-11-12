// TRELLIS.2: ObjektRunner stub for builds without HIP.
// AI-assisted (Claude Code), 2026-11-12.
#include "src/models/trellis2/runner.hpp"

namespace gufo::trellis2 {

std::unique_ptr<triposg::ObjektRunner> CreateObjektRunner(const std::filesystem::path& dir, bool, std::string* error) {
    if (error) *error = "TRELLIS.2 needs the HIP build (GPU): " + dir.string();
    return nullptr;
}

}  // namespace gufo::trellis2
