// TRELLIS.2: ObjektRunner (the HIP-free job interface of gufo serve objekt, shared with
// TripoSG) on top of the TRELLIS.2 pipeline. CreateObjektRunner lives in the HIP library;
// the CPU-only build links a stub that reports the missing GPU path.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "src/models/triposg/runner.hpp"

namespace gufo::trellis2 {

// `weights_dir` holds the converted trellis2-*.safetensors; `with_texture` also loads the
// texture flow and decoder (stage 8).
std::unique_ptr<triposg::ObjektRunner> CreateObjektRunner(const std::filesystem::path& weights_dir, bool with_texture,
                                                          std::string* error);

}  // namespace gufo::trellis2
