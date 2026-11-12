// TripoSG: mesh simplification by quadric-error edge collapse (Garland-Heckbert),
// for the `max_faces` request parameter. Closed meshes stay closed; collapses that
// would flip a triangle are rejected. Degenerate faces are dropped and unused
// vertices removed.
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstddef>

#include "src/models/triposg/marching_cubes.hpp"

namespace gufo::triposg {

struct DecimateOptions {
    std::size_t target_faces = 200000;
    // Maximum quadric error allowed for a collapse (relative to the bbox diagonal squared);
    // 0 = no limit (keep collapsing until the target is reached).
    double max_error = 0.0;
};

Mesh decimate(const Mesh& mesh, const DecimateOptions& options);

// Removes vertices referenced by no face and faces with repeated vertices.
Mesh compact(const Mesh& mesh);

}  // namespace gufo::triposg
