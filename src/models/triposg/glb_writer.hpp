// TripoSG: minimal glTF 2.0 binary (GLB) and OBJ writers for triangle meshes,
// plus a GLB reader for round-trip tests and the /v1/3d API (vertex/face counts).
// AI-assisted (Claude Code), 2026-11-12.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "src/models/triposg/marching_cubes.hpp"

namespace gufo::triposg {

struct GlbWriteOptions {
    bool write_normals = true;        // smooth per-vertex normals (area weighted)
    std::string generator = "gufo triposg";
    std::string mesh_name = "triposg";
};

// Serialises the mesh as one GLB (12-byte header, JSON chunk, BIN chunk);
// positions VEC3/FLOAT with min/max, indices SCALAR/UNSIGNED_INT, optional normals.
std::vector<std::uint8_t> write_glb(const Mesh& mesh, const GlbWriteOptions& options = {});
bool write_glb_file(const std::filesystem::path& path, const Mesh& mesh, std::string* error,
                    const GlbWriteOptions& options = {});

// Wavefront OBJ (positions + faces, 1-based indices).
std::string write_obj(const Mesh& mesh);

// Reads the first triangle primitive of the first mesh of a GLB written by any
// conforming exporter with FLOAT positions and UNSIGNED_INT/UNSIGNED_SHORT indices.
// Returns false and fills *error on malformed input.
bool read_glb(std::span<const std::uint8_t> bytes, Mesh* mesh, std::string* error,
              std::string* json_text = nullptr);

// Area-weighted vertex normals (unit length; zero for isolated vertices).
std::vector<MeshVertex> vertex_normals(const Mesh& mesh);

}  // namespace gufo::triposg
