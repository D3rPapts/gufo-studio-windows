// Written with AI assistance (Claude Code).
#include "voxcpm2/hip/model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ming_tts/hip/ops.hpp"
#include "voxcpm2/hip/ops.hpp"

namespace gufo::models::voxcpm2::hip {
namespace {

namespace ming = ming_tts::hip;

// Up to this many rows a product runs as a BF16 GEMV per row; above it the
// FP32-activation tiled GEMM takes over (keeps the math FP32 like the CPU).
constexpr std::int64_t kGemvRows = 8;
// Default row limit for the split-K skinny GEMM (32-row passes). GPU run 2:
// DiT (22 rows) 202.7 -> 138.1 ms/step with skinny v1, but prefill (86 rows,
// 3 passes) 344 -> 650 ms, so larger products keep the 64x64 tile GEMM.
// GUFO_VOXCPM_SKINNY_MAX_ROWS overrides it (for measuring).
constexpr std::int64_t kSkinnyMaxRowsDefault = 32;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("HIP ") + what + ": " + hipGetErrorString(status));
  }
}

struct DevLinear {
  const std::uint16_t* weight{nullptr};
  const float* bias{nullptr};
  std::int64_t out{0};
  std::int64_t in{0};
};

struct DevLayer {
  DevLinear q, k, v, o, gate, up, down;
  DevLinear gate_up;  // [gate; up] in one allocation (gate/up point into it)
  const float* input_norm{nullptr};
  const float* post_norm{nullptr};
};

struct DevStack {
  StackConfig config;
  std::vector<DevLayer> layers;
  const float* final_norm{nullptr};
};

// Grow-only device buffer.
struct Scratch {
  float* data{nullptr};
  std::size_t capacity{0};
  float* Ensure(std::size_t elements) {
    if (elements > capacity) {
      if (data != nullptr) {
        (void)hipFree(data);
        data = nullptr;
      }
      Check(hipMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(elements, 64) *
                                                           sizeof(float)),
            "scratch alloc");
      capacity = std::max<std::size_t>(elements, 64);
    }
    return data;
  }
  Scratch() = default;
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    if (data != nullptr) {
      (void)hipFree(data);
    }
  }
};

// Per-call working buffers of one transformer stack forward.
struct StackScratch {
  Scratch normed, q, k, v, attn, proj, gate, up;
};

// KV cache of one LM: per layer [capacity, kv_heads, head_dim], grown with copy.
struct DevCache {
  std::vector<float*> k, v;
  std::int64_t capacity{0};
  std::int64_t length{0};
  std::size_t width{0};
  void Ensure(std::int64_t positions, std::size_t layers, std::size_t kv_width,
              hipStream_t stream) {
    width = kv_width;
    if (k.size() != layers) {
      k.assign(layers, nullptr);
      v.assign(layers, nullptr);
    }
    if (positions <= capacity) {
      return;
    }
    const std::int64_t grown = std::max<std::int64_t>((positions + 1023) / 1024 * 1024, 2 * capacity);
    for (std::size_t l = 0; l < layers; ++l) {
      for (auto* list : {&k, &v}) {
        float* fresh = nullptr;
        Check(hipMalloc(reinterpret_cast<void**>(&fresh),
                        static_cast<std::size_t>(grown) * kv_width * sizeof(float)),
              "cache alloc");
        if ((*list)[l] != nullptr) {
          if (length > 0) {
            Check(hipMemcpyAsync(fresh, (*list)[l],
                                 static_cast<std::size_t>(length) * kv_width * sizeof(float),
                                 hipMemcpyDeviceToDevice, stream),
                  "cache copy");
          }
          Check(hipStreamSynchronize(stream), "cache sync");
          (void)hipFree((*list)[l]);
        }
        (*list)[l] = fresh;
      }
    }
    capacity = grown;
  }
  ~DevCache() {
    for (float* p : k) {
      if (p != nullptr) {
        (void)hipFree(p);
      }
    }
    for (float* p : v) {
      if (p != nullptr) {
        (void)hipFree(p);
      }
    }
  }
};

// One executed Euler step of the sampler (zero-init steps are skipped).
struct FlowStep {
  float t{0.0F};
  float dt{0.0F};
};

std::vector<FlowStep> FlowSchedule(const FlowOptions& options) {
  const std::vector<float> span = cpu::Model::TimeSpan(options.timesteps, options.sway_sampling_coef);
  const auto steps = static_cast<std::int32_t>(span.size());
  const std::int32_t zero_init = std::max(1, static_cast<std::int32_t>(static_cast<double>(steps) * 0.04));
  std::vector<FlowStep> schedule;
  float t = span[0];
  float dt = span[0] - span[1];
  for (std::int32_t step = 1; step < steps; ++step) {
    if (!(options.cfg_zero_star && step <= zero_init)) {
      schedule.push_back({t, dt});
    }
    t = t - dt;
    if (step < steps - 1) {
      dt = t - span[static_cast<std::size_t>(step + 1)];
    }
  }
  return schedule;
}

}  // namespace

bool DeviceAvailable() noexcept {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

struct Model::Impl {
  const cpu::Model* cpu{nullptr};
  hipStream_t stream{nullptr};
  std::vector<void*> allocations;

  DevStack base, residual, locenc, dit;
  DevLinear locenc_in, dit_in, dit_cond, dit_out, time1, time2;
  DevLinear fsq_in, fsq_out, enc_to_lm, lm_to_dit, res_to_dit, fusion, stop_proj, stop_head;
  const float* special{nullptr};
  const float* delta_time{nullptr};
  const float* rope_cos{nullptr};
  const float* rope_sin{nullptr};
  std::int64_t rope_positions{0};

  DevCache base_cache, residual_cache;

  bool use_graphs{true};
  // Split-K skinny GEMM for 9..128 rows (GUFO_VOXCPM_SKINNY=0: old tile GEMM).
  bool use_skinny{true};
  std::int64_t skinny_max_rows{kSkinnyMaxRowsDefault};
  // BF16 WMMA inside the skinny GEMM (GUFO_VOXCPM_WMMA=1; default off until
  // the oracle checks on the GPU say it holds).
  bool use_wmma{false};
  // Per-kernel profile (eager, synchronizing after every op): name -> ms, calls.
  bool profiling{false};
  std::map<std::string, std::pair<double, int>> profile;
  std::chrono::steady_clock::time_point profile_mark;
  void Tick(const char* name) {
    if (!profiling) {
      return;
    }
    Check(hipStreamSynchronize(stream), "profile sync");
    const auto now = std::chrono::steady_clock::now();
    auto& slot = profile[name];
    slot.first += std::chrono::duration<double, std::milli>(now - profile_mark).count();
    slot.second += 1;
    profile_mark = now;
  }
  bool Skinny(std::int64_t rows) const {
    return use_skinny && rows > kGemvRows && rows <= skinny_max_rows;
  }
  std::size_t max_partial{0};
  float* partial_main{nullptr};  // skinny partial sums (eager paths)
  float* partial_flow{nullptr};  // same for the graph-captured sampler
  bool in_flow{false};
  std::map<std::tuple<std::int32_t, float, bool>, hipGraphExec_t> graphs;
  std::map<std::tuple<std::int32_t, float, bool>, int> graph_seen;
  // Time-embedding rows per timesteps setting: [executed steps, dit_hidden].
  std::map<std::tuple<std::int32_t, float, bool>, std::pair<Scratch*, std::vector<FlowStep>>> time_rows;
  std::vector<std::unique_ptr<Scratch>> time_tables;

  // Working sets: `stack` for prefill/advance, `flow` reserved for the
  // (graph-captured) sampler so its pointers never move.
  StackScratch stack_scratch, flow_scratch;
  Scratch x, feats, mask, proj, seq, cls, feat_embed, fsq_tmp, fsq_all, fusion_in, residual_buf,
      hidden, small;
  // Sampler buffers (fixed after the first call).
  Scratch f_lm, f_res, f_prev, f_x, f_mu, f_cond, f_xrows, f_seq, f_vel, f_tmp;

  ~Impl() {
    for (auto& [key, exec] : graphs) {
      (void)hipGraphExecDestroy(exec);
    }
    for (void* pointer : allocations) {
      (void)hipFree(pointer);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  template <typename T>
  T* Upload(const T* host, std::size_t count) {
    void* device = nullptr;
    Check(hipMalloc(&device, count * sizeof(T)), "weight alloc");
    allocations.push_back(device);
    Check(hipMemcpy(device, host, count * sizeof(T), hipMemcpyHostToDevice), "weight upload");
    return static_cast<T*>(device);
  }
  const float* UploadVector(const std::vector<float>& host) {
    return host.empty() ? nullptr : Upload(host.data(), host.size());
  }
  DevLinear UploadLinear(const cpu::Linear& linear) {
    DevLinear out;
    out.out = linear.out_features;
    out.in = linear.in_features;
    out.weight = Upload(linear.weight, static_cast<std::size_t>(out.out * out.in));
    out.bias = UploadVector(linear.bias);
    max_partial = std::max(max_partial, SkinnyPartialFloats(static_cast<std::size_t>(out.out),
                                                            static_cast<std::size_t>(out.in)));
    if (out.in % 8 != 0) {
      throw std::runtime_error("VoxCPM2 HIP runtime needs in_features % 8 == 0");
    }
    return out;
  }
  DevStack UploadStack(const cpu::Stack& stack) {
    DevStack out;
    out.config = stack.config;
    for (const cpu::Layer& layer : stack.layers) {
      DevLayer d;
      d.q = UploadLinear(layer.q);
      d.k = UploadLinear(layer.k);
      d.v = UploadLinear(layer.v);
      d.o = UploadLinear(layer.o);
      // gate and up back to back: one [2f, h] matrix for the fused skinny GEMM
      {
        const std::size_t f = static_cast<std::size_t>(layer.gate.out_features);
        const std::size_t hin = static_cast<std::size_t>(layer.gate.in_features);
        void* fused = nullptr;
        Check(hipMalloc(&fused, 2 * f * hin * sizeof(std::uint16_t)), "weight alloc");
        allocations.push_back(fused);
        auto* base = static_cast<std::uint16_t*>(fused);
        Check(hipMemcpy(base, layer.gate.weight, f * hin * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice), "weight upload");
        Check(hipMemcpy(base + f * hin, layer.up.weight, f * hin * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice), "weight upload");
        if (!layer.gate.bias.empty() || !layer.up.bias.empty()) {
          throw std::runtime_error("VoxCPM2 HIP: MLP biases are not expected");
        }
        d.gate = {base, nullptr, static_cast<std::int64_t>(f), static_cast<std::int64_t>(hin)};
        d.up = {base + f * hin, nullptr, static_cast<std::int64_t>(f), static_cast<std::int64_t>(hin)};
        d.gate_up = {base, nullptr, static_cast<std::int64_t>(2 * f), static_cast<std::int64_t>(hin)};
        max_partial = std::max(max_partial, SkinnyPartialFloats(2 * f, hin));
      }
      d.down = UploadLinear(layer.down);
      d.input_norm = UploadVector(layer.input_norm);
      d.post_norm = UploadVector(layer.post_norm);
      out.layers.push_back(d);
    }
    out.final_norm = UploadVector(stack.final_norm);
    return out;
  }

  void Sync() { Check(hipStreamSynchronize(stream), "synchronize"); }
  void ToDevice(void* device, const void* host, std::size_t bytes) {
    Check(hipMemcpyAsync(device, host, bytes, hipMemcpyHostToDevice, stream), "upload");
  }
  void ToHost(void* host, const void* device, std::size_t bytes) {
    Check(hipMemcpyAsync(host, device, bytes, hipMemcpyDeviceToHost, stream), "download");
    Sync();
  }
  void Copy(float* dst, const float* src, std::size_t count) {
    Check(hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToDevice, stream),
          "copy");
  }

  // y[rows, out] = x[rows, in] W^T (+ b)
  void Linear(const DevLinear& layer, const float* xin, std::int64_t rows, float* y) {
    if (rows <= kGemvRows) {
      ming_tts::hip::LaunchBf16Gemv(layer.weight, xin, y,
                                               static_cast<std::size_t>(rows),
                                               static_cast<std::size_t>(layer.out),
                                               static_cast<std::size_t>(layer.in), stream);
    } else if (Skinny(rows)) {
      LaunchSkinnyGemm(xin, layer.weight, layer.bias, y, static_cast<std::size_t>(rows),
                       static_cast<std::size_t>(layer.out), static_cast<std::size_t>(layer.in),
                       in_flow ? partial_flow : partial_main, stream, SkinnyEpilogue::kStore,
                       use_wmma);
      return;  // bias added in the reduction
    } else {
      ming::LaunchGemmF32Bf16(xin, layer.weight, y, static_cast<std::size_t>(rows),
                              static_cast<std::size_t>(layer.out),
                              static_cast<std::size_t>(layer.in), stream);
    }
    if (layer.bias != nullptr) {
      ming::LaunchAddBias(y, layer.bias, static_cast<std::size_t>(rows),
                          static_cast<std::size_t>(layer.out), stream);
    }
  }

  // MiniCPM-4 stack on xin [batch * seq, hidden] in place (final norm applied).
  void StackForward(const DevStack& stack, float* xin, std::int64_t batch, std::int64_t seqlen,
                    bool causal, DevCache* cache, StackScratch& s) {
    const StackConfig& c = stack.config;
    const auto rows = static_cast<std::size_t>(batch * seqlen);
    const auto h = static_cast<std::size_t>(c.hidden_size);
    const auto hd = static_cast<std::size_t>(c.head_dim);
    const std::size_t qw = static_cast<std::size_t>(c.num_heads) * hd;
    const std::size_t kvw = static_cast<std::size_t>(c.num_kv_heads) * hd;
    const auto f = static_cast<std::size_t>(c.intermediate_size);
    const float eps = cpu->config().rms_norm_eps;
    float* nrm = s.normed.Ensure(rows * h);
    float* qb = s.q.Ensure(rows * qw);
    float* kb = s.k.Ensure(rows * kvw);
    float* vb = s.v.Ensure(rows * kvw);
    float* ab = s.attn.Ensure(rows * qw);
    float* pb = s.proj.Ensure(rows * h);
    float* gb = s.gate.Ensure(rows * f);
    float* ub = s.up.Ensure(rows * f);
    const std::int64_t first = cache != nullptr ? cache->length : 0;
    if (cache != nullptr) {
      cache->Ensure(first + seqlen, stack.layers.size(), kvw, stream);
    }
    if (c.rope && first + seqlen > rope_positions) {
      throw std::runtime_error("VoxCPM2 HIP: position beyond the RoPE table");
    }
    if (cpu->config().use_mup) {
      throw std::runtime_error("VoxCPM2 HIP: muP residual scaling is not implemented");
    }
    for (std::size_t index = 0; index < stack.layers.size(); ++index) {
      const DevLayer& layer = stack.layers[index];
      const bool skinny = Skinny(static_cast<std::int64_t>(rows));
      float* partial = in_flow ? partial_flow : partial_main;
      ming::LaunchRmsNorm(xin, layer.input_norm, nrm, rows, h, eps, stream);
      Tick("rmsnorm");
      Linear(layer.q, nrm, static_cast<std::int64_t>(rows), qb);
      Tick("gemm q");
      Linear(layer.k, nrm, static_cast<std::int64_t>(rows), kb);
      Linear(layer.v, nrm, static_cast<std::int64_t>(rows), vb);
      Tick("gemm k+v");
      if (c.rope) {
        LaunchRopeTable(qb, rows, static_cast<std::size_t>(seqlen),
                        static_cast<std::size_t>(c.num_heads), hd,
                        static_cast<std::size_t>(first), rope_cos, rope_sin, stream);
        LaunchRopeTable(kb, rows, static_cast<std::size_t>(seqlen),
                        static_cast<std::size_t>(c.num_kv_heads), hd,
                        static_cast<std::size_t>(first), rope_cos, rope_sin, stream);
        Tick("rope");
      }
      if (cache != nullptr) {
        Copy(cache->k[index] + static_cast<std::size_t>(first) * kvw, kb, rows * kvw);
        Copy(cache->v[index] + static_cast<std::size_t>(first) * kvw, vb, rows * kvw);
        LaunchAttention(qb, cache->k[index], cache->v[index], ab, 1, rows,
                        static_cast<std::size_t>(first) + rows,
                        static_cast<std::size_t>(c.num_heads),
                        static_cast<std::size_t>(c.num_kv_heads), hd, 0, 0, causal,
                        static_cast<std::size_t>(first), stream);
      } else {
        LaunchAttention(qb, kb, vb, ab, static_cast<std::size_t>(batch),
                        static_cast<std::size_t>(seqlen), static_cast<std::size_t>(seqlen),
                        static_cast<std::size_t>(c.num_heads),
                        static_cast<std::size_t>(c.num_kv_heads), hd,
                        static_cast<std::size_t>(seqlen) * qw,
                        static_cast<std::size_t>(seqlen) * kvw, causal, 0, stream);
      }
      Tick("attention (+cache copy)");
      if (skinny) {  // residual add and SiLU*up fused into the split-K reduction
        LaunchSkinnyGemm(ab, layer.o.weight, layer.o.bias, xin, rows,
                         static_cast<std::size_t>(layer.o.out), static_cast<std::size_t>(layer.o.in),
                         partial, stream, SkinnyEpilogue::kAccumulate, use_wmma);
        Tick("gemm o (+residual)");
        ming::LaunchRmsNorm(xin, layer.post_norm, nrm, rows, h, eps, stream);
        Tick("rmsnorm");
        LaunchSkinnyGemm(nrm, layer.gate_up.weight, nullptr, gb, rows,
                         static_cast<std::size_t>(layer.gate_up.out),
                         static_cast<std::size_t>(layer.gate_up.in), partial, stream,
                         SkinnyEpilogue::kSiluMulPairs, use_wmma);
        Tick("gemm gate+up (+silu*up)");
        LaunchSkinnyGemm(gb, layer.down.weight, layer.down.bias, xin, rows,
                         static_cast<std::size_t>(layer.down.out),
                         static_cast<std::size_t>(layer.down.in), partial, stream,
                         SkinnyEpilogue::kAccumulate, use_wmma);
        Tick("gemm down (+residual)");
        continue;
      }
      Linear(layer.o, ab, static_cast<std::int64_t>(rows), pb);
      ming::LaunchAdd(xin, pb, rows * h, stream);
      Tick("gemm o + add");
      ming::LaunchRmsNorm(xin, layer.post_norm, nrm, rows, h, eps, stream);
      Tick("rmsnorm");
      Linear(layer.gate, nrm, static_cast<std::int64_t>(rows), gb);
      Linear(layer.up, nrm, static_cast<std::int64_t>(rows), ub);
      ming::LaunchSiluMul(gb, ub, rows * f, stream);
      Tick("gemm gate+up + silu*up");
      Linear(layer.down, gb, static_cast<std::int64_t>(rows), pb);
      ming::LaunchAdd(xin, pb, rows * h, stream);
      Tick("gemm down + add");
    }
    ming::LaunchRmsNorm(xin, stack.final_norm, xin, rows, h, eps, stream);
    Tick("rmsnorm");
    if (cache != nullptr) {
      cache->length = first + seqlen;
    }
  }

  // Local encoder on `count` device patches [count, patch, latent] ->
  // CLS rows [count, enc_hidden] in `out`.
  void LocalEncodeDevice(const float* patches, std::int64_t count, float* out) {
    const ModelConfig& c = cpu->config();
    const auto p = static_cast<std::size_t>(c.patch_size);
    const auto eh = static_cast<std::size_t>(c.encoder.hidden_size);
    const auto n = static_cast<std::size_t>(count);
    float* pr = proj.Ensure(n * p * eh);
    Linear(locenc_in, patches, static_cast<std::int64_t>(n * p), pr);
    float* sq = seq.Ensure(n * (p + 1) * eh);
    LaunchAssembleLocEnc(special, pr, sq, n, p, eh, stream);
    StackForward(locenc, sq, count, static_cast<std::int64_t>(p + 1), false, nullptr,
                 stack_scratch);
    LaunchGatherRows(sq, out, n, p + 1, eh, stream);
  }

  void FsqDevice(const float* xin, std::int64_t rows, float* out) {
    const auto latent = static_cast<std::size_t>(cpu->config().fsq_latent_dim);
    float* tmp = fsq_tmp.Ensure(static_cast<std::size_t>(rows) * latent);
    Linear(fsq_in, xin, rows, tmp);
    LaunchFsqQuantize(tmp, static_cast<std::size_t>(rows) * latent,
                      static_cast<float>(cpu->config().fsq_scale), stream);
    Linear(fsq_out, tmp, rows, out);
  }

  // Time rows for a sampler setting (computed once, pointer stable).
  const std::pair<Scratch*, std::vector<FlowStep>>& TimeRows(const FlowOptions& options) {
    const auto key = std::make_tuple(options.timesteps, options.sway_sampling_coef,
                                     options.cfg_zero_star);
    if (const auto found = time_rows.find(key); found != time_rows.end()) {
      return found->second;
    }
    const auto dh = static_cast<std::size_t>(cpu->config().dit.hidden_size);
    std::vector<FlowStep> schedule = FlowSchedule(options);
    time_tables.push_back(std::make_unique<Scratch>());
    Scratch* table = time_tables.back().get();
    float* rows = table->Ensure(std::max<std::size_t>(1, schedule.size()) * dh);
    std::vector<float> sinus(dh);
    float* dev_sinus = small.Ensure(2 * dh);
    float* dev_hidden = dev_sinus + dh;
    for (std::size_t s = 0; s < schedule.size(); ++s) {
      cpu->TimeEmbedding(schedule[s].t, sinus.data());
      ToDevice(dev_sinus, sinus.data(), dh * sizeof(float));
      Linear(time1, dev_sinus, 1, dev_hidden);
      ming::LaunchSilu(dev_hidden, dh, stream);
      Linear(time2, dev_hidden, 1, rows + s * dh);
      ming::LaunchAdd(rows + s * dh, delta_time, dh, stream);
      Sync();  // `sinus` is reused by the next upload
    }
    return time_rows.emplace(key, std::make_pair(table, std::move(schedule))).first->second;
  }

  // Sampler buffers, sized once for the fixed DiT shapes.
  void EnsureFlowBuffers() {
    const ModelConfig& c = cpu->config();
    const auto h = static_cast<std::size_t>(c.base_lm.hidden_size);
    const auto dh = static_cast<std::size_t>(c.dit.hidden_size);
    const auto pd = static_cast<std::size_t>(c.patch_size * c.feat_dim);
    const auto p = static_cast<std::size_t>(c.patch_size);
    const std::size_t mu_tokens = h / dh;
    const std::size_t seq_len = mu_tokens + 1 + 2 * p;
    f_lm.Ensure(h);
    f_res.Ensure(h);
    f_prev.Ensure(pd);
    f_x.Ensure(pd);
    f_mu.Ensure(h);
    f_cond.Ensure(p * dh);
    f_xrows.Ensure(p * dh);
    f_seq.Ensure(2 * seq_len * dh);
    f_vel.Ensure(2 * pd);
    f_tmp.Ensure(dh);
  }

  // One CFG estimator evaluation on f_x at time row t_row -> f_vel [2, pd].
  void VelocityDevice(const float* t_row) {
    const ModelConfig& c = cpu->config();
    const auto h = static_cast<std::size_t>(c.base_lm.hidden_size);
    const auto dh = static_cast<std::size_t>(c.dit.hidden_size);
    const auto p = static_cast<std::size_t>(c.patch_size);
    const auto pd = static_cast<std::size_t>(c.patch_size * c.feat_dim);
    const std::size_t mu_tokens = h / dh;
    const std::size_t seq_len = mu_tokens + 1 + 2 * p;
    Linear(dit_in, f_x.data, static_cast<std::int64_t>(p), f_xrows.data);
    LaunchAssembleDit(f_mu.data, t_row, f_cond.data, f_xrows.data, f_seq.data, mu_tokens, p, dh,
                      stream);
    Tick("dit in_proj + assemble");
    StackForward(dit, f_seq.data, 2, static_cast<std::int64_t>(seq_len), false, nullptr,
                 flow_scratch);
    for (std::size_t b = 0; b < 2; ++b) {
      Linear(dit_out, f_seq.data + (b * seq_len + mu_tokens + 1 + p) * dh,
             static_cast<std::int64_t>(p), f_vel.data + b * pd);
    }
    Tick("dit out_proj");
  }
};

Model::Model(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Model::~Model() = default;

std::unique_ptr<Model> Model::Create(const cpu::Model& cpu_model, std::string* error) {
  if (!DeviceAvailable()) {
    if (error != nullptr) {
      *error = "no HIP device";
    }
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  try {
    Impl& m = *impl;
    m.cpu = &cpu_model;
    const ModelConfig& c = cpu_model.config();
    const char* graphs_env = std::getenv("GUFO_VOXCPM_GRAPHS");
    m.use_graphs = !(graphs_env != nullptr && std::string(graphs_env) == "0");
    Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking), "stream");
    m.base = m.UploadStack(cpu_model.base_lm());
    m.residual = m.UploadStack(cpu_model.residual_lm());
    m.locenc = m.UploadStack(cpu_model.local_encoder().stack);
    m.dit = m.UploadStack(cpu_model.local_dit().stack);
    m.locenc_in = m.UploadLinear(cpu_model.local_encoder().in_proj);
    m.special = m.UploadVector(cpu_model.local_encoder().special_token);
    m.dit_in = m.UploadLinear(cpu_model.local_dit().in_proj);
    m.dit_cond = m.UploadLinear(cpu_model.local_dit().cond_proj);
    m.dit_out = m.UploadLinear(cpu_model.local_dit().out_proj);
    m.time1 = m.UploadLinear(cpu_model.local_dit().time_mlp1);
    m.time2 = m.UploadLinear(cpu_model.local_dit().time_mlp2);
    m.delta_time = m.UploadVector(cpu_model.delta_time());
    m.fsq_in = m.UploadLinear(cpu_model.fsq().in_proj);
    m.fsq_out = m.UploadLinear(cpu_model.fsq().out_proj);
    m.enc_to_lm = m.UploadLinear(cpu_model.enc_to_lm());
    m.lm_to_dit = m.UploadLinear(cpu_model.lm_to_dit());
    m.res_to_dit = m.UploadLinear(cpu_model.res_to_dit());
    m.fusion = m.UploadLinear(cpu_model.fusion());
    m.stop_proj = m.UploadLinear(cpu_model.stop_proj());
    m.stop_head = m.UploadLinear(cpu_model.stop_head());
    cpu_model.PrepareRope(c.max_length + 64);
    m.rope_positions = cpu_model.rope_positions();
    m.rope_cos = m.UploadVector(cpu_model.rope_cos());
    m.rope_sin = m.UploadVector(cpu_model.rope_sin());
    m.EnsureFlowBuffers();
    const char* skinny_env = std::getenv("GUFO_VOXCPM_SKINNY");
    m.use_skinny = !(skinny_env != nullptr && std::string(skinny_env) == "0");
    if (const char* rows_env = std::getenv("GUFO_VOXCPM_SKINNY_MAX_ROWS"); rows_env != nullptr) {
      m.skinny_max_rows = std::max<std::int64_t>(kGemvRows, std::atoll(rows_env));
    }
    const char* wmma_env = std::getenv("GUFO_VOXCPM_WMMA");
    m.use_wmma = wmma_env != nullptr && std::string(wmma_env) == "1";
    for (float** buffer : {&m.partial_main, &m.partial_flow}) {
      void* device = nullptr;
      Check(hipMalloc(&device, std::max<std::size_t>(m.max_partial, 64) * sizeof(float)),
            "partial alloc");
      m.allocations.push_back(device);
      *buffer = static_cast<float*>(device);
    }
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = std::string("VoxCPM2 HIP runtime: ") + exception.what();
    }
    return nullptr;
  }
  return std::unique_ptr<Model>(new Model(std::move(impl)));
}

void Model::SetGraphs(bool enabled) { impl_->use_graphs = enabled; }
void Model::SetSkinnyGemm(bool enabled) {
  Impl& m = *impl_;
  if (m.use_skinny != enabled) {
    for (auto& [key, exec] : m.graphs) {  // recorded with the other kernels
      (void)hipGraphExecDestroy(exec);
    }
    m.graphs.clear();
    m.graph_seen.clear();
  }
  m.use_skinny = enabled;
}
bool Model::skinny_gemm() const noexcept { return impl_->use_skinny; }

void Model::SetSkinnyMaxRows(std::int64_t rows) {
  impl_->skinny_max_rows = std::max<std::int64_t>(kGemvRows, rows);
}

void Model::SetWmma(bool enabled) {
  Impl& m = *impl_;
  if (m.use_wmma != enabled) {
    for (auto& [key, exec] : m.graphs) {
      (void)hipGraphExecDestroy(exec);
    }
    m.graphs.clear();
    m.graph_seen.clear();
  }
  m.use_wmma = enabled;
}

bool Model::wmma() const noexcept { return impl_->use_wmma; }

void Model::SetProfiling(bool enabled) {
  Impl& m = *impl_;
  m.profiling = enabled;
  m.profile.clear();
  Check(hipStreamSynchronize(m.stream), "profile sync");
  m.profile_mark = std::chrono::steady_clock::now();
}

std::string Model::ProfileReport() const {
  const Impl& m = *impl_;
  std::vector<std::pair<std::string, std::pair<double, int>>> rows(m.profile.begin(),
                                                                   m.profile.end());
  std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
  double total = 0.0;
  for (const auto& row : rows) {
    total += row.second.first;
  }
  std::string out;
  char line[200];
  for (const auto& [name, value] : rows) {
    std::snprintf(line, sizeof(line), "  %-28s %8.2f ms %5.1f %%  %5d calls  %.3f ms/call\n",
                  name.c_str(), value.first, total > 0.0 ? 100.0 * value.first / total : 0.0,
                  value.second, value.first / std::max(1, value.second));
    out += line;
  }
  std::snprintf(line, sizeof(line), "  %-28s %8.2f ms (eager, synchronized after every op)\n",
                "total", total);
  out += line;
  return out;
}
bool Model::graphs_enabled() const noexcept { return impl_->use_graphs; }

void Model::Prefill(const std::int64_t* ids, const float* feats, const std::uint8_t* audio_mask,
                    std::int64_t rows, float* lm_hidden, float* residual_hidden) {
  Impl& m = *impl_;
  const ModelConfig& c = m.cpu->config();
  const auto n = static_cast<std::size_t>(rows);
  const auto h = static_cast<std::size_t>(c.base_lm.hidden_size);
  const auto eh = static_cast<std::size_t>(c.encoder.hidden_size);
  const auto pd = static_cast<std::size_t>(c.patch_size * c.feat_dim);
  m.base_cache.length = 0;
  m.residual_cache.length = 0;
  // text embeddings on the host (73k x 2048 table stays off the device)
  std::vector<float> embedded(n * h);
  m.cpu->Embed(std::span<const std::int64_t>(ids, n), embedded.data());
  float* fe = m.feats.Ensure(n * pd);
  m.ToDevice(fe, feats, n * pd * sizeof(float));
  auto* mask = reinterpret_cast<std::uint8_t*>(m.mask.Ensure((n + 3) / 4));
  m.ToDevice(mask, audio_mask, n);
  float* cls = m.cls.Ensure(n * eh);
  m.LocalEncodeDevice(fe, rows, cls);
  float* femb = m.feat_embed.Ensure(n * h);
  m.Linear(m.enc_to_lm, cls, rows, femb);
  float* xs = m.x.Ensure(n * h);
  m.ToDevice(xs, embedded.data(), n * h * sizeof(float));
  LaunchSelectRows(xs, femb, mask, n, h, m.stream);
  m.StackForward(m.base, xs, 1, rows, true, &m.base_cache, m.stack_scratch);
  float* fsq = m.fsq_all.Ensure(n * h);
  m.FsqDevice(xs, rows, fsq);
  LaunchSelectRows(xs, fsq, mask, n, h, m.stream);
  float* fin = m.fusion_in.Ensure(n * 2 * h);
  LaunchConcatRows(xs, femb, mask, fin, n, h, m.stream);
  float* res = m.residual_buf.Ensure(n * h);
  m.Linear(m.fusion, fin, rows, res);
  m.StackForward(m.residual, res, 1, rows, true, &m.residual_cache, m.stack_scratch);
  m.ToHost(lm_hidden, xs + (n - 1) * h, h * sizeof(float));
  m.ToHost(residual_hidden, res + (n - 1) * h, h * sizeof(float));
}

void Model::SamplePatch(const float* lm_hidden, const float* residual_hidden,
                        const float* prev_patch, const float* noise, const FlowOptions& options,
                        float* patch) {
  Impl& m = *impl_;
  const ModelConfig& c = m.cpu->config();
  const auto h = static_cast<std::size_t>(c.base_lm.hidden_size);
  const auto dh = static_cast<std::size_t>(c.dit.hidden_size);
  const auto p = static_cast<std::size_t>(c.patch_size);
  const auto pd = static_cast<std::size_t>(c.patch_size * c.feat_dim);
  const auto& [table, schedule] = m.TimeRows(options);
  const float* t_rows = table->data;
  m.ToDevice(m.f_lm.data, lm_hidden, h * sizeof(float));
  m.ToDevice(m.f_res.data, residual_hidden, h * sizeof(float));
  m.ToDevice(m.f_prev.data, prev_patch, pd * sizeof(float));
  m.ToDevice(m.f_x.data, noise, pd * sizeof(float));
  auto body = [&] {
    m.in_flow = true;
    m.Linear(m.lm_to_dit, m.f_lm.data, 1, m.f_mu.data);
    m.Linear(m.res_to_dit, m.f_res.data, 1, m.f_mu.data + dh);
    m.Linear(m.dit_cond, m.f_prev.data, static_cast<std::int64_t>(p), m.f_cond.data);
    m.Tick("dit condition (mu, cond)");
    for (std::size_t s = 0; s < schedule.size(); ++s) {
      m.VelocityDevice(t_rows + s * dh);
      LaunchCfgEuler(m.f_x.data, m.f_vel.data, options.cfg, schedule[s].dt, pd,
                     options.cfg_zero_star, m.stream);
      m.Tick("cfg euler");
    }
    m.in_flow = false;
  };
  const auto key = std::make_tuple(options.timesteps, options.cfg, options.cfg_zero_star);
  if (!m.use_graphs || m.profiling) {
    body();
  } else if (const auto found = m.graphs.find(key); found != m.graphs.end()) {
    Check(hipGraphLaunch(found->second, m.stream), "graph launch");
  } else if (m.graph_seen[key]++ == 0) {
    body();  // first call: eager (allocates every scratch buffer)
  } else {
    hipGraph_t captured = nullptr;
    hipGraphExec_t exec = nullptr;
    bool ok = hipStreamBeginCapture(m.stream, hipStreamCaptureModeThreadLocal) == hipSuccess;
    if (ok) {
      body();
      ok = hipStreamEndCapture(m.stream, &captured) == hipSuccess && captured != nullptr &&
           hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0) == hipSuccess;
    }
    if (captured != nullptr) {
      (void)hipGraphDestroy(captured);
    }
    if (ok) {
      m.graphs.emplace(key, exec);
      Check(hipGraphLaunch(exec, m.stream), "graph launch");
    } else {
      m.use_graphs = false;  // capture failed: the body did not run
      body();
    }
  }
  m.ToHost(patch, m.f_x.data, pd * sizeof(float));
}

void Model::StopLogits(const float* lm_hidden, float* logits) {
  Impl& m = *impl_;
  const auto h = static_cast<std::size_t>(m.cpu->config().base_lm.hidden_size);
  float* buf = m.hidden.Ensure(2 * h + 64);
  m.ToDevice(buf, lm_hidden, h * sizeof(float));
  m.Linear(m.stop_proj, buf, 1, buf + h);
  ming::LaunchSilu(buf + h, h, m.stream);
  m.Linear(m.stop_head, buf + h, 1, buf + 2 * h);
  m.ToHost(logits, buf + 2 * h, 2 * sizeof(float));
}

void Model::Advance(const float* patch, float* lm_hidden, float* residual_hidden) {
  Impl& m = *impl_;
  const ModelConfig& c = m.cpu->config();
  const auto h = static_cast<std::size_t>(c.base_lm.hidden_size);
  const auto eh = static_cast<std::size_t>(c.encoder.hidden_size);
  const auto pd = static_cast<std::size_t>(c.patch_size * c.feat_dim);
  float* fe = m.feats.Ensure(pd);
  m.ToDevice(fe, patch, pd * sizeof(float));
  float* cls = m.cls.Ensure(eh);
  m.LocalEncodeDevice(fe, 1, cls);
  float* curr = m.feat_embed.Ensure(h);
  m.Linear(m.enc_to_lm, cls, 1, curr);
  float* xs = m.x.Ensure(h);
  m.Copy(xs, curr, h);
  m.StackForward(m.base, xs, 1, 1, true, &m.base_cache, m.stack_scratch);
  float* lm = m.fsq_all.Ensure(h);
  m.FsqDevice(xs, 1, lm);
  float* fin = m.fusion_in.Ensure(2 * h);
  LaunchConcatRows(lm, curr, nullptr, fin, 1, h, m.stream);
  float* res = m.residual_buf.Ensure(h);
  m.Linear(m.fusion, fin, 1, res);
  m.StackForward(m.residual, res, 1, 1, true, &m.residual_cache, m.stack_scratch);
  m.ToHost(lm_hidden, lm, h * sizeof(float));
  m.ToHost(residual_hidden, res, h * sizeof(float));
}

void Model::Velocity(const float* x, const float* mu, const float* cond, float t, float* out) {
  Impl& m = *impl_;
  const ModelConfig& c = m.cpu->config();
  const auto h = static_cast<std::size_t>(c.base_lm.hidden_size);
  const auto dh = static_cast<std::size_t>(c.dit.hidden_size);
  const auto p = static_cast<std::size_t>(c.patch_size);
  const auto pd = static_cast<std::size_t>(c.patch_size * c.feat_dim);
  std::vector<float> sinus(dh);
  m.cpu->TimeEmbedding(t, sinus.data());
  float* dev_sinus = m.small.Ensure(2 * dh);
  m.ToDevice(dev_sinus, sinus.data(), dh * sizeof(float));
  m.Linear(m.time1, dev_sinus, 1, dev_sinus + dh);
  ming::LaunchSilu(dev_sinus + dh, dh, m.stream);
  m.Linear(m.time2, dev_sinus + dh, 1, m.f_tmp.data);
  ming::LaunchAdd(m.f_tmp.data, m.delta_time, dh, m.stream);
  m.ToDevice(m.f_mu.data, mu, h * sizeof(float));
  m.ToDevice(m.f_prev.data, cond, pd * sizeof(float));
  m.ToDevice(m.f_x.data, x, pd * sizeof(float));
  m.Linear(m.dit_cond, m.f_prev.data, static_cast<std::int64_t>(p), m.f_cond.data);
  m.in_flow = true;
  m.VelocityDevice(m.f_tmp.data);
  m.in_flow = false;
  m.ToHost(out, m.f_vel.data, 2 * pd * sizeof(float));
}

void Model::LocalEncode(const float* patches, std::int64_t count, float* out) {
  Impl& m = *impl_;
  const ModelConfig& c = m.cpu->config();
  const auto eh = static_cast<std::size_t>(c.encoder.hidden_size);
  const auto pd = static_cast<std::size_t>(c.patch_size * c.feat_dim);
  const auto n = static_cast<std::size_t>(count);
  float* fe = m.feats.Ensure(n * pd);
  m.ToDevice(fe, patches, n * pd * sizeof(float));
  float* cls = m.cls.Ensure(n * eh);
  m.LocalEncodeDevice(fe, count, cls);
  m.ToHost(out, cls, n * eh * sizeof(float));
}

}  // namespace gufo::models::voxcpm2::hip
