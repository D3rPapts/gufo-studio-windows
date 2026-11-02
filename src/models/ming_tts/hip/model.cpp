// Written with AI assistance (Claude Code).
#include "src/models/ming_tts/hip/model.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/models/fish_drama3/hip/backbone_ops.hpp"
#include "src/models/ming_tts/hip/ops.hpp"

namespace gufo::models::ming_tts::hip {
namespace {

// The 512-position static cache of the reference runtime.
constexpr std::int64_t kMaxCache = 512;
// Up to this many rows a matrix product runs as a BF16 GEMV per row; above
// it rocBLAS (BF16 inputs, FP32 accumulate/output) takes over.
constexpr std::int64_t kGemvRows = 8;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("HIP ") + what + ": " +
                             hipGetErrorString(status));
  }
}

void Check(rocblas_status status, const char* what) {
  if (status != rocblas_status_success) {
    throw std::runtime_error(std::string("rocBLAS ") + what + " failed (" +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}

struct DevLinear {
  const std::uint16_t* weight{nullptr};
  const float* bias{nullptr};
  std::int64_t out{0};
  std::int64_t in{0};
};

struct DevQwen2Layer {
  DevLinear q, k, v, o, gate, up, down;
  const float* input_norm{nullptr};
  const float* post_norm{nullptr};
};

struct DevQwen2 {
  Qwen2Config config;
  std::vector<DevQwen2Layer> layers;
  const float* final_norm{nullptr};
};

struct DevDitBlock {
  DevLinear q, k, v, out, ff_in, ff_out;
  const float* norm1{nullptr};
  const float* norm2{nullptr};
};

struct DevDit {
  DitConfig config;
  std::vector<DevDitBlock> blocks;
  const float* inv_freq{nullptr};
  const float* final_norm{nullptr};
  DevLinear final_linear;
};

// Grow-only device scratch buffer.
struct Scratch {
  float* data{nullptr};
  std::size_t capacity{0};
  float* Ensure(std::size_t elements) {
    if (elements > capacity) {
      if (data != nullptr) {
        (void)hipFree(data);
        data = nullptr;
      }
      Check(hipMalloc(reinterpret_cast<void**>(&data), elements * sizeof(float)),
            "scratch alloc");
      capacity = elements;
    }
    return data;
  }
  ~Scratch() {
    if (data != nullptr) {
      (void)hipFree(data);
    }
  }
};

}  // namespace

bool DeviceAvailable() noexcept {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

struct Model::Impl {
  const cpu::Model* cpu{nullptr};
  hipStream_t stream{nullptr};
  rocblas_handle blas{nullptr};
  std::vector<void*> allocations;

  DevQwen2 llm, vae;
  DevDit flow, aggregator;
  DevLinear time_mlp0, time_mlp2, flow_x, flow_c, agg_x, vae_fc1, head_out;
  const float* agg_cls{nullptr};
  const float* window{nullptr};
  const float* cos_sin{nullptr};

  std::vector<float*> cache_k, cache_v;
  std::int64_t cache_length{0};

  // Cached t-embedding rows for a given number of flow steps: [steps, d].
  std::int32_t t_steps{0};
  std::vector<float> t_values;
  Scratch t_table;

  // scratch
  Scratch hidden, normed, q, k, v, attn, proj, gate, up, bf16, misc, head,
      frames, audio;

  ~Impl() {
    if (blas != nullptr) {
      (void)rocblas_destroy_handle(blas);
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
    Check(hipMemcpy(device, host, count * sizeof(T), hipMemcpyHostToDevice),
          "weight upload");
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
    if (out.in % 8 != 0) {
      throw std::runtime_error("Ming HIP runtime needs in_features % 8 == 0");
    }
    return out;
  }
  DevQwen2 UploadQwen2(const cpu::Qwen2Stack& stack) {
    DevQwen2 out;
    out.config = stack.config;
    for (const cpu::Qwen2Layer& layer : stack.layers) {
      DevQwen2Layer d;
      d.q = UploadLinear(layer.q);
      d.k = UploadLinear(layer.k);
      d.v = UploadLinear(layer.v);
      d.o = UploadLinear(layer.o);
      d.gate = UploadLinear(layer.gate);
      d.up = UploadLinear(layer.up);
      d.down = UploadLinear(layer.down);
      d.input_norm = UploadVector(layer.input_norm);
      d.post_norm = UploadVector(layer.post_norm);
      out.layers.push_back(d);
    }
    out.final_norm = UploadVector(stack.final_norm);
    return out;
  }
  DevDit UploadDit(const cpu::DitStack& stack) {
    DevDit out;
    out.config = stack.config;
    for (const cpu::DitBlock& block : stack.blocks) {
      DevDitBlock d;
      d.q = UploadLinear(block.q);
      d.k = UploadLinear(block.k);
      d.v = UploadLinear(block.v);
      d.out = UploadLinear(block.out);
      d.ff_in = UploadLinear(block.ff_in);
      d.ff_out = UploadLinear(block.ff_out);
      d.norm1 = UploadVector(block.norm1);
      d.norm2 = UploadVector(block.norm2);
      out.blocks.push_back(d);
    }
    out.inv_freq = UploadVector(stack.inv_freq);
    out.final_norm = UploadVector(stack.final_norm);
    out.final_linear = UploadLinear(stack.final_linear);
    return out;
  }

  // y[rows, out] = x[rows, in] W^T (+ b)
  void Linear(const DevLinear& layer, const float* x, std::int64_t rows, float* y) {
    if (rows <= kGemvRows) {
      fish_drama3::hip::LaunchBackboneBf16Gemv(
          layer.weight, x, y, static_cast<std::size_t>(rows),
          static_cast<std::size_t>(layer.out), static_cast<std::size_t>(layer.in),
          stream);
    } else {
      // bf16 scratch holds uint16 values; size in floats rounded up.
      auto* x_bf16 = reinterpret_cast<std::uint16_t*>(
          bf16.Ensure(static_cast<std::size_t>((rows * layer.in + 1) / 2)));
      fish_drama3::hip::LaunchBackboneFloatToBf16(
          x, x_bf16, static_cast<std::size_t>(rows * layer.in), stream);
      const float alpha = 1.0F;
      const float beta = 0.0F;
      Check(rocblas_gemm_ex(blas, rocblas_operation_transpose,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(layer.out),
                            static_cast<rocblas_int>(rows),
                            static_cast<rocblas_int>(layer.in), &alpha,
                            layer.weight, rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(layer.in), x_bf16,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(layer.in), &beta, y,
                            rocblas_datatype_f32_r,
                            static_cast<rocblas_int>(layer.out), y,
                            rocblas_datatype_f32_r,
                            static_cast<rocblas_int>(layer.out),
                            rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                            0, 0),
            "gemm_ex");
    }
    if (layer.bias != nullptr) {
      LaunchAddBias(y, layer.bias, static_cast<std::size_t>(rows),
                    static_cast<std::size_t>(layer.out), stream);
    }
  }

  void Qwen2Forward(const DevQwen2& stack, float* x, std::int64_t rows,
                    std::int64_t first_position, bool use_cache) {
    const Qwen2Config& c = stack.config;
    const auto h = static_cast<std::size_t>(c.hidden_size);
    const auto n = static_cast<std::size_t>(rows);
    const std::size_t q_width = static_cast<std::size_t>(c.num_heads) * c.head_dim;
    const std::size_t kv_width = static_cast<std::size_t>(c.num_kv_heads) * c.head_dim;
    float* nrm = normed.Ensure(n * h);
    float* qb = q.Ensure(n * q_width);
    float* kb = k.Ensure(n * kv_width);
    float* vb = v.Ensure(n * kv_width);
    float* ab = attn.Ensure(n * q_width);
    float* pb = proj.Ensure(n * h);
    float* gb = gate.Ensure(n * static_cast<std::size_t>(c.intermediate_size));
    float* ub = up.Ensure(n * static_cast<std::size_t>(c.intermediate_size));
    for (std::size_t index = 0; index < stack.layers.size(); ++index) {
      const DevQwen2Layer& layer = stack.layers[index];
      LaunchRmsNorm(x, layer.input_norm, nrm, n, h, c.rms_norm_eps, stream);
      Linear(layer.q, nrm, rows, qb);
      Linear(layer.k, nrm, rows, kb);
      Linear(layer.v, nrm, rows, vb);
      LaunchRopeHalf(qb, n, static_cast<std::size_t>(c.num_heads),
                     static_cast<std::size_t>(c.head_dim),
                     static_cast<std::size_t>(first_position), c.rope_theta, stream);
      LaunchRopeHalf(kb, n, static_cast<std::size_t>(c.num_kv_heads),
                     static_cast<std::size_t>(c.head_dim),
                     static_cast<std::size_t>(first_position), c.rope_theta, stream);
      const float* keys = kb;
      const float* values = vb;
      std::size_t kv_len = n;
      std::size_t q_offset = 0;
      if (use_cache) {
        float* ck = cache_k[index] + static_cast<std::size_t>(first_position) * kv_width;
        float* cv = cache_v[index] + static_cast<std::size_t>(first_position) * kv_width;
        Check(hipMemcpyAsync(ck, kb, n * kv_width * sizeof(float),
                             hipMemcpyDeviceToDevice, stream),
              "cache k");
        Check(hipMemcpyAsync(cv, vb, n * kv_width * sizeof(float),
                             hipMemcpyDeviceToDevice, stream),
              "cache v");
        keys = cache_k[index];
        values = cache_v[index];
        kv_len = static_cast<std::size_t>(first_position) + n;
        q_offset = static_cast<std::size_t>(first_position);
      }
      LaunchAttention(qb, keys, values, ab, 1, n, kv_len,
                      static_cast<std::size_t>(c.num_heads),
                      static_cast<std::size_t>(c.num_kv_heads),
                      static_cast<std::size_t>(c.head_dim), 0, 0, true, q_offset,
                      static_cast<std::size_t>(c.sliding_window), stream);
      Linear(layer.o, ab, rows, pb);
      LaunchAdd(x, pb, n * h, stream);
      LaunchRmsNorm(x, layer.post_norm, nrm, n, h, c.rms_norm_eps, stream);
      Linear(layer.gate, nrm, rows, gb);
      Linear(layer.up, nrm, rows, ub);
      LaunchSiluMul(gb, ub, n * static_cast<std::size_t>(c.intermediate_size), stream);
      Linear(layer.down, gb, rows, pb);
      LaunchAdd(x, pb, n * h, stream);
    }
  }

  // DiT/aggregator blocks on x [batch * seq, d] (in place).
  void DitForward(const DevDit& stack, float* x, std::int64_t batch,
                  std::int64_t seq, float* nrm, float* qb, float* kb, float* vb,
                  float* ab, float* pb, float* fb) {
    const auto d = static_cast<std::size_t>(stack.config.hidden_size);
    const auto heads = static_cast<std::size_t>(stack.config.num_heads);
    const std::size_t head_dim = d / heads;
    const std::int64_t rows = batch * seq;
    const auto n = static_cast<std::size_t>(rows);
    const std::size_t inner = d * static_cast<std::size_t>(stack.config.mlp_ratio);
    for (const DevDitBlock& block : stack.blocks) {
      LaunchRmsNorm(x, block.norm1, nrm, n, d, 1e-6F, stream);
      Linear(block.q, nrm, rows, qb);
      Linear(block.k, nrm, rows, kb);
      Linear(block.v, nrm, rows, vb);
      LaunchRopeInterleaved(qb, n, static_cast<std::size_t>(seq), heads, head_dim,
                            stack.inv_freq, stream);
      LaunchRopeInterleaved(kb, n, static_cast<std::size_t>(seq), heads, head_dim,
                            stack.inv_freq, stream);
      LaunchAttention(qb, kb, vb, ab, static_cast<std::size_t>(batch),
                      static_cast<std::size_t>(seq), static_cast<std::size_t>(seq),
                      heads, heads, head_dim, static_cast<std::size_t>(seq) * d,
                      static_cast<std::size_t>(seq) * d, false, 0, 0, stream);
      Linear(block.out, ab, rows, pb);
      LaunchAdd(x, pb, n * d, stream);
      LaunchRmsNorm(x, block.norm2, nrm, n, d, 1e-6F, stream);
      Linear(block.ff_in, nrm, rows, fb);
      LaunchGeluTanh(fb, n * inner, stream);
      Linear(block.ff_out, fb, rows, pb);
      LaunchAdd(x, pb, n * d, stream);
    }
  }

  void Sync() { Check(hipStreamSynchronize(stream), "synchronize"); }

  void ToDevice(float* device, const float* host, std::size_t count) {
    Check(hipMemcpyAsync(device, host, count * sizeof(float), hipMemcpyHostToDevice,
                         stream),
          "upload");
  }
  void ToHost(float* host, const float* device, std::size_t count) {
    Check(hipMemcpyAsync(host, device, count * sizeof(float), hipMemcpyDeviceToHost,
                         stream),
          "download");
    Sync();
  }

  // Buffers for the DiT path, carved from `misc`:
  struct DitBuffers {
    float *cond, *hist, *y, *c_cond, *x_hist, *x_now, *seq, *nrm, *q, *k, *v,
        *attn, *proj, *ff, *last, *vel, *sde, *sinus, *t_hidden;
  };
  DitBuffers CarveDit() {
    const std::size_t d = static_cast<std::size_t>(flow.config.hidden_size);
    const std::size_t inner = d * static_cast<std::size_t>(flow.config.mlp_ratio);
    const std::size_t hdim = static_cast<std::size_t>(cpu->config().llm.hidden_size);
    const std::size_t lat = static_cast<std::size_t>(cpu->config().latent_dim);
    // Every slice is rounded to 64 floats to keep 256-byte alignment.
    auto round = [](std::size_t n) { return (n + 63) / 64 * 64; };
    const std::size_t sizes[] = {hdim, lat, lat, d, d, d, 6 * d, 6 * d, 6 * d,
                                 6 * d, 6 * d, 6 * d, 6 * d, 6 * inner, 2 * d,
                                 2 * lat, 32 * lat, 256, d};
    std::size_t total = 0;
    for (const std::size_t s : sizes) {
      total += round(s);
    }
    float* base = misc.Ensure(total);
    float* slices[19];
    std::size_t offset = 0;
    for (int i = 0; i < 19; ++i) {
      slices[i] = base + offset;
      offset += round(sizes[i]);
    }
    return DitBuffers{slices[0],  slices[1],  slices[2],  slices[3],  slices[4],
                      slices[5],  slices[6],  slices[7],  slices[8],  slices[9],
                      slices[10], slices[11], slices[12], slices[13], slices[14],
                      slices[15], slices[16], slices[17], slices[18]};
  }

  // t-embedding rows for the sampler timesteps (cached per step count).
  const float* TimeRows(std::int32_t steps, const DitBuffers& b) {
    if (steps != t_steps) {
      const std::size_t d = static_cast<std::size_t>(flow.config.hidden_size);
      t_values = cpu::Model::FlowTimesteps(steps);
      float* table = t_table.Ensure(static_cast<std::size_t>(steps) * d);
      for (std::int32_t s = 0; s < steps; ++s) {
        LaunchTimestepSinus(t_values[static_cast<std::size_t>(s)], b.sinus, stream);
        Linear(time_mlp0, b.sinus, 1, b.t_hidden);
        LaunchSilu(b.t_hidden, d, stream);
        Linear(time_mlp2, b.t_hidden, 1, table + static_cast<std::size_t>(s) * d);
      }
      Sync();
      t_steps = steps;
    }
    return t_table.data;
  }

  void Velocity(const DitBuffers& b, const float* t_row, const float* y_dev) {
    const std::size_t d = static_cast<std::size_t>(flow.config.hidden_size);
    Linear(flow_x, y_dev, 1, b.x_now);
    LaunchAssembleDitInput(t_row, b.c_cond, flow_c.bias, b.x_hist, b.x_now, b.seq,
                           d, stream);
    DitForward(flow, b.seq, 2, 3, b.nrm, b.q, b.k, b.v, b.attn, b.proj, b.ff);
    LaunchRmsNorm(b.seq + 2 * d, flow.final_norm, b.last, 1, d, 1e-6F, stream);
    LaunchRmsNorm(b.seq + 5 * d, flow.final_norm, b.last + d, 1, d, 1e-6F, stream);
    Linear(flow.final_linear, b.last, 2, b.vel);
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
    Check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking), "stream");
    Check(rocblas_create_handle(&m.blas), "create handle");
    Check(rocblas_set_atomics_mode(m.blas, rocblas_atomics_not_allowed), "atomics");
    Check(rocblas_set_stream(m.blas, m.stream), "set stream");
    m.llm = m.UploadQwen2(cpu_model.llm());
    m.vae = m.UploadQwen2(cpu_model.vae_decoder().stack);
    m.flow = m.UploadDit(cpu_model.flow_head().stack);
    m.time_mlp0 = m.UploadLinear(cpu_model.flow_head().time_mlp0);
    m.time_mlp2 = m.UploadLinear(cpu_model.flow_head().time_mlp2);
    m.flow_x = m.UploadLinear(cpu_model.flow_head().x_embedder);
    m.flow_c = m.UploadLinear(cpu_model.flow_head().c_embedder);
    m.aggregator = m.UploadDit(cpu_model.aggregator().stack);
    m.agg_x = m.UploadLinear(cpu_model.aggregator().x_embedder);
    m.agg_cls = m.UploadVector(cpu_model.aggregator().cls);
    m.vae_fc1 = m.UploadLinear(cpu_model.vae_decoder().fc1);
    m.head_out = m.UploadLinear(cpu_model.vae_decoder().head_out);
    m.window = m.UploadVector(cpu_model.vae_decoder().window);
    const std::size_t n_fft = cpu_model.vae_decoder().window.size();
    std::vector<float> table(2 * n_fft);
    for (std::size_t i = 0; i < n_fft; ++i) {
      const double angle = 2.0 * std::numbers::pi * static_cast<double>(i) /
                           static_cast<double>(n_fft);
      table[2 * i] = static_cast<float>(std::cos(angle));
      table[2 * i + 1] = static_cast<float>(std::sin(angle));
    }
    m.cos_sin = m.UploadVector(table);
    const Qwen2Config& c = cpu_model.config().llm;
    const std::size_t kv_width = static_cast<std::size_t>(c.num_kv_heads) * c.head_dim;
    for (std::int32_t layer = 0; layer < c.num_layers; ++layer) {
      void* kp = nullptr;
      void* vp = nullptr;
      Check(hipMalloc(&kp, kMaxCache * kv_width * sizeof(float)), "cache alloc");
      m.allocations.push_back(kp);
      Check(hipMalloc(&vp, kMaxCache * kv_width * sizeof(float)), "cache alloc");
      m.allocations.push_back(vp);
      m.cache_k.push_back(static_cast<float*>(kp));
      m.cache_v.push_back(static_cast<float*>(vp));
    }
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = std::string("Ming HIP runtime: ") + exception.what();
    }
    return nullptr;
  }
  return std::unique_ptr<Model>(new Model(std::move(impl)));
}

void Model::ResetBackbone() { impl_->cache_length = 0; }

void Model::RunBackbone(const float* embeddings, std::int64_t rows, float* normed) {
  Impl& m = *impl_;
  const auto h = static_cast<std::size_t>(m.llm.config.hidden_size);
  if (m.cache_length + rows > kMaxCache) {
    throw std::runtime_error("Ming HIP backbone exceeds the 512-position cache");
  }
  const auto n = static_cast<std::size_t>(rows);
  float* x = m.hidden.Ensure(n * h);
  m.ToDevice(x, embeddings, n * h);
  m.Qwen2Forward(m.llm, x, rows, m.cache_length, true);
  m.cache_length += rows;
  float* out = m.audio.Ensure(n * h);
  LaunchRmsNorm(x, m.llm.final_norm, out, n, h, m.llm.config.rms_norm_eps, m.stream);
  m.ToHost(normed, out, n * h);
}

float Model::StopAt(const float* z) {
  // 2 x 896: cheaper on the host than a device round trip.
  const cpu::Linear& head = impl_->cpu->stop_head();
  float logits[2];
  cpu::LinearForward(head, z, 1, logits);
  return 1.0F / (1.0F + std::exp(logits[0] - logits[1]));
}

void Model::FlowVelocity(const float* x, float t, const float* cond,
                         const float* history, float* cond_uncond) {
  Impl& m = *impl_;
  const Impl::DitBuffers b = m.CarveDit();
  const std::size_t d = static_cast<std::size_t>(m.flow.config.hidden_size);
  const std::size_t lat = static_cast<std::size_t>(m.cpu->config().latent_dim);
  m.ToDevice(b.cond, cond, static_cast<std::size_t>(m.cpu->config().llm.hidden_size));
  m.ToDevice(b.hist, history, lat);
  m.ToDevice(b.y, x, lat);
  m.Linear(m.flow_c, b.cond, 1, b.c_cond);
  m.Linear(m.flow_x, b.hist, 1, b.x_hist);
  LaunchTimestepSinus(t, b.sinus, m.stream);
  m.Linear(m.time_mlp0, b.sinus, 1, b.t_hidden);
  LaunchSilu(b.t_hidden, d, m.stream);
  float* t_row = m.proj.Ensure(d);
  m.Linear(m.time_mlp2, b.t_hidden, 1, t_row);
  m.Velocity(b, t_row, b.y);
  m.ToHost(cond_uncond, b.vel, 2 * lat);
}

void Model::SampleFlow(const float* cond, const float* history, const float* noise,
                       const float* sde_noise, const FlowOptions& options,
                       float* latent) {
  Impl& m = *impl_;
  const Impl::DitBuffers b = m.CarveDit();
  const std::size_t d = static_cast<std::size_t>(m.flow.config.hidden_size);
  const std::size_t lat = static_cast<std::size_t>(m.cpu->config().latent_dim);
  const float* t_rows = m.TimeRows(options.steps, b);
  m.ToDevice(b.cond, cond, static_cast<std::size_t>(m.cpu->config().llm.hidden_size));
  m.ToDevice(b.hist, history, lat);
  m.ToDevice(b.y, noise, lat);
  const bool stochastic = options.temperature > 0.0F && sde_noise != nullptr;
  if (stochastic) {
    m.ToDevice(b.sde, sde_noise, static_cast<std::size_t>(options.steps) * lat);
  }
  m.Linear(m.flow_c, b.cond, 1, b.c_cond);
  m.Linear(m.flow_x, b.hist, 1, b.x_hist);
  for (std::int32_t s = 0; s < options.steps; ++s) {
    const float dt = m.t_values[static_cast<std::size_t>(s) + 1] -
                     m.t_values[static_cast<std::size_t>(s)];
    m.Velocity(b, t_rows + static_cast<std::size_t>(s) * d, b.y);
    LaunchEulerStep(b.y, b.vel, options.cfg, dt, lat, m.stream);
    if (stochastic) {
      const float scale = options.sigma * std::sqrt(options.temperature) *
                          std::sqrt(std::fabs(dt));
      LaunchAxpy(b.y, b.sde + static_cast<std::size_t>(s) * lat, scale, lat, m.stream);
    }
  }
  m.ToHost(latent, b.y, lat);
}

void Model::AggregateLatent(const float* latent, float* out) {
  Impl& m = *impl_;
  const Impl::DitBuffers b = m.CarveDit();
  const std::size_t d = static_cast<std::size_t>(m.aggregator.config.hidden_size);
  const std::size_t lat = static_cast<std::size_t>(m.cpu->config().latent_dim);
  const std::size_t h = static_cast<std::size_t>(m.cpu->config().llm.hidden_size);
  m.ToDevice(b.y, latent, lat);
  Check(hipMemcpyAsync(b.seq, m.agg_cls, d * sizeof(float), hipMemcpyDeviceToDevice,
                       m.stream),
        "cls");
  m.Linear(m.agg_x, b.y, 1, b.seq + d);
  m.DitForward(m.aggregator, b.seq, 1, 2, b.nrm, b.q, b.k, b.v, b.attn, b.proj, b.ff);
  LaunchRmsNorm(b.seq, m.aggregator.final_norm, b.last, 1, d, 1e-6F, m.stream);
  m.Linear(m.aggregator.final_linear, b.last, 1, b.cond);
  m.ToHost(out, b.cond, h);
}

std::vector<float> Model::DecodeAudio(const float* latents, std::int64_t frames) {
  Impl& m = *impl_;
  const ModelConfig& c = m.cpu->config();
  const auto h = static_cast<std::size_t>(m.vae.config.hidden_size);
  const auto lat = static_cast<std::size_t>(c.latent_dim);
  const auto factor = static_cast<std::size_t>(c.vae_patch_size);
  const auto hop = static_cast<std::size_t>(c.vae_hop_size);
  const std::size_t n_fft = hop * 4;
  const auto s = static_cast<std::size_t>(frames);
  const std::size_t steps = s * factor;
  // Latents and fc1 output share `misc`-independent buffers: use `head` for
  // the latents/fc1 stage before the ISTFT head reuses it.
  float* staging = m.head.Ensure(std::max(steps * (n_fft + 2), s * (lat + h)));
  float* lat_dev = staging;
  float* fc1 = staging + s * lat;
  m.ToDevice(lat_dev, latents, s * lat);
  m.Linear(m.vae_fc1, lat_dev, frames, fc1);
  float* x = m.hidden.Ensure(steps * h);
  LaunchLinearUpsample(fc1, x, s, h, factor, m.stream);
  m.Qwen2Forward(m.vae, x, static_cast<std::int64_t>(steps), 0, false);
  float* nrm = m.audio.Ensure(steps * h);
  LaunchRmsNorm(x, m.vae.final_norm, nrm, steps, h, m.vae.config.rms_norm_eps, m.stream);
  float* head = m.head.Ensure(steps * (n_fft + 2));
  m.Linear(m.head_out, nrm, static_cast<std::int64_t>(steps), head);
  float* framed = m.frames.Ensure(steps * n_fft);
  LaunchIrfftWindow(head, m.window, m.cos_sin, framed, steps, n_fft, m.stream);
  float* audio = m.audio.Ensure(steps * hop);
  LaunchOverlapAdd(framed, m.window, audio, steps, n_fft, hop, m.stream);
  std::vector<float> out(steps * hop);
  m.ToHost(out.data(), audio, out.size());
  return out;
}

}  // namespace gufo::models::ming_tts::hip
