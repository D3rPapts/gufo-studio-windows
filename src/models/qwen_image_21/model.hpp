#ifndef GUFO_MODELS_QWEN_IMAGE_21_MODEL_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_MODEL_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/preview_hook.hpp"

namespace gufo::models::qwen_image_21 {

struct Image {
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgba;
};

Image DecodeImage(std::span<const std::uint8_t> bytes);
std::vector<std::uint8_t> EncodePng(const Image& image);
Image ResizeImage(const Image& image, int width, int height);
/// Keep edit noise independent of the noise that generated its references.
/// Supply the canonical resized RGBA references; generation retains its seed.
/// Image to image additionally binds the start image (already resized to the
/// output size), so its noise never replays the noise that generated it.
std::uint64_t InitialNoiseSeed(std::uint64_t seed,
                               std::span<const Image> images,
                               const Image* init_image = nullptr);

/// ComfyUI KSampler treats denoise above this as a full schedule.
inline constexpr double kFullStrength = 0.9999;

/// Which mask channel marks the area to regenerate.
/// kAuto: alpha when any pixel is translucent (OpenAI `mask`: transparent =
/// edit; ComfyUI LoadImage MASK), otherwise luminance (white = edit).
/// kAlpha: 1 - alpha (a fully opaque mask keeps everything).
/// kRed/kGreen/kBlue: that channel, white = edit (ComfyUI LoadImageMask).
enum class MaskChannel { kAuto, kAlpha, kRed, kGreen, kBlue };
/// "auto", "alpha", "red", "green" or "blue"; throws std::invalid_argument.
MaskChannel ParseMaskChannel(std::string_view name);
/// One latent cell per 16 x 16 output pixels, row-major
/// (height / 16) x (width / 16): the largest edit weight (0 keep .. 1 edit)
/// of the mask pixels the cell covers, after scaling the mask to
/// width x height (nearest area, no filter overshoot). Rounding outward keeps
/// every edited pixel editable; cells the mask does not touch stay exactly 0.
std::vector<float> MaskWeights(const Image& mask, MaskChannel channel,
                               int width, int height);
/// FP32 reference of the per-step blend (the GPU rounds the result to BF16):
/// row r of `latent` becomes m * latent + (1 - m) * kept with m = weights[r].
void BlendMasked(std::span<float> latent, std::span<const float> kept,
                 std::span<const float> weights, int cols);
/// Widens cell weights by `radius` cells with a linear falloff:
/// w = max(w, max over neighbors (w_n - d / (radius + 1))), d the Chebyshev
/// cell distance. Edited cells never lose weight; radius 0 returns `weights`.
std::vector<float> SpreadMask(std::span<const float> weights, int rows,
                              int cols, int radius);
/// Per output pixel weight of the generated image for the final composite:
/// the mask's edit value at that pixel (nearest mask pixel), widened by
/// `feather` pixels with a linear falloff (as SpreadMask, in pixels). 0
/// outside keeps the start image's pixel exactly.
std::vector<float> CompositeWeights(const Image& mask, MaskChannel channel,
                                    int width, int height, int feather);
/// result = w * result + (1 - w) * original per RGBA byte, rounded.
void CompositeMasked(Image& result, const Image& original,
                     std::span<const float> weights);
/// Default soft edge of the final composite, in output pixels.
inline constexpr int kDefaultMaskFeather = 16;
/// Default widening of the latent mask, in 16-pixel cells.
inline constexpr int kDefaultMaskBlur = 1;

/// Fun ControlNet-Union input (pose skeleton, depth, canny, lineart, HED,
/// MLSD, scribble or grayscale image, already preprocessed by the caller).
/// Needs a server started with a control branch (Model::SetControlnet).
struct ControlInput {
  /// Resized to the output canvas, alpha dropped, VAE-encoded.
  Image image;
  /// control_context_scale: multiplies every control skip; 0 disables.
  float strength{1};
  /// Fraction of the steps [start, end) that run the control branch.
  double start{0};
  double end{1};
  /// Also feed the request's start image and mask into the branch's inpaint
  /// channels (the checkpoint's inpaint recipe); needs init_image and mask.
  bool inpaint{false};
};

/// A registered adapter selected by name; strength multiplies its alpha/rank.
struct Lora {
  std::string name;
  float scale{1};
  bool operator==(const Lora&) const = default;
};

/// Regional conditioning (ComfyUI ConditioningSetArea / SetAreaPercentage /
/// SetMask, combined with ConditioningCombine). Each region is a prompt with
/// its own reference images; every step evaluates the transformer once per
/// region and blends the velocities like ComfyUI's calc_cond_batch:
///   v = sum_i(w_i * v_i) / sum_i(w_i)  per latent cell,
/// with w_i = strength * mask (or the area with ComfyUI's soft edge). The
/// base prompt (Request::prompt, Request::images) takes part with
/// `base_strength` over the whole image. Cells no conditioning covers get
/// velocity 0, as in ComfyUI.
struct RegionArea {
  double x{0}, y{0}, width{0}, height{0};
  /// false: pixels of the output image (ConditioningSetArea); true:
  /// fractions 0..1 of it (ConditioningSetAreaPercentage).
  bool fraction{false};
};
/// A ComfyUI MASK, row-major, values 0..1; stretched to the output image.
struct RegionMask {
  int width{0};
  int height{0};
  std::vector<float> values;
};
struct Region {
  std::string prompt;
  std::vector<Image> images;  // <image1>.. of this region's prompt
  float strength{1};
  std::optional<RegionArea> area;
  std::optional<RegionMask> mask;
  /// set_cond_area "mask bounds": evaluate only the mask's bounding box.
  bool mask_bounds{false};
};
/// Cells (16 x 16 output pixels) a region is evaluated on: the rectangle
/// [top, top + rows) x [left, left + cols) of the (height / 16) x
/// (width / 16) grid, and its blend weight per rectangle cell (row-major).
/// `crop` = the transformer sees only the rectangle (ComfyUI evaluates area
/// conditionings on the cropped latent); otherwise it is the whole grid.
struct RegionCells {
  int top{0}, left{0}, rows{0}, cols{0};
  bool crop{false};
  std::vector<float> weights;
  /// No cell has weight: the region is skipped.
  bool Empty() const;
};
inline constexpr std::size_t kMaxRegions = 16;
/// Reference images of a request, base and regions together.
inline constexpr std::size_t kMaxReferenceImages = 20;
/// Smallest crop the transformer is run on, in cells (crops grow to it).
inline constexpr int kMinRegionCells = 128;
/// ComfyUI's soft area edge (8 latent pixels = 4 cells here).
inline constexpr int kRegionEdgeCells = 4;
/// The cells and weights of `region` for a width x height output:
/// - area without mask: the area's cells (rounded outward), weight
///   strength, fading over kRegionEdgeCells at edges inside the image;
/// - mask: mask averaged per cell times strength, on the whole grid, on the
///   area when one is set, or on the mask's bounding box with mask_bounds;
/// - neither: the whole grid at strength.
/// Crops smaller than kMinRegionCells grow (weight 0 on the added cells).
RegionCells PlanRegion(const Region& region, int width, int height);
/// The full-strength base prompt over the whole grid.
RegionCells FullRegion(int width, int height, float strength);
/// Rows of a row-major latent (grid_cols cells per grid row, `channels`
/// values per cell) inside the crop, row-major.
std::vector<float> CropLatent(std::span<const float> latent, int grid_cols,
                              int channels, const RegionCells& cells);
/// sum += w * velocity and total += w over the region's cells; `velocity`
/// covers the crop (crop) or the whole grid.
void AccumulateRegion(std::span<float> sum, std::span<float> total,
                      std::span<const float> velocity, int grid_cols,
                      int channels, const RegionCells& cells);
/// sum / total per cell; cells with total 0 become 0.
std::vector<float> FinishRegions(std::span<const float> sum,
                                 std::span<const float> total, int channels);
/// Per-pixel edit values (0..1) of a mask image, see MaskChannel.
RegionMask RegionMaskFromImage(const Image& mask, MaskChannel channel,
                               bool invert);

struct Request {
  std::string prompt;
  std::vector<Image> images;
  int width{1024};
  int height{1024};
  // Zero selects the checkpoint's default; HTTP clients cannot request zero.
  int steps{0};
  std::uint64_t seed{0};
  /// True classifier-free guidance (Diffusers `true_cfg_scale`, ComfyUI
  /// KSampler `cfg`). Above 1 every step also evaluates `negative_prompt`
  /// (with the same reference images; empty encodes like an empty prompt)
  /// and combines `negative + guidance * (positive - negative)`. 1 keeps the
  /// unguided single-pass path.
  std::string negative_prompt;
  float guidance{1};
  /// Image to image (ComfyUI VAEEncode -> KSampler denoise). The image is
  /// resized to width x height and VAE-encoded; `strength` below 1 runs
  /// `steps` steps from the tail of an int(steps / strength) schedule,
  /// starting at sigma * noise + (1 - sigma) * image. Ignored at full
  /// strength.
  std::optional<Image> init_image;
  double strength{1};
  /// Masked editing (inpainting) of `init_image`, which it requires; any
  /// size, see MaskWeights for its channels. After every step the cells the
  /// mask keeps return to the start image noised to the next sigma, so they
  /// end as the start image's latent; edited cells denoise freely. Works at
  /// full strength (also on Turbo) and with strength below 1.
  std::optional<Image> mask;
  MaskChannel mask_channel{MaskChannel::kAuto};
  /// Copy the start image's pixels back outside the mask after decoding,
  /// with a `mask_feather` pixel soft edge (CompositeWeights), so the kept
  /// area is the start image exactly instead of its VAE round trip.
  bool mask_composite{true};
  int mask_feather{kDefaultMaskFeather};
  /// Widen the latent mask by this many cells with a falloff (SpreadMask).
  /// One cell (half weight 16 px past the mask) measured a smoother seam
  /// under the default composite; the feather covers it.
  int mask_blur{kDefaultMaskBlur};
  /// Structure control (Fun ControlNet-Union); text-only prompts.
  std::optional<ControlInput> control;
  /// Live preview of the denoised estimate; never changes the image.
  models::PreviewHook preview;
  // Applied in order; scale 0 entries are ignored.
  std::vector<Lora> loras{};
  /// Regional prompts (see Region); empty = the single prompt as before.
  std::vector<Region> regions{};
  /// Weight of the base prompt beside regions; 0 leaves it out (every
  /// conditioning of the graph has an area or a mask).
  float base_strength{1};
};

struct Metrics {
  double prompt_ms{0};
  double denoise_ms{0};
  double vae_ms{0};
  double total_ms{0};  // includes lora_ms
  std::size_t resident_weight_bytes{0};
  /// DiT condition rows (text + reference tokens) of the positive prompt and
  /// the negative prompt's own rows (references are shared).
  int condition_rows{0};
  int negative_condition_rows{0};
  /// The text encoder left GPU memory for this request's caches.
  bool released_text_encoder{false};
  double lora_ms{0};  // adapter switch, before prompt encoding
  /// Regional prompts: their condition rows and transformer passes.
  int region_condition_rows{0};
  int region_passes{0};
};

struct Result {
  Image image;
  Metrics metrics;
};

struct LoraInfo {
  std::string format;
  std::size_t targets{0};  // base weights the adapter patches
  int min_rank{0};
  int max_rank{0};
};

using CancellationCheck = std::function<bool()>;
/// The request does not fit into free GPU memory (checked before the
/// condition caches fill, and per step for large requests). The model stays
/// usable; servers answer 503.
class InsufficientMemory : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};
/// Development observer; tensors are FP32 views of the actual BF16 boundary.
struct Observer {
  std::function<void(std::string_view, std::span<const float>, int, int)> write;
  // Filter before copying tensors to the host, so trajectory checks do not
  // download every intermediate block.
  std::function<bool(std::string_view)> include;
  bool Wants(std::string_view name) const {
    return write && (!include || include(name));
  }
};

void ValidateRequest(const Request& request);
std::vector<float> FlowSigmas(int steps, int image_tokens);
/// `steps` + 1 sigmas: FlowSigmas for full strength, otherwise the last
/// `steps` + 1 of FlowSigmas(int(steps / strength)) as in ComfyUI KSampler.
std::vector<float> DenoiseSigmas(int steps, double strength, int image_tokens);

/// Base flow matching or the qualified checkpoint-supplied Turbo schedule.
struct SamplingConfig {
  std::vector<float> sample_sigmas;
  bool IsTurbo() const { return !sample_sigmas.empty(); }
  int DefaultSteps() const { return IsTurbo() ? 8 : 40; }
  int ResolveSteps(int requested) const;
  std::vector<float> Sigmas(int requested, int image_tokens) const;
};

class Model {
public:
  explicit Model(const std::filesystem::path& root);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// One execution at a time. Requests have independent RNG and prefix caches.
  Result Generate(const Request& request,
                  const CancellationCheck& cancelled = {},
                  const Observer& observer = {});
  std::vector<std::uint32_t> Tokenize(std::string_view text) const;
  /// True after a failed request left the HIP context unusable (a sticky
  /// launch failure or a failed library call); only a new process recovers.
  bool DeviceLost() const;
  /// Diagnostic lines (memory per phase, memory checks).
  void SetLog(std::function<void(const std::string&)> log);
  const SamplingConfig& sampling() const;
  /// Validate and register an adapter before serving; requests select it by
  /// name.
  LoraInfo AddLora(const std::string& name, const std::filesystem::path& path);
  /// Adapters merged into the transformer weights now (does not wait for
  /// Generate); empty while a switch is in progress or after one failed.
  std::vector<Lora> ActiveLoras() const;
  /// Map a Fun ControlNet-Union file (validated, uploaded on first use)
  /// before serving; returns its layout. Once per model.
  struct ControlnetLayout {
    int blocks{0};
    std::size_t bytes{0};
  };
  ControlnetLayout SetControlnet(const std::filesystem::path& path);
  bool HasControlnet() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen_image_21
#endif
