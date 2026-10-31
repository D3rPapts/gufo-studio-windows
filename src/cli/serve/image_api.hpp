#ifndef GUFO_SERVER_IMAGE_API_HPP_
#define GUFO_SERVER_IMAGE_API_HPP_

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/preview.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace gufo::server {

struct HttpRequest;
struct HttpResponse;

/// Adapters registered at startup and the set applied when a request omits
/// `loras`.
struct ImageLoraOptions {
  std::vector<std::pair<std::string, std::filesystem::path>> files;
  std::vector<models::qwen_image_21::Lora> defaults;
};

class ImageService {
public:
  using Request = models::qwen_image_21::Request;
  using Result = models::qwen_image_21::Result;
  using CancellationCheck = models::qwen_image_21::CancellationCheck;
  using Runner =
      std::function<Result(const Request&, const CancellationCheck&)>;

  explicit ImageService(const std::filesystem::path& model_root,
                        std::string model_id = "Qwen-Image-2.1",
                        Runner runner = {},
                        models::qwen_image_21::SamplingConfig sampling = {},
                        ImageLoraOptions loras = {});
  /// Adapters without an explicit sampling config (a loaded model takes it
  /// from its checkpoint).
  ImageService(const std::filesystem::path& model_root, std::string model_id,
               Runner runner, ImageLoraOptions loras);
  ~ImageService();
  const std::string& model_id() const { return model_id_; }
  const models::qwen_image_21::SamplingConfig& sampling() const {
    return sampling_;
  }
  const ImageLoraOptions& loras() const { return loras_; }
  bool HasLora(std::string_view name) const;
  /// Empty with an injected runner (no files are read).
  const models::qwen_image_21::LoraInfo& lora_info(std::string_view name) const;
  std::vector<models::qwen_image_21::Lora> ActiveLoras() const;
  Result Generate(const Request& request, const CancellationCheck& cancelled);
  /// The model's GPU context failed; the server must restart.
  bool device_lost() const { return model_ && model_->DeviceLost(); }
  /// Latent layout for live previews (MOSS-Image: kFlux2Patched).
  LatentFormat preview_format() const { return preview_format_; }
  void set_preview_format(LatentFormat format) { preview_format_ = format; }

private:
  std::string model_id_;
  models::qwen_image_21::SamplingConfig sampling_;
  Runner runner_;
  ImageLoraOptions loras_;
  std::map<std::string, models::qwen_image_21::LoraInfo, std::less<>>
      lora_info_;
  std::unique_ptr<models::qwen_image_21::Model> model_;
  std::timed_mutex gate_;
  LatentFormat preview_format_{LatentFormat::kQwenImage21};
};

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service);
/// `GET /sdapi/v1/loras`: registered adapters in registration order.
HttpResponse HandleImageLorasRequest(const HttpRequest& request,
                                     const ImageService& service);

}  // namespace gufo::server
#endif
