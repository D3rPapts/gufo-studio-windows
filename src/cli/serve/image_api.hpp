#ifndef GUFO_SERVER_IMAGE_API_HPP_
#define GUFO_SERVER_IMAGE_API_HPP_

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "src/cli/serve/preview.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace gufo::server {

struct HttpRequest;
struct HttpResponse;

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
                        models::qwen_image_21::SamplingConfig sampling = {});
  ~ImageService();
  const std::string& model_id() const { return model_id_; }
  const models::qwen_image_21::SamplingConfig& sampling() const {
    return sampling_;
  }
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
  std::unique_ptr<models::qwen_image_21::Model> model_;
  std::timed_mutex gate_;
  LatentFormat preview_format_{LatentFormat::kQwenImage21};
};

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service);

}  // namespace gufo::server
#endif
