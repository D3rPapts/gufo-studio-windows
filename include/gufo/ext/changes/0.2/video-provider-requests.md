# Provider-owned video requests

Extend the video capability (`include/gufo/ext/video.hpp`) so a video
provider can accept conditioning the core does not model:

- `VideoAttachment`: a named binary input (uploaded multipart file or bytes a
  provider decoded from the request body).
- `VideoRequest::options_json` and `VideoRequest::attachments`: provider-owned
  options and inputs handed to `VideoRunner::Generate`.
- `VideoRequestInput`, `VideoPreparedRequest`, `VideoRequestError` and the
  optional `VideoProvider::prepare_request` hook: the provider parses and
  validates a whole `POST /v1/videos` request; the server keeps the job
  lifecycle and treats the provider id (plus configured aliases) as the model.
- `VideoProvider::probe_priority`: when several providers accept a model
  directory, `ProbeVideoProvider` returns the highest priority (ties keep
  registration order).

All additions default to the previous behaviour. Providers built against the
earlier layout must be rebuilt because `VideoRequest` and `VideoProvider`
gained members.
