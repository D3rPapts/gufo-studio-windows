#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#include <cstdio>
#endif

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

#include "src/core/platform/env.hpp"
#include "src/models/deepseek_v4_flash/runtime/model.h"
#if defined(_WIN32)
// Upstream's POSIX environment helper, for the redirected-sink assertions.
static int setenv(const char* name, const char* value, int) {
  return _putenv_s(name, value);
}
#endif

namespace {
void Integer(std::ostream& out, std::uint64_t value, unsigned bytes) {
  for (unsigned i = 0; i < bytes; ++i)
    out.put(static_cast<char>((value >> (i * 8)) & 255));
}

void String(std::ostream& out, std::string_view text) {
  Integer(out, text.size(), 8);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::size_t OpenDescriptors() {
#if !defined(_WIN32)
  int count = 0;
  for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd"))
    ++count;
  return count;
#else
  return 0;  // /proc does not exist; the leak check is POSIX-only.
#endif
}
}  // namespace

int main() {
#if defined(_WIN32)
  // The lifecycle test leans on /proc fd accounting, mkdtemp, and POSIX
  // setenv; revisit with the DeepSeek Windows bring-up phase.
  std::puts("SKIP: POSIX-only lifecycle test");
  return 77;
#else
  char directory[] = "/tmp/gufo-ds4-load-XXXXXX";
  assert(::mkdtemp(directory));
  const auto root = std::filesystem::path(directory);
  const auto path = root / "invalid.gguf";
  const auto lock_path = root / "instance.lock";
  assert(gufo::platform::SetEnv("DS4_LOCK_FILE", lock_path.string()));
  const auto descriptors = OpenDescriptors();
  for (int fixture = 0; fixture < 3; ++fixture) {
    {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (fixture == 0) {
        out << "truncated";
      } else {
        Integer(out, 0x46554747, 4);
        Integer(out, 3, 4);
        Integer(out, 0, 8);  // No tensors.
        Integer(out, fixture == 1 ? 0 : 2, 8);
        if (fixture == 2) {
          String(out, "tokenizer.ggml.tokens");
          Integer(out, 9, 4);  // Array.
          Integer(out, 8, 4);  // Strings.
          Integer(out, 1, 8);
          String(out, "hello");  // Required special tokens are missing.
          String(out, "tokenizer.ggml.merges");
          Integer(out, 9, 4);
          Integer(out, 8, 4);
          Integer(out, 0, 8);
        }
        while (out.tellp() < 32 || out.tellp() % 32)
          out.put(0);
      }
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
      ds4_engine* engine = nullptr;
      const ds4_engine_options options{path.c_str(), nullptr};
      assert(ds4_engine_open(&engine, &options) != 0);
      assert(engine == nullptr);
      // Failure must release both mapped files and the process lock.
#if defined(_WIN32)
      const int lock = _open(lock_path.c_str(), _O_RDWR);
      assert(lock >= 0);
      _lseek(lock, 0, SEEK_SET);
      assert(_locking(lock, _LK_NBLCK, 1) == 0);
      // A separately held lock must cause another ordinary load failure.
      assert(ds4_engine_open(&engine, &options) != 0 && engine == nullptr);
      _close(lock);
#else
      const int lock = ::open(lock_path.c_str(), O_RDWR);
      assert(lock >= 0 && ::flock(lock, LOCK_EX | LOCK_NB) == 0);
      // A separately held lock must cause another ordinary load failure.
      assert(ds4_engine_open(&engine, &options) != 0 && engine == nullptr);
      ::close(lock);
#endif
      assert(OpenDescriptors() == descriptors);
    }
  }
  std::filesystem::remove_all(root);
#endif  // !defined(_WIN32)
}
