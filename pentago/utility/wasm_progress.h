// WebAssembly progress bar callbacks
#pragma once

#include "pentago/utility/debug.h"
#include "pentago/utility/wasm.h"
namespace pentago {

#ifdef __wasm__
extern "C" void wasm_progress(const int percent);
#endif  // !__wasm__

class wasm_progress_t {
private:
  const uint64_t total;
  int percent, next;
  uint64_t count;

  wasm_progress_t(const wasm_progress_t&) = delete;
  void operator=(const wasm_progress_t&) = delete;
public:

  wasm_progress_t(const uint64_t total)
    : total(total), percent(0), next(0), count(0) {}

  ~wasm_progress_t() {
    if (count != total)
      die("wasm_progress_t not completed");
  }

  void step() {
    count++;
    if (count >= next) {
#ifdef __wasm__
      wasm_progress(percent);
#endif
      percent++;
      next = ceil_div(percent * total, 100);
    }
  }
};

}  // namespace pentago
