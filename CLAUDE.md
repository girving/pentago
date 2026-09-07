# Pentago

A massively parallel solver for the board game Pentago. Results at https://perfect-pentago.net.

## Build

Use `bin/bazel` which wraps bazelisk with sandbox-friendly cache directories:

    bin/bazel build -c opt //pentago/...
    bin/bazel test -c opt //pentago/...

Always use `-c opt` for tests — some are slow without optimizations.

On macOS, when running `bin/bazel` via the Bash tool, always use `dangerouslyDisableSandbox: true`. Claude Code's `permissions.allow` doesn't bypass the OS-level macOS sandbox (`sandbox-exec`), which blocks Bazel from writing to its output base and binding to localhost.

In Claude remote containers (claude.ai/code), run `bin/sandbox-setup` once first: it installs bazelisk, clang/lld, and MPI, and routes bazel around download hosts the container's network proxy blocks. After that `bin/bazel` works normally.

## Project structure

- `pentago/utility/` — general utilities (threads, arrays, memory, etc.)
- `pentago/base/` — core game logic (boards, moves, symmetries, scoring)
- `pentago/search/` — forward tree search
- `pentago/data/` — file I/O, compression, serialization
- `pentago/end/` — backward endgame solver (retrograde analysis)
- `pentago/mid/` — midgame solver (18+ stones)
- `pentago/high/` — high-level public API
- `pentago/mpi/` — MPI distributed computation
- `pentago/learn/` — ML (TensorFlow ops)
- `third_party/` — external dependency BUILD files and detection (.bzl)
- `web/server/` — Node.js backend (Google Cloud Functions)
- `web/client/` — Svelte + WebAssembly frontend

## IMPORTANT: Do not commit until the user has reviewed the code

Always wait for explicit user approval before running `git commit`. Show the
user what changed and let them review. Never auto-commit after implementing,
even if all tests pass.

## Build conventions

- C++20, `-Wall -Werror`
- Compiler-specific flags use `select()` on `@platforms//os:macos` (Clang) vs `//conditions:default` (GCC)
- Common copts in `pentago/pentago.bzl` (`COPTS`), test helper `cc_tests()`
- Use `tfm::format` (tinyformat), not bare `format`
- Prefer fixing root causes over suppressing warnings
- Preserve existing comments when rewriting files — don't silently delete comments, speed logs, or explanatory notes
- Don't add includes speculatively — only if the build actually fails
- System includes go after project includes, sorted alphabetically
- Mark const arguments as const
- In tests, use `PENTAGO_ASSERT_EQ/LT/GT/LE/GE` instead of gtest `ASSERT_EQ` etc. to avoid sign-compare and dangling-else warnings
- Never use `std::` prefix when a `using` declaration suffices
- Prefer `Array`/`RawArray` over `vector` for POD types
- Even trivial destructors should be declared in the header and defined in the .cc to reduce code size
- Order function arguments with slowly-varying parameters first
- Use unnamed namespaces for file-local types; use `static` for file-local functions
- Do not commit until the user has reviewed the code
- No `__attribute__((target(...)))` — we compile with `-march=native` (in COPTS) so AVX2 is always available
- Don't allocate large intermediate buffers when direct access suffices
- Merge consecutive `GEODE_ASSERT`s into one (each assert has overhead, so `GEODE_ASSERT(a && b)` is better than two separate calls)
- Validate untrusted input upfront (e.g. assert stream lengths sum correctly) rather than clamping during use
- When changing a serialization format, commit the format change first with determinism hashes, then optimize — hashes must not change during optimization
- `-march=native` is already in COPTS in `pentago/pentago.bzl`, so `--copt=-march=native` is not needed on the command line
- Simplify loops: iterate over containers directly instead of indices into them, and avoid intermediate variables when the expression is clear (e.g. `for (const auto& r : readers)` not `for (const int i : range(readers.size())) { ... readers[i] ... }`)

## Scratch files

Write scratch/throwaway C++ programs to `tmp/` in the repo (not `/tmp`). The sandbox allows writing within the repo but blocks `/tmp/claude-1000` for `g++` output. Compile and run from there:

    g++ -O2 -std=c++20 -march=native -o tmp/foo tmp/foo.cc && tmp/foo

## Profiling

Use `perf` for line-level profiling. Requires `sudo sysctl kernel.perf_event_paranoid=-1` first (sandbox blocks this, so the user must run it via `!`). Then:

    perf record -g -o /tmp/claude-1000/perf.data bazel-bin/pentago/data/some_test --gtest_filter='...'
    perf annotate -i /tmp/claude-1000/perf.data -s 'pentago::function_name' --stdio
    perf report -i /tmp/claude-1000/perf.data --stdio --sort=symbol --no-children

The `perf record` command must run with `dangerouslyDisableSandbox: true` in Claude Code. `perf report` and `perf annotate` can run inside the sandbox.

## SIMD optimization lessons

- On Cascade Lake (this machine), `vpmulld` (mullo_epi32) is **10 cycles latency** — the dominant bottleneck in rANS encode/decode. On Zen2+ it's 5 cycles.
- `permutevar8x32` (3c) is better than `cmpeq`+`blendv` chains for 3-element table lookups. Don't try to replace it with comparison-based selection — more ops at lower latency still loses.
- Derive values instead of looking them up when cheap: e.g. `xmax = freq << 17` saves one permutevar.
- For encoder renorm (emit bytes), SIMD extract+blend beats scalar spill/reload because state8 feeds directly into the encode step.
- For decoder renorm (read bytes), scalar spill/reload beats SIMD cmpeq/blend per lane — the byte reads are inherently scalar anyway.
- Scatter-write to precomputed offsets (8 indexed byte stores per group) is cheaper than a bulk 160-byte transpose pass. Same for scatter-read on the encoder side.
- Benchmark with min-of-N iterations (N=10) for stable numbers. Single runs have ~15% noise on this machine.
- `#pragma GCC unroll N` is critical for loops over `constexpr` arrays in SIMD code — GCC won't constant-fold array indices through unrolled iterations without it. Measured 44% speedup for forward8.
- Write tests that measure the actual use case, not proxy metrics. E.g. batch diversity (how many distinct positions per training batch) is more relevant than chi-squared uniformity for ML training quality.
- Profile with `perf annotate` before guessing at bottlenecks — intuition about what's slow is often wrong (e.g. the 160-byte transpose was assumed cheap but was 15% of decoder time).

## Settled design decisions (don't re-investigate)

- **Forward search for 18+ stone boards is a dead end; the backward midsolve (`pentago/mid`) stays.**
  Re-examined in 2026 and the original rejection holds for structural reasons, not old-engine
  inefficiency. From an 18-stone root there are 86.6M downstream positions (rotations abstracted)
  vs 18! = 6.4e15 raw tree leaves, so the DAG collapses the tree by ~10^11 while best-case
  alpha-beta only collapses it by ~10^8 (sqrt(18!) = 80M leaves, about the size of the whole DAG).
  The midsolve visits every position via a perfect hash-free combinatorial index at ~18 ns per
  position per pass covering 128 rotation states; a forward node (superstandardize, hashed TT,
  super win detection) costs 100-1000x more, so forward only wins when it visits under ~1% of
  positions. Random boards do; boards on best-play lines don't, because a node is cut only when
  all 128 rotation states are decided and different rotations want different refutations, so the
  proof DAG approaches the full reachable set. Ties also need two proofs. df-pn, a lazy top-down
  memoized midsolve, forward-then-midsolve hybrids (18 children x 29.6M positions = 6x the root
  sweep), and threat-space search all fail on the same tail. Real speedups are parallelizing the
  backward sweep (outer `s0` loop in `midsolve_loop` is embarrassingly parallel per slice; Web
  Workers or a WebGPU port of the existing Metal kernel) or serving slice 18 from the bucket so the
  client only midsolves 19+ stones (3x smaller, but Coldline retrieval cost).

## Web server (web/server)

- **Zero npm dependencies, by design.** Auth (`auth.js`), HTTP (`request.js`), logging (`log.js`), the
  LRU cache (`lru.js`), and xz decoding (`xz.js` + bazel-built `xz.wasm`) are all small in-repo modules.
  Don't add packages; extend these instead. `package.json` stays because Cloud Functions requires it.
- `make test` runs the offline suite, `make test-all` adds lookups against the real bucket (needs
  `gcloud auth application-default login`). Both build `xz.wasm` via bazel first; in the Claude
  sandbox use `make BAZEL=../../bin/bazel ...` with `dangerouslyDisableSandbox: true`.
- `./deploy` needs a gcloud project: set `CLOUDSDK_CORE_PROJECT` to the project of the service account
  in `deploy` (the local gcloud config has no default). Cloud Functions gen1 installs only
  `dependencies`, never devDependencies, and ships its own functions-framework.
  `gcloud meta list-files-for-upload .` shows exactly what a deploy uploads.
- Smoke test a deploy with boards the function has never served (responses carry a one-year
  cache-control, so repeats may not reach the function), compare against a local `Values.values`
  run, and read `gcloud functions logs read pentago --region us-central1`.
- Add ad hoc checks to `unit.js` as permanent tests when generally useful; `token_source` takes
  `{adc, metadata_host}` options so tests never touch global environment.
- Cost exposure: the function is public; `--max-instances` in `deploy` is the spend bound (gen1
  serves one request per instance). Slices 15-18 are Coldline, so cache-miss floods cost retrieval
  plus 2.5x per-op fees. A kill switch that only revokes `allUsers` invoker on the function (not
  billing) was discussed but not built.

## Freestanding wasm builds (web/client/build-wasm, web/server/build-wasm)

- Sources of a Bazel Central Registry module are private to it; expose them with a
  `single_version_override` patch adding a public `filegroup` (see `third_party/xz_srcs.patch`).
  Bazel's native patcher needs exact hunk line counts.
- clang's builtin `inttypes.h` does `#include_next <inttypes.h>` and fails without a libc one; supply
  a one-line shim that includes `<stdint.h>`. `-ffreestanding` implies `-fno-builtin`, so hand-written
  `mem*` loops won't be compiled back into calls to themselves.
- liblzma: define `HAVE_CONFIG_H` and supply `config.h` (its fallback header includes `<inttypes.h>`
  unconditionally otherwise); `filter_decoder.c` includes the simple/delta filter headers even when
  those filters are compiled out, so keep their `-I` paths.
