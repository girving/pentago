# Pentago

A massively parallel solver for the board game Pentago. Results at https://perfect-pentago.net.

## Build

Use `bin/bazel` which wraps bazelisk with sandbox-friendly cache directories:

    bin/bazel build -c opt //pentago/...
    bin/bazel test -c opt //pentago/...

Always use `-c opt` for tests — some are slow without optimizations.

On macOS, when running `bin/bazel` via the Bash tool, always use `dangerouslyDisableSandbox: true`. Claude Code's `permissions.allow` doesn't bypass the OS-level macOS sandbox (`sandbox-exec`), which blocks Bazel from writing to its output base and binding to localhost.

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

## Build conventions

- C++20, `-Wall -Werror`
- Compiler-specific flags use `select()` on `@platforms//os:macos` (Clang) vs `//conditions:default` (GCC)
- Common copts in `pentago/pentago.bzl` (`COPTS`), test helper `cc_tests()`
- Use `tfm::format` (tinyformat), not bare `format`
- Prefer fixing root causes over suppressing warnings
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
- Profile with `perf annotate` before guessing at bottlenecks — intuition about what's slow is often wrong (e.g. the 160-byte transpose was assumed cheap but was 15% of decoder time).
