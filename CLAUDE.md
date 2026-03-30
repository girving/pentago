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
- No `__attribute__((target(...)))` — we compile with `-march=native` so AVX2 is always available
- Always build and test with `--copt=-march=native` (needed for AVX2 SIMD paths)
- Don't allocate large intermediate buffers when direct access suffices
- Validate untrusted input upfront (e.g. assert stream lengths sum correctly) rather than clamping during use
- When changing a serialization format, commit the format change first with determinism hashes, then optimize — hashes must not change during optimization
