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
