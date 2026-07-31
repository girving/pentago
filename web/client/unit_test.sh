#!/bin/bash
# Run the client unit tests under node (see BUILD.bazel)

set -e
# Bazel test actions get a minimal PATH, so add the usual node locations
PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"
# unit.js expects cwd src/, with the wasm binaries in ../public and ../build
cd "$TEST_SRCDIR/$TEST_WORKSPACE/web/client/src"
exec node unit.js
