// xz (LZMA2) decompression via WebAssembly
//
// xz.wasm is liblzma's decoder compiled freestanding by bazel (//web/server:xz_wasm, built from
// xz/xz.c by ./build-wasm).  `make xz.wasm` copies it next to this file, and ./deploy ships it.

'use strict'
const fs = require('fs')
const path = require('path')

// lzma_ret values from liblzma's base.h, plus XZ_TRAILING_DATA from xz/xz.c
const errors = {
  2: 'input stream has no integrity check',
  3: 'cannot calculate the integrity check',
  5: 'cannot allocate memory',
  6: 'memory usage limit was reached',
  7: 'file format not recognized',
  8: 'invalid or unsupported options',
  9: 'data is corrupt',
  10: 'no progress is possible',
  11: 'programming error',
  100: 'trailing data after end of stream',
}

const wasm_path = path.join(__dirname, 'xz.wasm')
let code
try {
  code = fs.readFileSync(wasm_path)
} catch (e) {
  throw Error(wasm_path + " is missing: build it with 'make xz.wasm' (bazel build //web/server:xz_wasm)")
}
const {memory, xz_alloc, xz_reset, xz_decompress} = new WebAssembly.Instance(new WebAssembly.Module(code), {}).exports

// Decompress an .xz stream that must produce exactly size bytes, returning a Buffer.
// Synchronous.  The wasm arena is reset on every call, so memory stays bounded.
exports.decompress = (compressed, size) => {
  xz_reset()
  const in_p = xz_alloc(compressed.length)
  const out_p = xz_alloc(size)
  if (!in_p || !out_p)
    throw Error('xz decompression failed: cannot allocate memory')
  // memory.buffer is detached whenever the wasm memory grows, so reacquire it after each call
  new Uint8Array(memory.buffer, in_p, compressed.length).set(compressed)
  const r = xz_decompress(in_p, compressed.length, out_p, size)
  if (r < 0)
    throw Error('xz decompression failed: ' + (errors[-r] || 'lzma error ' + -r))
  if (r != size)
    throw Error('xz decompression produced ' + r + ' bytes, expected ' + size)
  return Buffer.from(memory.buffer.slice(out_p, out_p + size))
}
