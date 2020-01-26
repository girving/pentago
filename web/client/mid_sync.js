// Javascript interface to WebAssembly midsolver

'use strict'

const board_t = require('./board.js')

// Compile mid.wasm
const mid_module = (typeof fetch === 'function'
  ? fetch('mid.wasm').then(r => r.arrayBuffer())
  : Promise.resolve(require('fs').readFileSync('public/mid.wasm')))
  .then(bytes => WebAssembly.compile(bytes))

function read_char_p(M, p) {
  const chars = new Uint8Array(M.exports.memory.buffer)
  let s = ''
  for (;;) {
    let c = chars[p++]
    if (!c) break
    s += String.fromCharCode(c)
  }
  return s
}

function read_int(M, p) {
  return new Int32Array(M.exports.memory.buffer, p, 1)[0]
}

function read_board(M, p) {
  const quads = new Uint16Array(M.exports.memory.buffer, p, 4)
  const middle = (quads[3] & 1 << 15) != 0
  const x = middle ? 0xffff : 0
  return new board_t([quads[0] ^ x, quads[1] ^ x, quads[2] ^ x, quads[3] ^ x], middle)
}

function write_board(M, p, board) {
  const x = board.middle ? 0xffff : 0
  const quads = new Uint16Array(M.exports.memory.buffer, p, 4)
  for (let i = 0; i < 4; i++)
    quads[i] = board.quads[i] ^ x
}

// Instantiate a fresh copy
async function instantiate() {
  const module = await mid_module
  let M = null
  M = await WebAssembly.instantiate(module, {
    env: {
      die: msg_p => { throw Error(read_char_p(M, msg_p)) },
      wasm_log: (name_p, value) => console.log(read_char_p(M, name_p) + value),
    }
  })
  return M
}

// Raw interface to wasm_midsolve
async function midsolve(board) {
  const M = await instantiate()

  // Allocate memory for arguments and results
  const limit = M.exports.midsolve_results_limit()
  const board_p = M.exports.wasm_malloc(12 + 12 * limit)
  const results_p = board_p + 8
  write_board(M, board_p, board)

  // Compute!
  M.exports.wasm_midsolve(board_p, results_p)

  // Read results
  const results = {}
  const num_results = read_int(M, results_p)
  for (let i = 0; i < num_results; i++) {
    const tuple_p = results_p + 4 + 12 * i
    results[read_board(M, tuple_p).name] = read_int(M, tuple_p + 8)
  }
  return results
}

// Exports
exports.instantiate = instantiate
exports.midsolve = midsolve
