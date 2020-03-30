// Javascript interface to WebAssembly midsolver

// rollup will strip this in the browser case
import fs from 'fs'

// Compile mid.wasm
const code = fs.readFileSync('../build/mid.wasm')
const mid_module = WebAssembly.compile(code)

// Instantiate a fresh copy
export async function instantiate(module) {
  let M = null

  const read_char_p = p => {
    const chars = new Uint8Array(M.exports.memory.buffer)
    let s = ''
    for (;;) {
      let c = chars[p++]
      if (!c) break
      s += String.fromCharCode(c)
    }
    return s
  }

  M = await WebAssembly.instantiate(await (module || mid_module), {
    env: {
      die: msg_p => { throw Error(read_char_p(msg_p)) },
    }
  })
  return M
}

// Raw interface to midsolve
export async function midsolve(board) {
  const M = await instantiate()

  const read_int = p => (new Int32Array(M.exports.memory.buffer, p, 1))[0]
  const read_board = p => [...new Uint16Array(M.exports.memory.buffer, p, 9)]

  function write_board(p, board) {
    // Write 9 uint16's, then an extra zero at the end to complete a uint32
    const high = new Uint16Array(M.exports.memory.buffer, p, 10)
    for (let i = 0; i < 10; i++)
      high[i] = board[i] || 0
  }

  // Allocate memory for arguments and results
  const limit = 1+18+8*18
  const board_p = M.exports.malloc(24 + 8 + 32 * limit)
  const results_p = board_p + 24
  write_board(board_p, board)

  // Compute!
  M.exports.midsolve(board_p, results_p)

  // Read results
  const results = {}
  const num_results = read_int(results_p)
  for (let i = 0; i < num_results; i++) {
    const tuple_p = results_p + 8 + 32 * i
    results[read_board(tuple_p)] = read_int(tuple_p + 24)
  }
  return results
}
