// Javascript interface to WebAssembly midsolver

// rollup will strip this in the browser case
import fs from 'fs'

// Compile mid.wasm
const code = fs.readFileSync ? Promise.resolve(fs.readFileSync('../public/mid.wasm'))
                             : fetch('/mid.wasm').then(r => r.arrayBuffer())
const mid_module = code.then(b => WebAssembly.compile(b))

// Instantiate a fresh copy
export async function instantiate(module) {
  let M = null

  const read_char_p = p => {
    const chars = new Uint8Array(M.memory.buffer)
    let s = ''
    for (;;) {
      let c = chars[p++]
      if (!c) break
      s += String.fromCharCode(c)
    }
    return s
  }

  M = (await WebAssembly.instantiate(await (module || mid_module), {
    env: {
      die: msg_p => { throw Error(read_char_p(msg_p)) },
    }
  })).exports
  return M
}

// Raw interface to midsolve
export async function midsolve(board) {
  const M = await instantiate()

  const read_int = p => (new Int32Array(M.memory.buffer, p, 1))[0]
  const read_board = p => {
    const b = new Uint32Array(M.memory.buffer, p, 2)
    return BigInt(b[0]) | BigInt(b[1]) << 32n
  }

  // Allocate memory for arguments and results
  const limit = 1 + 18 + 8*18
  const results_p = M.malloc(8 + 16 * limit)

  // Compute!
  M.midsolve(BigInt(board), results_p)

  // Read results
  const results = {}
  const num_results = read_int(results_p)
  for (let i = 0; i < num_results; i++) {
    const tuple_p = results_p + 8 + 16 * i
    results[read_board(tuple_p)] = read_int(tuple_p + 8)
  }
  return results
}
