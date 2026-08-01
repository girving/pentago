// Web worker wasm midsolver.  Self-contained (unlike src/mid_sync.js, which
// node unit tests use) to keep the shipped bundle small: fetch and compile
// mid.wasm once, then instantiate a fresh copy per solve since the wasm bump
// allocator never frees.

const module = WebAssembly.compileStreaming(fetch('/mid.wasm'))

onmessage = async e => {
  try {
    const M = (await WebAssembly.instantiate(await module, {
      env: {die: p => {
        const chars = new Uint8Array(M.memory.buffer)
        let s = ''
        for (; chars[p]; p++)
          s += String.fromCharCode(chars[p])
        throw Error(s)
      }},
    })).exports
    const limit = 1 + 18 + 8 * 18
    const ptr = M.malloc(8 + 16 * limit)
    M.midsolve(BigInt(e.data), ptr)
    const ints = p => new Uint32Array(M.memory.buffer, p, 3)
    const results = {}
    for (let i = 0, n = ints(ptr)[0]; i < n; i++) {
      const t = ints(ptr + 8 + 16 * i)
      results[BigInt(t[0]) | BigInt(t[1]) << 32n] = t[2] | 0
    }
    postMessage(results)
  } catch (x) {
    postMessage(x)
  }
}
