// Javascript interface to the WebAssembly midsolvers
//
// midsolve is the original dense solver for boards with 18 or more stones.  midsolve_tiled is the
// compressed, tiled solver (pentago/mid/tiled.h), which handles 16 or more stones in a few hundred MB
// and can spread its work over helper workers sharing one WebAssembly memory (mid-threads.wasm).

// Read a wasm binary from disk under node (unit tests, which run with cwd src/) or by fetching in
// the browser, and compile it once
const node = globalThis.process?.versions?.node
const read = name => node
  ? import('fs').then(fs => fs.readFileSync('../public/' + name))
  : fetch('/' + name).then(r => r.arrayBuffer())
const compile = name => read(name).then(b => WebAssembly.compile(b))
const mid_module = compile('mid.wasm')
let threads_module = null  // Compiled on first use

const read_char_p = (memory, p) => {
  const chars = new Uint8Array(memory.buffer)
  let s = ''
  for (; chars[p]; p++)
    s += String.fromCharCode(chars[p])
  return s
}

// Instantiate a fresh copy, on the given shared memory if any
export async function instantiate(module, memory) {
  let M = null
  const env = {die: p => { throw Error(read_char_p(M.memory, p)) }}
  if (memory)
    env.memory = memory
  // Copy the exports so we can record an imported memory alongside them
  M = {...(await WebAssembly.instantiate(await (module || mid_module), {env})).exports}
  if (memory)
    M.memory = memory
  return M
}

// Room for the results of one solve, matching mid_values_t
const limit = 1 + 20 + 8*20  // 1 + MID_MAX_SPOTS + 8*MID_MAX_SPOTS

function read_results(M, results_p) {
  const read_int = p => (new Int32Array(M.memory.buffer, p, 1))[0]
  const read_board = p => {
    const b = new Uint32Array(M.memory.buffer, p, 2)
    return BigInt(b[0]) | BigInt(b[1]) << 32n
  }
  const results = {}
  const num_results = read_int(results_p)
  for (let i = 0; i < num_results; i++) {
    const tuple_p = results_p + 8 + 16 * i
    results[read_board(tuple_p)] = read_int(tuple_p + 8)
  }
  return results
}

// Dense solver, for 18 or more stones
export async function midsolve(board) {
  const M = await instantiate()
  const results_p = M.malloc(8 + 16 * limit)
  M.midsolve(BigInt(board), results_p)
  return read_results(M, results_p)
}

// Start a helper worker on the shared memory, with a stack of its own
async function spawn_helper(module, memory, stack) {
  const data = {module, memory, stack}
  if (node) {
    const {Worker} = await import('worker_threads')
    const worker = new Worker(new URL('./mid_helper.js', import.meta.url), {workerData: data})
    return {worker, ready: new Promise((res, rej) => { worker.once('message', res); worker.once('error', rej) })}
  }
  const worker = new Worker(new URL('mid_helper.js', import.meta.url), {type: 'module'})
  const ready = new Promise((res, rej) => { worker.onmessage = res; worker.onerror = rej })
  worker.postMessage(data)
  return {worker, ready}
}

// Tiled solver, for 16 or more stones.  threads counts the caller, so threads=1 is single threaded
// and uses mid.wasm; more needs SharedArrayBuffer (cross-origin isolation in browsers).  Merged mode
// computes win and not-lose in one pass, about 25% faster for 1.5x the compressed memory, so we use
// it only when the platform grants a 2 GB memory ceiling (merged = null); pass true or false to force.
export async function midsolve_tiled(board, threads = 1, merged = null) {
  if (threads <= 1 || typeof SharedArrayBuffer == 'undefined') {
    const M = await instantiate()
    const results_p = M.malloc(8 + 16 * limit)
    M.midsolve_tiled(BigInt(board), 1, merged === true ? 1 : 0, results_p)
    return read_results(M, results_p)
  }
  threads_module ||= compile('mid-threads.wasm')
  const module = await threads_module
  // 16 MB to start, growing on demand.  The hardest 16 stone boards found so far peak near 920 MB,
  // so ask for a 2 GB ceiling and fall back if the platform refuses to reserve that much.
  let memory = null, maximum = 0
  for (maximum of [32768, 16384, 8192]) {
    try { memory = new WebAssembly.Memory({initial: 256, maximum, shared: true}); break }
    catch (e) { if (maximum == 8192) throw e }
  }
  if (merged === null)
    merged = maximum >= 32768
  const M = await instantiate(module, memory)
  const stack_size = 1 << 20
  const helpers = []
  for (let i = 1; i < threads; i++)
    helpers.push(spawn_helper(module, memory, (M.malloc(stack_size + 16) + stack_size) & ~15))
  const started = await Promise.all(helpers)
  try {
    await Promise.all(started.map(h => h.ready))
    const results_p = M.malloc(8 + 16 * limit)
    M.midsolve_tiled(BigInt(board), threads, merged ? 1 : 0, results_p)
    return read_results(M, results_p)
  } finally {
    for (const h of started)
      h.worker.terminate()
  }
}
