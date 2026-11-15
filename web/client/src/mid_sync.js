// Javascript interface to the WebAssembly midsolvers
//
// midsolve is the original dense solver for boards with 18 or more stones.  midsolve_tiled is the
// compressed, tiled solver (pentago/mid/tiled.h), which handles 16 or more stones in a few hundred MB
// and can spread its work over helper workers sharing one WebAssembly memory.  All of them, and the
// prover cascade, live in one module (mid.wasm) that imports a shared memory, so even single threaded
// solves get a fresh shared memory of their own.

// Read a wasm binary from disk under node (unit tests, which run with cwd src/) or by fetching in
// the browser, and compile it once
const node = globalThis.process?.versions?.node
const read = name => node
  ? import('fs').then(fs => fs.readFileSync('../public/' + name))
  : fetch('/' + name).then(r => r.arrayBuffer())
const compile = name => read(name).then(b => WebAssembly.compile(b))
const mid_module = compile('mid.wasm')

// A fresh shared memory: 16 MB to start, growing on demand.  The hardest 16 stone boards found so
// far peak near 920 MB in the tiled solver, so ask for a 2 GB ceiling and fall back if the platform
// refuses to reserve that much.  ceiling records the maximum (in 64 KB pages) that worked.
let ceiling = 0
function shared_memory() {
  for (const maximum of ceiling ? [ceiling] : [32768, 16384, 8192]) {
    try {
      const memory = new WebAssembly.Memory({initial: 256, maximum, shared: true})
      ceiling = maximum
      return memory
    } catch (e) { if (ceiling || maximum == 8192) throw e }
  }
}

const read_char_p = (memory, p) => {
  const chars = new Uint8Array(memory.buffer)
  let s = ''
  for (; chars[p]; p++)
    s += String.fromCharCode(chars[p])
  return s
}

// Instantiate a fresh copy, on the given shared memory or a new one
export async function instantiate(module, memory = shared_memory()) {
  const env = {memory, die: p => { throw Error(read_char_p(memory, p)) }, now: () => performance.now() / 1000}
  // Copy the exports so we can record the imported memory alongside them
  return {...(await WebAssembly.instantiate(await (module || mid_module), {env})).exports, memory}
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

// Prover cascade for 17 stone boards (pentago/cascade/cascade.h): cheap provers settle most
// positions in well under a second, and leave the rest (null) to midsolve_tiled.  Single threaded.
export async function cascade(board) {
  const M = await instantiate()
  const results_p = M.malloc(8 + 16 * limit)
  return M.cascade_children(BigInt(board), 18, results_p) ? read_results(M, results_p) : null
}

// Start a helper worker on the shared memory, with a stack of its own
async function spawn_helper(module, memory, stack) {
  const data = {module, memory, stack}
  if (node) {
    const {Worker} = await import('worker_threads')
    const worker = new Worker(new URL('./mid_helper.js', import.meta.url), {workerData: data})
    return {worker, ready: new Promise((res, rej) => { worker.once('message', res); worker.once('error', rej) })}
  }
  // In browsers we run inside mid_worker.js, and some browsers can't start workers from a worker
  // (Safari before 15.5, some embedded Chromiums), so ask the page to start the helper (app.js's
  // start_helper) and talk to it over a channel: it answers 'ready' or an error message, and we send
  // anything to terminate it.
  const {port1, port2} = new MessageChannel()
  const ready = new Promise((res, rej) => {
    port1.onmessage = e => e.data == 'ready' ? res() : rej(Error('Helper worker failed: ' + e.data))
  })
  postMessage({helper: data, port: port2}, [port2])
  return {worker: {terminate: () => { port1.postMessage('terminate'); port1.close() }}, ready}
}

// Tiled solver, for 16 or more stones.  threads counts the caller, so threads=1 is single threaded
// and uses mid.wasm; more needs SharedArrayBuffer (cross-origin isolation in browsers).  Merged mode
// computes win and not-lose in one pass, about 25% faster for 1.5x the compressed memory, so we use
// it only when the platform grants a 2 GB memory ceiling (merged = null); pass true or false to force.
export async function midsolve_tiled(board, threads = 1, merged = null) {
  const memory = shared_memory()
  const M = await instantiate(mid_module, memory)
  if (threads <= 1 || typeof SharedArrayBuffer == 'undefined') {
    const results_p = M.malloc(8 + 16 * limit)
    M.midsolve_tiled(BigInt(board), 1, merged === true ? 1 : 0, results_p)
    return read_results(M, results_p)
  }
  if (merged === null)
    merged = ceiling >= 32768
  const module = await mid_module
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
