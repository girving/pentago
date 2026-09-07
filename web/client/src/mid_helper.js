// Helper worker for the threaded tiled midsolver: instantiate mid-threads.wasm on the shared memory
// it is handed, take the stack region reserved for it, report ready, and join the worker pool
// forever (midsolve_tiled_worker blocks in atomic waits between jobs).  The owner terminates us
// when the solve finishes.  Runs under both node worker_threads and browser Workers.

const node = globalThis.process?.versions?.node

async function run({module, memory, stack}, post) {
  const M = (await WebAssembly.instantiate(module, {
    env: {
      memory,
      die: p => {
        const chars = new Uint8Array(memory.buffer)
        let s = ''
        for (; chars[p]; p++)
          s += String.fromCharCode(chars[p])
        throw Error(s)
      },
    },
  })).exports
  M.__stack_pointer.value = stack
  post('ready')
  M.midsolve_tiled_worker()
}

if (node) {
  const {parentPort, workerData} = await import('worker_threads')
  run(workerData, m => parentPort.postMessage(m))
} else
  onmessage = e => run(e.data, m => postMessage(m))
