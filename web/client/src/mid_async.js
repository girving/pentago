// Javascript interface to WebAssembly midsolver

import pending from './pending.js'

// Run midsolve in a web worker, merging simultaneous duplicate requests.
// The worker is created lazily so that most visitors never fetch mid.wasm.
let worker = null
const callbacks = []
const inner = pending(board => {
  if (!worker) {
    worker = new Worker(new URL('mid_worker.js', import.meta.url), {type: 'module'})
    worker.onmessage = e => callbacks.shift()(e.data)
  }
  const p = new Promise((resolve, reject) =>
    callbacks.push(x => x instanceof Error ? reject(x) : resolve(x))
  )
  worker.postMessage(board)
  return p
})
export const midsolve = board => inner(board.raw + '')
