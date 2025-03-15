// Javascript interface to WebAssembly midsolver

import pending from './pending.js'

// Make a web worker, and handle onmessage in order
const worker = new Worker(new URL('mid_worker.js', import.meta.url))
const callbacks = []
worker.onmessage = e => callbacks.shift()(e.data)

// Run midsolve in the worker, merging simultaneous duplicate requests
const inner = pending(board => {
  const p = new Promise((resolve, reject) =>
    callbacks.push(x => x instanceof Error ? reject(x) : resolve(x))
  )
  worker.postMessage(board)
  return p
})
export const midsolve = board => inner(board.raw + '')
