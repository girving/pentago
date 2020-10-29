// Javascript interface to WebAssembly midsolver

import pending from './pending.js'
import { from_high_map } from './board.js'

// Make a web worker, and handle onmessage in order
const worker = new Worker('./mid_worker.js')
const callbacks = []
worker.onmessage = e => callbacks.shift()(e.data)

// Run midsolve in the worker, merging simultaneous duplicate requests
export const midsolve = pending(board => {
  const p = new Promise((resolve, reject) =>
    callbacks.push(x => x instanceof Error ? reject(x) : resolve(from_high_map(x)))
  )
  worker.postMessage(board.high())
  return p
})
