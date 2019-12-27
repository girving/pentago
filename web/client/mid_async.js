// Javascript interface to WebAssembly midsolver

'use strict'

const board_t = require('./board.js').board_t
const pending = require('./pending.js')

// Make a web worker, and handle onmessage in order
const worker = new Worker('mid_worker.js')
const callbacks = []
worker.onmessage = e => callbacks.shift()(e.data)

// Run midsolve in the worker, merging simultaneous duplicate requests
const midsolve = pending(board => {
  const p = new Promise((resolve, reject) =>
    callbacks.push(x => (x instanceof Error ? reject : resolve)(x))
  )
  worker.postMessage(board.name)
  return p
})

// Exports
exports.midsolve = midsolve
