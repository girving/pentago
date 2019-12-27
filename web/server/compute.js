// Pentago tree search worker process

'use strict'
const pentago = require('./build/Release/pentago')

// Allocate workspace
const workspace = Buffer.alloc(pentago.midsolve_workspace_memory_usage(18))

// Process compute requests
process.on('message', board => {
  const start = Date.now()
  const pile = pentago.midsolve(pentago.high_board_t(board), workspace)
  const results = Object.fromEntries(pile)
  results['time'] = (Date.now() - start) / 1000
  process.send(results)
})

// Ready to process requests
process.send('READY')
