// Pentago tree search worker process

'use strict'
const pentago = require('./build/Release/pentago')

// Allocate workspace
const workspace = Buffer.alloc(pentago.midsolve_workspace_memory_usage(18))

// Process compute requests
process.on('message', ({root, boards}) => {
  const start = Date.now()
  const results = pentago.high_midsolve(pentago.high_board_t(root), boards.map(pentago.high_board_t), workspace)
  results['time'] = (Date.now() - start) / 1000
  process.send(results)
})

// Ready to process requests
process.send('READY')
