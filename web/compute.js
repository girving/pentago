// Pentago tree search worker process

'use strict'
var pentago = require('./build/Release/pentago')

// Allocate workspace
var workspace = Buffer.alloc(pentago.midsolve_workspace_memory_usage(18))

// Process compute requests
process.on('message',function (job) {
  var root = pentago.high_board_t(job[0])
  var boards = job[1].map(pentago.high_board_t)
  var start = Date.now()
  var results = pentago.high_midsolve(root,boards,workspace)
  results['time'] = (Date.now()-start)/1000
  process.send(results)
})

// Ready to process requests
process.send('READY')
