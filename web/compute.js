// Pentago tree search worker process

'use strict'
var pentago = require('./pentago/build/Release/pentago')

// Initialize supertable
var bits = parseInt(process.env['PENTAGO_WORKER_BITS'])
if (isNaN(bits))
  throw 'supertable bits must be specified via PENTAGO_WORKER_BITS environment variable'
pentago.init_supertable(bits)

// Initialize timing system
pentago.init_threads(0,0)

// No block cache here
var cache = pentago.empty_block_cache()

// Process compute requests
process.on('message',function (boards) {
  var ordered = boards.slice(0)
  ordered.sort()
  var results = {}
  for (var i=0;i<boards.length;i++) {
    var start = Date.now()
    var v = pentago.high_board_t(boards[i]).value(cache)
    results[boards[i]] = {v:v,time:(Date.now()-start)/1000}
  }
  process.send(results)
})

// Ready to process requests
process.send('READY')
