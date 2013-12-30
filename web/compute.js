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
    console.log('\nevaluating '+boards[i]+' ('+i+'/'+boards.length+') ')
    pentago.clear_supertable() // Don't let high depth entries block new low depth entries
    pentago.clear_stats()
    var v = pentago.high_board_t(boards[i]).value(cache)
    var t = (Date.now()-start)/1000
    results[boards[i]] = {v:v,time:t}
    console.log('done evaluating '+boards[i]+': value '+v+', time '+t+' s ('+i+'/'+boards.length+') ')
    pentago.print_stats()
  }
  process.send(results)
})

// Ready to process requests
process.send('READY')
