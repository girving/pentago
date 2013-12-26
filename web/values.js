// Asynchronous board value lookup and compute

'use strict'
var os = require('os')
var http = require('http')
var time = require('time')
var WorkQueue = require('mule').WorkQueue
var pentago = require('./pentago/build/Release/pentago')

// Pull in math
var max = Math.max

exports.defaults = {
  bits: 22,
  pool: os.cpus().length,
  cache: '250M',
  maxSlice: 18
}

exports.add_options = function (options) {
  var d = exports.defaults
  options.option('-b,--bits <n>','Size of transposition table in bits (actual size is 80<<bits)',parseInt,d.bits)
         .option('--pool <n>','Number of worker compute processes (defaults to cpu count)',parseInt,d.pool)
         .option('--cache <size>','Size of block cache (suffixes M/MB and G/GB are understood)',d.cache)
         .option('--max-slice <n>','Maximum slice available in database (for debugging use only)',parseInt,d.maxSlice)
}

// Create an evaluation routine with calling convention
//   values(board,cont)
// The options are
//   bits: Size of transposition table in bits (actual size is 80<<bits)
//   pool: Number of worker compute processes (defaults to cpu count)
//   cache: Size of block cache (suffixes M/MB and G/GB are understood)
//   maxSlice: Maximum slice available in database (for debugging use only)
exports.values = function (options,log) {
  // Incorporate defaults
  var opts = {}
  for (var k in exports.defaults)
    opts[k] = options[k] || exports.defaults[k]

  // Prepare for opening book lookups
  var m = opts.cache.match(/^(\d+)(M|MB|G|GB)$/)
  if (!m) {
    log.error("invalid --cache size '%s', expect something like 256M or 1G",opts.cache)
    process.exit(1)
  }

  // Initialize timing system
  pentago.init_threads(0,0)

  pentago.descendent_sections([[0,0],[0,0],[0,0],[0,0]],opts.maxSlice)

  var indices = pentago.descendent_sections([[0,0],[0,0],[0,0],[0,0]],opts.maxSlice).map(pentago.supertensor_index_t)
  var cache = pentago.async_block_cache_t(parseInt(m[0])<<{'M':20,'G':30}[m[1][0]])
  var cache_pending = {} // Map from block to callbacks to call once block is available
  var slice_host = '582aa28f4f000f497ad5-81c103f827ca6373fd889208ea864720.r52.cf5.rackcdn.com'

  // Prepare for computations
  process.env['PENTAGO_WORKER_BITS'] = opts.bits
  var pool = new WorkQueue(__dirname+'/compute.js',opts.pool)

  // Useful counters
  var active_gets = 0

  // http.get, but bail on all errors, and call cont(data) once the full data is available
  function lazy_get(req,cont) {
    req['encoding'] = null // Set binary mode
    req['agent'] = false // Don't leave connection alive
    log.debug('range request %s, %s, active %d',req.path,req.headers.range,active_gets++)
    http.get(req,function (res) {
      var body = []
      res.on('data',function (chunk) { body.push(chunk) })
      res.on('end',function () {
        body = Buffer.concat(body)
        log.debug('range response %s, %s, length %d, active %d',req.path,req.headers.range,body.length,--active_gets)
        cont(body)
      })
    }).on('error',function (e) {
      log.error('http get failed: request %j, error %s',req,e)
      process.exit(1)
    })
  }

  // Lookup or compute the value or board and its children, then call cont(results) with a board -> value map.
  function values(board,cont) {
    // Collect the leaf boards whose values we need
    var results = {}
    var requests = []
    function traverse(board,children,cont) {
      if (board.done()) { // Done, so 
        var v = board.immediate_value()
        results[board.name()] = v
        cont(v)
      } else if (!children && !board.middle()) { // Request board
        requests.push({'board':board,'cont':cont})
      } else { // Traverse into children
        var value = -1
        var moves = board.moves()
        var left = moves.length
        var scale = board.middle() ? -1 : 1
        for (var i=0;i<moves.length;i++) {
          traverse(moves[i],false,function (v) {
            value = max(value,scale*v)
            if (!--left) {
              results[board.name()] = value
              cont(value)
            }
          })
        }
      }
    }
    traverse(board,true,function (v) {
      cont(results)
    })

    // Lookup a value in the database
    function remote_request(board,cont) {
      var block = cache.board_block(board) 
      var rest = function () {
        var v = board.value(cache)
        results[board.name()] = v
        cont(v)
      }
      if (cache.contains(block))
        rest()
      else if (block in cache_pending)
        cache_pending[block].push(rest)
      else {
        // Add one pending callback
        cache_pending[block] = [rest]
        // Grab blob via an http range request.  This tells us where to look for the block.
        var slice = board.count()
        var range = indices[slice].blob_range_header(block)
        var blob_req = {host: slice_host,
                        path: '/slice-'+slice+'.pentago.index',
                        headers: {range:range}}
        lazy_get(blob_req,function (blob) {
          // Grab block via an http range request
          var data_req = {host: slice_host,
                          path: '/slice-'+slice+'.pentago',
                          headers: {range: indices[slice].block_range_header(blob)}}
          lazy_get(data_req,function (data) {
            var csize = indices[slice].block_compressed_length(blob)
            if (data.length != csize)
              throw 'block '+block+': expected compressed size '+csize+', got '+data.length
            cache.set(block,data)
            var pending = cache_pending[block]
            delete cache_pending[block]
            for (var i=0;i<pending.length;i++)
              pending[i]()
          })
        })
      }
    }

    // At this point, all needed values have been added to the requests list.
    if (requests.length) {
      // Verify that all requested have the same slice
      var slice = requests[0].board.count()
      for (var i=0;i<requests.length;i++)
        if (slice != requests[i].board.count())
          throw 'slice mismatch: '+slice+' != '+requests[i].board.count()
      // Asychronously request or compute all needed values
      if (slice <= opts.maxSlice) // If the slice is in the database, make asynchronous http requests
        for (var i=0;i<requests.length;i++)
          remote_request(requests[i].board,requests[i].cont)
      else { // Dispatch all requests to a single worker process to exploit coherence
        log.debug('compute launch %s',board.name())
        var boards = requests.map(function (r) { return r.board.name() })
        pool.enqueue(boards, function (res) {
          log.debug('compute success %s',board.name())
          for (var i=0;i<requests.length;i++) {
            var name = requests[i].board.name()
            var v = res[name].v
            log.info('computed %s = %d, elapsed %s s',name,v,res[name].time)
            results[name] = v
            requests[i].cont(v)
          }
        })
      }
    }
  }
  return values
}
