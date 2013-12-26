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
         .option('--http-timeout <s>','http request timeout in seconds',parseFloat,0)
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

  // Allow a lot of simultaneous http connections
  http.globalAgent.maxSockets = 64

  // Useful counters
  var active_gets = 0

  // http.get, but bail on all errors, and call cont(data) once the full data is available
  function lazy_get(path,blob,cont) {
    var opts = {host: slice_host,
               path: path,
               encoding: null, // Binary mode
               headers: {range: 'bytes='+blob.offset+'-'+(blob.offset+blob.size-1)}}
    var name = path+', '+blob.offset+"+"+blob.size
    log.debug('range request %s, active %d',name,active_gets++)
    function hcont (res) {
      var body = []
      res.on('data',function (chunk) { body.push(chunk) })
      res.on('end',function () {
        body = Buffer.concat(body)
        if (body.length != blob.size)
          throw 'http size mismatch: '+name+', got size '+body.length
        log.debug('range response %s, active %d',name,--active_gets)
        cont(body)
      })
    }
    function launch () {
      var req = http.get(opts,hcont)
        .on('error',function (e) {
          log.error('http get failed, relaunching: %s, error %s',name,e)
          launch()
        })
      if (options.httpTimeout)
        req.setTimeout(1000*options.httpTimeout,function () {
          log.error('http get timed out, relaunching: %s, time limit %s s',name,options.httpTimeout)
          req.abort()
          launch()
        })
    }
    // Launch first request, then retry on failure
    launch()
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
        // Grab block location
        var slice = board.count()
        lazy_get('/slice-'+slice+'.pentago.index',indices[slice].blob_location(block),function (blob) {
          // Grab block data
          lazy_get('/slice-'+slice+'.pentago',indices[slice].block_location(blob),function (data) {
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
        log.debug('computing board %s, slice %d',board.name(),board.count())
        var boards = requests.map(function (r) { return r.board.name() })
        pool.enqueue(boards, function (res) {
          var total = 0
          for (var name in res)
            total += res[name].time
          log.info('computed board %s, slice %d, time %s s',board.name(),board.count(),total)
          for (var i=0;i<requests.length;i++) {
            var name = requests[i].board.name()
            var v = res[name].v
            results[name] = v
            requests[i].cont(v)
          }
        })
      }
    }
  }
  return values
}
