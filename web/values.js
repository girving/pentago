// Asynchronous board value lookup and compute

'use strict'
var os = require('os')
var https = require('https')
var time = require('time')
var request = require('request')
var LRU = require('lru-cache')
var WorkQueue = require('mule').WorkQueue
var Pending = require('./pending')
var storage = require('./storage')
var pentago = require('./build/Release/pentago')

// Pull in math
var min = Math.min
var max = Math.max
var floor = Math.floor

exports.defaults = {
  pool: os.cpus().length,
  cache: '250M',
  ccache: '250M',
  maxSlice: 18,
  maxSockets: 64,
  apiKey: '',
  external: false
}

exports.add_options = function (options) {
  var d = exports.defaults
  options.option('--pool <n>','Number of worker compute processes (defaults to cpu count)',parseInt,d.pool)
         .option('--cache <size>','Size of block cache (suffixes M/MB and G/GB are understood)',d.cache)
         .option('--ccache <size>','Size of compute cache (suffixes M/MB and G/GB are understood)',d.ccache)
         .option('--max-slice <n>','Maximum slice available in database (for debugging use only)',parseInt,d.maxSlice)
         .option('--max-sockets <n>','Maximum number of simultaneous https connections',parseInt,d.maxSockets)
         .option('--api-key <key>','Rackspace API key',d.apiKey)
         .option('--external','Work outside of Rackspace')
}

// Useful counters
var stats = {
  active_gets: 0
}
exports.stats = stats

function parseSize (s,name) {
  var m = s.match(/^(\d+)(K|KB|M|MB|G|GB)$/)
  if (!m) {
    log.error("invalid %ssize '%s', expect something like 256M or 1G",name?name+' ':'',opts.cache)
    throw 'invalid '+(name?name+' ':'')+'size '+s
  }
  return parseInt(m[1])<<{'K':10,'M':20,'G':30}[m[2][0]]
}

// Create an evaluation routine with calling convention
//   values(board,cont)
// The options are
//   pool: Number of worker compute processes (defaults to cpu count)
//   cache: Size of block cache (suffixes M/MB and G/GB are understood)
//   maxSlice: Maximum slice available in database (for debugging use only)
exports.values = function (options,log) {
  // Incorporate defaults
  var opts = {}
  for (var k in exports.defaults)
    opts[k] = options[k] || exports.defaults[k]
  var cache_limit = parseSize(opts.cache,'--cache')
  var ccache_limit = parseSize(opts.ccache,'--ccache')

  // Print information
  log.info('cache memory limit = %d (%s)',cache_limit,opts.cache)
  log.info('compute cache memory limit = %d (%s)',ccache_limit,opts.ccache)
  log.info('max slice = %d',opts.maxSlice)
  log.info('max sockets = %d',opts.maxSockets)
  log.info('compute pool = %d',opts.pool)
  log.info('external = %d',opts.external)

  // Prepare for opening book lookups
  var indices = pentago.descendent_sections([[0,0],[0,0],[0,0],[0,0]],opts.maxSlice).map(pentago.supertensor_index_t)
  var cache = pentago.async_block_cache_t(cache_limit)
  var cache_pending = {} // Map from block to callbacks to call once block is available

  // Prepare for Cloud Files access via pkgcloud
  if (!opts.apiKey)
    throw 'no --api-key specified'
  var container = 'pentago-edison-all'
  var download = storage.downloader({
    username: 'pentago',
    region: 'IAD',
    apiKey: opts.apiKey,
    useInternal: !opts.external,
  }, stats, log)

  // Allow more simultaneous connections
  if (!(0 < opts.maxSockets && opts.maxSockets <= 1024))
    throw 'invalid --max-sockets value '+opts.maxSockets
  https.globalAgent.maxSockets = opts.maxSockets

  // Initialize timing system
  pentago.init_threads(0,0)

  // Prepare for computations
  var pool = new WorkQueue(__dirname+'/compute.js',opts.pool)
  var compute_cache = LRU({
    max: floor(ccache_limit/1.2),
    length: function (s) { return s.length }
  })
  // Cache and don't simultaneously duplicate work. b = (root,boards),
  // where boards are to be evaluated and root is their nearest common ancestor.
  var pending_compute = Pending(function (b,cont) {
    var board = b[0]
    var results = compute_cache.get(board)
    if (results)
      cont(JSON.parse(results))
    else
      pool.enqueue(b,function (results) {
        compute_cache.set(board,JSON.stringify(results))
        cont(results)
      })
  })

  // Get a section of a file
  function range_get(object, blob, cont) {
    download(container, object, blob.offset, blob.size, cont)
  }

  // Get a block if necessary, merging simultaneous requests
  var pending_block = Pending(function (block,cont) {
    if (cache.contains(block))
      cont()
    else {
      // Compute slice
      var slice = 0
      for (var q=0;q<4;q++)
        for (var s=0;s<2;s++)
          slice += block[0][q][s]
      // Grab block location
      range_get('slice-'+slice+'.pentago.index',indices[slice].blob_location(block),function (blob) {
        // Grab block data, retrying if the data is corrupt
        function try_get() {
          range_get('slice-'+slice+'.pentago',indices[slice].block_location(blob),function (data) {
            try {
              cache.set(block,data)
            } catch (error) {
              log.warning("corrupt block, retrying: slice %d, block [%s], error '%s'",slice,block,error)
              return try_get()
            }
            cont()
          })
        }
        try_get()
      })
    }
  })

  // Lookup or compute the value or board and its children, then call cont(results) with a board -> value map.
  function values(board,cont) {
    // Collect the leaf boards whose values we need
    var results = {}
    var requests = []
    function traverse(board,children,cont) {
      if (board.done()) { // Done, so no lookup required
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

    // Look up a remote value, caching the block in the process
    function remote_request(board,cont) {
      pending_block(cache.board_block(board), function () {
        var v = board.value(cache)
        results[board.name()] = v
        cont(v)
      })
    }

    // At this point, all needed values have been added to the requests list.
    if (requests.length) {
      // Verify that all requested have the same slice
      var slice = requests[0].board.count()
      for (var i=0;i<requests.length;i++)
        if (slice != requests[i].board.count())
          throw 'slice mismatch: '+slice+' != '+requests[i].board.count()
      // Asychronously request or compute all needed values
      if (slice <= opts.maxSlice) // If the slice is in the database, make asynchronous https requests
        for (var i=0;i<requests.length;i++)
          remote_request(requests[i].board,requests[i].cont)
      else { // Dispatch all requests to a single worker process to exploit coherence
        log.debug('computing board %s, slice %d',board.name(),board.count())
        var boards = requests.map(function (r) { return r.board.name() })
        pending_compute([board.name(),boards], function (res) {
          var stime = res['time']
          log.info('computed board %s, slice %d, time %s s',board.name(),board.count(),stime)
          results['search-time'] = stime
          for (var i=0;i<requests.length;i++) {
            var name = requests[i].board.name()
            var v = res[name]
            results[name] = v
            requests[i].cont(v)
          }
        })
      }
    }
  }
  return values
}
