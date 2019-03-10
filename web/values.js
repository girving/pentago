// Asynchronous board value lookup and compute

'use strict'
const os = require('os')
const https = require('https')
const LRU = require('lru-cache')
const WorkQueue = require('mule').WorkQueue
const Pending = require('./pending')
const storage = require('./storage')
const pentago = require('./build/Release/pentago')

// Public version
const version = '0.0.2'

// Pull in math
const min = Math.min
const max = Math.max
const floor = Math.floor

exports.defaults = {
  pool: os.cpus().length,
  cache: '250M',
  ccache: '250M',
  maxSlice: 18,
  maxSockets: 64,
  apiKey: '',
  external: false
}

exports.add_options = options => {
  const d = exports.defaults
  options.option('--pool <n>','Number of worker compute processes (defaults to cpu count)',parseInt,d.pool)
         .option('--cache <size>','Size of block cache (suffixes M/MB and G/GB are understood)',d.cache)
         .option('--ccache <size>','Size of compute cache (suffixes M/MB and G/GB are understood)',d.ccache)
         .option('--max-slice <n>','Maximum slice available in database (for debugging use only)',parseInt,d.maxSlice)
         .option('--max-sockets <n>','Maximum number of simultaneous https connections',parseInt,d.maxSockets)
         .option('--api-key <key>','Rackspace API key',d.apiKey)
         .option('--external','Work outside of Rackspace')
}

// Useful counters
const stats = {
  active_gets: 0
}
exports.stats = stats

function parseSize (s,name) {
  const m = s.match(/^(\d+)(K|KB|M|MB|G|GB)$/)
  if (!m) {
    log.error("invalid %ssize '%s', expect something like 256M or 1G",name?name+' ':'',opts.cache)
    throw 'invalid '+(name?name+' ':'')+'size '+s
  }
  return parseInt(m[1])<<{'K':10,'M':20,'G':30}[m[2][0]]
}

// Create an evaluation routine with calling convention
//   values(board) : Promise
// The options are
//   pool: Number of worker compute processes (defaults to cpu count)
//   cache: Size of block cache (suffixes M/MB and G/GB are understood)
//   maxSlice: Maximum slice available in database (for debugging use only)
exports.values = (options, log) => {
  // Incorporate defaults
  const opts = {}
  for (const k in exports.defaults)
    opts[k] = options[k] || exports.defaults[k]
  const cache_limit = parseSize(opts.cache,'--cache')
  const ccache_limit = parseSize(opts.ccache,'--ccache')

  // Print information
  log.info('cache memory limit = %d (%s)', cache_limit, opts.cache)
  log.info('compute cache memory limit = %d (%s)', ccache_limit, opts.ccache)
  log.info('max slice = %d', opts.maxSlice)
  log.info('max sockets = %d', opts.maxSockets)
  log.info('compute pool = %d', opts.pool)
  log.info('external = %d', opts.external)
  log.info('version = %s', version)

  // Prepare for opening book lookups
  const indices = pentago.descendent_sections([[0,0],[0,0],[0,0],[0,0]],opts.maxSlice).map(
      pentago.supertensor_index_t)
  const cache = pentago.async_block_cache_t(cache_limit)
  const cache_pending = {} // Map from block to callbacks to call once block is available

  // Prepare for Cloud Files access via pkgcloud
  if (!opts.apiKey)
    throw 'no --api-key specified'
  const container = 'pentago-edison-all'
  const download = storage.downloader({
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
  const pool = new WorkQueue(__dirname+'/compute.js',opts.pool)
  const compute_cache = LRU({
    max: floor(ccache_limit/1.2),
    length: s => s.length,
  })
  // Cache and don't simultaneously duplicate work. b = (root,boards),
  // where boards are to be evaluated and root is their nearest common ancestor.
  const pending_compute = Pending(async b => {
    const results = compute_cache.get(b.root)
    if (results)
      return JSON.parse(results)
    else
      return await new Promise((resolve, reject) =>
        pool.enqueue(b, results => {
          compute_cache.set(b.root, JSON.stringify(results))
          resolve(results)
        })
      )
  })

  // Get a section of a file
  function range_get(object, blob) {
    return download(container, object, blob.offset, blob.size)
  }

  // Get a block if necessary, merging simultaneous requests
  const pending_block = Pending(async block => {
    if (cache.contains(block))
      return

    // Compute slice
    let slice = 0
    for (let q = 0; q < 4; q++)
      for (let s = 0; s < 2; s++)
        slice += block[0][q][s]

    // Grab block location
    const blob = await range_get('slice-'+slice+'.pentago.index', indices[slice].blob_location(block))

    // Grab block data, retrying if the data is corrupt
    for (;;) {
      const data = await range_get('slice-'+slice+'.pentago', indices[slice].block_location(blob))
      try {
        cache.set(block, data)
        return
      } catch (error) {
        log.warning("corrupt block, retrying: slice %d, block [%s], error '%s'", slice, block, error)
      }
    }
  })

  // Lookup or compute the value or board and its children, returning a promise of a board -> value map.
  async function values(board) {
    // Collect the leaf boards whose values we need
    const results = {version: version}
    const requests = []
    async function traverse(board, children) {
      if (board.done()) {  // Done, so no lookup required
        const value = board.immediate_value()
        results[board.name()] = value
        return value
      } else if (!children && !board.middle()) {  // Request board
        return await new Promise((resolve, reject) => requests.push({board: board, resolve: resolve}))
      } else {  // Traverse into children
        let value = -1
        const scale = board.middle() ? -1 : 1
        for (const v of await Promise.all(board.moves().map(m => traverse(m, false))))
          value = max(value, scale*v)
        results[board.name()] = value
        return value
      }
    }
    const top = traverse(board, true)

    if (requests.length) {
      // Verify that all requests have the same slice
      const slice = requests[0].board.count()
      requests.forEach(r => {
        if (slice != r.board.count())
          throw Error('slice mismatch: '+slice+' != '+r.board.count())
      })

      // If the slice is in the database, make asynchronous https requests
      if (slice <= opts.maxSlice)
       await Promise.all(requests.map(async ({board, resolve}) => {
          // Look up a remote value, caching the block in the process
          await pending_block(cache.board_block(board))
          const value = board.value(cache)
          results[board.name()] = value
          resolve(value)
        }))
      else {  // Dispatch all requests to a single worker process to exploit coherence
        log.debug('computing board %s, slice %d', board.name(), board.count())
        const res = await pending_compute({root: board.name(), boards: requests.map(({board}) => board.name())})
        const stime = res['time']
        log.info('computed board %s, slice %d, time %s s', board.name(), board.count(), stime)
        results['search-time'] = stime
        requests.forEach(({board, resolve}) => {
          const value = res[board.name()]
          results[board.name()] = value
          resolve(value)
        })
      }
    }

    await top
    return results
  }
  return values
}
