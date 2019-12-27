// Asynchronous board value lookup

'use strict'
const os = require('os')
const https = require('https')
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
  cache: '250M',
  maxSlice: 18,
  maxSockets: 64,
  apiKey: '',
  external: false
}

exports.add_options = options => {
  const d = exports.defaults
  options.option('--cache <size>','Size of block cache (suffixes M/MB and G/GB are understood)',d.cache)
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
    throw Error('invalid '+(name?name+' ':'')+'size '+s)
  }
  return parseInt(m[1])<<{'K':10,'M':20,'G':30}[m[2][0]]
}

// Create an evaluation routine with calling convention
//   values(board) : Promise
// The options are
//   cache: Size of block cache (suffixes M/MB and G/GB are understood)
//   maxSlice: Maximum slice available in database (for debugging use only)
exports.values = (options, log) => {
  // Incorporate defaults
  const opts = {}
  for (const k in exports.defaults)
    opts[k] = options[k] || exports.defaults[k]
  const cache_limit = parseSize(opts.cache,'--cache')

  // Print information
  log.info('cache memory limit = %d (%s)', cache_limit, opts.cache)
  log.info('max slice = %d', opts.maxSlice)
  log.info('max sockets = %d', opts.maxSockets)
  log.info('external = %d', opts.external)
  log.info('version = %s', version)

  // Prepare for opening book lookups
  const indices = pentago.descendent_sections([[0,0],[0,0],[0,0],[0,0]],opts.maxSlice).map(
      pentago.supertensor_index_t)
  const cache = pentago.async_block_cache_t(cache_limit)
  const cache_pending = {} // Map from block to callbacks to call once block is available

  // Prepare for Cloud Files access via pkgcloud
  if (!opts.apiKey)
    throw Error('no --api-key specified')
  const container = 'pentago-edison-all'
  const download = storage.downloader({
    username: 'pentago',
    region: 'IAD',
    apiKey: opts.apiKey,
    useInternal: !opts.external,
  }, stats, log)

  // Allow more simultaneous connections
  if (!(0 < opts.maxSockets && opts.maxSockets <= 1024))
    throw Error('invalid --max-sockets value '+opts.maxSockets)
  https.globalAgent.maxSockets = opts.maxSockets

  // Initialize timing system
  pentago.init_threads(0,0)

  // Get a section of a file
  function range_get(object, blob) {
    return download({container: container, object: object, offset: blob.offset, size: blob.size})
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

  // Lookup the value or board and its children, returning a promise of a board -> value map.
  async function values(board) {
    // Boards with more stones should be handled on the client
    if (board.count() >= opts.maxSlice)
      throw Error('board ' + board.name() + ' has ' + board.count() +
                  ' >= ' + opts.maxSlice + ' stones, and should be computed locally')

    // Collect the leaf boards whose values we need
    const results = {version: version}
    async function traverse(board, children) {
      let value
      if (board.done()) {  // Done, so no lookup required
        value = board.immediate_value()
      } else if (!children && !board.middle()) {  // Look up a remote value, caching the block in the process
        await pending_block(cache.board_block(board))
        value = board.value(cache)
      } else {  // Traverse into children
        value = -1
        const scale = board.middle() ? -1 : 1
        for (const v of await Promise.all(board.moves().map(m => traverse(m, false))))
          value = max(value, scale*v)
      }
      results[board.name()] = value
      return value
    }
    await traverse(board, true)
    return results
  }
  return values
}
