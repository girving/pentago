// Asynchronous board value lookup

'use strict'
const os = require('os')
const https = require('https')
const Pending = require('./pending')
const {Storage} = require('@google-cloud/storage')
const block_cache = require('./block_cache.js')

// Pull in math
const min = Math.min
const max = Math.max
const floor = Math.floor

exports.defaults = {
  cache: '250M',
  maxSlice: 18,
  maxSockets: 64,
}

exports.add_options = options => {
  const d = exports.defaults
  options.option('--cache <size>', 'Size of block cache (suffixes M/MB and G/GB are understood)', d.cache)
         .option('--max-slice <n>', 'Maximum slice available in database (for debugging use only)', parseInt,
                 d.maxSlice)
         .option('--max-sockets <n>', 'Maximum number of simultaneous https connections', parseInt, d.maxSockets)
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

  // Prepare for opening book lookups
  const indices = block_cache.descendent_sections(opts.maxSlice).map(s => new block_cache.supertensor_index_t(s))
  const cache = new block_cache.block_cache_t(cache_limit)
  const cache_pending = {} // Map from block to callbacks to call once block is available
  const bucket = new Storage().bucket('pentago')

  // Allow more simultaneous connections
  if (!(0 < opts.maxSockets && opts.maxSockets <= 1024))
    throw Error('invalid --max-sockets value '+opts.maxSockets)
  https.globalAgent.maxSockets = opts.maxSockets

  // Get a section of a file
  async function range_get(object, blob) {
    const data = await bucket.file(object).download({start: blob.offset, end: blob.offset + blob.size - 1})
    return data[0]
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
        await cache.set(block, data)
        return
      } catch (error) {
        log.warning("corrupt block, retrying: slice %d, block [%s], error '%s'", slice, ''+block, error.message)
      }
    }
  })

  // Lookup the value or board and its children, returning a promise of a board -> value map.
  async function values(board) {
    // Boards with more stones should be handled on the client
    if (board.count >= opts.maxSlice)
      throw Error('board ' + board.name + ' has ' + board.count +
                  ' >= ' + opts.maxSlice + ' stones, and should be computed locally')

    // Collect the leaf boards whose values we need
    const results = {}
    async function traverse(board, children) {
      let value
      if (board.done()) {  // Done, so no lookup required
        value = board.immediate_value()
      } else if (!children && !board.middle) {  // Look up a remote value, caching the block in the process
        const block = cache.board_block(board)
        await pending_block(block)
        value = cache.value(board)
      } else {  // Traverse into children
        value = -1
        const scale = board.middle ? -1 : 1
        for (const v of await Promise.all(board.moves().map(m => traverse(m, false))))
          value = max(value, scale*v)
      }
      results[board.name] = value
      return value
    }
    await traverse(board, true)
    return results
  }
  return values
}
