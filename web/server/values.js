// Asynchronous board value lookup

'use strict'
const https = require('https')
const {setTimeout: sleep} = require('timers/promises')
const Pending = require('./pending')
const auth = require('./auth.js')
const {request} = require('./request.js')
const block_cache = require('./block_cache.js')

// Pull in math
const min = Math.min
const max = Math.max
const floor = Math.floor

exports.defaults = {
  // Keep well below the function's memory limit (256M): the Node runtime, lzma buffers,
  // and transient request state need the rest, and lru eviction only sees cache entries.
  cache: '128M',
  maxSlice: 18,
  maxSockets: 64,
}

// Useful counters
const stats = {
  active_gets: 0
}
exports.stats = stats

function parseSize (s,name) {
  const m = s.match(/^(\d+)(K|KB|M|MB|G|GB)$/)
  if (!m)
    throw Error('invalid '+(name?name+' ':'')+'size '+s+", expect something like 256M or 1G")
  return parseInt(m[1])<<{'K':10,'M':20,'G':30}[m[2][0]]
}

// GET a url into a Buffer.  Errors are marked transient if a retry might help.
async function https_get(url, headers) {
  const {status, body} = await request(url, {headers})
  if (status == 200 || status == 206)
    return body
  const error = Error('GET ' + url + ' failed with status ' + status + ': ' + body.toString().trim())
  if (status == 429 || status >= 500)
    error.transient = true
  throw error
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
  const token = auth.token_source()
  const bucket_url = 'https://storage.googleapis.com/storage/v1/b/pentago-us-central1/o/'

  // Allow more simultaneous connections
  if (!(0 < opts.maxSockets && opts.maxSockets <= 1024))
    throw Error('invalid --max-sockets value '+opts.maxSockets)
  https.globalAgent.maxSockets = opts.maxSockets

  // Get a section of a file, retrying transient failures (network errors, 429, 5xx) with
  // exponential backoff.  This mirrors the @google-cloud/storage defaults we used to rely on.
  const get_tries = 4
  async function range_get(object, blob) {
    const url = bucket_url + encodeURIComponent(object) + '?alt=media'
    for (let attempt = 1;; attempt++) {
      try {
        const headers = {authorization: 'Bearer ' + await token(),
                         range: 'bytes=' + blob.offset + '-' + (blob.offset + blob.size - 1)}
        const data = await https_get(url, headers)
        if (data.length != blob.size)
          throw Error('range get of ' + object + ' returned ' + data.length + ' bytes, expected ' + blob.size)
        return data
      } catch (error) {
        if (attempt == get_tries || !error.transient)
          throw error
        log.warning("transient error, attempt %d/%d: %s", attempt, get_tries, error.message)
        await sleep(1000 * 2 ** (attempt - 1) * (0.5 + Math.random()))
      }
    }
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
    const tries = 3
    for (let attempt = 1;; attempt++) {
      const data = await range_get('slice-'+slice+'.pentago', indices[slice].block_location(blob))
      try {
        await cache.set(block, data)
        return
      } catch (error) {
        log.warning("corrupt block, attempt %d/%d: slice %d, block [%s], error '%s'",
                    attempt, tries, slice, ''+block, error.message)
        if (attempt == tries)
          throw Error('corrupt block after '+tries+' attempts: slice '+slice+', block ['+block+']')
      }
    }
  })

  // Lookup the value or board and its children, returning a promise of a board -> value map.
  async function values(board) {
    // Boards with more stones should be handled on the client
    if (board.count >= opts.maxSlice) {
      const e = Error('board ' + board.name + ' has ' + board.count +
                      ' >= ' + opts.maxSlice + ' stones, and should be computed locally')
      e.status = 400  // Client error, not server error
      throw e
    }

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
