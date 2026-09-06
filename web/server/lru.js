// Tiny byte-weighted LRU cache
//
// Replaces the lru-cache package for the three methods we use (has, get, set).  Entries are
// weighted by a caller-supplied size, and the least recently used entries are evicted once the
// total exceeds max_size.  Backed by a Map, whose iteration order is insertion order: moving a
// key to the back on get/set makes the front the least recently used.

'use strict'

exports.LRU = (max_size, size_of) => {
  if (!(max_size > 0))
    throw Error('LRU: max_size must be positive, got ' + max_size)
  const map = new Map()  // key → {value, size}, least recently used first
  let total = 0          // Sum of sizes of all entries

  // Is key present?  Does not affect recency.
  const has = key => map.has(key)

  // Look up key, marking it most recently used.  Returns undefined if missing.
  const get = key => {
    const entry = map.get(key)
    if (entry === undefined)
      return undefined
    map.delete(key)
    map.set(key, entry)
    return entry.value
  }

  // Insert or replace key, marking it most recently used, then evict until we fit.  Values
  // larger than max_size are not stored (matching lru-cache); the key is removed if present.
  const set = (key, value) => {
    const size = size_of(value)
    const old = map.get(key)
    if (old !== undefined) {
      map.delete(key)
      total -= old.size
    }
    if (size > max_size)
      return
    map.set(key, {value, size})
    total += size
    for (const [k, e] of map) {
      if (total <= max_size)
        break
      map.delete(k)
      total -= e.size
    }
  }

  return {has, get, set, get size() { return total }, get count() { return map.size }}
}
