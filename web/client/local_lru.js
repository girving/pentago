// Lazy LRU cache using Window.localStorage (if available)

'use strict'
const {storageFactory} = require('storage-factory')
const storage = storageFactory(() => localStorage)

// We prune down to lower whenever we exceed upper
const lower = 10000
const upper = 1.1 * lower

// Discard oldest entries
function collect() {
  // Collect all [time,key] pairs, and sort
  const pairs = []
  for (let n = 0;; n++) {
    const key = storage.key(n)
    if (!key)
      break
    pairs.push([JSON.parse(storage.getItem(key)).t, key])
  }
  pairs.sort()

  // Remove all old pairs
  const dead = pairs.length - lower
  console.log('Pruning ' + dead + ' old cache entries')
  for (let i = 0; i < dead; i++)
    storage.removeItem(pairs[i][1])
}

// Read a value, updating last access time
exports.get = key => {
  const data = JSON.parse(storage.getItem(key))
  if (!data)
    return undefined
  const value = data.v
  storage.setItem(key, JSON.stringify({t: Date.now(), v: value}))
  return value
}

// Write a value, garbage collecting if desired
exports.set = (key, value) => {
  storage.setItem(key, JSON.stringify({t: Date.now(), v: value}))
  if (storage.length > upper)
    collect()
}
