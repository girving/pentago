// Lazy LRU cache using Window.localStorage (if available)

import factory from './storage-factory.js'
const storage = factory(() => localStorage)
const max = Math.max

// Constants
const version = '2'
const special = {_version: true}

// Wipe entries if we're not at the right version
if (storage.getItem('_version') != version) {
  console.log('New cache version ' + version + '; wiping existing entries')
  storage.clear()
  storage.setItem('_version', version)
}

// We prune down to lower whenever we exceed upper
let lower = 10000
let upper = 1.1 * lower

// Iterate over all nonspecial keys
function for_keys(f) {
  for (let n = 0;; n++) {
    const key = storage.key(n)
    if (key in special)
      continue
    if (!key)
      break
    f(key)
  }
}

// Track time using an integer counter to avoid duplicate time stamps
let last_time = 0
for_keys(k => last_time = max(last_time, JSON.parse(storage.getItem(k)).t))
const now = () => ++last_time

// Set lower and upper bounds
export const set_lohi = (lo, hi) => {
  lower = lo
  upper = hi
}

// Current number of cached values
export const size = () => storage.length - Object.keys(special).length

// Discard oldest entries
function collect() {
  // Collect all [time,key] pairs, and sort
  const pairs = []
  for_keys(key => pairs.push({t: JSON.parse(storage.getItem(key)).t, k: key}))
  pairs.sort((a, b) => a.t - b.t)  // Sort in ascending order of time

  // Remove all old pairs
  const dead = pairs.length - lower
  for (let i = 0; i < dead; i++)
    storage.removeItem(pairs[i].k)
}

// Read a value, updating last access time
export const get = key => {
  const data = JSON.parse(storage.getItem(key))
  if (!data)
    return undefined
  const value = data.v
  storage.setItem(key, JSON.stringify({t: now(), v: value}))
  return value
}

// Read a value, but don't update access time
export const peek = key => {
  const data = JSON.parse(storage.getItem(key))
  return data ? data.v : undefined
}

// Dump all values via console.log
export const dump = () => {
  console.log('cache (version ' + storage.getItem('version') + '):')
  for_keys(key => {
    const {t, v} = JSON.parse(storage.getItem(key))
    console.log('  ' + key + ': ' + v + ', ' + t)
  })
}

// Write a value, garbage collecting if desired
export const set = (key, value) => {
  storage.setItem(key, JSON.stringify({t: now(), v: value}))
  if (size() > upper)
    collect()
}
