// Lazy LRU cache using Window.localStorage (if available)
// Actually, it used to be an LRU cache, but now it just resets once full.  Simplicity FTW.

import factory from './storage-factory.js'
const storage = factory(() => localStorage)

// Constants
const version = '5'
const limit = 10000  // We clear if we exceed this

// Clear and set version
function clear() {
  storage.clear()
  storage.setItem('_version', version)
}

// Wipe entries if we're not at the right version
if (storage.getItem('_version') != version) {
  console.log('New cache version ' + version)
  clear()
}

// Read a value, updating last access time
export const get = key => {
  return JSON.parse(storage.getItem(key))
}

// Write a value, garbage collecting if desired
export const set = (key, value) => {
  if (storage.length > limit)
    clear()
  storage.setItem(key, JSON.stringify(value))
}
