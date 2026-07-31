// Lazy cache using Window.localStorage.
// The cache resets once full.  Simplicity FTW.
// This module also evaluates inside the web worker (which bundles the same
// script but never calls us), so tolerate missing localStorage.

// Constants
const version = 5
const limit = 10000  // We clear if we exceed this
const storage = globalThis.localStorage

// Clear and set version
const clear = () => {
  storage.clear()
  set('_version', version)
}

// Read a value, updating last access time
const get = key => JSON.parse(storage.getItem(key))

// Write a value, garbage collecting if desired
const set = (key, value) => {
  if (storage.length > limit)
    clear()
  storage.setItem(key, JSON.stringify(value))
}

// Wipe entries if we're not at the right version
if (storage && get('_version') != version)
  clear()

// Exports
export { get, set}
