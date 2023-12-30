// Lazy cache using Window.localStorage.
// The cache resets once full.  Simplicity FTW.
// This file works only in the browser.

// Constants
const version = 5
const limit = 10000  // We clear if we exceed this
const storage = localStorage

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
if (get('_version') != version)
  clear()

// Exports
export { get, set}
