// Modified from https://github.com/MichalZalecki/storage-factory
// to play nice with modules.  Also heavily cleaned.

export default get_storage => {
  try {
    // Test persistent storage
    const storage = get_storage()
    const test = "__some_random_key_you_are_not_going_to_use__"
    storage.setItem(test, test)
    storage.removeItem(test)

    // It works!
    return storage
  } catch (e) {
    // Otherwise, use in-memory storage
    let memory = {}
    return {
      clear: () => memory = {},
      getItem: name => name in memory ? memory[name] : null,
      key: index => Object.keys(memory)[index] || null,
      removeItem: name => delete memory[name],
      setItem: (name, value) => memory[name] = String(value),
      get length() { return Object.keys(memory).length }
    }
  }
}
