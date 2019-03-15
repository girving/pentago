// If two people ask for the same asynchronous computation, do it only once.

'use strict'

// The compute function should take an input and return a value or promise
// Example usage:
//
//   const pending = require('./pending')
//
//   const slow = pending(x => new Promise((resolve, reject) =>
//       setTimeout(() => { console.log(2*x); resolve(2*x) },1000)))
//   // The following prints 14 once 
//   slow(7)
//   slow(7)

module.exports = compute => {
  const pending = {}  // Map from input to promise
  return input => {
    const name = JSON.stringify(input)
    if (name in pending) {
      return pending[name]
    } else {
      const p = Promise.resolve(compute(input))
      pending[name] = p
      p.then(() => delete pending[name])
      return p
    }
  }
}
