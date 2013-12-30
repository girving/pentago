// If two people ask for the same asynchronous computation, do it only once.

'use strict'

// The compute function should take two arguments: the input and a callback for the results. 
// Example usage:
//
//   var slow = function(x,cont) { setTimeout(function () { console.log(2*x); cont(2*x) },1000) }
//   slow = require('./pending')(slow)
//   // The following prints 14 once 
//   slow(7,function () {})
//   slow(7,function () {})

module.exports = function (compute) {
  var pending = {} // Map from input to callbacks to callbacks to call
  return function (input,cont) {
    var name = ''+input
    if (name in pending)
      pending[name].push(cont)
    else {
      pending[name] = [cont]
      compute(input,function (results) {
        var cs = pending[name]
        delete pending[name]
        for (var i=0;i<cs.length;i++)
          cs[i](results)
      })
    }
  }
}
