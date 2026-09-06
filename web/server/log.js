// Tiny leveled logger
//
// Replaces the `log` npm package (last published 2013) with the subset we use, keeping its
// output format: "[<date>] LEVEL <message>" on stdout, where <message> is util.format'd.

'use strict'
const {format} = require('util')

const levels = ['error', 'warning', 'info', 'debug']  // Most to least severe

// Log(level) returns an object with one method per level, e.g. log.info('request %s', name).
// Messages at or above the given severity are written; the rest are dropped.
exports.Log = (level = 'debug', stream = process.stdout) => {
  const max = levels.indexOf(level)
  if (max < 0)
    throw Error('unknown log level ' + level + ', expected one of ' + levels)
  const log = {}
  levels.forEach((name, i) => {
    log[name] = (...args) => {
      if (i <= max)
        stream.write('[' + new Date() + '] ' + name.toUpperCase() + ' ' + format(...args) + '\n')
    }
  })
  return log
}
