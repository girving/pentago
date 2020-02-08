// Inside of web worker midsolver

'use strict'

const board_t = require('./board.js')
const mid_sync = require('./mid_sync.js')

module.exports = self => {
  self.addEventListener('message', async e => {
    try {
      self.postMessage(await mid_sync.midsolve(new board_t(e.data)))
    } catch (e) {
      self.postMessage(e)
    }
  })
}
