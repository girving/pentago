// Inside of web worker midsolver

'use strict'

const board_t = require('./board.js')
const mid_sync = require('./mid_sync.js')

global.onmessage = async e => {
  try {
    global.postMessage(await mid_sync.midsolve(new board_t(e.data)))
  } catch (e) {
    global.postMessage(e)
  }
}
