// Web worker for the wasm midsolvers.  Each message is {board, threads}: threads = 0 runs the dense
// midsolve, otherwise the tiled solver with that many participants.  Instances are fresh per solve
// since the wasm bump allocator never frees.  Answers strictly in request order.

import {midsolve, midsolve_tiled} from './mid_sync.js'

onmessage = async e => {
  try {
    const {board, threads} = e.data
    postMessage(await (threads ? midsolve_tiled(board, threads) : midsolve(board)))
  } catch (x) {
    postMessage(x)
  }
}
