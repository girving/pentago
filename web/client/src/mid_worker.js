// Web worker for the wasm midsolvers.  Each message is {board, threads, cascade}: threads = 0 runs
// the dense midsolve, otherwise the tiled solver with that many participants, after trying the cheap
// prover cascade first if cascade is set.  Instances are fresh per solve since the wasm bump
// allocator never frees.  Answers strictly in request order.

import {cascade, midsolve, midsolve_tiled} from './mid_sync.js'

onmessage = async e => {
  try {
    const {board, threads, cascade: cheap} = e.data
    postMessage((cheap && await cascade(board)) ||
                await (threads ? midsolve_tiled(board, threads) : midsolve(board)))
  } catch (x) {
    // Only Errors survive postMessage reliably (an ErrorEvent can't be cloned at all)
    postMessage(x instanceof Error ? x : Error(x?.message || String(x)))
  }
}
