// Inside of web worker midsolver

import { midsolve } from './mid_sync.js'

onmessage = async e => {
  try {
    self.postMessage(await midsolve(e.data))
  } catch (e) {
    self.postMessage(e)
  }
}
