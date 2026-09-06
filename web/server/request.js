// Minimal HTTP(S) request helper: fetch a url into a Buffer
//
// Network-level failures are marked error.transient = true, since a retry might help.
// HTTP status codes are returned as is; callers decide which statuses are retryable.

'use strict'
const http = require('http')
const https = require('https')

// request(url, {method, headers, body}) : Promise<{status, headers, body: Buffer}>
exports.request = (url, options = {}) => new Promise((resolve, reject) => {
  const transient = error => reject(Object.assign(error, {transient: true}))
  const {body, ...rest} = options
  const req = (url.startsWith('https:') ? https : http).request(url, rest, res => {
    const chunks = []
    res.on('data', chunk => chunks.push(chunk))
    res.on('error', transient)
    res.on('end', () => resolve({status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks)}))
  })
  req.on('error', transient)
  req.end(body)
})
