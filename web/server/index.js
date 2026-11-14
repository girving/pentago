#!/usr/bin/env node

'use strict'
const fs = require('fs')
const board_t = require('./board.js')
const {Log} = require('./log.js')
const Values = require('./values.js')

// Prepare
const log = Log('debug')
const values = Values.values(Values.defaults, log)
const usage = fs.readFileSync(__dirname + '/usage.txt')

// Board name from a request path: /<board> when called directly, or
// /backend/<board> via the perfect-pentago.net hosting rewrite.  The bare
// paths (/, /backend, /backend/) give the empty name.
const board_name = url => url.replace(/^\/(backend(\/|$))?/, '')
exports.board_name = board_name

// Server
exports.pentago = async (req, res) => {
  // No board: show usage
  if (!board_name(req.url)) {
    res.writeHead(200, {
      'content-type': 'text/plain; charset=utf-8',
      'cache-control': 'public, max-age=3600',  // Short, so deploys show up soon
      'x-content-type-options': 'nosniff'})
    res.end(usage)
    return
  }

  // Parse board
  let board, start
  try {
    board = new board_t(board_name(req.url))
    log.info('request %s', board.name)
    start = Date.now()
  } catch (e) {
    log.error('bad request %s', req.url)
    res.writeHead(404, {
      'content-type': 'text/plain; charset=utf-8',
      'cache-control': 'no-store',  // Firebase Hosting would otherwise cache errors for 10 minutes
      'x-content-type-options': 'nosniff'})
    res.end('bad url ' + req.url + ': expected (\\d+)m? representing valid board as described below\n\n' + usage)
    return
  }

  // Lookup!  Client mistakes (e.g., boards too deep for the database) get status 400;
  // anything else (GCS failures, corrupt data) is a genuine 500.
  let results
  try {
    results = await values(board)
  } catch (e) {
    const status = e.status || 500
    log.error('lookup failed for %s (status %d): %s', board.name, status, e.message)
    res.writeHead(status, {
      'content-type': 'text/plain; charset=utf-8',
      'cache-control': 'no-store',  // Transient failures must not stick in the CDN
      'x-content-type-options': 'nosniff'})
    res.end('lookup failed: ' + e.message + '\n')
    return
  }
  const elapsed = (Date.now() - start) / 1000

  // Compose response
  log.info('response %s, elapsed %s s', board.name, elapsed)
  // Send reply, following cache advice at https://developers.google.com/speed/docs/best-practices/caching.
  // Values never change, so cache for a year.  max-age (not just expires) lets the
  // Firebase Hosting CDN in front of perfect-pentago.net/backend cache responses too.
  res.writeHead(200, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'public, max-age=31536000',
    'access-control-allow-origin': '*', // All access from javascript is safe
    'expires': '' + new Date(Date.now() + 31536000000)})  // One year later
  res.write(JSON.stringify(results))
  res.end()
}
