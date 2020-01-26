#!/usr/bin/env node

'use strict'
const fs = require('fs')
const board_t = require('./board.js')
const Log = require('log')
const options = require('commander')
const Values = require('./values.js')

// Prepare
const log = new Log('debug')
const values = Values.values(Values.defaults, log)
const usage = fs.readFileSync(__dirname + '/usage.txt')

// Server
exports.pentago = async (req, res) => {
  // Parse board
  let board, start
  try {
    board = new board_t(req.url.substr(1))
    log.info('request %s', board.name)
    start = Date.now()
  } catch (e) {
    log.error('bad request %s', req.url)
    res.writeHead(404)
    res.end('bad url ' + req.url + ': expected (\d+)m? representing valid board as described below\n\n' + usage)
    return
  }

  // Lookup!
  const results = await values(board)
  const elapsed = (Date.now() - start) / 1000

  // Compose response
  log.info('response %s, elapsed %s s', board.name, elapsed)
  // Send reply, following cache advice at https://developers.google.com/speed/docs/best-practices/caching
  res.writeHead(200, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'public',
    'access-control-allow-origin': '*', // All access from javascript is safe
    'expires': '' + new Date(Date.now() + 31536000000)})  // One year later
  res.write(JSON.stringify(results))
  res.end()
}
