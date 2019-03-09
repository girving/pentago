#!/usr/bin/env node

'use strict'
const fs = require('fs')
const Log = require('log')
const http = require('http')
const https = require('https')
const options = require('commander')
const pentago = require('./build/Release/pentago')
const Values = require('./values.js')

// Parse options
options.option('-p,--port <p>', 'Port to listen on', s => parseInt(s), 2048)
       .option('--log <file>', 'Log file')
       .option('--no-https', 'Disable https (for testing only)')
Values.add_options(options)
options.parse(process.argv)

// Initialize logging
const log = options.log ? new Log('debug', fs.createWriteStream(options.log, {flags:'a'})) : new Log('debug')
log.info('command = %s', process.argv.join(' '))
log.info('https = %s', options.https)
log.info('port = %d', options.port)

// Log pentago configuration
log.info('config:')
const config = pentago.config()
Object.keys(config).sort().forEach(k => {
  log.info('  %s = %s', k, config[k])
})

// Prepare for evaluation
const values = Values.values(options, log)

// Usage information
const usage = fs.readFileSync(__dirname+'/usage.txt')

// Server logic
function listener(req, res) {
  // Parse board
  let board, start
  try {
    board = pentago.high_board_t(req.url.substr(1))
    log.info('request %s',board.name())
    start = Date.now()
  } catch (e) {
    log.error('bad request %s',req.url)
    res.writeHead(404)
    res.end('bad url '+req.url+': expected (\d+)m? representing valid board as described below\n\n'+usage)
    return
  }

  values(board).then(results => {
    const elapsed = (Date.now()-start)/1000
    log.info('response %s, elapsed %s s',board.name(),elapsed)
    // Send reply, following cache advice at https://developers.google.com/speed/docs/best-practices/caching
    res.writeHead(200, {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'public',
      'access-control-allow-origin': '*', // All access from javascript is safe
      'expires': '' + new Date(Date.now()+31536000000)})  // One year later
    res.write(JSON.stringify(results))
    res.end()
  })
}

// Create server
const server = !options.https ? http.createServer(listener) : https.createServer({
  ca: [fs.readFileSync('ssl/chain-1.crt'), fs.readFileSync('ssl/chain-2.crt')],
  key: fs.readFileSync('ssl/pentago.key'),
  cert: fs.readFileSync('ssl/pentago.crt')
})

// Listen forever
log.info('listening on port %d', options.port)
server.listen(options.port)
