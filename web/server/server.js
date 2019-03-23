#!/usr/bin/env node

'use strict'
const fs = require('fs')
const http = require('http')
const https = require('https')
const Log = require('log')
const options = require('commander')
const pentago = require('./build/Release/pentago')
const Values = require('./values.js')

// Parse options
options.option('-p,--port <p>', 'Port to listen on', s => parseInt(s), 2048)
       .option('--green-port <p>', 'Greenlock HTTP port to listen on', s => parseInt(s), 2049)
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
    log.info('request %s', board.name())
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

// Create server and listen forever
if (options.https) {
  const greenlock = require('greenlock').create({
    email: 'irving@naml.us',
    approveDomains: ['backend.perfect-pentago.net'],
    agreeTos: true,
    configDir: 'acme',
    communityMember: true,
    securityUpdates: true,
  })
  const redir = require('redirect-https')()
  const green = http.createServer(greenlock.middleware(redir))
  log.info('greenlock listening on port %d', options.greenPort)
  green.listen(options.greenPort)
  const server = https.createServer(greenlock.tlsOptions, listener)
  log.info('listening on port %d', options.port)
  server.listen(options.port)
} else {  // http
  const server = http.createServer(listener)
  log.info('listening on port %d', options.port)
  server.listen(options.port)
}
