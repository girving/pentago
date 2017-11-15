#!/usr/bin/env node

'use strict'
var fs = require('fs')
var Log = require('log')
var https = require('https')
var time = require('time')
var options = require('commander')
var pentago = require('./build/Release/pentago')
var Values = require('./values.js')

// Parse options
options.option('-p,--port <p>','Port to listen on',parseInt,2048)
       .option('--log <file>','Log file')
Values.add_options(options)
options.parse(process.argv)

// Initialize logging
var log = options.log ? new Log('debug',fs.createWriteStream(options.log,{flags:'a'})) : new Log('debug')
log.info('command = %s',process.argv.join(' '))
log.info('port = %d',options.port)

// Log pentago configuration
log.info('config:')
var config = pentago.config()
Object.keys(config).sort().forEach(function (k) {
  log.info('  %s = %s',k,config[k])
})

// Prepare for evaluation
var values = Values.values(options,log)

// Usage information
var usage = fs.readFileSync(__dirname+'/usage.txt')

// Load certificate
var slurp = fs.readFileSync
var cert = {
  ca: [slurp('ssl/chain-1.crt'),slurp('ssl/chain-2.crt')],
  key: slurp('ssl/pentago.key'),
  cert: slurp('ssl/pentago.crt')
}

// Create server
var server = https.createServer(cert, function (req,res) {
  // Parse board
  try {
    var board = pentago.high_board_t(req.url.substr(1))
    log.info('request %s',board.name())
    var start = Date.now()
  } catch (e) {
    log.error('bad request %s',req.url)
    res.writeHead(404)
    res.end('bad url '+req.url+': expected (\d+)m? representing valid board as described below\n\n'+usage)
    return
  }

  values(board,function (results) {
    var elapsed = (Date.now()-start)/1000
    log.info('response %s, elapsed %s s',board.name(),elapsed)
    // Send reply, following cache advice at https://developers.google.com/speed/docs/best-practices/caching
    res.writeHead(200,{
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'public',
      'access-control-allow-origin': '*', // All access from javascript is safe
      'expires': time.Date(Date.now()+31536000000)}) // One year later
    res.write(JSON.stringify(results))
    res.end()
  })
})

// Listen forever
log.info('listening on port %d',options.port)
server.listen(options.port)
