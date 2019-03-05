// Minimal interface to Rackspace Cloud Files

'use strict'
const concat = require('concat-stream')
const request = require('request')

const early_token_timeout = 1000 * 60 * 5  // 5 minutes in ms

function match_region(a, b) {
  if (!a && !b)
    return true
  else if ((!a && b) || (a && !b))
    return false
  else
    return a.toLowerCase() === b.toLowerCase()
}

exports.downloader = function (options, stats, log) {
  const region = options.region
  const username = options.username
  const apiKey = options.apiKey
  const useInternal = options.useInternal

  // Authorization.  We use a promised token so that simultaneous auth requests get merged,
  // and wipe the promise on expiration.
  let promised_token = null
  function auth(callback) {
    // Launch a token request if necessary
    if (!promised_token)
      promised_token = new Promise((resolve, reject) => {
        // Request a new token
        request({
          uri: 'https://identity.api.rackspacecloud.com/v2.0/tokens',
          method: 'POST',
          strictSSL: true,
          headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
          json: {auth: {'RAX-KSKEY:apiKeyCredentials': {username: username, apiKey: apiKey}}}
        }, function (err, res, body) {
          if (err) return reject(err)
          if (res.statusCode >= 400) return reject('authentication failed: code ' + res.statusCode)

          // Build token
          const token = body.access.token
          if (!token || !token.id || !token.expires) return reject('Invalid token')
          token.expires = new Date(token.expires)

          // Add service url
          const service = body.access.serviceCatalog.find(service => service.type.toLowerCase() == 'object-store')
          if (!service) return reject('Unable to find matching endpoint for requested service')
          const endpoint = service.endpoints.find(endpoint => match_region(endpoint.region, region))
          if (!endpoint) return reject('Unable to identify endpoint url')
          token.service_url = useInternal && endpoint.internalURL ? endpoint.internalURL : endpoint.publicURL

          // Schedule the destruction of the promise
          const timeout = token.expires.getTime() - new Date().getTime() - early_token_timeout
          setTimeout(() => { promised_token = null }, timeout)

          // All done!
          resolve(token)
        })
      })
    // Add new callback
    promised_token.then(callback)
  }

  function download(options, callback) {
    return new Promise((resolve, reject) =>
      auth(token => {
        const opts = {}
        opts.method = 'GET'
        opts.headers = options.headers
        opts.headers['x-auth-token'] = token.id
        opts.uri = token.service_url + '/' + options.container + '/' + options.path
        request(opts, err => { if (err) reject(err) }).pipe(concat(resolve))
      })
    )
  }

  function range_download(container, object, offset, size, cont) {
    const name = object + ', ' + offset + '+' + size
    log.debug('range request %s, active %d', name, stats.active_gets++)
    download({
      container: container,
      path: object,
      headers: {range: 'bytes=' + offset + '-' + (offset+size-1)}
    }).then(body => {
      if (body.length != size)
        log.error('range request failed: %s, got size %d != %d', name, body.length, size)
      else {
        log.debug('range response %s, active %d', name, --stats.active_gets)
        cont(body)
      }
    }).catch(error => log.error("range request failed: %s, error '%s'", name, error))
  }
  return range_download
}
