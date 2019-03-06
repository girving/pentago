// Minimal interface to Rackspace Cloud Files

'use strict'
const concat = require('concat-stream')
const https = require('https')

const early_token_timeout = 1000 * 60 * 5  // 5 minutes in ms

function match_region(a, b) {
  if (!a && !b)
    return true
  else if ((!a && b) || (a && !b))
    return false
  else
    return a.toLowerCase() === b.toLowerCase()
}

// Simple promise-based https request.  Options:
//   uri: URL
//   method: GET, POST, etc.
//   headers: optional http headers
//   body: optional body (already encoded if json)
function request(options) {
  return new Promise((resolve, reject) => {
    const req = https.request(options.uri, options, res => {
      if (res.statusCode >= 400) {
        res.resume()
        reject('https request failed, code ' + res.statusCode)
      } else {
        const data = []
        res.on('data', chunk => data.push(chunk))
        res.on('end', () => resolve(Buffer.concat(data)))
      }
    })
    req.on('error', reject)
    if (options.body)
      req.write(options.body)
    req.end()
  })
}

exports.downloader = function (options, stats, log) {
  const region = options.region
  const username = options.username
  const apiKey = options.apiKey
  const useInternal = options.useInternal

  // Authorization.  We use a promised token so that simultaneous auth requests get merged,
  // and wipe the promise on expiration.
  let promised_token = null
  function auth() {
    // Launch a token request if necessary
    if (!promised_token)
      promised_token = request({
        uri: 'https://identity.api.rackspacecloud.com/v2.0/tokens',
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
        body: JSON.stringify({auth: {'RAX-KSKEY:apiKeyCredentials': {username: username, apiKey: apiKey}}}),
      }).then(body => {
        const access = JSON.parse(body).access

        // Build token
        const token = access.token
        if (!token || !token.id || !token.expires) throw Error('Invalid token')
        token.expires = new Date(token.expires)

        // Add service url
        const service = access.serviceCatalog.find(service => service.type.toLowerCase() == 'object-store')
        if (!service) throw Error('Unable to find matching endpoint for requested service')
        const endpoint = service.endpoints.find(endpoint => match_region(endpoint.region, region))
        if (!endpoint) throw Error('Unable to identify endpoint url')
        token.service_url = useInternal && endpoint.internalURL ? endpoint.internalURL : endpoint.publicURL

        // Schedule the destruction of the promise
        const timeout = token.expires.getTime() - new Date().getTime() - early_token_timeout
        setTimeout(() => { promised_token = null }, timeout)

        // All done!
        return token
      })
    return promised_token
  }

  async function download(options) {
    const token = await auth()
    options.headers['x-auth-token'] = token.id
    return request({
      uri: token.service_url + '/' + options.container + '/' + options.path,
      method: 'GET',
      headers: options.headers,
    })
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
