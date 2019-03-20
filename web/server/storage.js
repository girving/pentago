// Minimal interface to Rackspace Cloud Files

'use strict'
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

// Simple promise-based https request
function request(options) {
  return new Promise((resolve, reject) => {
    const req = https.request(options, res => {
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

exports.downloader = (options, stats, log) => {
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
        host: 'identity.api.rackspacecloud.com',
        path: '/v2.0/tokens',
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
        body: JSON.stringify({auth: {'RAX-KSKEY:apiKeyCredentials': {username: username, apiKey: apiKey}}}),
      }).then(body => {
        const access = JSON.parse(body).access

        // Build token
        const token = access.token
        if (!token || !token.id || !token.expires) throw Error('rackspace auth: Invalid token')
        token.expires = new Date(token.expires)

        // Add service url
        const service = access.serviceCatalog.find(service => service.type.toLowerCase() == 'object-store')
        if (!service) throw Error('rackspace auth: Unable to find matching endpoint for requested service')
        const endpoint = service.endpoints.find(endpoint => match_region(endpoint.region, region))
        if (!endpoint) throw Error('rackspace auth: Unable to identify endpoint url')
        token.service_url = useInternal && endpoint.internalURL ? endpoint.internalURL : endpoint.publicURL

        // Schedule the destruction of the promise
        const timeout = token.expires.getTime() - new Date().getTime() - early_token_timeout
        setTimeout(() => { promised_token = null }, timeout).unref()

        // All done!
        return token
      })
    return promised_token
  }

  async function download(options) {
    const token = await auth()
    const url = new URL(token.service_url)
    options.headers['x-auth-token'] = token.id
    return request({
      host: url.hostname,
      path: url.pathname + '/' + options.container + '/' + options.path,
      method: 'GET',
      headers: options.headers,
    })
  }

  async function range_download({container, object, offset, size, allow_truncate}) {
    const name = object + ', ' + offset + '+' + size
    log.debug('range request %s, active %d', name, stats.active_gets++)
    let body
    try {
      body = await download({
        container: container,
        path: object,
        headers: {range: 'bytes=' + offset + '-' + (offset+size-1)}
      })
    } catch (e) {
      log.error('range request failed: ' + name + ", error '" + e.message + "'")
      throw e
    }
    if (allow_truncate ? body.length > size : body.length != size) {
      const op = allow_truncate ? ' > ' : ' != '
      throw Error('range request failed: ' + name + ', got size ' + body.length + op + size)
    } else {
      log.debug('range response %s, active %d', name, --stats.active_gets)
      return body
    }
  }
  return range_download
}
