// Google Cloud access tokens without google-auth-library
//
// Two credential sources are supported, mirroring the two environments the server runs in:
//   1. Inside Cloud Functions, the metadata server hands out tokens for the function's
//      service account (pentago-read).
//   2. Locally, application default credentials from `gcloud auth application-default login`
//      give us a refresh token, which we exchange for access tokens.
// Tokens are cached and refreshed shortly before they expire.

'use strict'
const fs = require('fs')
const os = require('os')
const path = require('path')
const {request} = require('./request.js')

const scope = 'https://www.googleapis.com/auth/devstorage.read_only'
const refresh_margin = 60  // Refresh tokens this many seconds before they expire

// Path to the application default credentials file, or null if it doesn't exist
function adc_path() {
  const env = process.env.GOOGLE_APPLICATION_CREDENTIALS
  if (env)
    return env
  const config = process.env.CLOUDSDK_CONFIG || path.join(os.homedir(), '.config', 'gcloud')
  const p = path.join(config, 'application_default_credentials.json')
  return fs.existsSync(p) ? p : null
}

// Fetch a token as {access_token, expires_in}, marking retryable failures as transient
async function fetch_token(adc) {
  let res
  if (adc) {  // Exchange the ADC refresh token for an access token
    const creds = JSON.parse(fs.readFileSync(adc))
    if (creds.type != 'authorized_user')
      throw Error('unsupported credentials type ' + creds.type + ' in ' + adc + ', expected authorized_user')
    const body = new URLSearchParams({
      client_id: creds.client_id,
      client_secret: creds.client_secret,
      refresh_token: creds.refresh_token,
      grant_type: 'refresh_token',
    }).toString()
    res = await request('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: {'content-type': 'application/x-www-form-urlencoded'},
      body,
    })
  } else {  // Ask the metadata server for a service account token
    const host = process.env.GCE_METADATA_HOST || 'metadata.google.internal'
    res = await request('http://' + host + '/computeMetadata/v1/instance/service-accounts/default/token?scopes=' + scope,
                        {headers: {'metadata-flavor': 'Google'}})
  }
  if (res.status != 200) {
    const error = Error('token request failed with status ' + res.status + ': ' + res.body.toString().trim())
    if (res.status == 429 || res.status >= 500)
      error.transient = true
    throw error
  }
  const token = JSON.parse(res.body)
  if (typeof token.access_token != 'string' || typeof token.expires_in != 'number')
    throw Error('malformed token response: ' + res.body.toString().slice(0, 200))
  return token
}

// Create a token source.  token() returns a Promise of a valid access token, sharing one
// in-flight refresh between concurrent callers.
exports.token_source = () => {
  const adc = adc_path()
  let token = null      // Current access token
  let expires = 0       // Unix time (seconds) at which it expires
  let refreshing = null // In-flight refresh, if any
  return async () => {
    if (token && Date.now() / 1000 < expires - refresh_margin)
      return token
    if (!refreshing) {
      refreshing = fetch_token(adc).then(t => {
        token = t.access_token
        expires = Date.now() / 1000 + t.expires_in
        return token
      }).finally(() => { refreshing = null })
    }
    return refreshing
  }
}
