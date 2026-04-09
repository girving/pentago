// Internal API for testing — not for external use
#pragma once

#include "pentago/utility/array.h"
#include <functional>
namespace pentago {
namespace gcs_internal {

using std::function;
using std::string;

// Minimal JSON string extraction (flat objects only)
string json_string(const string& json, const string& key);

// Base64url encoding (no padding, URL-safe alphabet)
string base64url_encode(RawArray<const uint8_t> data);
string base64url_encode(const string& s);

// HTTP response from GCS
struct http_response_t {
  long status = 0;
  vector<uint8_t> body;
};

// HTTP request with retries (exponential backoff on 429/500/503).
// method: "GET" or "POST". upload_data: null for GET.
http_response_t gcs_request(const string& url, const string& method,
                             const string& range_header,
                             const uint8_t* upload_data, size_t upload_size);

}  // namespace gcs_internal
}  // namespace pentago
