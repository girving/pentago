// Internal API for testing — not for external use
#pragma once

#include "pentago/data/file.h"
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

// Create a chunk-cached read_file_t backed by a custom fetch function.
// fetch(chunk_index) returns the chunk data (may be shorter than chunk_bytes for the last chunk).
shared_ptr<const read_file_t> read_chunked_file(
    const string& name, int64_t chunk_bytes, int64_t max_cache_bytes,
    const function<Array<const uint8_t>(int64_t)>& fetch);

}  // namespace gcs_internal
}  // namespace pentago
