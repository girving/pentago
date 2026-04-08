// Google Cloud Storage access via libcurl + service account JWT auth
//
// Provides read_gcs_file() with a chunk cache for efficient random-access reads,
// and gcs_upload() for whole-object writes. Authentication uses a service account
// JSON key file with RS256 JWT signing.
#pragma once

#include "pentago/data/file.h"
#include <cstdint>
namespace pentago {

// Initialize GCS auth from a service account JSON key file.
// Must be called before any GCS operations. Not thread-safe.
void gcs_init(const string& credentials_path);

// Is this a gs:// URI?
bool is_gcs_path(const string& path);

// Parse "gs://bucket/object" into (bucket, object). Throws on invalid format.
std::pair<string, string> parse_gcs_uri(const string& path);

// Open a GCS object for random-access reading with a chunk cache.
// Path format: "gs://bucket/object"
shared_ptr<const read_file_t> read_gcs_file(const string& path);

// Upload a complete byte array as a GCS object. Retries on transient errors.
// Path format: "gs://bucket/object"
void gcs_upload(const string& path, RawArray<const uint8_t> data);

// URI routing: returns read_gcs_file for gs:// paths, read_local_file otherwise
shared_ptr<const read_file_t> open_file(const string& path);

}  // namespace pentago
