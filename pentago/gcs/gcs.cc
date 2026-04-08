// Google Cloud Storage access via libcurl + service account JWT auth

#include "pentago/gcs/gcs.h"
#include "pentago/gcs/internal.h"
#include "pentago/data/lru.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/log.h"
#include <curl/curl.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_set>
namespace pentago {

using std::ifstream;
using std::lock_guard;
using std::mutex;
using std::pair;

// --- Testable internals (exposed via internal.h) ---

namespace gcs_internal {

string json_string(const string& json, const string& key) {
  const string needle = "\"" + key + "\"";
  const auto pos = json.find(needle);
  if (pos == string::npos) die("gcs: missing JSON key \"%s\"", key);
  auto start = json.find('"', pos + needle.size() + 1);
  GEODE_ASSERT(start != string::npos);
  start++;
  string result;
  for (size_t i = start; i < json.size() && json[i] != '"'; i++) {
    if (json[i] == '\\' && i + 1 < json.size()) {
      i++;
      if (json[i] == 'n') result += '\n';
      else result += json[i];
    } else {
      result += json[i];
    }
  }
  return result;
}

string base64url_encode(RawArray<const uint8_t> data) {
  static constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  const int n = data.size();
  string result;
  result.reserve((n + 2) / 3 * 4);
  for (int i = 0; i < n; i += 3) {
    const uint32_t b0 = data[i];
    const uint32_t b1 = i + 1 < n ? data[i + 1] : 0;
    const uint32_t b2 = i + 2 < n ? data[i + 2] : 0;
    const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
    result += table[(triple >> 18) & 63];
    result += table[(triple >> 12) & 63];
    if (i + 1 < n) result += table[(triple >> 6) & 63];
    if (i + 2 < n) result += table[triple & 63];
  }
  return result;
}

// --- Chunk-cached reader ---

struct read_chunked_file_t final : public read_file_t {
  const string name_;
  const int64_t chunk_bytes_;
  const int64_t max_cache_bytes_;
  const function<Array<const uint8_t>(int64_t)> fetch_;

  mutable mutex cache_mutex;
  mutable std::condition_variable cache_cv;
  mutable lru_t<int64_t, Array<const uint8_t>> cache;
  mutable std::unordered_set<int64_t> pending;
  mutable int64_t cache_total = 0;

  read_chunked_file_t(const string& name, const int64_t chunk_bytes,
                       const int64_t max_cache_bytes,
                       const function<Array<const uint8_t>(int64_t)>& fetch)
    : name_(name), chunk_bytes_(chunk_bytes), max_cache_bytes_(max_cache_bytes), fetch_(fetch) {}

  ~read_chunked_file_t() override = default;

  string name() const override { return name_; }

  string pread(RawArray<uint8_t> data, const uint64_t offset) const override {
    uint64_t pos = offset;
    uint8_t* dst = data.data();
    int64_t remaining = data.size();

    while (remaining > 0) {
      const int64_t ci = pos / chunk_bytes_;
      const int64_t chunk_offset = pos % chunk_bytes_;
      const auto chunk = get_chunk(ci);

      // Compute copy size from actual chunk size (last chunk may be short)
      if (chunk_offset >= int64_t(chunk.size()))
        return tfm::format("read past end of %s", name_);
      const int64_t to_copy = std::min(remaining, int64_t(chunk.size()) - chunk_offset);
      memcpy(dst, chunk.data() + chunk_offset, to_copy);

      dst += to_copy;
      pos += to_copy;
      remaining -= to_copy;
    }
    return "";
  }

private:
  // Get a chunk from cache, or fetch it. If another thread is already fetching
  // the same chunk, wait for it rather than issuing a duplicate request.
  // Returns a refcounted copy so the data stays alive after cache eviction.
  Array<const uint8_t> get_chunk(const int64_t ci) const {
    std::unique_lock<mutex> lock(cache_mutex);

    if (const auto* p = cache.get(ci)) return *p;

    // Wait if another thread is fetching this chunk
    while (pending.count(ci)) cache_cv.wait(lock);
    if (const auto* p = cache.get(ci)) return *p;

    // We'll fetch it
    pending.insert(ci);
    lock.unlock();

    auto chunk = fetch_(ci);

    // Insert into cache and notify waiters
    lock.lock();
    pending.erase(ci);
    cache_total += chunk.size();
    cache.add(ci, chunk);
    while (cache_total > max_cache_bytes_) {
      auto [key, val] = cache.drop();
      cache_total -= val.size();
    }
    cache_cv.notify_all();
    return chunk;
  }
};

shared_ptr<const read_file_t> read_chunked_file(
    const string& name, const int64_t chunk_bytes, const int64_t max_cache_bytes,
    const function<Array<const uint8_t>(int64_t)>& fetch) {
  return std::make_shared<read_chunked_file_t>(name, chunk_bytes, max_cache_bytes, fetch);
}

string base64url_encode(const string& s) {
  return base64url_encode({CHECK_CAST_INT(s.size()), reinterpret_cast<const uint8_t*>(s.data())});
}

}  // namespace gcs_internal

// --- GCS-specific internals ---

namespace {

using gcs_internal::json_string;
using gcs_internal::base64url_encode;

// Default chunk cache configuration
static constexpr int64_t chunk_bytes = 64 << 20;   // 64 MB per chunk
static constexpr int64_t max_cache_bytes = 10LL << 30;  // 10 GB total cache

// --- Auth: service account JWT with RS256 ---

static string sa_client_email;
static EVP_PKEY* sa_private_key = nullptr;

static string make_jwt() {
  GEODE_ASSERT(sa_private_key);
  const auto now = time(nullptr);
  const string header = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
  const string claims = tfm::format(
      "{\"iss\":\"%s\",\"scope\":\"https://www.googleapis.com/auth/devstorage.full_control\","
      "\"aud\":\"https://oauth2.googleapis.com/token\",\"iat\":%d,\"exp\":%d}",
      sa_client_email, now, now + 3600);
  const string payload = base64url_encode(header) + "." + base64url_encode(claims);

  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  GEODE_ASSERT(ctx);
  GEODE_ASSERT(EVP_DigestSignInit(ctx, nullptr, EVP_sha256(), nullptr, sa_private_key) == 1);
  GEODE_ASSERT(EVP_DigestSignUpdate(ctx, payload.data(), payload.size()) == 1);
  size_t sig_len = 0;
  GEODE_ASSERT(EVP_DigestSignFinal(ctx, nullptr, &sig_len) == 1);
  vector<uint8_t> sig(sig_len);
  GEODE_ASSERT(EVP_DigestSignFinal(ctx, sig.data(), &sig_len) == 1);
  EVP_MD_CTX_free(ctx);

  return payload + "." + base64url_encode({int(sig_len), sig.data()});
}

// Cached bearer token
static mutex token_mutex;
static string cached_token;
static time_t token_expiry = 0;

static size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
  static_cast<string*>(userdata)->append(ptr, size * nmemb);
  return size * nmemb;
}

static string get_bearer_token() {
  {
    lock_guard<mutex> lock(token_mutex);
    if (time(nullptr) < token_expiry - 60)
      return cached_token;
  }

  const string jwt = make_jwt();
  const string post_data = "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"
                           "&assertion=" + jwt;

  CURL* curl = curl_easy_init();
  GEODE_ASSERT(curl);
  string response;
  curl_easy_setopt(curl, CURLOPT_URL, "https://oauth2.googleapis.com/token");
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_string);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  const CURLcode res = curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  if (res != CURLE_OK) die("gcs: token exchange failed: %s", curl_easy_strerror(res));

  const string token = json_string(response, "access_token");
  if (token.empty()) die("gcs: no access_token in token response");

  lock_guard<mutex> lock(token_mutex);
  cached_token = token;
  token_expiry = time(nullptr) + 3600;
  return cached_token;
}

// --- HTTP helper with retries ---

struct http_response_t {
  long status = 0;
  vector<uint8_t> body;
};

static size_t write_to_vector(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* v = static_cast<vector<uint8_t>*>(userdata);
  v->insert(v->end(), ptr, ptr + size * nmemb);
  return size * nmemb;
}

struct upload_state_t {
  const uint8_t* data;
  size_t remaining;
};

static size_t read_upload(char* buffer, size_t size, size_t nmemb, void* userdata) {
  auto* state = static_cast<upload_state_t*>(userdata);
  const size_t to_copy = std::min(size * nmemb, state->remaining);
  memcpy(buffer, state->data, to_copy);
  state->data += to_copy;
  state->remaining -= to_copy;
  return to_copy;
}

static http_response_t gcs_request(const string& url, const string& method,
                                   const string& range_header,
                                   const uint8_t* upload_data, size_t upload_size) {
  static constexpr int max_retries = 5;
  for (int attempt = 0; attempt <= max_retries; attempt++) {
    if (attempt > 0) {
      const int delay_ms = (1 << attempt) * 500;
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

    const string token = get_bearer_token();
    CURL* curl = curl_easy_init();
    GEODE_ASSERT(curl);

    http_response_t resp;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_vector);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp.body);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + token).c_str());
    if (!range_header.empty())
      headers = curl_slist_append(headers, ("Range: " + range_header).c_str());

    upload_state_t upload_state{upload_data, upload_size};
    if (method == "PUT") {
      curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
      curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(upload_size));
      curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_upload);
      curl_easy_setopt(curl, CURLOPT_READDATA, &upload_state);
      headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
    }

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    const CURLcode res = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
      if (attempt < max_retries) continue;
      die("gcs: %s %s failed: %s", method, url, curl_easy_strerror(res));
    }
    if (resp.status == 429 || resp.status == 500 || resp.status == 503) {
      if (attempt < max_retries) continue;
    }
    return resp;
  }
  die("gcs: unreachable");
}

static string url_encode(const string& s) {
  CURL* curl = curl_easy_init();
  GEODE_ASSERT(curl);
  char* escaped = curl_easy_escape(curl, s.c_str(), s.size());
  string result(escaped);
  curl_free(escaped);
  curl_easy_cleanup(curl);
  return result;
}

}  // namespace

// --- Public API ---

void gcs_init(const string& credentials_path) {
  curl_global_init(CURL_GLOBAL_DEFAULT);
  ifstream f(credentials_path);
  if (!f.good()) die("gcs: can't open credentials file: %s", credentials_path);
  const string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  sa_client_email = gcs_internal::json_string(json, "client_email");
  const string pem = gcs_internal::json_string(json, "private_key");
  BIO* bio = BIO_new_mem_buf(pem.data(), pem.size());
  sa_private_key = PEM_read_bio_PrivateKey(bio, nullptr, nullptr, nullptr);
  BIO_free(bio);
  if (!sa_private_key) die("gcs: failed to parse private key from %s", credentials_path);
  slog("gcs: authenticated as %s", sa_client_email);
}

bool is_gcs_path(const string& path) {
  return path.size() > 5 && path.substr(0, 5) == "gs://";
}

pair<string, string> parse_gcs_uri(const string& path) {
  GEODE_ASSERT(is_gcs_path(path));
  const auto slash = path.find('/', 5);
  GEODE_ASSERT(slash != string::npos && slash > 5);
  return {path.substr(5, slash - 5), path.substr(slash + 1)};
}

shared_ptr<const read_file_t> read_gcs_file(const string& path) {
  const auto [bucket, object] = parse_gcs_uri(path);
  const string base_url = "https://storage.googleapis.com/storage/v1/b/" + bucket +
                           "/o/" + url_encode(object) + "?alt=media";
  return gcs_internal::read_chunked_file(path, chunk_bytes, max_cache_bytes,
      [base_url, path](const int64_t ci) -> Array<const uint8_t> {
        const int64_t start = ci * chunk_bytes;
        const int64_t end = start + chunk_bytes - 1;
        const string range = tfm::format("bytes=%lld-%lld", start, end);
        const auto resp = gcs_request(base_url, "GET", range, nullptr, 0);
        if (resp.status != 200 && resp.status != 206)
          die("gcs: GET %s range %s returned %ld", path, range, resp.status);
        Array<uint8_t> result(int(resp.body.size()), uninit);
        memcpy(result.data(), resp.body.data(), resp.body.size());
        return result;
      });
}

void gcs_upload(const string& path, RawArray<const uint8_t> data) {
  const auto [bucket, object] = parse_gcs_uri(path);
  const string url = "https://storage.googleapis.com/upload/storage/v1/b/" + bucket +
                     "/o?uploadType=media&name=" + url_encode(object);
  const auto resp = gcs_request(url, "PUT", "", data.data(), data.size());
  if (resp.status != 200 && resp.status != 201)
    die("gcs: upload %s returned %ld", path, resp.status);
}

shared_ptr<const read_file_t> open_file(const string& path) {
  return is_gcs_path(path) ? read_gcs_file(path) : read_local_file(path);
}

}  // namespace pentago
