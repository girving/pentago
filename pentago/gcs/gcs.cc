// Google Cloud Storage access via libcurl + service account JWT auth

#include "pentago/gcs/gcs.h"
#include "pentago/gcs/internal.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/log.h"
#include <curl/curl.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <cstring>
#include <ctime>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
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

string base64url_encode(const string& s) {
  return base64url_encode({CHECK_CAST_INT(s.size()), reinterpret_cast<const uint8_t*>(s.data())});
}

}  // namespace gcs_internal

// --- GCS-specific internals ---

namespace {

using gcs_internal::json_string;
using gcs_internal::base64url_encode;

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

}  // namespace

// --- HTTP helper with retries (in gcs_internal for use by stream.cc) ---

namespace gcs_internal {

static size_t write_to_vector(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* v = static_cast<vector<uint8_t>*>(userdata);
  v->insert(v->end(), ptr, ptr + size * nmemb);
  return size * nmemb;
}

http_response_t gcs_request(const string& url, const string& method,
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

    if (upload_data) {
      curl_easy_setopt(curl, CURLOPT_POST, 1L);
      curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(upload_size));
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, upload_data);
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

}  // namespace gcs_internal

namespace {

using gcs_internal::gcs_request;

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

Array<const uint8_t> gcs_download(const string& path) {
  const auto [bucket, object] = parse_gcs_uri(path);
  const string url = "https://storage.googleapis.com/storage/v1/b/" + bucket +
                     "/o/" + url_encode(object) + "?alt=media";
  const auto resp = gcs_request(url, "GET", "", nullptr, 0);
  if (resp.status != 200)
    die("gcs: GET %s returned %ld", path, resp.status);
  Array<uint8_t> result(int(resp.body.size()), uninit);
  memcpy(result.data(), resp.body.data(), resp.body.size());
  return result;
}

void gcs_upload(const string& path, RawArray<const uint8_t> data) {
  const auto [bucket, object] = parse_gcs_uri(path);
  const string url = "https://storage.googleapis.com/upload/storage/v1/b/" + bucket +
                     "/o?uploadType=media&name=" + url_encode(object);
  const auto resp = gcs_request(url, "POST", "", data.data(), data.size());
  if (resp.status != 200 && resp.status != 201)
    die("gcs: upload %s returned %ld", path, resp.status);
}

}  // namespace pentago
