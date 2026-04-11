// Full-scale shard test: compare ~1000 random entries against the pentago server
//
// Uses shard_iterator_t with a fixed seed to deterministically sample n_samples entries
// from shard 0 of 1,048,576 (max_slice=18), and compares against hardcoded server values.
//
// To (re)generate server_values:
//   bin/bazel test -c opt //pentago/shard:full_test --test_arg=--gtest_also_run_disabled_tests --test_arg='--gtest_filter=full.DISABLED_query_server' --test_output=streamed 2>&1 | grep '^{'
// Copy the printed map initializer into server_values below.
// When these values change, bump the shard file format version in shard.h/shard.cc.

#include "pentago/shard/shard.h"
#include "pentago/high/board.h"
#include "pentago/mid/midengine.h"
#include "pentago/shard/parallel.h"
#include "pentago/utility/log.h"
#include "pentago/utility/portable_hash.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <curl/curl.h>
#include <thread>
#include <unordered_map>
namespace pentago {
namespace {

using std::find_if;
using std::get;
using std::unordered_map;

static constexpr int total_shards = 1 << 20;  // 1,048,576
static constexpr int shard_id = 0;
static constexpr int n_samples = 100;
static constexpr uint128_t seed = 42;

// n_samples boards sampled from shard 0 of 1,048,576 (max_slice=18) via shard_iterator_t
// with seed=42, verified against the pentago server (or midsolve for 18-stone boards).
// Values: 1=current player wins, 0=tie, -1=current player loses.
static const unordered_map<board_t, int> server_values = {{66333366,1},{644959458,1},{48134819087804,-1},{319210650050627806,1},{214971201,1},{1254255932689682470,1},{71643969,1},{76572686998915984,1},{45610556028168679,-1},{3718929843174843519,1},{1232025954888982709,1},{1859434434822013259,1},{639237494397809978,1},{211117636807496147,1},{1983843105206582008,0},{39734299037,1},{617580451938048481,0},{3943132718108790,1},{74315707744340224,-1},{1231176080231711415,1},{691614688464666650,1},{1044368214480,1},{45606253072156454,1},{2872799944807233848,-1},{3080462351577336064,1},{22811109384192101,0},{1033577004640055599,1},{45601510403687095,1},{662074062468623680,-1},{889217234398019610,1},{957589472228147862,1},{4910232058,0},{137927145524443331,1},{35475305460609096,-1},{1847650825994114283,-1},{3761929866376199861,1},{250842806980852736,-1},{16060897074815246,1},{1299590531894290469,1},{616161828649375678,0},{45657534198918641,1},{45655696074942951,1},{828395833933509946,1},{217467122,1},{24547843023577767,1},{7602329293099575,1},{2693749977303684103,1},{45614602198007086,1},{28180215777710,1},{306535721007268442,1},{1960758175583636944,1},{820839560166644619,1},{1940507534367074570,1},{624386647811758953,1},{32992895987552292,0},{45630957670837239,1},{691625958695044586,1},{625739838207700019,1},{1855264545160180078,1},{1850145833175620709,-1},{3131713339753,1},{2462412237370034543,0},{68746084665263758,1},{69271840778173983,1},{205206833436570723,1},{703413087,0},{137670383780497935,1},{319257808839117022,1},{12985653282555584,1},{155172352509,1},{3718874644215309439,1},{1254257968541536294,1},{45667867645778135,0},{639245808609479178,1},{45655653045314704,1},{3132249556937,-1},{274168224876283584,1},{45649082416306391,1},{211158069712851411,0},{1859425277095051391,1},{617578210389793249,1},{1983837178286455544,1},{228887634259298826,1},{9393457152785,0},{2872751367962704184,1},{228051867130921755,1},{3956387552628854,1},{1847057596517122812,1},{1300448838044814800,1},{45633431144311799,-1},{2054521740419466730,1},{662089026226695488,-1},{7940790,1},{1033577004032667951,1},{137927146087528643,1},{957580015140602518,1},{1231479515471821824,1},{2632293,1},{114005389512359605,1},{9573404827389392,1}};

TEST(full, vs_server) {
  shard_iterator_t it("data", total_shards, range(shard_id, shard_id + 1), seed);

  const Array<board_value_t> samples(n_samples, uninit);
  it.next_batch(samples);
  for (const auto& bv : samples) {
    const auto found = server_values.find(bv.board());
    ASSERT_NE(found, server_values.end()) << "board " << bv.board() << " not in server_values";
    PENTAGO_ASSERT_EQ(shard_to_server_value(bv.board(), bv.value()), found->second);
  }
  slog("checked %d server-verified boards", int(server_values.size()));
}

// Query the pentago server (or midsolve for 18-stone boards) to generate or verify server_values.
// When server_values is empty: prints the map initializer to paste into this file.
// When server_values is populated: verifies each entry against the live server/midsolve.
// Run with: bin/bazel test -c opt //pentago/shard:full_test --test_arg=--gtest_also_run_disabled_tests --test_arg='--gtest_filter=full.DISABLED_query_server' --test_output=streamed 2>&1 | grep '^{'
TEST(full, DISABLED_query_server) {
  curl_global_init(CURL_GLOBAL_DEFAULT);

  // Collect samples from the iterator
  shard_iterator_t it("data", total_shards, range(shard_id, shard_id + 1), seed);
  const Array<board_value_t> samples(n_samples, uninit);
  it.next_batch(samples);

  // Query server or midsolve in parallel
  const auto write_cb = [](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
    static_cast<string*>(userdata)->append(ptr, size * nmemb);
    return size * nmemb;
  };
  static constexpr int max_retries = 8;

  vector<int> results(n_samples, INT_MIN);
  const int num_threads = int(std::thread::hardware_concurrency());
  parallel_for(num_threads, n_samples, [&](const size_t i) {
    const board_t board = samples[i].board();

    if (count_stones(board) == 18) {
      // 18-stone boards: use midsolve (server doesn't handle these)
      thread_local Array<halfsuper_s> workspace;
      if (!workspace.size()) workspace = midsolve_workspace(18);
      const auto high = high_board_t::from_board(board, false);
      const auto mid = midsolve(high, workspace);
      const auto found = find_if(mid.begin(), mid.end(),
          [raw = high.raw()](const auto& x) { return get<0>(x) == raw; });
      GEODE_ASSERT(found != mid.end());
      results[i] = get<1>(*found);
    } else {
      // <18-stone boards: query the server with exponential backoff
      const string url = tfm::format(
          "https://us-central1-naml-148801.cloudfunctions.net/pentago/%llu",
          (unsigned long long)board);
      for (int attempt = 0; attempt <= max_retries; attempt++) {
        if (attempt > 0)
          std::this_thread::sleep_for(std::chrono::milliseconds((1 << attempt) * 100));
        CURL* curl = curl_easy_init();
        GEODE_ASSERT(curl);
        string response;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        const CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        if (res != CURLE_OK || response.find('{') == string::npos) {
          if (attempt < max_retries) continue;
          ADD_FAILURE() << "curl failed for board " << board << " after retries: "
                        << (res != CURLE_OK ? curl_easy_strerror(res) : response);
          return;
        }
        // Extract integer value for key "board_id" from flat JSON response
        const string key = "\"" + std::to_string(board) + "\"";
        const auto pos = response.find(key);
        if (pos == string::npos) {
          if (attempt < max_retries) continue;
          ADD_FAILURE() << "board " << board << " key not in response: " << response;
          return;
        }
        const auto colon = response.find(':', pos + key.size());
        GEODE_ASSERT(colon != string::npos);
        results[i] = std::stoi(response.substr(colon + 1));
        break;
      }
    }
  });

  const bool generating = server_values.empty();
  if (generating) fprintf(stderr, "{");
  int mismatches = 0;
  for (const int i : range(n_samples)) {
    if (results[i] == INT_MIN) continue;  // already failed
    const board_t board = samples[i].board();
    if (generating) {
      fprintf(stderr, "{%llu,%d},", (unsigned long long)board, results[i]);
    } else {
      const auto found = server_values.find(board);
      ASSERT_NE(found, server_values.end()) << "board " << board << " not in server_values";
      if (found->second != results[i]) {
        slog("MISMATCH board %llu: stored %d got %d",
             (unsigned long long)board, found->second, results[i]);
        mismatches++;
      }
    }
  }
  if (generating) {
    fprintf(stderr, "};\n");
    GTEST_SKIP() << "printed map initializer; paste into server_values";
  }
  ASSERT_EQ(mismatches, 0);
  slog("verified %d boards against server/midsolve", n_samples);
}

TEST(full, portable_hash) {
  // Count total entries in this shard across all slices
  static constexpr int max_slice = 18;
  const shard_locator_t loc(total_shards, range(shard_id, shard_id + 1));
  uint64_t total_entries = 0;
  for (const int s : range(max_slice + 1))
    total_entries += loc.shard_size(shard_mapping_t(s).total(), shard_id);

  // Expand the entire shard into board_value_t entries via shard_iterator_t
  shard_iterator_t it("data", total_shards, range(shard_id, shard_id + 1), seed);
  const Array<board_value_t> entries(int(total_entries), uninit);
  it.next_batch(entries);
  const auto h = portable_hash(entries);
  slog("shard %d: %d entries, hash %s", shard_id, entries.size(), h);
  ASSERT_EQ(h, "ca427dc51297b3ef810580ac65d2ee0af0e211ce");
}

}  // namespace
}  // namespace pentago
