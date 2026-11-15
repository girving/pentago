// Full-scale shard test: compare ~1000 random entries against the pentago server
//
// Uses shard_iterator_t with a fixed seed to deterministically sample n_samples entries
// from shard 0 of 1,048,576 (max_slice=17), and compares against hardcoded server values.
//
// portable_hash speed (Cascade Lake, 114M entries, -c opt):
//   initial: iterate 16.3 s, 142 ns/entry
//   precompute section_info + fast_div + local transform_board: iterate 11.6 s, 101 ns/entry
//   direct section lookup table (replace binary search): iterate 10.1 s, 88 ns/entry
//
// To (re)generate server_values:
//   bin/bazel test -c opt //pentago/shard:full_test --test_arg=--gtest_also_run_disabled_tests --test_arg='--gtest_filter=full.DISABLED_query_server' --test_output=streamed 2>&1 | grep '^{'
// Copy the printed map initializer into server_values below.
// When these values change, bump the shard file format version in shard.h/shard.cc.

#include "pentago/shard/shard.h"
#include "pentago/high/board.h"
#include "pentago/mid/tiled.h"
#include "pentago/shard/parallel.h"
#include "pentago/utility/log.h"
#include "pentago/utility/portable_hash.h"
#include "pentago/utility/range.h"
#include "pentago/utility/wall_time.h"
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

// n_samples boards sampled from shard 0 of 1,048,576 (max_slice=17) via shard_iterator_t
// with seed=42, verified against the pentago server (or the tiled solver for 17-stone boards).
// Values: 1=current player wins, 0=tie, -1=current player loses.
static const unordered_map<board_t, int> server_values = {{214971201,1},{76572686998915984,1},{45610556028168679,-1},{639237494397809978,1},{71643969,1},{39734299037,1},{74315707744340224,-1},{1231176080231711415,1},{691614688464666650,1},{7940790,1},{1232025954888982709,1},{45606253072156454,1},{1847650825994114283,-1},{217467122,1},{24547843023577767,1},{1044368214480,1},{3080462351577336064,1},{45614602198007086,1},{45601510403687095,1},{889217234398019610,1},{4910232058,0},{820839560166644619,1},{35475305460609096,-1},{703413087,0},{45655696074942951,1},{828395833933509946,1},{28180215777710,1},{1960758175583636944,1},{137670383780497935,1},{45655653045314704,1},{45630957670837239,1},{228051867130921755,1},{691625958695044586,1},{3131713339753,1},{12985653282555584,1},{1723871064359381,1},{45667867645778135,0},{639245808609479178,1},{282804489,0},{3132249556937,-1},{274168224876283584,1},{45649082416306391,1},{228887634259298826,1},{1436388865980765495,1},{45633388552992160,1},{410391233384029376,1},{9393457152785,0},{1300448838044814800,1},{45633431144311799,-1},{66333366,1},{2054521740419466730,1},{8472763658152659,1},{9573404827389392,1},{9393360421935,-1},{48134819087804,-1},{616745240082580970,1},{22811109384192101,0},{837304204,1},{45627375300848631,1},{2006383088969533120,1},{117228521633,1},{2120364714151201290,1},{1231232197833201879,0},{1849869550148077248,1},{3132250796496,0},{2054526721640711742,-1},{53205411001279143,1},{1231200423902529414,-1},{1231186507585624279,1},{1847113972835555995,1},{13527554437,1},{121616328356335056,1},{28179901198816,1},{17749390821170895,1},{639252105978648318,1},{410394232163085303,1},{1852135266358610832,1},{5389428914,1},{45618776644848590,1},{2667549337365840382,1},{3693543537438697257,-1},{210543463831775584,1},{1869613227427630419,-1},{45637137188325588,1},{7602329293099575,1},{4893458787,1},{349016703166,1},{3696084452280112439,1},{1754602140935032,1},{1869877255752843326,1},{15203518589645217,-1},{24773020084939104,-1},{39369195704,0},{1869886336171573310,1},{410391697894093217,-1},{72650849314360208,1},{1847057596517122812,1},{4993781275,1},{206051250520468179,0},{2485178110581735934,1},};

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

// Query the pentago server (or the tiled solver for 17-stone boards) to generate or verify server_values.
// When server_values is empty: prints the map initializer to paste into this file.
// When server_values is populated: verifies each entry against the live server/tiled solver.
// Run with: bin/bazel test -c opt //pentago/shard:full_test --test_arg=--gtest_also_run_disabled_tests --test_arg='--gtest_filter=full.DISABLED_query_server' --test_output=streamed 2>&1 | grep '^{'
TEST(full, DISABLED_query_server) {
  curl_global_init(CURL_GLOBAL_DEFAULT);

  // Collect samples from the iterator
  shard_iterator_t it("data", total_shards, range(shard_id, shard_id + 1), seed);
  const Array<board_value_t> samples(n_samples, uninit);
  it.next_batch(samples);

  // Query the server or tiled solver in parallel
  const auto write_cb = [](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
    static_cast<string*>(userdata)->append(ptr, size * nmemb);
    return size * nmemb;
  };
  static constexpr int max_retries = 8;

  vector<int> results(n_samples, INT_MIN);
  const int num_threads = int(std::thread::hardware_concurrency());
  parallel_for(num_threads, n_samples, [&](const size_t i) {
    const board_t board = samples[i].board();

    if (count_stones(board) == 17) {
      // 17-stone boards: the server refuses these, so solve them (one thread each, since
      // parallel_for already spreads the samples over the cores)
      const auto high = high_board_t::from_board(board, false);
      const auto mid = midsolve_tiled(high, tiled_options_t());
      const auto found = find_if(mid.begin(), mid.end(),
          [raw = high.raw()](const auto& x) { return get<0>(x) == raw; });
      GEODE_ASSERT(found != mid.end());
      results[i] = get<1>(*found);
    } else {
      // <17-stone boards: query the server with exponential backoff
      const string url = tfm::format(
          "https://perfect-pentago.net/backend/%llu",
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
  slog("verified %d boards against server/tiled solver", n_samples);
}

TEST(full, portable_hash) {
  // Count total entries in this shard across all slices
  static constexpr int max_slice = 17;
  const shard_locator_t loc(total_shards, range(shard_id, shard_id + 1));
  uint64_t total_entries = 0;
  for (const int s : range(max_slice + 1))
    total_entries += loc.shard_size(shard_mapping_t(s).total(), shard_id);

  // Expand the entire shard into board_value_t entries via shard_iterator_t
  const auto t0 = wall_time();
  shard_iterator_t it("data", total_shards, range(shard_id, shard_id + 1), seed);
  const Array<board_value_t> entries(int(total_entries), uninit);
  it.next_batch(entries);
  const auto t1 = wall_time();
  const auto h = portable_hash(entries);
  const auto t2 = wall_time();
  slog("shard %d: %d entries, iterate %.3f s, hash %.3f s, total %.3f s, %.0f ns/entry",
       shard_id, entries.size(), (t1 - t0).seconds(), (t2 - t1).seconds(), (t2 - t0).seconds(),
       (t1 - t0).seconds() / entries.size() * 1e9);
  ASSERT_EQ(h, "f2bd9dbcd4990ec12d0d0b49c80445b80aa7459d");
}

}  // namespace
}  // namespace pentago
