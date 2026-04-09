// Streaming reader with multi-threaded readahead

#include "pentago/gcs/stream.h"
#include "pentago/base/blocks.h"
#include "pentago/gcs/gcs.h"
#include "pentago/gcs/internal.h"
#include "pentago/gcs/mpmc_queue.h"
#include "pentago/data/supertensor.h"
#include "pentago/data/compress.h"
#include "pentago/data/filter.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/log.h"
#include "pentago/utility/range.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <curl/curl.h>
#include <numeric>
#include <sys/stat.h>
#include <thread>
namespace pentago {

using std::atomic;
using std::sort;
using std::thread;
using gcs_internal::gcs_request;

// --- streamer_t ---
//
// Sorts requests by offset, partitions into non-overlapping groups that each
// span slightly more than chunk_bytes, then reader threads claim groups,
// fetch the byte range, extract all requests in the group, and push results
// into an mpmc_queue_t.

struct queued_result_t {
  int id;
  Array<const uint8_t> data;
  int64_t size() const { return data.size(); }
};

// A group of requests that will be fetched together in one read
struct group_t {
  Range<int64_t> bytes;     // byte range to fetch
  Range<int> requests;      // indices into sorted requests array
};

struct streamer_t::impl_t {
  vector<request_t> requests;  // sorted by offset
  vector<group_t> groups;
  mpmc_queue_t<queued_result_t> results;
  atomic<int> next_group{0};
  vector<thread> readers;

  impl_t(const fetch_fn_t& fetch, RawArray<const request_t> reqs,
         const int64_t chunk_bytes, const int64_t readahead_bytes, const int num_threads)
    : results(readahead_bytes, reqs.size()) {
    // Sort requests by offset and verify non-overlapping
    requests.assign(reqs.data(), reqs.data() + reqs.size());
    sort(requests.begin(), requests.end(),
         [](const request_t& a, const request_t& b) { return a.offset < b.offset; });
    for (size_t i = 1; i < requests.size(); i++)
      GEODE_ASSERT(requests[i].offset >= requests[i-1].offset + requests[i-1].size);

    // Partition into groups, each spanning slightly more than chunk_bytes
    const int n = int(requests.size());
    if (!n) return;
    int first = 0;
    int64_t group_start = requests[0].offset;
    for (int i = 0; i < n; i++) {
      const int64_t req_end = requests[i].offset + requests[i].size;
      if (req_end - group_start >= chunk_bytes || i + 1 == n) {
        groups.push_back({range(group_start, req_end), range(first, i + 1)});
        first = i + 1;
        if (first < n) group_start = requests[first].offset;
      }
    }

    for (int t = 0; t < num_threads; t++)
      readers.emplace_back([this, fetch]() { reader_loop(fetch); });
  }

  ~impl_t() {
    for (auto& t : readers) t.join();
  }

  void reader_loop(const fetch_fn_t& fetch) {
    for (;;) {
      const int gi = next_group.fetch_add(1);
      if (gi >= int(groups.size())) return;
      const auto& g = groups[gi];

      auto chunk = fetch(g.bytes.lo, g.bytes.size());
      GEODE_ASSERT(int64_t(chunk.size()) == g.bytes.size());

      for (const int ri : g.requests) {
        const auto& req = requests[ri];
        const int local = int(req.offset - g.bytes.lo);
        results.push({req.id, chunk.slice(local, local + req.size).copy()});
      }
    }
  }
};

streamer_t::streamer_t(const fetch_fn_t& fetch, RawArray<const request_t> requests,
                       const int64_t chunk_bytes, const int64_t readahead_bytes,
                       const int num_threads)
  : impl(std::make_unique<impl_t>(fetch, requests, chunk_bytes, readahead_bytes, num_threads)) {}

streamer_t::~streamer_t() = default;

streamer_t::result_t streamer_t::next() {
  auto item = impl->results.pop();
  if (!item) return {};
  return {item->id, item->data};
}

// --- Create fetch functions for local and GCS ---

namespace {

static fetch_fn_t make_fetch(const string& path) {
  if (is_gcs_path(path)) {
    const auto [bucket, object] = parse_gcs_uri(path);
    CURL* curl = curl_easy_init();
    GEODE_ASSERT(curl);
    char* escaped = curl_easy_escape(curl, object.c_str(), object.size());
    const string url = "https://storage.googleapis.com/storage/v1/b/" + bucket +
                       "/o/" + escaped + "?alt=media";
    curl_free(escaped);
    curl_easy_cleanup(curl);
    return [url](const int64_t offset, const int64_t size) -> Array<const uint8_t> {
      const string range = tfm::format("bytes=%lld-%lld", offset, offset + size - 1);
      const auto resp = gcs_request(url, "GET", range, nullptr, 0);
      if (resp.status == 416) return {};
      if (resp.status != 200 && resp.status != 206)
        die("gcs: GET %s range %s returned %ld", url, range, resp.status);
      Array<uint8_t> result(int(resp.body.size()), uninit);
      memcpy(result.data(), resp.body.data(), resp.body.size());
      return result;
    };
  } else {
    const auto fd = read_local_file(path);
    struct stat st;
    GEODE_ASSERT(stat(path.c_str(), &st) == 0);
    const int64_t file_size = st.st_size;
    return [fd, file_size](const int64_t offset, const int64_t size) -> Array<const uint8_t> {
      if (offset >= file_size) return {};
      const int64_t actual = std::min(size, file_size - offset);
      Array<uint8_t> buf(int(actual), uninit);
      const auto err = fd->pread(buf, offset);
      if (!err.empty()) die("pread failed on %s: %s", fd->name(), err);
      return buf;
    };
  }
}

}  // namespace

// --- supertensor_stream_t ---

struct section_info_t {
  supertensor_header_t header;
  Array<uint64_t,4> offsets;
  Array<uint32_t,4> compressed_sizes;
};

struct supertensor_stream_t::impl_t {
  vector<section_info_t> sections;
  struct block_meta_t {
    int section_index;
    Vector<uint8_t,4> block;
  };
  vector<block_meta_t> block_metas;
  unique_ptr<streamer_t> streamer;
};

// Parse a section index from raw decompressed blob data
static void parse_index(section_info_t& si, RawArray<const uint8_t> raw) {
  const auto blocks = Vector<int,4>(si.header.blocks);
  GEODE_ASSERT(raw.size() == int(sizeof(supertensor_blob_t)) * blocks.product());
  Array<supertensor_blob_t,4> index(blocks,
      shared_ptr<supertensor_blob_t>(reinterpret_cast<supertensor_blob_t*>(
          const_cast<uint8_t*>(raw.data())), [](supertensor_blob_t*) {}));
  to_little_endian_inplace(index.flat());
  si.offsets = Array<uint64_t,4>(blocks, uninit);
  si.compressed_sizes = Array<uint32_t,4>(blocks, uninit);
  for (const int i : range(blocks.product())) {
    si.offsets.flat()[i] = index.flat()[i].offset;
    si.compressed_sizes.flat()[i] = uint32_t(index.flat()[i].compressed_size);
  }
}

supertensor_stream_t::supertensor_stream_t(const string& path,
                                           const int64_t readahead_bytes,
                                           const int num_threads)
  : impl(std::make_unique<impl_t>()) {
  const auto fetch = make_fetch(path);

  // Phase 1: Read file preheader + all section headers in one streamer pass.
  // For single-section files the header starts at offset 0.
  // For multi-section files we need the preheader (32 bytes) to learn
  // section count and stride, then all section headers.
  // Since all headers are at the start of the file and contiguous, one
  // chunk covers them all.
  {
    // First fetch just the preheader to determine layout
    static const char single_magic[] = "pentago supertensor\n";
    static const char multi_magic[] = "pentago sections   \n";
    static constexpr int preheader_size = 20 + 12;  // magic + version/count/stride

    vector<streamer_t::request_t> header_reqs;
    header_reqs.push_back({0, preheader_size, 0});
    streamer_t preheader_stream(fetch, asarray(header_reqs),
                                 preheader_size, readahead_bytes, 1);
    auto pre = preheader_stream.next();
    GEODE_ASSERT(bool(pre));

    if (!memcmp(pre.data.data(), single_magic, 20)) {
      // Single section: header at offset 0
      vector<streamer_t::request_t> reqs;
      reqs.push_back({0, supertensor_header_t::header_size, 0});
      streamer_t s(fetch, asarray(reqs), supertensor_header_t::header_size,
                   readahead_bytes, 1);
      auto r = s.next();
      GEODE_ASSERT(bool(r));
      section_info_t si;
      si.header = supertensor_header_t::unpack(r.data);
      impl->sections.push_back(std::move(si));
    } else if (!memcmp(pre.data.data(), multi_magic, 20)) {
      uint32_t version, section_count, section_header_size;
      memcpy(&version, pre.data.data() + 20, 4);
      memcpy(&section_count, pre.data.data() + 24, 4);
      memcpy(&section_header_size, pre.data.data() + 28, 4);
      version = little_to_native_endian(version);
      section_count = little_to_native_endian(section_count);
      section_header_size = little_to_native_endian(section_header_size);
      GEODE_ASSERT(version == 3);
      const int64_t base = preheader_size;

      // Fetch all section headers
      vector<streamer_t::request_t> reqs;
      for (uint32_t s = 0; s < section_count; s++)
        reqs.push_back({base + int64_t(section_header_size) * s,
                        supertensor_header_t::header_size, int(s)});
      streamer_t hdr_stream(fetch, asarray(reqs), readahead_bytes, readahead_bytes, num_threads);
      impl->sections.resize(section_count);
      for (uint32_t s = 0; s < section_count; s++) {
        auto r = hdr_stream.next();
        GEODE_ASSERT(bool(r));
        impl->sections[r.id].header = supertensor_header_t::unpack(r.data);
      }
    } else {
      die("invalid supertensor file: %s", path);
    }
  }

  // Phase 2: Read all section indices via streamer
  {
    vector<streamer_t::request_t> reqs;
    for (int si = 0; si < int(impl->sections.size()); si++) {
      const auto& hdr = impl->sections[si].header;
      reqs.push_back({int64_t(hdr.index.offset), int32_t(hdr.index.compressed_size), si});
    }
    streamer_t idx_stream(fetch, asarray(reqs), readahead_bytes, readahead_bytes, num_threads);
    for (int si = 0; si < int(impl->sections.size()); si++) {
      auto r = idx_stream.next();
      GEODE_ASSERT(bool(r));
      auto& sec = impl->sections[r.id];
      const auto raw = pentago::decompress(r.data, sec.header.index.uncompressed_size, unevent);
      parse_index(sec, raw);
    }
  }

  // Phase 3: Build block request list
  vector<streamer_t::request_t> requests;
  for (int si = 0; si < int(impl->sections.size()); si++) {
    const auto& sec = impl->sections[si];
    const auto blocks = Vector<int,4>(sec.header.blocks);
    for (const int b0 : range(blocks[0]))
      for (const int b1 : range(blocks[1]))
        for (const int b2 : range(blocks[2]))
          for (const int b3 : range(blocks[3])) {
            const auto bv = Vector<uint8_t,4>(vec(b0, b1, b2, b3));
            const auto iv = Vector<int,4>(bv);
            const int id = int(impl->block_metas.size());
            impl->block_metas.push_back({si, bv});
            streamer_t::request_t req;
            req.offset = sec.offsets[iv];
            req.size = sec.compressed_sizes[iv];
            req.id = id;
            requests.push_back(req);
          }
  }

  slog("stream: %d sections, %d blocks from %s",
       int(impl->sections.size()), int(requests.size()), path);

  static constexpr int64_t default_chunk_bytes = 64 << 20;
  impl->streamer = std::make_unique<streamer_t>(fetch, asarray(requests),
                                                  default_chunk_bytes, readahead_bytes,
                                                  num_threads);
}

supertensor_stream_t::~supertensor_stream_t() = default;

int64_t supertensor_stream_t::total_blocks() const {
  return impl->block_metas.size();
}

supertensor_stream_t::block_t supertensor_stream_t::next() {
  auto result = impl->streamer->next();
  if (!result) return {};
  const auto& meta = impl->block_metas[result.id];
  const auto& sec = impl->sections[meta.section_index];
  return {sec.header.section, meta.block, int(sec.header.filter), result.data};
}

Array<Vector<super_t,2>,4> supertensor_stream_t::decompress(const block_t& block) {
  const auto bs = block_shape(block.section.shape(), block.block);
  auto raw = pentago::decompress(block.compressed,
      sizeof(Vector<super_t,2>) * bs.product(), unevent);
  return unfilter(block.filter, bs, raw);
}

}  // namespace pentago
