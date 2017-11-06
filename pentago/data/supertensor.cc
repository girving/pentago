// In memory and out-of-core operations on large four dimensional arrays of superscores

#include "pentago/data/supertensor.h"
#include "pentago/data/compress.h"
#include "pentago/data/filter.h"
#include "pentago/utility/aligned.h"
#include "pentago/utility/char_view.h"
#include "pentago/utility/const_cast.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/index.h"
#include "pentago/utility/random.h"
#include "pentago/utility/curry.h"
#include "pentago/utility/str.h"
#include "pentago/utility/endian.h"
namespace pentago {

using std::make_shared;

// Spaces appended so that sizes match
const char single_supertensor_magic[21]   = "pentago supertensor\n";
const char multiple_supertensor_magic[21] = "pentago sections   \n";

Vector<int,4> supertensor_header_t::block_shape(Vector<uint8_t,4> block) const {
  for (const int i : range(4))
    GEODE_ASSERT(block[i] <= blocks[i]);
  Vector<int,4> bs;
  for (const int i : range(4))
    bs[i] = block[i]+1<blocks[i]?block_size:shape[i]-block_size*(blocks[i]-1);
  return bs;
}

void read_and_uncompress(const read_file_t* fd, supertensor_blob_t blob,
                         const function<void(Array<uint8_t>)>& cont) {
  // Check consistency
  GEODE_ASSERT(blob.compressed_size<(uint64_t)1<<31);
  GEODE_ASSERT(!blob.uncompressed_size || blob.offset);

  // Read using pread for thread safety
  Array<uint8_t> compressed;
  {
    thread_time_t time(read_kind,unevent);
    compressed = Array<uint8_t>(int(blob.compressed_size),uninit);
    const auto error = fd->pread(compressed,blob.offset);
    if (error.size())
      THROW(IOError, "read_and_uncompress pread failed: %s", error);
  }

  // Schedule decompression
  threads_schedule(CPU, compose(cont, curry(decompress,compressed,blob.uncompressed_size,unevent)));
}

void supertensor_writer_t::pwrite(supertensor_blob_t* blob, Array<const uint8_t> data) {
  GEODE_ASSERT(thread_type() == IO);

  // Choose offset
  blob->offset = next_offset->reserve(data.size());

  // Write using pwrite for thread safety
  thread_time_t time(write_kind, unevent);
  const auto error = fd->pwrite(data, blob->offset);
  if (error.size())
    THROW(IOError,"failed to write compressed block to supertensor file: %s", error);
}

void supertensor_writer_t::compress_and_write(supertensor_blob_t* blob, RawArray<const uint8_t> data) {
  GEODE_ASSERT(thread_type()==CPU);

  // Compress
  blob->uncompressed_size = data.size();
  Array<uint8_t> compressed = compress(data, level, unevent);
  blob->compressed_size = compressed.size();

  // Schedule write
  threads_schedule(IO, curry(&Self::pwrite, this, blob, compressed));
}

static const string& check_extension(const string& path) {
  GEODE_ASSERT(path.size() >= 8);
  GEODE_ASSERT(path.substr(path.size()-8) == ".pentago");
  return path;
}

#define HEADER_FIELDS() \
  FIELD(magic) \
  FIELD(version) \
  FIELD(valid) \
  FIELD(stones) \
  FIELD(section) \
  FIELD(shape) \
  FIELD(block_size) \
  FIELD(blocks) \
  FIELD(filter) \
  FIELD(index)

void supertensor_header_t::pack(RawArray<uint8_t> buffer) const {
  GEODE_ASSERT(buffer.size()==header_size);
  int next = 0;
  #define FIELD(f) ({ \
    GEODE_ASSERT(next+sizeof(f)<=size_t(header_size)); \
    const auto le = boost::endian::native_to_little(f); \
    memcpy(buffer.data()+next,&le,sizeof(le)); \
    next += sizeof(le); });
  HEADER_FIELDS()
  #undef FIELD
  GEODE_ASSERT(next==header_size);
}

supertensor_header_t supertensor_header_t::unpack(RawArray<const uint8_t> buffer) {
  GEODE_ASSERT(buffer.size()==header_size);
  supertensor_header_t h;
  int next = 0;
  #define FIELD(f) ({ \
    GEODE_ASSERT(next+sizeof(h.f)<=size_t(header_size)); \
    decltype(h.f) le; \
    memcpy(&le,buffer.data()+next,sizeof(le)); \
    h.f = boost::endian::little_to_native(le); \
    next += sizeof(le); });
  HEADER_FIELDS()
  #undef FIELD
  GEODE_ASSERT(next==header_size);
  return h;
}

static void save_index(Vector<int,4> blocks, Array<supertensor_blob_t,4>* dst, Array<uint8_t> src) {
  GEODE_ASSERT(src.size()==(int)sizeof(supertensor_blob_t)*blocks.product());
  *dst = Array<supertensor_blob_t,4>(blocks, shared_ptr<supertensor_blob_t>(
      src.owner(), reinterpret_cast<supertensor_blob_t*>(src.data())));
}

supertensor_reader_t::supertensor_reader_t(const string& path, const thread_type_t io)
  : fd(read_local_file(check_extension(path))) {
  initialize(path, 0, io);
}

supertensor_reader_t::supertensor_reader_t(const string& path, const shared_ptr<const read_file_t>& fd,
                                           const uint64_t header_offset, const thread_type_t io)
  : fd(fd) {
  initialize(path, header_offset, io);
}

void supertensor_reader_t::initialize(const string& path, const uint64_t header_offset,
                                      const thread_type_t io) {
  // Read header
  const int header_size = supertensor_header_t::header_size;
  uint8_t buffer[header_size];
  {
    const auto error = fd->pread(asarray(buffer), header_offset);
    if (error.size())
      THROW(IOError, "invalid supertensor file \"%s\": error reading header, %s", path, error);
  }
  const auto h = supertensor_header_t::unpack(RawArray<const uint8_t>(header_size,buffer));
  const_cast_(header) = h;

  // Verify header
  if (memcmp(&h.magic,single_supertensor_magic,20))
    THROW(IOError,"invalid supertensor file \"%s\": incorrect magic string",path);
  if (h.version!=2 && h.version!=3)
    THROW(IOError,"supertensor file \"%s\" has unknown version %d",path,h.version);
  if (!h.valid)
    THROW(IOError,"supertensor file \"%s\" is marked invalid",path);
  GEODE_ASSERT(h.stones==h.section.counts.sum().sum());
  for (int i=0;i<4;i++) {
    GEODE_ASSERT(h.section.counts[i].max()<=9);
    GEODE_ASSERT(h.section.counts[i].sum()<=9);
  }
  GEODE_ASSERT((Vector<int,4>(h.shape)==h.section.shape()));
  GEODE_ASSERT(1<=h.block_size && h.block_size<=27);
  GEODE_ASSERT((h.block_size&1)==0);
  GEODE_ASSERT(h.blocks==(h.shape+h.block_size-1)/h.block_size);

  // Read block index
  Array<supertensor_blob_t,4> index;
  threads_schedule(io, curry(read_and_uncompress, &*fd, h.index,
                             curry(save_index, Vector<int,4>(h.blocks), &index)));
  threads_wait_all();
  GEODE_ASSERT((index.shape() == Vector<int,4>(h.blocks)));
  to_little_endian_inplace(index.flat());

  // Compact block index
  const Array<uint64_t,4> offset(index.shape(), uninit);
  const Array<uint32_t,4> compressed_size(index.shape(), uninit);
  for (const int i : range(index.flat().size())) {
    offset.flat()[i] = index.flat()[i].offset;
    compressed_size.flat()[i] = CHECK_CAST_INT(index.flat()[i].compressed_size);
    const auto block = Vector<uint8_t,4>(decompose(index.shape(),i));
    GEODE_ASSERT(sizeof(Vector<super_t,2>)*header.block_shape(block).product() ==
                 index.flat()[i].uncompressed_size);
  }
  const_cast_(this->offset) = offset;
  const_cast_(this->compressed_size_) = compressed_size;
}

supertensor_reader_t::~supertensor_reader_t() {}

static void save(Array<Vector<super_t,2>,4>* dst, Vector<uint8_t,4> block,
                 Array<Vector<super_t,2>,4> src) {
  *dst = src;
}

Array<Vector<super_t,2>,4> supertensor_reader_t::read_block(Vector<uint8_t,4> block) const {
  Array<Vector<super_t,2>,4> data;
  schedule_read_block(block,curry(save,&data));
  threads_wait_all();
  GEODE_ASSERT(data.shape() == header.block_shape(block));
  return data;
}

Array<Vector<super_t,2>,4> unfilter(int filter, Vector<int,4> block_shape, Array<uint8_t> raw_data) {
  GEODE_ASSERT(raw_data.size()==(int)sizeof(Vector<super_t,2>)*block_shape.product());
  Array<Vector<super_t,2>,4> data(block_shape, shared_ptr<Vector<super_t,2>>(
      raw_data.owner(), reinterpret_cast<Vector<super_t,2>*>(raw_data.data())));
  to_little_endian_inplace(data.flat());
  switch (filter) {
    case 0: break;
    case 1: uninterleave(data.flat()); break;
    default: THROW(ValueError,"supertensor_reader_t::read_block: unknown filter %d",filter);
  }
  return data;
}

void supertensor_reader_t::schedule_read_block(Vector<uint8_t,4> block,
                                               const read_cont_t& cont) const {
  schedule_read_blocks(RawArray<const Vector<uint8_t,4>>(1,&block), cont);
}

void supertensor_reader_t::schedule_read_blocks(RawArray<const Vector<uint8_t,4>> blocks,
                                                const read_cont_t& cont) const {
  for (auto block : blocks)
    threads_schedule(IO, curry(read_and_uncompress, &*fd, blob(block),
                               compose(curry(cont, block),
                                       curry(unfilter, header.filter, header.block_shape(block)))));
}

uint64_t supertensor_reader_t::total_size() const {
  uint64_t total = supertensor_header_t::header_size + header.index.compressed_size;
  for (const auto cs : compressed_size_.flat())
    total += cs;
  return total;
}

// Write header at given offset
static void write_header(write_file_t& fd, const supertensor_header_t& h, const uint64_t offset) {
  uint8_t buffer[supertensor_header_t::header_size];
  h.pack(asarray(buffer));
  const auto error = fd.pwrite(asarray(buffer), offset);
  if (error.size())
    THROW(IOError,"failed to write header to supertensor file: %s",error);
}

supertensor_header_t::supertensor_header_t() {}

supertensor_header_t::supertensor_header_t(section_t section, int block_size, int filter)
  : version(3)
  , valid(false)
  , stones(section.sum())
  , section(section)
  , shape(section.shape())
  , block_size(block_size)
  , blocks((shape+block_size-1)/block_size)
  , filter(filter) {
  memcpy(&magic,single_supertensor_magic,20);
  index.uncompressed_size = index.compressed_size = index.offset = 0;
  // valid and index must be filled in later
}

supertensor_writer_t::supertensor_writer_t(const string& path, section_t section, int block_size,
                                           int filter, int level)
  : supertensor_writer_t(path, write_local_file(check_extension(path)), 0,
                         make_shared<next_offset_t>(supertensor_header_t::header_size),
                         section, block_size, filter, level) {}

supertensor_writer_t::supertensor_writer_t(const string& path, const shared_ptr<write_file_t>& fd,
                                           const uint64_t header_offset,
                                           const shared_ptr<next_offset_t>& next_offset,
                                           section_t section, int block_size, int filter, int level)
  : path(path), fd(fd)
  , header(section, block_size, filter) // Set all but valid and index, which finalize fills in later
  , level(level)
  , header_offset(header_offset)
  , next_offset(next_offset)
  , index(Vector<int,4>(header.blocks)) {
  if (block_size & 1)
    THROW(ValueError, "supertensor block size must be even (not %d) to support block-wise reflection",
          block_size);
}

supertensor_writer_t::~supertensor_writer_t() {
  if (!header.valid) {
    // File wasn't finished (due to an error or a failure to call finalize), so delete it
    int r = unlink(path.c_str());
    if (r < 0 && errno != ENOENT)
      std::cerr << format("failed to remove incomplete supertensor file \"%s\": %s",
                          path, strerror(errno)) << std::endl;
  }
}

static Array<uint8_t> filter(int filter, Array<Vector<super_t,2>,4> data) {
  GEODE_ASSERT(thread_type()==CPU);
  switch (filter) {
    case 0: break;
    case 1: interleave(data.flat()); break;
    default: THROW(ValueError,"supertensor_writer_t::write_block: unknown filter %d",filter);
  }
  to_little_endian_inplace(data.flat());
  return char_view_own(data.flat_own());
}

void supertensor_writer_t::write_block(Vector<uint8_t,4> block, Array<Vector<super_t,2>,4> data) {
  schedule_write_block(block,data);
  threads_wait_all();
}

void supertensor_writer_t::schedule_write_block(Vector<uint8_t,4> block,
                                                Array<Vector<super_t,2>,4> data) {
  GEODE_ASSERT(data.shape() == header.block_shape(block));
  const Vector<int,4> block_(block);
  GEODE_ASSERT(index.valid(block_) && !index[block_].offset); // Don't write the same block twice
  threads_schedule(CPU, compose(curry(&Self::compress_and_write, this, &index[block_]),
                                curry(filter, header.filter, data)));
}

void supertensor_writer_t::finalize() {
  if (!fd || header.valid)
    return;

  // Check if all blocks have been written
  threads_wait_all();
  for (auto blob : index.flat())
    if (!blob.offset)
      THROW(RuntimeError,"can't finalize incomplete supertensor file \"%s\"", path);

  // Write index
  supertensor_header_t h = header;
  to_little_endian_inplace(index.flat());
  threads_schedule(CPU, curry(&Self::compress_and_write, this, &h.index,
                              char_view_own(index.flat_own())));
  threads_wait_all();

  // Finalize header
  h.valid = true;
  write_header(*fd, h, header_offset);
  header = h;
  fd.reset();
}

uint64_t supertensor_reader_t::compressed_size(Vector<uint8_t,4> block) const {
  header.block_shape(block); // check validity
  return compressed_size_[Vector<int,4>(block)];
}

uint64_t supertensor_reader_t::uncompressed_size(Vector<uint8_t,4> block) const {
  const auto shape = header.block_shape(block); // checks validity
  return sizeof(Vector<super_t,2>)*shape.product();
}

supertensor_blob_t supertensor_reader_t::blob(Vector<uint8_t,4> block) const {
  supertensor_blob_t blob;
  blob.uncompressed_size = uncompressed_size(block); // checks validity
  blob.compressed_size = compressed_size_[Vector<int,4>(block)];
  blob.offset = offset[Vector<int,4>(block)];
  return blob;
}

uint64_t supertensor_writer_t::compressed_size(Vector<uint8_t,4> block) const {
  header.block_shape(block); // check validity
  return index[Vector<int,4>(block)].compressed_size;
}

uint64_t supertensor_writer_t::uncompressed_size(Vector<uint8_t,4> block) const {
  header.block_shape(block); // check validity
  return index[Vector<int,4>(block)].uncompressed_size;
}

vector<shared_ptr<const supertensor_reader_t>>
open_supertensors(const string& path, const thread_type_t io) {
  return open_supertensors(read_local_file(check_extension(path)),io);
}

vector<shared_ptr<const supertensor_reader_t>>
open_supertensors(const shared_ptr<const read_file_t>& fd, const thread_type_t io) {
  const string path = fd->name();

  // Read magic string to determine file type (single or multiple supertensors)
  uint8_t buffer[20];
  {
    const auto error = fd->pread(asarray(buffer), 0);
    if (error.size())
      THROW(IOError, "invalid supertensor file \"%s\": error reading magic string, %s", path, error);
  }

  // Branch on type
  vector<shared_ptr<const supertensor_reader_t>> readers;
  if (!memcmp(buffer,single_supertensor_magic,20))
    readers.push_back(make_shared<supertensor_reader_t>(path, fd, 0, io));
  else if (!memcmp(buffer,multiple_supertensor_magic,20)) {
    uint32_t header[3];
    {
      const auto error = fd->pread(char_view(asarray(header)), 20);
      if (error.size())
        THROW(IOError, "invalid multiple supertensor file \"%s\": error reading global header, %s",
              path, error);
    }
    for (auto& h : header)
      h = boost::endian::little_to_native(h);
    if (header[0] != 3)
      THROW(IOError,"multiple supertensor file \"%s\" has unknown version %d",path,header[0]);
    if (header[1] >= 8239)
      THROW(IOError,"multiple supertensor file \"%s\" has weird section count %d",path,header[1]);
    const size_t offset = 20+3*sizeof(uint32_t);
    for (int s=0;s<(int)header[1];s++)
      readers.push_back(make_shared<supertensor_reader_t>(path, fd, offset+header[2]*s, io));
  } else
    THROW(IOError,"invalid supertensor file \"%s\": bad magic string",path);
  return readers;
}

vector<shared_ptr<supertensor_writer_t>> supertensor_writers(
    const string& path, RawArray<const section_t> sections, const int block_size, const int filter,
    const int level, Array<const uint64_t> padding) {
  // Open shared file and write pre-header
  const auto fd = write_local_file(check_extension(path));
  const int preheader_size = supertensor_magic_size + 3*sizeof(uint32_t);
  const auto next_offset = make_shared<next_offset_t>(
      preheader_size + supertensor_header_t::header_size*sections.size(), padding);
  fd->pwrite(multiple_supertensor_header(
      sections, Array<const supertensor_blob_t>(sections.size()), block_size, filter)
      .slice(0, preheader_size), 0);

  // Fill in padding
  const uint8_t zero[1] = {0};
  for (const auto p : padding)
    fd->pwrite(asarray(zero), p);

  // Create one writer per section
  vector<shared_ptr<supertensor_writer_t>> writers;
  for (const int s : range(sections.size())) {
    writers.push_back(make_shared<supertensor_writer_t>(
        path, fd, preheader_size + supertensor_header_t::header_size * s, next_offset, sections[s],
        block_size, filter, level));
  }
  return writers;
}

int supertensor_slice(const string& path) {
  const auto fd = read_local_file(check_extension(path));

  // Read magic string to determine file type (single or multiple supertensors)
  uint8_t buffer[20];
  {
    const auto error = fd->pread(asarray(buffer), 0);
    if (error.size())
      THROW(IOError, "invalid supertensor file \"%s\": error reading magic string, %s", path, error);
  }

  // Branch on type
  uint64_t header_offset;
  vector<shared_ptr<const supertensor_reader_t>> readers;
  if (!memcmp(buffer,single_supertensor_magic,20))
    header_offset = 0;
  else if (!memcmp(buffer,multiple_supertensor_magic,20))
    header_offset = 20+3*sizeof(uint32_t);
  else
    THROW(IOError,"invalid supertensor file \"%s\": bad magic string",path);

  // Extract slice from the first header
  {
    const int header_size = supertensor_header_t::header_size;
    uint8_t buffer[header_size];
    const auto error = fd->pread(asarray(buffer), header_offset);
    if (error.size())
      THROW(IOError, "invalid supertensor file \"%s\": error reading header, %s", path, error);
    const auto header = supertensor_header_t::unpack(RawArray<const uint8_t>(header_size,buffer));
    return header.stones;
  }
}

uint64_t supertensor_reader_t::index_offset() const {
  return header.index.offset;
}

Array<const uint64_t,4> supertensor_reader_t::block_offsets() const {
  return offset;
}

uint64_t multiple_supertensor_header_size(const int sections) {
  return supertensor_magic_size + 3*sizeof(uint32_t) +
         supertensor_header_t::header_size * sections;
}

Array<uint8_t> multiple_supertensor_header(
    RawArray<const section_t> sections, RawArray<const supertensor_blob_t> index_blobs,
    const int block_size, const int filter) {
  GEODE_ASSERT(sections.size() == index_blobs.size());
  const auto header_size = multiple_supertensor_header_size(sections.size());
  Array<uint8_t> headers(CHECK_CAST_INT(header_size), uninit);
  size_t offset = 0;
  #define HEADER(pointer,size) \
    memcpy(headers.data()+offset, pointer, size); \
    offset += size;
  #define LE_HEADER(value) \
    value = boost::endian::native_to_little(value); \
    HEADER(&value, sizeof(value));
  uint32_t version = 3;
  uint32_t section_count = sections.size();
  uint32_t section_header_size = supertensor_header_t::header_size;
  HEADER(multiple_supertensor_magic, supertensor_magic_size);
  LE_HEADER(version)
  LE_HEADER(section_count)
  LE_HEADER(section_header_size)
  for (const int s : range(sections.size())) {
    supertensor_header_t sh(sections[s], block_size, filter);
    sh.valid = true;
    sh.index = index_blobs[s];
    sh.pack(headers.slice(int(offset)+range(sh.header_size)));
    offset += sh.header_size;
  }
  GEODE_ASSERT(offset==header_size);
  return headers;
}

uint64_t next_offset_t::reserve(const uint64_t size) {
  spin_t spin(lock);
  if (padding.size()) {
    const uint64_t* p;
    for (;;) {
      p = std::lower_bound(padding.begin(), padding.end(), offset);
      if (p == padding.end() || *p > offset)
        break;
      offset++;
    }
    GEODE_ASSERT(p == padding.end() || *p >= offset + size);
  }
  const auto r = offset;
  offset += size;
  return r;
}

}  // namespace pentago
