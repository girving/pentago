// In memory and out-of-core operations on large four dimensional arrays of superscores

#include <pentago/data/supertensor.h>
#include <pentago/data/compress.h>
#include <pentago/utility/convert.h>
#include <pentago/data/filter.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/char_view.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/endian.h>
#include <geode/array/IndirectArray.h>
#include <geode/python/Class.h>
#include <geode/python/to_python.h>
#include <geode/python/stl.h>
#include <geode/random/Random.h>
#include <geode/utility/compose.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/curry.h>
#include <geode/utility/Log.h>
#include <geode/utility/str.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
namespace pentago {

using std::cout;
using std::cerr;
using std::endl;

// Spaces appended so that sizes match
const char single_supertensor_magic[21]   = "pentago supertensor\n";
const char multiple_supertensor_magic[21] = "pentago sections   \n";

struct supertensor_header_py : public Object, public supertensor_header_t {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
protected:
  supertensor_header_py(supertensor_header_t h)
    : supertensor_header_t(h) {}
};

GEODE_DEFINE_TYPE(supertensor_header_py)
GEODE_DEFINE_TYPE(supertensor_reader_t)
GEODE_DEFINE_TYPE(supertensor_writer_t)

#ifdef GEODE_PYTHON
static PyObject* to_python(const supertensor_header_t& h) {
  return to_python(new_<supertensor_header_py>(h));
}
#endif

Vector<int,4> supertensor_header_t::block_shape(Vector<uint8_t,4> block) const {
  GEODE_ASSERT(all_less_equal(Vector<uint16_t,4>(block),blocks));
  Vector<int,4> bs;
  for (int i=0;i<4;i++)
    bs[i] = block[i]+1<blocks[i]?block_size:shape[i]-block_size*(blocks[i]-1);
  return bs;
}

void read_and_uncompress(int fd, supertensor_blob_t blob, const function<void(Array<uint8_t>)>& cont) {
  // Check consistency
  GEODE_ASSERT(blob.compressed_size<(uint64_t)1<<31);
  GEODE_ASSERT(!blob.uncompressed_size || blob.offset);

  // Read using pread for thread safety
  Array<uint8_t> compressed;
  {
    thread_time_t time(read_kind,unevent);
    compressed.resize(blob.compressed_size,false,false);
    ssize_t r = pread(fd,compressed.data(),blob.compressed_size,blob.offset);
    if (r<0 || r!=(ssize_t)blob.compressed_size)
      THROW(IOError,"read_and_uncompress pread failed: %s",r<0?strerror(errno):"incomplete read");
  }

  // Schedule decompression
  threads_schedule(CPU,compose(cont,curry(decompress,compressed,blob.uncompressed_size,unevent)));
}

void supertensor_writer_t::pwrite(supertensor_blob_t* blob, Array<const uint8_t> data) {
  GEODE_ASSERT(thread_type()==IO);

  // Choose offset
  {
    spin_t spin(offset_lock);
    blob->offset = next_offset;
    next_offset += data.size();
  }

  // Write using pwrite for thread safety
  thread_time_t time(write_kind,unevent);
  ssize_t w = ::pwrite(fd->fd,data.data(),data.size(),blob->offset);
  if (w < 0 || w < (ssize_t)data.size())
    THROW(IOError,"failed to write compressed block to supertensor file: %s",w<0?strerror(errno):"incomplete write");
}

void supertensor_writer_t::compress_and_write(supertensor_blob_t* blob, RawArray<const uint8_t> data) {
  GEODE_ASSERT(thread_type()==CPU);

  // Compress
  thread_time_t time(compress_kind,unevent);
  blob->uncompressed_size = data.size();
  Array<uint8_t> compressed = compress(data,level,unevent);
  blob->compressed_size = compressed.size();

  // Schedule write
  threads_schedule(IO,curry(&Self::pwrite,this,blob,compressed));
}

fildes_t::fildes_t(const string& path, int flags, mode_t mode)
  : fd(open(path.c_str(),flags,mode)) {}

fildes_t::~fildes_t() {
  close();
}

void fildes_t::close() {
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }
}

static const string& check_extension(const string& path) {
  GEODE_ASSERT(path.size()>=8);
  GEODE_ASSERT(path.substr(path.size()-8)==".pentago");
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
    const auto le = to_little_endian(f); \
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
    h.f = to_little_endian(le); \
    next += sizeof(le); });
  HEADER_FIELDS()
  #undef FIELD
  GEODE_ASSERT(next==header_size);
  return h;
}

static void save_index(Vector<int,4> blocks, Array<const supertensor_blob_t,4>* dst, Array<const uint8_t> src) {
  GEODE_ASSERT(src.size()==(int)sizeof(supertensor_blob_t)*blocks.product());
  *dst = Array<const supertensor_blob_t,4>(blocks,(const supertensor_blob_t*)src.data(),src.borrow_owner());
}

supertensor_reader_t::supertensor_reader_t(const string& path, const thread_type_t io)
  : fd(new_<fildes_t>(check_extension(path),O_RDONLY)) {
  initialize(path,0,io);
}

supertensor_reader_t::supertensor_reader_t(const string& path, const fildes_t& fd, const uint64_t header_offset, const thread_type_t io)
  : fd(ref(fd)) {
  initialize(path,header_offset,io);
}

void supertensor_reader_t::initialize(const string& path, const uint64_t header_offset, const thread_type_t io) {
  if (fd->fd < 0)
    THROW(IOError,"can't open supertensor file \"%s\" for reading: %s",path,strerror(errno));

  // Read header
  const int header_size = supertensor_header_t::header_size;
  uint8_t buffer[header_size];
  ssize_t r = pread(fd->fd,buffer,header_size,header_offset);
  if (r < 0)
    THROW(IOError,"invalid supertensor file \"%s\": error reading header, %s",path,strerror(errno));
  if (r < header_size)
    THROW(IOError,"invalid supertensor file \"%s\": unexpected end of file during header",path);
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
  Array<const supertensor_blob_t,4> index;
  threads_schedule(io,curry(read_and_uncompress,fd->fd,h.index,function<void(Array<uint8_t>)>(curry(save_index,Vector<int,4>(h.blocks),&index))));
  threads_wait_all();
  GEODE_ASSERT((index.shape==Vector<int,4>(h.blocks)));
  to_little_endian_inplace(index.flat.const_cast_());
  const_cast_(this->index) = index;
}

supertensor_reader_t::~supertensor_reader_t() {}

static void save(Array<Vector<super_t,2>,4>* dst, Vector<uint8_t,4> block, Array<Vector<super_t,2>,4> src) {
  *dst = src;
}

Array<Vector<super_t,2>,4> supertensor_reader_t::read_block(Vector<uint8_t,4> block) const {
  Array<Vector<super_t,2>,4> data;
  schedule_read_block(block,curry(save,&data));
  threads_wait_all();
  GEODE_ASSERT(data.shape==header.block_shape(block));
  return data;
}

static Array<Vector<super_t,2>,4> unfilter(int filter, Vector<int,4> block_shape, Array<uint8_t> raw_data) {
  GEODE_ASSERT(thread_type()==CPU);
  GEODE_ASSERT(raw_data.size()==(int)sizeof(Vector<super_t,2>)*block_shape.product());
  Array<Vector<super_t,2>,4> data(block_shape,(Vector<super_t,2>*)raw_data.data(),raw_data.borrow_owner());
  to_little_endian_inplace(data.flat);
  switch (filter) {
    case 0: break;
    case 1: uninterleave(data.flat); break;
    default: THROW(ValueError,"supertensor_reader_t::read_block: unknown filter %d",filter);
  }
  return data;
}

void supertensor_reader_t::schedule_read_block(Vector<uint8_t,4> block, const function<void(Vector<uint8_t,4>,Array<Vector<super_t,2>,4>)>& cont) const {
  schedule_read_blocks(RawArray<const Vector<uint8_t,4>>(1,&block),cont);
}

void supertensor_reader_t::schedule_read_blocks(RawArray<const Vector<uint8_t,4>> blocks, const function<void(Vector<uint8_t,4>,Array<Vector<super_t,2>,4>)>& cont) const {
  for (auto block : blocks) {
    const Vector<int,4> block_(block);
    GEODE_ASSERT(index.valid(block_));
    threads_schedule(IO,curry(read_and_uncompress,fd->fd,index[block_],compose(curry(cont,block),curry(unfilter,header.filter,header.block_shape(block)))));
  }
}

uint64_t supertensor_reader_t::total_size() const {
  uint64_t total = supertensor_header_t::header_size + header.index.compressed_size;
  for (const auto& blob : index.flat)
    total += blob.compressed_size;
  return total;
}

// Write header at offset 0, and return the header size
static uint64_t write_header(int fd, const supertensor_header_t& h) {
  const int header_size = supertensor_header_t::header_size;
  uint8_t buffer[header_size];
  h.pack(RawArray<uint8_t>(header_size,buffer));
  ssize_t w = pwrite(fd,buffer,header_size,0);
  if (w < 0 || w < header_size)
    THROW(IOError,"failed to write header to supertensor file: %s",w<0?strerror(errno):"incomplete write");
  return header_size;
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

supertensor_writer_t::supertensor_writer_t(const string& path, section_t section, int block_size, int filter, int level)
  : path(check_extension(path))
  , fd(new_<fildes_t>(path,O_WRONLY|O_CREAT|O_TRUNC,0644))
  , header(section,block_size,filter) // Initialize everything except for valid and index, which finalize will fill in later
  , level(level) {
  if (fd->fd < 0)
    THROW(IOError,"can't open supertensor file \"%s\" for writing: %s",path,strerror(errno));
  if (block_size&1)
    THROW(ValueError,"supertensor block size must be even (not %d) to support block-wise reflection",block_size);

  // Write preliminary header
  next_offset = write_header(fd->fd,header);

  // Initialize block index to all undefined
  const_cast_(index) = Array<supertensor_blob_t,4>(Vector<int,4>(header.blocks));
}

supertensor_writer_t::~supertensor_writer_t() {
  if (!header.valid) {
    // File wasn't finished (due to an error or a failure to call finalize), so delete it
    int r = unlink(path.c_str());
    if (r < 0 && errno != ENOENT)
      cerr << format("failed to remove incomplete supertensor file \"%s\": %s",path,strerror(errno)) << endl;
  }
}

static Array<uint8_t> filter(int filter, Array<Vector<super_t,2>,4> data) {
  GEODE_ASSERT(thread_type()==CPU);
  switch (filter) {
    case 0: break;
    case 1: interleave(data.flat); break;
    default: THROW(ValueError,"supertensor_writer_t::write_block: unknown filter %d",filter);
  }
  to_little_endian_inplace(data.flat);
  return char_view_own(data.flat);
}

void supertensor_writer_t::write_block(Vector<uint8_t,4> block, Array<Vector<super_t,2>,4> data) {
  schedule_write_block(block,data);
  threads_wait_all();
}

void supertensor_writer_t::schedule_write_block(Vector<uint8_t,4> block, Array<Vector<super_t,2>,4> data) {
  GEODE_ASSERT(data.shape==header.block_shape(block));
  const Vector<int,4> block_(block);
  GEODE_ASSERT(index.valid(block_) && !index[block_].offset); // Don't write the same block twice
  threads_schedule(CPU,compose(curry(&Self::compress_and_write,this,&index[block_]),curry(filter,header.filter,data)));
}

void supertensor_writer_t::finalize() {
  if (fd->fd < 0 || header.valid)
    return;

  // Check if all blocks have been written
  threads_wait_all();
  for (auto blob : index.flat)
    if (!blob.offset)
      THROW(RuntimeError,"can't finalize incomplete supertensor file \"%s\"",path);

  // Write index
  supertensor_header_t h = header;
  to_little_endian_inplace(index.flat);
  threads_schedule(CPU,curry(&Self::compress_and_write,this,&h.index,char_view_own(index.flat)));
  threads_wait_all();

  // Finalize header
  h.valid = true;
  write_header(fd->fd,h);
  header = h;
  fd->close();
}

uint64_t supertensor_reader_t::compressed_size(Vector<uint8_t,4> block) const {
  header.block_shape(block); // check validity
  return index[Vector<int,4>(block)].compressed_size;
}

uint64_t supertensor_reader_t::uncompressed_size(Vector<uint8_t,4> block) const {
  header.block_shape(block); // check validity
  return index[Vector<int,4>(block)].uncompressed_size;
}

uint64_t supertensor_writer_t::compressed_size(Vector<uint8_t,4> block) const {
  header.block_shape(block); // check validity
  return index[Vector<int,4>(block)].compressed_size;
}

uint64_t supertensor_writer_t::uncompressed_size(Vector<uint8_t,4> block) const {
  header.block_shape(block); // check validity
  return index[Vector<int,4>(block)].uncompressed_size;
}

vector<Ref<const supertensor_reader_t>> open_supertensors(const string& path, const thread_type_t io) {
  const auto fd = new_<fildes_t>(check_extension(path),O_RDONLY);
  if (fd->fd < 0)
    THROW(IOError,"can't open supertensor file \"%s\" for reading: %s",path,strerror(errno));

  // Read magic string to determine file type (single or multiple supertensors)
  char buffer[20];
  ssize_t r = pread(fd->fd,buffer,20,0);
  if (r < 20)
    THROW(IOError,"invalid supertensor file \"%s\": error reading magic string, %s",path,r<0?strerror(errno):"unexpected eof");

  // Branch on type
  vector<Ref<const supertensor_reader_t>> readers;
  if (!memcmp(buffer,single_supertensor_magic,20))
    readers.push_back(new_<supertensor_reader_t>(path,fd,0,io));
  else if (!memcmp(buffer,multiple_supertensor_magic,20)) {
    uint32_t header[3];
    ssize_t r = pread(fd->fd,header,sizeof(header),20);
    if (r < (int)sizeof(header))
      THROW(IOError,"invalid multiple supertensor file \"%s\": error reading global header, %s",path,r<0?strerror(errno):"unexpected eof");
    for (auto& h: header)
      h = to_little_endian(h);
    if (header[0] != 3)
      THROW(IOError,"multiple supertensor file \"%s\" has unknown version %d",path,header[0]);
    if (header[1] >= 8239)
      THROW(IOError,"multiple supertensor file \"%s\" has weird section count %d",path,header[1]);
    const size_t offset = 20+3*sizeof(uint32_t);
    for (int s=0;s<(int)header[1];s++)
      readers.push_back(new_<supertensor_reader_t>(path,fd,offset+header[2]*s,io));
  } else
    THROW(IOError,"invalid supertensor file \"%s\": bad magic string",path);
  return readers;
}

static vector<Ref<const supertensor_reader_t>> open_supertensors_py(const string& path) {
  return open_supertensors(path);
}

}
using namespace pentago;

void wrap_supertensor() {
  {typedef supertensor_header_py Self;
  Class<Self>("supertensor_header_t")
    .GEODE_FIELD(version)
    .GEODE_FIELD(valid)
    .GEODE_FIELD(stones)
    .GEODE_FIELD(section)
    .GEODE_FIELD(shape)
    .GEODE_FIELD(block_size)
    .GEODE_FIELD(blocks)
    .GEODE_FIELD(filter)
    .GEODE_METHOD(block_shape)
    ;}

  {typedef supertensor_reader_t Self;
  Class<Self>("supertensor_reader_t")
    .GEODE_INIT(const string&)
    .GEODE_FIELD(header)
    .GEODE_METHOD(read_block)
    .GEODE_METHOD(compressed_size)
    .GEODE_METHOD(uncompressed_size)
    .GEODE_METHOD(total_size)
    ;}

  {typedef supertensor_writer_t Self;
  Class<Self>("supertensor_writer_t")
    .GEODE_INIT(const string&,section_t,int,int,int)
    .GEODE_CONST_FIELD(header)
    .GEODE_METHOD(write_block)
    .GEODE_METHOD(finalize)
    .GEODE_METHOD(compressed_size)
    .GEODE_METHOD(uncompressed_size)
    ;}

  GEODE_FUNCTION_2(compress,pentago::compress)
  GEODE_FUNCTION_2(open_supertensors,open_supertensors_py)
}
