// Organized index for supertensor files

#include <pentago/high/index.h>
#include <pentago/end/blocks.h>
#include <pentago/data/compress.h>
#include <pentago/data/supertensor.h>
#include <pentago/utility/convert.h>
#include <pentago/utility/index.h>
#include <geode/python/Class.h>
#include <geode/python/stl.h>
#ifdef BOOST_LITTLE_ENDIAN
namespace pentago {

GEODE_DEFINE_TYPE(supertensor_index_t)
const int filter = 1; // interleave filtering
using namespace pentago::end;

Array<const uint64_t> make_offsets(const sections_t& sections) {
  Array<uint64_t> offsets(sections.sections.size()+1,false);
  offsets[0] = 20+sizeof(uint32_t);
  for (const int i : range(sections.sections.size()))
    offsets[i+1] = offsets[i]+sizeof(compact_blob_t)*section_blocks(sections.sections[i]).product();
  return offsets;
}

supertensor_index_t::supertensor_index_t(const sections_t& sections)
  : sections(ref(sections))
  , section_offset(make_offsets(sections)) {
  // Make sure we have a complete slice
  GEODE_ASSERT(descendent_sections(section_t(),sections.slice).at(sections.slice)->sections==sections.sections);
}

supertensor_index_t::~supertensor_index_t() {}

string supertensor_index_t::header() const {
  string header = "pentago index      \n    ";
  GEODE_ASSERT(header.size()==20+4,"");
  const uint32_t slice = sections->slice;
  memcpy(&header[20],&slice,sizeof(slice));
  return header;
}

compact_blob_t supertensor_index_t::blob_location(const block_t block) const {
  const uint64_t base = section_offset[sections->section_id.get(block.x)];
  const int i = index(section_blocks(block.x),Vector<int,4>(block.y));
  const uint64_t offset = base+sizeof(compact_blob_t)*i;
  compact_blob_t b;
  b.offset[0] = uint32_t(offset);
  b.offset[1] = uint32_t(offset>>32);
  b.size = 12;
  return b;
}

compact_blob_t supertensor_index_t::block_location(RawArray<const uint8_t> blob) {
  compact_blob_t b;
  GEODE_ASSERT(blob.size()==sizeof(b),format("expected size %d, got size %d, data %s",sizeof(b),blob.size(),str(blob)));
  memcpy(&b,blob.data(),sizeof(b));
  return b;
}

Array<Vector<super_t,2>,4> supertensor_index_t::unpack_block(const block_t block, RawArray<const uint8_t> compressed) {
  const auto shape = block_shape(block.x.shape(),block.y);
  const auto data = decompress(compressed,sizeof(Vector<super_t,2>)*shape.product(),unevent);
  return unfilter(filter,shape,data);
}

namespace {
struct compact_blob_py : public Object {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  const uint64_t offset;
  const uint32_t size;
protected:
  compact_blob_py(const compact_blob_t b)
    : offset(b.offset[0]|uint64_t(b.offset[1])<<32)
    , size(b.size) {}
};
GEODE_DEFINE_TYPE(compact_blob_py)
}

static inline PyObject* to_python(const compact_blob_t b) {
  return to_python(new_<compact_blob_py>(b));
}

void write_supertensor_index(const string& name, const vector<Ref<const supertensor_reader_t>>& readers) {
  // Check consistency
  GEODE_ASSERT(readers.size());
  const uint32_t slice = readers[0]->header.stones;
  const auto sections = descendent_sections(section_t(),slice).at(slice);
  GEODE_ASSERT(sections->slice==int(slice));
  Hashtable<section_t,Ref<const supertensor_reader_t>> section_reader;
  for (const auto reader : readers) {
    GEODE_ASSERT(int(reader->header.filter)==filter);
    section_reader.set(reader->header.section,reader);
  }
  for (const auto section : sections->sections)
    GEODE_ASSERT(section_reader.contains(section));

  // Write index
  FILE* file = fopen(name.c_str(),"wb");
  if (!file)
    throw IOError(format("write_supertensor_index: can't open '%s' for writing",name));
  fwrite("pentago index      \n",1,20,file);
  fwrite(&slice,sizeof(uint32_t),1,file);
  GEODE_ASSERT(ftell(file)==24);
  for (const auto section : sections->sections) {
    const auto reader = section_reader.get(section);
    Array<compact_blob_t> blobs(reader->offset.flat.size(),false);
    for (const int i : range(blobs.size())) {
      const uint64_t offset = reader->offset.flat[i];
      blobs[i].offset[0] = uint32_t(offset);
      blobs[i].offset[1] = uint32_t(offset>>32);
      blobs[i].size = reader->compressed_size_.flat[i];
    }
    fwrite(blobs.data(),sizeof(compact_blob_t),blobs.size(),file);
  }
  const auto index = new_<supertensor_index_t>(sections);
  GEODE_ASSERT(uint64_t(ftell(file))==index->section_offset.back());
  fclose(file);
}

}
#endif
using namespace pentago;

void wrap_index() {
#ifdef BOOST_LITTLE_ENDIAN
  {
    typedef compact_blob_py Self;
    Class<Self>("compact_blob_t")
      .GEODE_FIELD(offset)
      .GEODE_FIELD(size)
      ;
  } {
    typedef supertensor_index_t Self;
    Class<Self>("supertensor_index_t")
      .GEODE_INIT(const sections_t&)
      .GEODE_METHOD(header)
      .GEODE_METHOD(blob_location)
      .GEODE_METHOD(block_location)
      .GEODE_METHOD(unpack_block)
      ;
  }

  GEODE_FUNCTION(write_supertensor_index)
#endif
}

