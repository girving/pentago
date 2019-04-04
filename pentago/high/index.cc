// Organized index for supertensor files

#include "pentago/high/index.h"
#include "pentago/end/blocks.h"
#include "pentago/data/compress.h"
#include "pentago/data/supertensor.h"
#include "pentago/utility/index.h"
#include "pentago/utility/endian.h"
#ifdef BOOST_LITTLE_ENDIAN
namespace pentago {

using namespace pentago::end;
using std::get;

const int filter = 1; // interleave filtering

Array<const uint64_t> make_offsets(const sections_t& sections) {
  Array<uint64_t> offsets(sections.sections.size()+1,uninit);
  offsets[0] = 20+sizeof(uint32_t);
  for (const int i : range(sections.sections.size()))
    offsets[i+1] = offsets[i]+sizeof(compact_blob_t)*section_blocks(sections.sections[i]).product();
  return offsets;
}

supertensor_index_t::supertensor_index_t(const shared_ptr<const sections_t>& sections)
  : sections(sections)
  , section_offset(make_offsets(*sections)) {
  // Make sure we have a complete slice
  GEODE_ASSERT(descendent_sections(section_t(), sections->slice).at(sections->slice)->sections ==
               sections->sections);
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
  const auto section = get<0>(block);
  const auto shape = section_blocks(get<0>(block));
  const auto I = Vector<int,4>(get<1>(block));
  GEODE_ASSERT(valid(shape, I), format("section %s, shape %s, invalid block %s", section, shape, I));
  const uint64_t base = section_offset[check_get(sections->section_id, section)];
  const uint64_t offset = base+sizeof(compact_blob_t)*index(shape, I);
  compact_blob_t b;
  b.set_offset(offset);
  b.size = 12;
  return b;
}

compact_blob_t supertensor_index_t::block_location(RawArray<const uint8_t> blob) {
  compact_blob_t b;
  GEODE_ASSERT(blob.size() == sizeof(b), format("expected size %d, got size %d, data %s",
                                                sizeof(b), blob.size(), blob));
  memcpy(&b,blob.data(),sizeof(b));
  return b;
}

Array<Vector<super_t,2>,4> supertensor_index_t::unpack_block(const block_t block,
                                                             RawArray<const uint8_t> compressed) {
  const auto shape = block_shape(get<0>(block).shape(), get<1>(block));
  const auto data = decompress(compressed, sizeof(Vector<super_t,2>)*shape.product(), unevent);
  return unfilter(filter, shape, data);
}

void write_supertensor_index(const string& name,
                             const vector<shared_ptr<const supertensor_reader_t>>& readers) {
  // Check consistency
  GEODE_ASSERT(readers.size());
  const uint32_t slice = readers[0]->header.stones;
  const auto sections = descendent_sections(section_t(),slice).at(slice);
  GEODE_ASSERT(sections->slice==int(slice));
  unordered_map<section_t,shared_ptr<const supertensor_reader_t>> section_reader;
  for (const auto reader : readers) {
    GEODE_ASSERT(int(reader->header.filter)==filter);
    section_reader.insert(make_pair(reader->header.section,reader));
  }
  for (const auto section : sections->sections)
    GEODE_ASSERT(section_reader.find(section) != section_reader.end());

  // Write index
  FILE* file = fopen(name.c_str(),"wb");
  if (!file)
    throw IOError(format("write_supertensor_index: can't open '%s' for writing",name));
  fwrite("pentago index      \n",1,20,file);
  fwrite(&slice,sizeof(uint32_t),1,file);
  GEODE_ASSERT(ftell(file)==24);
  for (const auto section : sections->sections) {
    const auto it = section_reader.find(section);
    GEODE_ASSERT(it != section_reader.end());
    const auto reader = it->second;
    Array<compact_blob_t> blobs(reader->offset.flat().size(),uninit);
    for (const int i : range(blobs.size())) {
      const uint64_t offset = reader->offset.flat()[i];
      blobs[i].set_offset(offset);
      blobs[i].size = reader->compressed_size_.flat()[i];
    }
    fwrite(blobs.data(),sizeof(compact_blob_t),blobs.size(),file);
  }
  const supertensor_index_t index(sections);
  GEODE_ASSERT(uint64_t(ftell(file))==index.section_offset.back());
  fclose(file);
}

}
#endif  // BOOST_LITTLE_ENDIAN
