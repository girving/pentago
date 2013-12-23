// Uncompressed index for supertensor files
#pragma once

/* A supertensor index file (.pentago.index) contains uncompressed index information
 * for a supertensor file suitable for uncompressed random access.  The format is
 *
 *   char magic[20] = "pentago index      \n";
 *   uint32_t slice; // .pentago.index files always contain a complete slice
 *   compact_blob_t blobs[section_id][block]; // compact blobs in standard order
 *
 * where compact_blob_t packs offset and compressed_size into 12 bytes as defined below,
 * and the section order is as computed by descendent_sections in pentago/end/sections.h.
 * All data is little endian.
 */

#include <pentago/end/sections.h>
#include <pentago/data/supertensor.h>
#include <boost/detail/endian.hpp>
#ifdef BOOST_LITTLE_ENDIAN
namespace pentago {
namespace end {

struct __attribute__ ((packed)) compact_blob_t {
  uint64_t offset;
  uint32_t compressed_size;
};
static_assert(sizeof(compact_blob_t)==12,"struct packing failed");

struct supertensor_index_t : public Object {
  GEODE_DECLARE_TYPE(GEODE_EXPORT)

  const Ref<const sections_t> sections;
  const Array<const uint64_t> section_offset;

protected:
  supertensor_index_t(const sections_t& sections);
public:
  ~supertensor_index_t();

  string header() const;
  uint64_t blob_offset(const section_t section, const Vector<uint8_t,4> block) const;
  string blob_range_header(const section_t section, const Vector<uint8_t,4> block) const;

  // Convert blob data to block range header
  static string block_range_header(const string& blob);

  // Decompress a compressed block
  static Array<Vector<super_t,2>,4> unpack_block(const section_t section, const Vector<uint8_t,4> block,
                                                 RawArray<const uint8_t> compressed);
};

void write_supertensor_index(const string& name, const vector<Ref<const supertensor_reader_t>>& readers);

}
}
#endif
