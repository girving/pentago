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

#include "pentago/end/sections.h"
#include "pentago/data/supertensor.h"
#include <boost/core/noncopyable.hpp>
#include <boost/endian/conversion.hpp>
#ifdef BOOST_LITTLE_ENDIAN
namespace pentago {

struct compact_blob_t {
  Vector<uint32_t,2> packed_offset; // Split into two to guarantee struct packing
  uint32_t size; // compressed for blocks, uncompressed for blob information

  uint64_t offset() const {
    return packed_offset[0]|uint64_t(packed_offset[1])<<32;
  }

  void set_offset(const uint64_t offset) {
    packed_offset = vec(uint32_t(offset),uint32_t(offset>>32));
  }
};
static_assert(sizeof(compact_blob_t)==12,"struct packing failed");

struct supertensor_index_t : private boost::noncopyable {
  typedef tuple<section_t,Vector<uint8_t,4>> block_t;

  const shared_ptr<const end::sections_t> sections;
  const Array<const uint64_t> section_offset;

public:
  supertensor_index_t(const shared_ptr<const end::sections_t>& sections);
  ~supertensor_index_t();

  string header() const;
  compact_blob_t blob_location(const block_t block) const;

  // Convert blob data to block location
  static compact_blob_t block_location(RawArray<const uint8_t> blob);

  // Decompress a compressed block
  static Array<Vector<super_t,2>,4> unpack_block(const block_t block, RawArray<const uint8_t> compressed);
};

void write_supertensor_index(const string& name,
                             const vector<shared_ptr<const supertensor_reader_t>>& readers);

}  // namespace pentago
#endif  // BOOST_LITTLE ENDIAN
