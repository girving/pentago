// Lists of sections
//
// See base/section.h for background on sections.
#pragma once

#include <pentago/base/section.h>
#include <geode/structure/Hashtable.h>
namespace pentago {
struct supertensor_reader_t;
namespace end {

// An ordered list of sections at the same slice
struct sections_t : public Object {
  GEODE_DECLARE_TYPE(GEODE_EXPORT)

  const int slice;
  const Array<const section_t> sections;
  const Hashtable<section_t,int> section_id; // The inverse of sections
  const uint64_t total_blocks, total_nodes;

protected:
  GEODE_EXPORT sections_t(const int slice, Array<const section_t> sections);
public:
  ~sections_t();

  uint64_t memory_usage() const;
};

// Compute all sections that root depends, organized by slice.
// Only 35 slices are returned, since computing slice 36 is unnecessary.
GEODE_EXPORT vector<Ref<const sections_t>> descendent_sections(const section_t root, const int max_slice);

GEODE_EXPORT Ref<const sections_t> sections_from_supertensors(const vector<Ref<const supertensor_reader_t>> tensors);

}
}
