// Lists of sections
#pragma once

#include <pentago/section.h>
#include <other/core/structure/Hashtable.h>
namespace pentago {
namespace mpi {

// An ordered list of sections at the same slice
struct sections_t : public Object {
  OTHER_DECLARE_TYPE(OTHER_NO_EXPORT)

  const int slice;
  const Array<const section_t> sections;
  const Hashtable<section_t,int> section_id; // The inverse of sections

protected:
  sections_t(const int slice, Array<const section_t> sections);
public:
  ~sections_t();

  uint64_t memory_usage() const;
};

// Compute all sections that root depends, organized by slice.
// Only 35 slices are returned, since computing slice 36 is unnecessary.
vector<Ref<const sections_t>> descendent_sections(const section_t root, const int max_slice);

}
}
