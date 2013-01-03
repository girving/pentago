// Partitioning of sections and lines for MPI purposes

#include <pentago/mpi/sections.h>
#include <pentago/utility/memory.h>
#include <other/core/array/sort.h>
#include <other/core/python/Class.h>
#include <other/core/python/stl.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/Log.h>
namespace pentago {
namespace mpi {

OTHER_DEFINE_TYPE(sections_t)

vector<Ref<const sections_t>> descendent_sections(const section_t root, const int max_slice) {
  Log::Scope scope("dependents");
  OTHER_ASSERT(0<=max_slice && max_slice<=35);
  OTHER_ASSERT(root.sum()<=max_slice);

  // Recursively compute all sections that root depends on
  vector<Array<section_t>> slices(max_slice+1);
  Hashtable<section_t> seen;
  Array<section_t> stack;
  stack.append(root);
  while (stack.size()) {
    section_t section = stack.pop().standardize<8>().x;
    if (seen.set(section)) {
      int n = section.sum();
      slices.at(n).append(section);
      if (n < max_slice)
        for (int i=0;i<4;i++)
          if (section.counts[i].sum()<9)
            stack.append(section.child(i));
    }
  }

  // Sort each slice
  for (auto& slice : slices)
    sort(slice);

  // Construct sections_t objects
  vector<Ref<const sections_t>> result;
  for (const int slice : range(max_slice+1))
    result.push_back(new_<sections_t>(slice,slices[slice]));
  return result;
}

sections_t::sections_t(const int slice, Array<const section_t> sections)
  : slice(slice)
  , sections(sections) {

  // Verify that caller didn't lie about slice
  for (const auto& section : sections)
    OTHER_ASSERT(section.sum()==slice);

  // Invert sections
  for (int s=0;s<sections.size();s++) 
    const_cast_(section_id).set(sections[s],s);
}

sections_t::~sections_t() {}

uint64_t sections_t::memory_usage() const {
  return sizeof(sections_t)
       + pentago::memory_usage(sections)
       + pentago::memory_usage(section_id);
}

}
}
using namespace pentago;
using namespace pentago::mpi;

void wrap_sections() {
  OTHER_FUNCTION(descendent_sections)

  typedef sections_t Self;
  Class<Self>("sections_t")
    .OTHER_INIT(int,Array<const section_t>)
    .OTHER_FIELD(slice)
    .OTHER_FIELD(sections)
    .OTHER_METHOD(memory_usage)
    ;
}
