// Partitioning of sections and lines for MPI purposes

#include <pentago/end/sections.h>
#include <pentago/end/blocks.h>
#include <pentago/data/supertensor.h>
#include <pentago/utility/memory.h>
#include <geode/array/sort.h>
#include <geode/python/Class.h>
#include <geode/python/stl.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/Log.h>
namespace pentago {
namespace end {

GEODE_DEFINE_TYPE(sections_t)

vector<Ref<const sections_t>> descendent_sections(const section_t root, const int max_slice) {
  GEODE_ASSERT(0<=max_slice && max_slice<=35);
  GEODE_ASSERT(root.sum()<=max_slice);

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
  , sections(sections)
  , total_blocks(0)
  , total_nodes(0) {

  for (const int s : range(sections.size())) {
    GEODE_ASSERT(sections[s].sum()==slice); // Verify that caller didn't lie about slice
    const_cast_(section_id).set(sections[s],s); // Invert sections
    const_cast_(total_blocks) += section_blocks(sections[s]).product();
    const_cast_(total_nodes) += Vector<uint64_t,4>(sections[s].shape()).product();
  }
}

sections_t::~sections_t() {}

uint64_t sections_t::memory_usage() const {
  return sizeof(sections_t)
       + pentago::memory_usage(sections)
       + pentago::memory_usage(section_id);
}

Ref<const sections_t> sections_from_supertensors(const vector<Ref<const supertensor_reader_t>> tensors) {
  GEODE_ASSERT(tensors.size());
  const int slice = tensors[0]->header.stones;
  Array<section_t> sections;
  for (const auto& tensor : tensors)
    sections.append(tensor->header.section);
  return new_<sections_t>(slice,sections);
}

}
}
using namespace pentago::end;

void wrap_sections() {
  GEODE_FUNCTION(descendent_sections)
  GEODE_FUNCTION(sections_from_supertensors)

  typedef sections_t Self;
  Class<Self>("sections_t")
    .GEODE_INIT(int,Array<const section_t>)
    .GEODE_FIELD(slice)
    .GEODE_FIELD(sections)
    .GEODE_FIELD(total_blocks)
    .GEODE_FIELD(total_nodes)
    .GEODE_METHOD(memory_usage)
    ;
}
