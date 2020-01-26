// Partitioning of sections and lines for MPI purposes

#include "pentago/end/sections.h"
#include "pentago/end/blocks.h"
#include "pentago/data/supertensor.h"
#include "pentago/utility/memory_usage.h"
#include "pentago/utility/sort.h"
#include "pentago/utility/const_cast.h"
#include "pentago/utility/log.h"
#include <unordered_set>
namespace pentago {
namespace end {

using std::get;
using std::make_shared;
using std::unordered_set;

vector<shared_ptr<const sections_t>> descendent_sections(const section_t root, const int max_slice) {
  GEODE_ASSERT(0<=max_slice && max_slice<=35);
  GEODE_ASSERT(root.sum()<=max_slice);

  // Recursively compute all sections that root depends on
  vector<vector<section_t>> slices(max_slice+1);
  unordered_set<section_t> seen;
  vector<section_t> stack;
  stack.push_back(root);
  while (stack.size()) {
    const section_t section = get<0>(stack.back().standardize<8>());
    stack.pop_back();
    if (seen.insert(section).second) {
      int n = section.sum();
      slices.at(n).push_back(section);
      if (n < max_slice)
        for (int i=0;i<4;i++)
          if (section.counts[i].sum()<9)
            stack.push_back(section.child(i));
    }
  }

  // Sort each slice
  for (auto& slice : slices)
    std::sort(slice.begin(), slice.end());

  // Construct sections_t objects
  vector<shared_ptr<const sections_t>> result;
  for (const int slice : range(max_slice+1))
    result.push_back(make_shared<sections_t>(slice, asarray(slices[slice]).copy()));
  return result;
}

sections_t::sections_t(const int slice, Array<const section_t> sections)
  : slice(slice)
  , sections(sections)
  , total_blocks(0)
  , total_nodes(0) {

  for (const int s : range(sections.size())) {
    GEODE_ASSERT(sections[s].sum()==slice); // Verify that caller didn't lie about slice
    const_cast_(section_id)[sections[s]] = s; // Invert sections
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

shared_ptr<const sections_t>
sections_from_supertensors(const vector<shared_ptr<const supertensor_reader_t>> tensors) {
  GEODE_ASSERT(tensors.size());
  const int slice = tensors[0]->header.stones;
  vector<section_t> sections;
  for (const auto& tensor : tensors)
    sections.push_back(tensor->header.section);
  return make_shared<sections_t>(slice, asarray(sections).copy());
}

}
}
