// Lists of sections
//
// See base/section.h for background on sections.
#pragma once

#include "pentago/base/section.h"
#include <boost/core/noncopyable.hpp>
#include <unordered_map>
namespace pentago {
struct supertensor_reader_t;
namespace end {

using std::unordered_map;

// An ordered list of sections at the same slice
struct sections_t : private boost::noncopyable {
  const int slice;
  const Array<const section_t> sections;
  const unordered_map<section_t,int> section_id; // The inverse of sections
  const uint64_t total_blocks, total_nodes;

  sections_t(const int slice, Array<const section_t> sections);
  ~sections_t();

  uint64_t memory_usage() const;
};

// Compute all sections that root depends on, organized by slice.
// Only 35 slices are returned, since computing slice 36 is unnecessary.
vector<shared_ptr<const sections_t>> descendent_sections(const section_t root, const int max_slice);

shared_ptr<const sections_t> sections_from_supertensors(
    const vector<shared_ptr<const supertensor_reader_t>> tensors);

}
}
