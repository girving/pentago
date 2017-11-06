// History-related utilities
#pragma once

#include "pentago/base/section.h"
#include "pentago/utility/thread.h"
#include <unordered_map>
namespace pentago {
namespace end {

using std::unordered_map;

static inline event_t block_event(section_t section, Vector<uint8_t,4> block) {
  return block_ekind
       | event_t(section.microsig())<<29
       | event_t(block[3])<<18
       | event_t(block[2])<<12
       | event_t(block[1])<<6
       | event_t(block[0])<<0;
}

static inline event_t line_event(section_t section, uint8_t dimension, Vector<uint8_t,3> block_base) {
  return line_ekind
       | event_t(section.microsig())<<29
       | event_t(dimension)<<24
       | event_t(block_base[2])<<12
       | event_t(block_base[1])<<6
       | event_t(block_base[0])<<0;
}

static inline event_t block_line_event(section_t section, uint8_t dimension, Vector<uint8_t,4> block) {
  assert(dimension<4);
  return block_line_ekind
       | event_t(section.microsig())<<29
       | event_t(dimension)<<24
       | event_t(block[3])<<18
       | event_t(block[2])<<12
       | event_t(block[1])<<6
       | event_t(block[0])<<0;
}

struct dimensions_t {
  uint8_t data; // 4*parent_to_child_symmetry + child_dimension

  dimensions_t(uint8_t parent_to_child_symmetry, uint8_t child_dimension)
    : data(4*parent_to_child_symmetry + child_dimension) {
    assert(parent_to_child_symmetry<8 && child_dimension<4);
  }

  static dimensions_t raw(uint8_t data) {
    dimensions_t d(0,0);
    d.data = data;
    return d;
  }
};

static inline event_t block_lines_event(section_t section, dimensions_t dimensions, Vector<uint8_t,4> block) {
  return block_lines_ekind
       | event_t(section.microsig())<<29
       | event_t(dimensions.data)<<24 // 4*parent_to_child_symmetry + child_dimension
       | event_t(block[3])<<18
       | event_t(block[2])<<12
       | event_t(block[1])<<6
       | event_t(block[0])<<0;
}

string str_event(const event_t event);

Vector<int,2> search_thread(const vector<Array<const history_t>>& thread, double time);

vector<tuple<int,int,history_t>> event_dependencies(
    const vector<vector<Array<const history_t>>>& event_sorted_history, const int direction,
    const int thread, const int kind, const history_t source);

void check_dependencies(const vector<vector<Array<const history_t>>>& event_sorted_history,
                        const int direction);

Array<double,3> estimate_bandwidth(const vector<vector<Array<const history_t>>>& event_sorted_history,
                                   const int threads, const double dt_seconds);

unordered_map<string,Array<const Vector<float,2>>>
message_statistics(const vector<vector<Array<const history_t>>>& event_sorted_history,
                   const int ranks_per_node, const int threads_per_rank,
                   const time_kind_t source_kind, const int steps,
                   RawArray<const double> slice_compression_ratio);

}
}
