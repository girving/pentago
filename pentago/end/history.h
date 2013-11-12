// History-related utilities
#pragma once

#include <pentago/base/section.h>
#include <pentago/utility/thread.h>
namespace pentago {
namespace end {

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

}
}
