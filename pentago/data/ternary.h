// Flat array of ternary values, packed 5 per byte (3^5 = 243 < 256)
#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/ceil_div.h"
namespace pentago {

struct ternaries_t {
  const uint64_t size;  // number of ternary values
  const Array<uint8_t> data;  // ceil(size/5) packed bytes

  ternaries_t();
  explicit ternaries_t(const uint64_t size);
  ~ternaries_t();

  int operator[](const uint64_t i) const;
  void set(const uint64_t i, const int v);
};

// Sequential reader: unpacks 5 values at a time, avoiding per-element division
struct ternary_reader_t {
  const uint8_t* ptr;
  const uint8_t* end;
  int buf[5];
  int pos;

  explicit ternary_reader_t(const ternaries_t& t);
  int next();
};

// Sequential writer: packs 5 values at a time
struct ternary_writer_t {
  uint8_t* ptr;
  int buf[5];
  int pos;

  explicit ternary_writer_t(ternaries_t& t);
  void put(const int v);
  void flush();
};

}  // namespace pentago
