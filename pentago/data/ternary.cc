// Flat array of ternary values, packed 5 per byte

#include "pentago/data/ternary.h"
#include "pentago/utility/debug.h"
#include <cassert>
namespace pentago {

// pow3[i] = 3^i
static const int pow3[5] = {1, 3, 9, 27, 81};

ternaries_t::ternaries_t()
    : size(0) {}

ternaries_t::ternaries_t(const uint64_t size)
    : size(size)
    , data(CHECK_CAST_INT(ceil_div(size, uint64_t(5)))) {}

ternaries_t::~ternaries_t() {}

int ternaries_t::operator[](const uint64_t i) const {
  assert(i < size);
  return data[int(i / 5)] / pow3[i % 5] % 3;
}

void ternaries_t::set(const uint64_t i, const int v) {
  assert(i < size && unsigned(v) < 3);
  const int byte = int(i / 5);
  const int pos = int(i % 5);
  const int old = data[byte] / pow3[pos] % 3;
  data[byte] += uint8_t((v - old) * pow3[pos]);
}

static void unpack5(const uint8_t byte, int out[5]) {
  int v = byte;
  for (int i = 0; i < 5; i++) {
    out[i] = v % 3;
    v /= 3;
  }
}

static uint8_t pack5(const int in[5]) {
  return uint8_t(in[0] + 3*(in[1] + 3*(in[2] + 3*(in[3] + 3*in[4]))));
}

ternary_reader_t::ternary_reader_t(const ternaries_t& t)
    : ptr(t.data.data())
    , end(ptr + t.data.size())
    , pos(5) {}

int ternary_reader_t::next() {
  if (pos >= 5) {
    unpack5(ptr < end ? *ptr++ : 0, buf);
    pos = 0;
  }
  return buf[pos++];
}

ternary_writer_t::ternary_writer_t(ternaries_t& t)
    : ptr(t.data.data())
    , pos(0) {
  for (int i = 0; i < 5; i++) buf[i] = 0;
}

void ternary_writer_t::put(const int v) {
  assert(unsigned(v) < 3);
  buf[pos++] = v;
  if (pos >= 5) {
    *ptr++ = pack5(buf);
    pos = 0;
  }
}

void ternary_writer_t::flush() {
  if (pos > 0) {
    for (int i = pos; i < 5; i++) buf[i] = 0;
    *ptr++ = pack5(buf);
    pos = 0;
  }
}

}  // namespace pentago
