#include "pentago/utility/hash.h"
#include <boost/uuid/sha1.hpp>
namespace pentago {

string sha1(RawArray<const uint8_t> data) {
  boost::uuids::detail::sha1 sha;
  sha.process_bytes(data.data(), data.size());
  uint32_t hash[5] = {0};
  sha.get_digest(hash);
  string s;
  for (const auto n : hash) s += format("%08x", n);
  return s;
}

}  // namespace pentago
