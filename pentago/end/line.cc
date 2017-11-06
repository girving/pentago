// Information about a single 1D block line

#include "pentago/end/line.h"
namespace pentago {
namespace end {

ostream& operator<<(ostream& output, const line_t& line) {
  return output<<line.section<<'-'<<int(line.dimension)<<'-'<<Vector<int,3>(line.block_base);
}

}
}
