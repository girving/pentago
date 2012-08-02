// Information about a single 1D block line

#include <pentago/mpi/line.h>
namespace pentago {
namespace mpi {

ostream& operator<<(ostream& output, const line_t& line) {
  return output<<line.section<<'-'<<line.dimension<<'-'<<line.block_base;
}

}
}
