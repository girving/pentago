// Partitioning of sections and lines for MPI purposes

#include <pentago/all_boards.h>
#include <other/core/geometry/Box.h>
#include <other/core/structure/Hashtable.h>
#include <vector>
namespace pentago {
namespace mpi {

using std::vector;

// Compute all sections that root depends, organized by slice.
// Only 35 slices are returned, since computing slice 36 is unnecessary.
vector<Array<const section_t>> descendent_sections(section_t root);

// Information about a set of 1D block lines
struct lines_t {
  // Section information
  section_t section;
  Vector<int,4> shape;

  // Line information
  int dimension; // Line dimension
  Box<Vector<int,3>> blocks; // Ranges of blocks along the other three dimensions
  int count; // Number of lines in this range = blocks.volume()
  uint64_t line_size; // All lines in this line range have the same size
};

// Given a set of sections, distribute all lines amongst a number of processors
struct partition_t : public Object {
  OTHER_DECLARE_TYPE

  const int ranks;
  const int block_size;
  const Array<const section_t> sections;
  const Array<const lines_t> owner_lines, other_lines;
  const Hashtable<section_t,int> first_owner_line;
  const Array<const Vector<int,2>> owner_starts, other_starts;
  const Array<const uint64_t> owner_work, other_work; // empty unless save_work is true
  const double owner_excess, total_excess; // initialized only if verbose

protected:
  partition_t(const int ranks, const int block_size, Array<const section_t> sections, bool save_work=false);
public:
  ~partition_t();

  // Find the rank which owns a given block
  int block_to_rank(section_t section, Vector<int,4> block);

private:
  // Can the remaining work fit within the given bound?
  template<bool record> static bool fit(RawArray<uint64_t> work, RawArray<const lines_t> lines, const uint64_t bound, RawArray<Vector<int,2>> starts=(RawArray<Vector<int,2>>()));

  // Divide a set of lines between processes
  static Array<const Vector<int,2>> partition_lines(RawArray<uint64_t> work, RawArray<const lines_t> lines);

  // Find the line that owns a given block
  Vector<int,2> block_to_line(section_t section, Vector<int,4> block);
};

}
}
