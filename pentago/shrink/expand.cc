#include "pentago/base/board.h"
#include "pentago/base/score.h"
#include "pentago/base/symmetry.h"
#include "pentago/data/numpy.h"
#include "pentago/utility/array.h"
#include "pentago/utility/range.h"
#include <iostream>
#include <string>
#include <fstream>

namespace pentago {

using std::cout;
using std::endl;
using std::string;

int main(int argc, char** argv) {
  try {
    // Default input/output paths
    string input_path = "/Users/irving/pentago/data/edison/project/all/sparse-12.npy";
    string output_path = "/Users/irving/pentago/data/edison/project/all/expanded-12.npy";

    // Parse command line arguments if provided
    if (argc > 1)
      input_path = argv[1];
    if (argc > 2)
      output_path = argv[2];

    // Load the numpy data
    cout << "Reading data from " << input_path << "..." << endl;
    auto raw_data = read_numpy<uint64_t, 9>(input_path);
    cout << "Loaded " << raw_data.size() << " samples" << endl;

    // Convert to sample_t for easier processing
    struct sample_t {
      board_t board;
      Vector<super_t, 2> wins;  // black wins, white wins
    };

    Array<sample_t> samples(raw_data.size());
    for (const int i : range(raw_data.size())) {
      samples[i].board = board_t(raw_data[i][0]);
      for (const int j : range(2)) {
        const int offset = 1 + 4*j;
        samples[i].wins[j] = super_t(raw_data[i][offset], raw_data[i][offset+1],
                                     raw_data[i][offset+2], raw_data[i][offset+3]);
      }
    }

    // Expand data with all rotations
    cout << "Processing all rotations..." << endl;
    Array<Vector<int64_t, 3>> expanded(256 * raw_data.size());
    int idx = 0;
    for (const auto& sample : samples) {
      // For each rotation
      for (const int r : range(256)) {
        // Apply rotation to the board
        const symmetry_t sym(0, r);  // 0 for global rotation, r for local rotations
        const board_t rotated = transform_board(sym, sample.board);

        // Store
        auto& exp = expanded[idx++];
        for (const int s : range(2))
          exp[s] = unpack(rotated, s);
        exp[2] = sample.wins[0](r) - sample.wins[1](r);
      }
    }
    GEODE_ASSERT(idx == expanded.size());

    // Write expanded data to a new NPY file
    cout << "Writing expanded data to " << output_path << endl;
    write_numpy(output_path, expanded);
    return 0;
  } catch (const std::exception& e) {
    cout << "Error: " << e.what() << endl;
    return 1;
  }
}

} // namespace pentago

int main(int argc, char** argv) {
  return pentago::main(argc, argv);
}