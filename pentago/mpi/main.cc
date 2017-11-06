// Actual main routine

#include "pentago/mpi/toplevel.h"
#include <stdexcept>
#include <iostream>

int main(int argc, char** argv) {
  try {
    return pentago::mpi::toplevel(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "uncaught exception: " << e.what() << std::endl;
    return 1;
  }
}
