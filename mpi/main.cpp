// Actual main routine

#include <pentago/mpi/toplevel.h>
#include <stdexcept>
#include <iostream>

using std::exception;
using std::cerr;
using std::endl;

int main(int argc, char** argv) {
  try {
    return pentago::mpi::toplevel(argc,argv);
  } catch (const exception& e) {
    cerr << "uncaught exception: " << e.what() << endl;
    return 1;
  }
}
