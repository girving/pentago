// MPI related utilities
#pragma once

#include <pentago/mpi/config.h>
#include <pentago/thread.h>
#include <pentago/section.h>
#include <other/core/utility/format.h>
#include <other/core/vector/Vector.h>
namespace pentago {
namespace mpi {

using std::string;

// Should we print?  Defaults to true.
bool verbose();
void set_verbose(bool verbose);

// Print a message and abort
void OTHER_NORETURN(die(const string& msg));

// Convenience version of die
template<class... Args> static inline void OTHER_NORETURN(die(const char* msg, const Args&... args));
template<class... Args> static inline void die(const char* msg, const Args&... args) {
  die(format(msg,args...));
}

void OTHER_NORETURN(check_failed(const char* file, const char* function, int line, const char* call, int result));

// Die if an MPI routine doesn't succeed
#define CHECK(call) ({ \
  int r_ = call; \
  if (r_ != MPI_SUCCESS) \
    check_failed(__FILE__,__FUNCTION__,__LINE__,#call,r_); \
  })

// Convert an error code into a string
string error_string(int code);

Vector<int,4> section_blocks(section_t section);

template<int d> static inline Vector<int,d> block_shape(Vector<int,d> shape, Vector<uint8_t,d> block) {
  const Vector<int,d> block_(block);
  return Vector<int,d>::componentwise_min(shape,block_size*(block_+1))-block_size*block_;
}

// Convenience functions
int comm_size(MPI_Comm comm);
int comm_rank(MPI_Comm comm);
MPI_Comm comm_dup(MPI_Comm comm);
int get_count(MPI_Status* status, MPI_Datatype datatype);
void send_empty(int rank, int tag, MPI_Comm comm); // Send an empty message

// Init and finalize
struct mpi_world_t : public boost::noncopyable {
  mpi_world_t(int& argc, char**& argv);
  ~mpi_world_t();
};

// Events
static inline event_t block_event(section_t section, Vector<uint8_t,4> block) {
  return block_ekind
       | event_t(section.microsig())<<28
       | event_t(block[3])<<18
       | event_t(block[2])<<12
       | event_t(block[1])<<6
       | event_t(block[0])<<0;
}

static inline event_t line_event(section_t section, uint8_t dimension, Vector<uint8_t,3> block_base) {
  return line_ekind
       | event_t(section.microsig())<<28
       | event_t(dimension)<<24
       | event_t(block_base[2])<<12
       | event_t(block_base[1])<<6
       | event_t(block_base[0])<<0;
}

static inline event_t block_line_event(section_t section, uint8_t dimension, Vector<uint8_t,4> block) {
  assert(dimension<4);
  return block_line_ekind
       | event_t(section.microsig())<<28
       | event_t(dimension)<<24
       | event_t(block[3])<<18
       | event_t(block[2])<<12
       | event_t(block[1])<<6
       | event_t(block[0])<<0;
}

static inline event_t block_lines_event(section_t section, uint8_t dimensions, Vector<uint8_t,4> block) {
  assert(dimensions<16);
  return block_lines_ekind
       | event_t(section.microsig())<<28
       | event_t(dimensions)<<24 // 4*child_dimension+parent_dimension
       | event_t(block[3])<<18
       | event_t(block[2])<<12
       | event_t(block[1])<<6
       | event_t(block[0])<<0;
}

}
}
