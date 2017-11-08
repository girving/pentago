// Tracing code for use in debugging inconsistency detections
//
// The superengine contains a code to detect certain kinds of inconsistencies
// in the results.  If such an inconsistency is found, this module can help
// to track it down to the source bug.  Since it is disabled at compile time,
// it may or may not work at any given time.
#pragma once

#include "pentago/base/board.h"
#include "pentago/base/superscore.h"
namespace pentago {

// Uncomment to enable tracing
//#define TRACING_ON

// Run a line of code only in tracing mode
#ifdef TRACING_ON
#define TRACE(...) __VA_ARGS__
#else
#define TRACE(...) ((void)0)
#endif

// Mark that the transposition table is clear
extern void trace_restart();

// Record that the value of board is computed inconsistently
__attribute__((noreturn)) void trace_error(bool aggressive, int depth, board_t board,
                                           const char* context);

#ifdef TRACING_ON

// Is tracing active for this position?
extern bool traced(bool aggressive, board_t board);

// -1 for silence, or [0,255] to print a bunch of information about the position
extern Array<uint8_t> trace_verbose_start(bool aggressive, int depth, board_t board);
extern string trace_verbose_prefix();
extern void trace_verbose_end();

struct TraceVerbose {
  bool active;
  TraceVerbose(bool active) : active(active) {}
  ~TraceVerbose() { if (active) trace_verbose_end(); }
};

// Record that the value of parent depends on the given value of child
extern void trace_dependency(bool parent_aggressive, int parent_depth, board_t parent, int child_depth, board_t child, superinfo_t child_info);

// Check that information is consistent with what we know
extern void trace_check(bool aggressive, int depth, board_t board, superinfo_t info, const char* context);

// Turn a subset of a super into a string
extern string subset(super_t s, RawArray<const uint8_t> w);

#endif

#define TRACE_VERBOSE_START(depth,board) TRACE(const Array<const uint8_t> verbose = trace_verbose_start(depth,board); TraceVerbose ender(verbose.size()>0))
#define TRACE_VERBOSE(...) TRACE(if (verbose.size()) std::cout << trace_verbose_prefix() << format(__VA_ARGS__) << std::endl)

// Do a clean evaluation of each board involved in an error or dependency
bool trace_learn();

}
