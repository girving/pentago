// Tracing code for use in debugging inconsistency detections

#pragma once

#include "board.h"
#include "superscore.h"
#include <other/core/utility/config.h>
namespace pentago {

// Switch comments to disable tracing
//#define TRACE(...) ((void)0)
#define TRACE(...) __VA_ARGS__

// Mark that the transposition table is clear
extern void trace_restart();

// Is tracing active for this position?
extern bool traced(board_t board);

// -1 for silence, or [0,255] to print a bunch of information about the position
extern Array<uint8_t> trace_verbose_start(int depth, board_t board);
extern string trace_verbose_prefix();
extern void trace_verbose_end();

struct TraceVerbose {
  bool active;
  TraceVerbose(bool active) : active(active) {}
  ~TraceVerbose() { if (active) trace_verbose_end(); }
};
#define TRACE_VERBOSE_START(depth,board) TRACE(const Array<const uint8_t> verbose = trace_verbose_start(depth,board); TraceVerbose ender(verbose.size()>0))
#define TRACE_VERBOSE(...) TRACE(if (verbose.size()) std::cout << trace_verbose_prefix() << format(__VA_ARGS__) << std::endl)

// Record that the value of board is computed inconsistently
extern void OTHER_NORETURN(trace_error(int depth, board_t board, const char* context));

// Record that the value of parent depends on the given value of child
extern void trace_dependency(int parent_depth, board_t parent, int child_depth, board_t child, superinfo_t child_info);

// Check that information is consistent with what we know
extern void trace_check(int depth, board_t board, superinfo_t info, const char* context);

// Turn a subset of a super into a string
extern string subset(super_t s, RawArray<const uint8_t> w);

}
