// Massively parallel in-core endgame database computation

#include "pentago/mpi/toplevel.h"
#include "pentago/mpi/flow.h"
#include "pentago/mpi/io.h"
#include "pentago/mpi/reduction.h"
#include "pentago/mpi/utility.h"
#include "pentago/base/all_boards.h"
#include "pentago/data/block_cache.h"
#include "pentago/data/compress.h"
#include "pentago/data/supertensor.h"
#include "pentago/end/check.h"
#include "pentago/end/config.h"
#include "pentago/end/load_balance.h"
#include "pentago/end/options.h"
#include "pentago/end/partition.h"
#include "pentago/end/predict.h"
#include "pentago/end/random_partition.h"
#include "pentago/end/simple_partition.h"
#include "pentago/end/store_block_cache.h"
#include "pentago/end/trace.h"
#include "pentago/search/superengine.h"
#include "pentago/search/supertable.h"
#include "pentago/utility/aligned.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/index.h"
#include "pentago/utility/join.h"
#include "pentago/utility/large.h"
#include "pentago/utility/memory.h"
#include "pentago/utility/memory_usage.h"
#include "pentago/utility/thread.h"
#include "pentago/utility/wall_time.h"
#include "pentago/utility/array.h"
#include "pentago/utility/uint128.h"
#include "pentago/utility/random.h"
#include "pentago/utility/curry.h"
#include "pentago/utility/log.h"
#include <boost/endian/conversion.hpp>
#include <sys/resource.h>
#include <sys/stat.h>
#include <getopt.h>
#include <errno.h>
#include <stdio.h>
namespace pentago {
namespace mpi {

using std::max;
#define error PENTAGO_OPTION_ERROR

static void report(MPI_Comm comm, const char* name) {
  const int rank = comm_rank(comm);
  Array<uint64_t> info = memory_info();
  CHECK(MPI_Reduce(rank?info.data():MPI_IN_PLACE,info.data(),info.size(),datatype<uint64_t>(),MPI_MAX,0,comm));
  if (!rank)
    slog("memory %s: %s", name, memory_report(info));
}

static void report_mpi_times(const MPI_Comm comm, const options_t& o, const thread_times_t local,
                             const wall_time_t elapsed, const uint64_t local_outputs,
                             const uint64_t local_inputs) {
  const int rank = comm_rank(comm),
            ranks = comm_size(comm);
  if (o.per_rank_times && ranks>1) {
    Array<wall_time_t,2> all_times(ranks,local.times.size(),uninit);
    CHECK(MPI_Gather((int64_t*)local.times.data(),local.times.size(),datatype<int64_t>(),
                     (int64_t*)all_times.data(),local.times.size(),datatype<int64_t>(),0,comm));
    if (!rank) {
      Scope scope("per rank times");
      for (int r=0;r<ranks;r++)
        report_thread_times(all_times[r],format("%d",r));
    }
  }
  const auto times = local.times.copy();
  CHECK(MPI_Reduce(rank?(int64_t*)times.data():MPI_IN_PLACE,
                   (int64_t*)times.data(),times.size(),datatype<int64_t>(),MPI_SUM,0,comm));
  const auto papi = local.papi.copy();
  CHECK(MPI_Reduce(rank?papi.data():MPI_IN_PLACE,papi.data(),papi.flat().size(),datatype<papi_t>(),MPI_SUM,0,comm));
  uint64_t counts[2] = {local_outputs,local_inputs};
  CHECK(MPI_Reduce(rank?counts:MPI_IN_PLACE,counts,2,datatype<uint64_t>(),MPI_SUM,0,comm));
  if (!rank) {
    report_thread_times(times);
    report_papi_counts(papi);
    const double core_time = elapsed.seconds() * o.threads * comm_size(comm);
    const uint64_t outputs = counts[0],
                   inputs = counts[1];
    const double output_speed = outputs/core_time,
                 input_speed = inputs/core_time,
                 speed = output_speed+input_speed;
    const uint64_t all_nodes = 13540337135288;
    slog("speeds\n  elapsed = %g, output nodes = %s, input nodes = %s\n"
         "  speeds (nodes/second/core): output = %g, input = %g, output+input = %g\n"
         "  grand estimate = %s core-hours",
         elapsed.seconds(), large(outputs), large(inputs), output_speed, input_speed, speed,
         large(uint64_t(2*all_nodes/speed/3600)));
  }
}

static int broadcast_supertensor_slice(const MPI_Comm comm, const string& path) {
  int slice;
  if (!comm_rank(comm))
    slice = supertensor_slice(path);
  CHECK(MPI_Bcast(&slice,1,MPI_INT,0,comm));
  return slice;
}

static shared_ptr<partition_t> make_simple_partition(
    const int ranks, const shared_ptr<const sections_t>& sections) {
  return make_shared<simple_partition_t>(ranks, sections);
}

static shared_ptr<partition_t>
make_random_partition(const uint128_t key, const int ranks,
                      const shared_ptr<const sections_t>& sections) {
  return make_shared<random_partition_t>(key, ranks, sections);
}

// Record whether we're meaningless
static void write_meaningless(const MPI_Comm comm, const string& dir, const int meaningless) {
  if (meaningless && !comm_rank(comm)) {
    // touch meaningless-<n>
    const auto name = format("%s/meaningless-%d", dir, meaningless);
    FILE* file = fopen(name.c_str(), "wb");
    if (!file)
      die("failed to touch '%s': %s", name, strerror(errno));
    fclose(file);
  }
}

int toplevel(int argc, char** argv) {
  // Initialize MPI
  mpi_world_t world(argc, argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);
  set_verbose(!rank);

  // Check tag space
  const int required_tag_ub = (1<<(14+6+2))-1;
  int tag_ub = 0;
  {
    int success = false;
    void* value = 0;
    CHECK(MPI_Comm_get_attr(comm,MPI_TAG_UB,(void*)&value,&success));
    if (!success)
      error("tag upper bound lookup failed");
    tag_ub = *(int*)value;
    if (tag_ub<required_tag_ub)
      error("tag upper bound is only %d, need at least %d: 14 bits for line, 6 for block, 2 for dimension",tag_ub,required_tag_ub);
  }

  // Parse command line options
  const options_t o = parse_options(argc, argv, ranks, rank);

  // Make directory, insisting that it's new
  {
    int r;
    if (!rank)
      r = mkdir(o.dir.c_str(), 0777);
    CHECK(MPI_Bcast(&r, 1, MPI_INT, 0, comm));
    if (r < 0)
      error("failed to make new directory '%s': %s", o.dir, strerror(errno));
  }

  // Configure logging.
  if (rank)
    suppress_log();
  if (!rank || o.log_all)
    copy_log_to_file(o.log_all ? format("%s/log-%d", o.dir, rank) : format("%s/log", o.dir));
  Scope scope("endgame");
  report(comm, "mpi");

  // Allocate thread pool
  const int workers = o.threads - 1;
  init_threads(workers, 0);
  report(comm, "threads");

  // Make sure the compression level is valid
  if (!rank) {
    Array<uint8_t> data(37,uninit);
    for (int i=0;i<data.size();i++)
      data[i] = i;
    GEODE_ASSERT(decompress(compress(data, o.level, unevent), data.size(), unevent)==data);
  }

  // Make partition factory
  const auto partition_factory =
      o.randomize ? partition_factory_t(curry(make_random_partition, o.randomize))
                  :                           make_simple_partition;

  // Run a unit test if requested.  See mpi_test.cc for invocation.
  if (o.test.size()) {
    if (o.test=="write-2" || o.test=="write-3" || o.test=="write-4") {
      check_directory(comm, o.dir);
      const int slice = o.test[6]-'0';
      write_meaningless(comm, o.dir, slice);
      const auto sections = make_shared<sections_t>(slice, all_boards_sections(slice,8));
      const auto partition = partition_factory(ranks, sections);
      const auto store = make_shared<compacting_store_t>(estimate_block_heap_size(*partition, rank));
      const auto sections_file = format("%s/slice-%d.pentago", o.dir, slice);
      {
        const auto blocks = meaningless_block_store(partition, rank, o.samples, store);
        Scope scope("write");
        write_counts(comm, format("%s/counts-%d.npy", o.dir, slice), *blocks);
        write_sparse_samples(comm, format("%s/sparse-%d.npy", o.dir, slice), *blocks);
        write_sections(comm, sections_file, *blocks, o.level);
      } {
        Scope scope("restart");
        const auto restart = read_sections(comm, sections_file, store, partition_factory);
        Scope write_scope("write");
        write_sections(comm, format("%s/slice-%d-restart.pentago", o.dir, slice), *restart, o.level);
      }
    } else if (o.test == "restart") {
      read_sections_test(comm, o.restart, partition_factory);
    } else
      error("unknown unit test '%s'", o.test);
    return 0;
  }

  // Report
  if (!rank) {
    Scope scope("parameters");
    slog("command = %s", join(" ", vector<string>(argv, argv+argc)));
    slog("ranks = %d", ranks);
    slog("cores = %d", ranks * o.threads);
    slog("threads / rank = %d", o.threads);
    slog("section = %s", o.section);
    slog("block size = %d", block_size);
    slog("saved slices = %d", o.save);
    slog("level = %d", o.level);
    slog("memory limit = %s", large(o.memory_limit));
    slog("gather limit = %d", o.gather_limit);
    slog("line limit = %d", o.line_limit);
    slog("mode = %s", GEODE_DEBUG_ONLY(1)+0?"debug":"optimized");
    slog("funnel = %d", PENTAGO_MPI_FUNNEL);
    slog("compress = %d", PENTAGO_MPI_COMPRESS);
    slog("compress outputs = %d", PENTAGO_MPI_COMPRESS_OUTPUTS);
    slog("timing = %d", PENTAGO_TIMING);
    slog("sse = %d", PENTAGO_SSE);
    slog("endian = %s", boost::endian::order::native == boost::endian::order::little ? "little"
                      : boost::endian::order::native == boost::endian::order::big    ? "big"
                                                                                     : "unknown");
    slog("history = %d", thread_history_enabled());
    slog("papi = %s", papi_enabled() ? join(" ",papi_event_names()) : "<disabled>");
    slog("wildcard recvs = %d", wildcard_recv_count);
    slog("meaningless = %d", o.meaningless);
    slog("randomize = %d", o.randomize);
    slog("tag ub = %d (%d required)", tag_ub, required_tag_ub);
    if (PENTAGO_MPI_DEBUG)
      slog("WARNING: EXPENSIVE DEBUGGING CODE ENABLED!");
  }

  if (PENTAGO_MPI_DEBUG)
    init_supertable(12);

  // For paranoia's sake, generate a dummy file to make sure the directory works.
  // It would be very sad to have the computation run for an hour and then choke.
  check_directory(comm, o.dir);

  // Record whether we're meaningless
  write_meaningless(comm, o.dir, o.meaningless);

  // Compute one list of sections per slice
  const auto slices = descendent_sections(o.section, o.meaningless ?: 35);

  // Allocate communicators
  flow_comms_t comms(comm);
  report(comm,"base");

  // Determine which slice we're going to start from
  const int restart_slice = o.restart.size() ? broadcast_supertensor_slice(comm, o.restart) : -1;
  if (o.restart.size())
    slog("restart: slice %d, file %s", restart_slice, o.restart);

  // Allocate the space needed for block storage
  uint64_t heap_size = 0;
  {
    Scope scope("estimate");
    uint64_t prev_size = 0;
    // Note that we compute first_slice differently from below in the meaningless case.
    const int first_slice = o.restart.size() ? restart_slice
                          : o.meaningless    ? o.meaningless
                                             : int(slices.size())-1;
    for (int slice=first_slice;slice>=o.stop_after;slice--) {
      if (!slices[slice]->sections.size())
        break;
      const auto size = estimate_block_heap_size(*partition_factory(ranks, slices[slice]), rank);
      heap_size = max(heap_size, prev_size+size);
      prev_size = size;
    }
    slog("heap size = %d", heap_size);
    const uint64_t min_heap_size = 1<<26;
    if (heap_size < min_heap_size) {
      slog("raising heap size to a minimum of %d", min_heap_size);
      heap_size = min_heap_size;
    }
  }
  const auto store = make_shared<compacting_store_t>(heap_size);

  // Compute each slice in turn
  wall_time_t total_elapsed;
  uint64_t total_local_outputs = 0,
           total_local_inputs = 0;
  {
    shared_ptr<const block_partition_t> prev_partition;
    shared_ptr<const readable_block_store_t> prev_blocks;

    // Load existing restart data if available
    if (o.restart.size()) {
      prev_blocks = read_sections(comm, o.restart, store, partition_factory);
      prev_partition = prev_blocks->partition;
      // Verify that we match the expected set of sections
      if (prev_partition->sections->slice!=restart_slice)
        die("internal error: inconsistent restart slices: %d vs %d", restart_slice,
            prev_partition->sections->slice);
      const auto expected = slices[restart_slice];
      for (const auto s : expected->sections)
        check_get(prev_partition->sections->section_id, s);
      for (const auto s : prev_partition->sections->sections)
        check_get(expected->section_id, s);
    }

    // Allocate meaningless data if desired
    if (o.meaningless) {
      if (o.restart.size()) {
        GEODE_ASSERT(prev_partition->sections->slice <= o.meaningless);
        GEODE_ASSERT(!PENTAGO_MPI_DEBUG);
      } else {
        prev_partition = partition_factory(ranks,slices[o.meaningless]);
        prev_blocks = meaningless_block_store(prev_partition, rank, o.samples, store);
        if (PENTAGO_MPI_DEBUG)
          set_block_cache(store_block_cache(prev_blocks, uint64_t(1)<<28));
      }
    }

    // Compute!
    const int first_slice = prev_partition ? prev_partition->sections->slice-1 : int(slices.size())-1;
    for (int slice=first_slice;slice>=o.stop_after;slice--) {
      if (!slices[slice]->sections.size())
        break;
      Scope scope(format("slice %d",slice));
      const auto start = wall_time();

      // Partition work among processors
      const auto partition = partition_factory(ranks,slices[slice]);

      // Allocate memory for all the blocks we own
      auto lines = partition->rank_lines(rank);
      auto local_blocks = partition->rank_blocks(rank);
      const auto blocks = make_shared<accumulating_block_store_t>(
          partition, rank, local_blocks, o.samples, store);
      {
        Scope scope("load balance");
        const auto load = load_balance(reduction<int64_t,max_op>(comm), lines, local_blocks);
        if (!rank)
          load->print();
        local_blocks.clean_memory();
        #define local_blocks hide_local_blocks
      }

      // Estimate peak memory usage ignoring active lines
      const int64_t store_memory = memory_usage(store),
                    partition_memory = memory_usage(prev_partition)+memory_usage(partition),
                    base_block_memory = (prev_blocks ? prev_blocks->base_memory_usage() : 0) +
                                        blocks->base_memory_usage(),
                    line_memory = memory_usage(lines)+base_compute_memory_usage(lines.size()),
                    base_memory = store_memory+partition_memory+base_block_memory+line_memory;
      if (o.memory_limit <= base_memory)
        die("memory limit exceeded: base = %s, limit = %s", large(base_memory), large(o.memory_limit));
      const int64_t free_memory = o.memory_limit - base_memory;
      {
        int64_t numbers[6] = {store_memory, partition_memory, base_block_memory, line_memory,
                              base_memory, -free_memory};
        CHECK(MPI_Reduce(rank ? numbers : MPI_IN_PLACE, numbers, 5, datatype<int64_t>(), MPI_MAX, 0,
                         comm));
        if (!rank) {
          slog("memory usage: store = %s, partitions = %d, blocks = %s, lines = %d, total = %s, "
               "free = %s", large(numbers[0]), numbers[1], large(numbers[2]), numbers[3],
               large(numbers[4]), large(-numbers[5]));
          slog("line parallelism = %d", -numbers[5]/(2*13762560));
        }
      }
      report(comm,"compute");

      // Count inputs
      const auto local_inputs = prev_blocks?prev_blocks->total_nodes:0;
      total_local_inputs += local_inputs;

      // Compute (and communicate)
      {
        Scope scope("compute");
        compute_lines(comms, prev_blocks, *blocks, lines, free_memory, o.gather_limit, o.line_limit);
      }

      // Deallocate obsolete slice
      prev_partition = partition;
      prev_blocks = blocks;
      lines.clean_memory();
      report(comm,"free");

      // Freeze newly computed blocks in preparation for use as inputs
      blocks->store.freeze();
      blocks->print_compression_stats(reduction<double,sum_op>(comm));

      // Count local outputs
      const auto local_outputs = blocks->total_nodes;
      total_local_outputs += local_outputs;

      // Write various information to disk
      {
        Scope scope("write");
        write_counts(comm, format("%s/counts-%d.npy", o.dir, slice), *blocks);
        write_sparse_samples(comm, format("%s/sparse-%d.npy", o.dir, slice), *blocks);
        if (slice <= o.save)
          write_sections(comm, format("%s/slice-%d.pentago", o.dir, slice), *blocks, o.level);
      }

      // Dump timing
      const auto elapsed = wall_time()-start;
      total_elapsed += elapsed;
      report_mpi_times(comm, o, clear_thread_times(), elapsed, local_outputs, local_inputs);
    }
  }

  // Dump total timing
  report_mpi_times(comm, o, total_thread_times(), total_elapsed, total_local_outputs,
                   total_local_inputs);
  report(comm, "final");
  write_thread_history(format("%s/history-r%d", o.dir, rank));
  return 0;
}

}  // namespace mpi
}  // namespace pentago
