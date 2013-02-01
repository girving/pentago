// Massively parallel in-core endgame database computation

#include <pentago/mpi/toplevel.h>
#include <pentago/mpi/flow.h>
#include <pentago/mpi/reduction.h>
#include <pentago/end/config.h>
#include <pentago/end/partition.h>
#include <pentago/end/predict.h>
#include <pentago/end/random_partition.h>
#include <pentago/end/simple_partition.h>
#include <pentago/end/load_balance.h>
#include <pentago/mpi/io.h>
#include <pentago/mpi/utility.h>
#include <pentago/end/check.h>
#include <pentago/mpi/trace.h>
#include <pentago/base/all_boards.h>
#include <pentago/data/block_cache.h>
#include <pentago/end/store_block_cache.h>
#include <pentago/utility/thread.h>
#include <pentago/data/compress.h>
#include <pentago/search/superengine.h>
#include <pentago/search/supertable.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/index.h>
#include <pentago/utility/large.h>
#include <pentago/utility/memory.h>
#include <pentago/utility/wall_time.h>
#include <other/core/array/Array2d.h>
#include <other/core/random/Random.h>
#include <other/core/utility/curry.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/process.h>
#include <boost/detail/endian.hpp>
#include <sys/resource.h>
#include <sys/stat.h>
#include <getopt.h>
namespace pentago {
namespace mpi {

using Log::cout;
using Log::cerr;
using std::endl;
using std::flush;

// File scope for use in report_mpi_times
static int threads = 0;
static bool per_rank_times = false;

static const section_t bad_section(Vector<Vector<uint8_t,2>,4>(Vector<uint8_t,2>(255,0),Vector<uint8_t,2>(),Vector<uint8_t,2>(),Vector<uint8_t,2>()));

static section_t parse_section(const string& s) {
  section_t r;
  if (s.size() != 8)
    return bad_section;
  for (int i=0;i<4;i++)
    for (int j=0;j<2;j++) {
      const char c = s[2*i+j];
      if (c<'0' || c>'9')
        return bad_section;
      r.counts[i][j] = c-'0';
    }
  return r;
}

#define error(...) ({ \
  if (!rank) \
    cerr << argv[0] << ": " << format(__VA_ARGS__) << endl; \
  return 1; })

static void report(MPI_Comm comm, const char* name) {
  const int rank = comm_rank(comm);
  Array<uint64_t> info = memory_info();
  CHECK(MPI_Reduce(rank?info.data():MPI_IN_PLACE,info.data(),info.size(),datatype<uint64_t>(),MPI_MAX,0,comm));
  if (!rank)
    cout << "memory "<<name<<": "<<memory_report(info)<<endl;
}

static void report_mpi_times(MPI_Comm comm, RawArray<wall_time_t> times, wall_time_t elapsed, uint64_t outputs, uint64_t inputs) {
  const int rank = comm_rank(comm),
            ranks = comm_size(comm);
  if (per_rank_times && ranks>1) {
    Array<wall_time_t,2> all_times(ranks,times.size(),false);
    CHECK(MPI_Gather((int64_t*)times.data(),times.size(),datatype<int64_t>(),(int64_t*)all_times.data(),times.size(),datatype<int64_t>(),0,comm));
    if (!rank) {
      Log::Scope scope("per rank times");
      for (int r=0;r<ranks;r++)
        report_thread_times(all_times[r],format("%d",r));
    }
  }
  CHECK(MPI_Reduce(rank?(int64_t*)times.data():MPI_IN_PLACE,(int64_t*)times.data(),times.size(),datatype<int64_t>(),MPI_SUM,0,comm));
  if (!rank) {
    report_thread_times(times);
    const double core_time = elapsed.seconds()*threads*comm_size(comm);
    const double output_speed = outputs/core_time,
                 input_speed = inputs/core_time,
                 speed = output_speed+input_speed;
    const uint64_t all_nodes = 13540337135288;
    cout << format("speeds\n  elapsed = %g, output nodes = %s, input nodes = %s\n  speeds (nodes/second/core): output = %g, input = %g, output+input = %g\n  grand estimate = %s core-hours",
                   elapsed.seconds(),large(outputs),large(inputs),output_speed,input_speed,speed,large(uint64_t(2*all_nodes/speed/3600)))<<endl;
  }
}

static Ref<partition_t> make_simple_partition(const int ranks, const sections_t& sections) {
  return new_<simple_partition_t>(ranks,sections);
}

static Ref<partition_t> make_random_partition(const uint128_t key, const int ranks, const sections_t& sections) {
  return new_<random_partition_t>(key,ranks,sections);
}

// Record whether we're meaningless
static void write_meaningless(const MPI_Comm comm, const string& dir, const int meaningless) {
  if (meaningless && !comm_rank(comm)) {
    // touch meaningless-<n>
    const auto name = format("%s/meaningless-%d",dir,meaningless);
    FILE* file = fopen(name.c_str(),"wb");
    if (!file)
      die("failed to touch '%s': %s",name,strerror(errno));
  }
}

int toplevel(int argc, char** argv) {
  // Initialize MPI
  mpi_world_t world(argc,argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);
  set_verbose(!rank);
  report(comm,"mpi");

  // Check tag space
  const int required_tag_ub = (1<<(15+6+2))-1;
  int tag_ub = 0;
  {
    int success = false;
    void* value = 0;
    CHECK(MPI_Comm_get_attr(comm,MPI_TAG_UB,(void*)&value,&success));
    if (!success)
      error("tag upper bound lookup failed");
    tag_ub = *(int*)value;
    if (tag_ub<required_tag_ub)
      error("tag upper bound is only %d, need at least %d: 15 bits for line, 6 for block, 2 for dimension",tag_ub,required_tag_ub);
  }

  // Parse command line options
  int save = -100;
  int specified_block_size = 8;
  int level = 26;
  int64_t memory_limit = 0;
  int gather_limit = 32;
  int line_limit = 32;
  int samples = 256;
  int specified_ranks = -1;
  string dir;
  string test;
  int meaningless = 0;
  int stop_after = 0;
  int randomize = 0;
  bool log_all = false;
  static const option options[] = {
    {"help",no_argument,0,'h'},
    {"threads",required_argument,0,'t'},
    {"block-size",required_argument,0,'b'},
    {"save",required_argument,0,'s'},
    {"dir",required_argument,0,'d'},
    {"level",required_argument,0,'l'},
    {"memory",required_argument,0,'m'},
    {"gather-limit",required_argument,0,'g'},
    {"line-limit",required_argument,0,'L'},
    {"samples",required_argument,0,'p'},
    {"ranks",required_argument,0,'r'},
    {"test",required_argument,0,'u'},
    {"meaningless",required_argument,0,'n'},
    {"per-rank-times",no_argument,0,'z'},
    {"stop-after",required_argument,0,'S'},
    {"randomize",required_argument,0,'R'},
    {"log-all",no_argument,0,'a'},
    {0,0,0,0}};
  for (;;) {
    int option = 0;
    int c = getopt_long(argc,argv,"ht:b:s:d:m:",options,&option);
    if (c == -1) // Are we out of options?
      break;
    char* end;
    switch (c) {
      case 'h':
        if (!rank) {
          cout << format("usage: %s [options...] <section>\n",argv[0])
               << "options:\n"
                  "  -h, --help                 Display usage information and quit\n"
                  "  -t, --threads <threads>    Number of threads per MPI rank (required)\n"
                  "  -b, --block-size <size>    4D block size for each section (default "<<block_size<<")\n"
                  "  -s, --save <n>             Save all slices with n stones for fewer (required)\n"
                  "  -d, --dir <dir>            Save and log to given new directory (required)\n"
                  "      --level <n>            Compression level: 1-9 is zlib, 20-29 is xz (default "<<level<<")\n"
                  "  -m, --memory <n>           Approximate memory usage limit per *rank* (required)\n"
                  "      --gather-limit <n>     Maximum number of simultaneous active line gathers (default "<<gather_limit<<")\n"
                  "      --line-limit <n>       Maximum number of simultaneously allocated lines (default "<<line_limit<<")\n"
                  "      --samples <n>          Number of sparse samples to save per section (default "<<samples<<")\n"
                  "      --ranks <n>            Allowed for compatibility with predict, but must match mpirun --np\n"
                  "      --test <name>          Run the MPI side of one of the unit tests\n"
                  "      --meaningless <n>      Use meaningless values the given slice\n"
                  "      --per-rank-times       Print a timing report for each rank\n"
                  "      --stop-after <n>       Stop after computing the given slice\n"
                  "      --randomize <key>      If nonzero, partition lines and blocks randomly using the given key\n"
                  "      --log-all              Write log files for every process\n"
               << flush;
        }
        return 0;
        break;
      #define INT_ARG(short_opt,long_opt,var) \
        case short_opt: { \
          var = strtol(optarg,&end,0); \
          if (!*optarg || *end) \
            error("--" #long_opt " expected int, got '%s'",optarg); \
          break; }
      INT_ARG('t',threads,threads)
      INT_ARG('b',block-size,specified_block_size)
      INT_ARG('s',save,save)
      INT_ARG('l',level,level)
      INT_ARG('p',samples,samples)
      INT_ARG('r',ranks,specified_ranks)
      INT_ARG('n',meaningless,meaningless)
      INT_ARG('g',gather-limit,gather_limit)
      INT_ARG('L',line-limit,line_limit)
      INT_ARG('S',stop-after,stop_after)
      INT_ARG('R',randomize,randomize)
      case 'm': {
        double memory = strtod(optarg,&end);
        if (!strcmp(end,"MB") || !strcmp(end,"M"))
          memory_limit = uint64_t(memory*pow(2.,20));
        else if (!strcmp(end,"GB") || !strcmp(end,"G"))
          memory_limit = uint64_t(memory*pow(2.,30));
        else
          error("don't understand memory limit \"%s\", use e.g. 1.5GB",optarg);
        break; }
      case 'd':
        dir = optarg;
        break;
      case 'u':
        test = optarg;
        break;
      case 'z':
        per_rank_times = true;
        break;
      case 'a':
        log_all = true;
        break;
      default:
        MPI_Abort(MPI_COMM_WORLD,1);
    }
  }
  if (argc-optind != 1 && !test.size())
    error("expected exactly one argument (section)");
  if (!threads)
    error("must specify --threads n (or -t n) with n > 1");
  if (threads < 1)
    error("need at least two threads for communication vs. compute");
  if (block_size != specified_block_size)
    error("block size is currently hard coded to %d, can't specify %s",block_size,specified_block_size);
  if (block_size < 2 || (block_size&1))
    error("invalid block size %d",block_size);
  if (block_size != 8)
    error("for now, block size is hard coded to 8 (see compute.cpp)");
  if (gather_limit < 1)
    error("--gather-limit %d must be at least 1",gather_limit);
  if (line_limit < 2)
    error("--line-limit %d must be at least 2",line_limit);
  if (samples < 0)
    error("must specify positive value for --samples, not %d",samples);
  if (save == -100 && !test.size())
    error("must specify how many slices to save with --save");
  if (specified_ranks>=0 && specified_ranks!=ranks)
    error("--ranks %d doesn't match actual number of ranks %d",specified_ranks,ranks);
  if (stop_after<0)
    error("--stop-after %d should be nonnegative",stop_after);
  if (!dir.size())
    error("must specify --dir",dir);
  if (meaningless>9)
    error("--meaningless %d is outside valid range of [1,9]",meaningless);
  const char* section_str = argv[optind++];
  section_t section = test.size()?section_t():parse_section(section_str);
  if (!section.valid())
    error("invalid section '%s'",section_str);
  if (section.sum()==36)
    error("refusing to compute a 36 stone section");

  // Make directory, insisting that it's new
  {
    int r;
    if (!rank)
      r = mkdir(dir.c_str(),0777);
    CHECK(MPI_Bcast(&r,1,MPI_INT,0,comm));
    if (r < 0)
      error("failed to make new directory '%s': %s",dir,strerror(errno));
  }

  // Configure logging
  if (log_all)
    Log::cache_initial_output();
  Log::configure("endgame",rank!=0,false,rank?0:1<<30);
  if (!rank || log_all)
    Log::copy_to_file(log_all?format("%s/log-%d",dir,rank):format("%s/log",dir),false);

  // Allocate thread pool
  const int workers = threads-1;
  init_threads(workers,0);
  report(comm,"threads");

  // Make sure the compression level is valid
  if (!rank) {
    Array<uint8_t> data(37,false);
    for (int i=0;i<data.size();i++)
      data[i] = i;
    OTHER_ASSERT(decompress(compress(data,level,unevent),data.size(),unevent)==data);
  }

  // Make partition factory
  typedef function<Ref<partition_t>(int,const sections_t&)> partition_factory_t;
  const auto partition_factory = randomize ? partition_factory_t(curry(make_random_partition,randomize))
                                           :                           make_simple_partition;

  // Run unit test if requested.  See test_mpi.py for the Python side of these tests.
  if (test.size()) {
    if (test=="write-3" || test=="write-4") {
      check_directory(comm,dir);
      const int slice = test[6]-'0';
      write_meaningless(comm,dir,slice);
      const auto sections = new_<sections_t>(slice,all_boards_sections(slice,8));
      const auto partition = partition_factory(ranks,sections);
      const auto blocks = meaningless_block_store(partition,rank,samples);
      write_counts(comm,format("%s/counts-%d.npy",dir,slice),blocks);
      write_sparse_samples(comm,format("%s/sparse-%d.npy",dir,slice),blocks);
      write_sections(comm,format("%s/slice-%d.pentago",dir,slice),blocks,level);
    } else
      error("unknown unit test '%s'",test);
    return 0;
  }

  // Report
  if (!rank) {
    Log::Scope scope("parameters");
    cout << "command =";
    for (int i=0;i<argc;i++)
      cout << ' '<<argv[i];
    cout << "\nranks = "<<ranks
         << "\ncores = "<<ranks*threads
         << "\nthreads / rank = "<<threads
         << "\nsection = "<<section
         << "\nblock size = "<<block_size
         << "\nsaved slices = "<<save
         << "\nlevel = "<<level
         << "\nmemory limit = "<<large(memory_limit)
         << "\ngather limit = "<<gather_limit
         << "\nline limit = "<<line_limit
         << "\nmode = "<<(OTHER_DEBUG_ONLY(1)+0?"debug":"optimized")
         << "\nfunnel = "<<PENTAGO_MPI_FUNNEL
         << "\ncompress = "<<PENTAGO_MPI_COMPRESS
         << "\ncompress outputs = "<<PENTAGO_MPI_COMPRESS_OUTPUTS
         << "\nsse = "<<PENTAGO_SSE
#ifdef BOOST_BIG_ENDIAN
         << "\nendian = big"
#else
         << "\nendian = little"
#endif
         << "\nhistory = "<<thread_history_enabled()
         << "\nwildcard recvs = "<<wildcard_recv_count
         << "\nmeaningless = "<<meaningless
         << "\nrandomize = "<<randomize
         << "\ntag ub = "<<tag_ub<<" ("<<required_tag_ub<<" required)"
         << endl;
    if (PENTAGO_MPI_DEBUG)
      cout << "WARNING: EXPENSIVE DEBUGGING CODE ENABLED!" << endl; 
  }

  if (PENTAGO_MPI_DEBUG)
    init_supertable(12);

  // For paranoia's sake, generate dummy file to make sure the directory works.
  // It would be very sad to have the computation run for an hour and then choke.
  check_directory(comm,dir);

  // Record whether we're meaningless
  write_meaningless(comm,dir,meaningless);

  // Compute one list of sections per slice
  const vector<Ref<const sections_t>> slices = descendent_sections(section,meaningless?:35);

  // Allocate communicators
  flow_comms_t comms(comm);
  report(comm,"base");

  // Compute each slice in turn
  wall_time_t total_elapsed;
  uint64_t total_outputs = 0,
           total_inputs = 0;
  {
    Ptr<partition_t> prev_partition;
    Ptr<block_store_t> prev_blocks;
    const int first_slice = meaningless?meaningless-1:(int)slices.size()-1;
    for (int slice=first_slice;slice>=stop_after;slice--) {
      if (!slices[slice]->sections.size())
        break;
      Log::Scope scope(format("slice %d",slice));
      const auto start = wall_time();

      // Allocate meaningless data if necessary
      if (slice+1==meaningless) {
        prev_partition = partition_factory(ranks,slices[slice+1]);
        prev_blocks = meaningless_block_store(*prev_partition,rank,samples);
        if (PENTAGO_MPI_DEBUG)
          set_block_cache(store_block_cache(ref(prev_blocks),uint64_t(1)<<28));
      }

      // Partition work among processors
      const auto partition = partition_factory(ranks,slices[slice]);

      // Allocate memory for all the blocks we own
      auto lines = partition->rank_lines(rank);
      auto local_blocks = partition->rank_blocks(rank);
      const auto blocks = new_<block_store_t>(partition,rank,local_blocks,samples);
      {
        Log::Scope scope("load balance");
        const auto load = load_balance(reduction<int64_t,max_op>(comm),lines,local_blocks);
        if (!rank)
          cout << str(*load) << endl;
        local_blocks.clean_memory();
        #define local_blocks hide_local_blocks
      }

      // Estimate peak memory usage ignoring active lines
      const int64_t partition_memory = memory_usage(prev_partition)+memory_usage(partition),
                    block_memory = (prev_blocks?prev_blocks->estimate_peak_memory_usage():0)+blocks->estimate_peak_memory_usage(),
                    line_memory = memory_usage(lines)+base_compute_memory_usage(lines.size()),
                    base_memory = partition_memory+block_memory+line_memory;
      if (memory_limit <= base_memory)
        die("memory limit exceeded: base = %s, limit = %s",large(base_memory),large(memory_limit));
      const int64_t free_memory = memory_limit-base_memory;
      {
        int64_t numbers[5] = {partition_memory,block_memory,line_memory,base_memory,-free_memory};
        CHECK(MPI_Reduce(rank?numbers:MPI_IN_PLACE,numbers,5,datatype<int64_t>(),MPI_MAX,0,comm));
        if (!rank) {
          cout << "memory usage: partitions = "<<numbers[0]<<", blocks = "<<large(numbers[1])<<", lines = "<<numbers[2]<<", total = "<<large(numbers[3])<<", free = "<<large(-numbers[4])<<endl;
          cout << "line parallelism = "<<-numbers[4]/(2*13762560)<<endl;
        }
      }
      report(comm,"compute");

      // Compute (and communicate)
      {
        Log::Scope scope("compute");
        compute_lines(comms,prev_blocks,blocks,lines,free_memory,gather_limit,line_limit);
      }
      blocks->print_compression_stats(reduction<double,sum_op>(comm));
      const auto outputs = blocks->total_nodes,
                 inputs = prev_blocks?prev_blocks->total_nodes:0;
      total_outputs += outputs;
      total_inputs += inputs;

      // Deallocate obsolete slice
      prev_partition = partition;
      prev_blocks = blocks;
      lines.clean_memory();
      report(comm,"free");

      // Write various information to disk
      {
        Log::Scope scope("write");
        write_counts(comm,format("%s/counts-%d.npy",dir,slice),blocks);
        write_sparse_samples(comm,format("%s/sparse-%d.npy",dir,slice),blocks);
        if (slice <= save)
          write_sections(comm,format("%s/slice-%d.pentago",dir,slice),blocks,level);
      }

      // Dump timing
      const auto elapsed = wall_time()-start;
      total_elapsed += elapsed;
      report_mpi_times(comm,clear_thread_times(),elapsed,outputs,inputs);
    }
  }

  // Dump total timing
  report_mpi_times(comm,total_thread_times(),total_elapsed,total_outputs,total_inputs);
  report(comm,"final");
  write_thread_history(format("%s/history-r%d",dir,rank));
  return 0;
}

}
}
