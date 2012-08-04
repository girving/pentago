// Massively parallel in-core endgame database computation

#include <pentago/mpi/flow.h>
#include <pentago/mpi/config.h>
#include <pentago/mpi/partition.h>
#include <pentago/mpi/io.h>
#include <pentago/mpi/utility.h>
#include <pentago/mpi/check.h>
#include <pentago/mpi/trace.h>
#include <pentago/all_boards.h>
#include <pentago/block_cache.h>
#include <pentago/thread.h>
#include <pentago/compress.h>
#include <pentago/superengine.h>
#include <pentago/supertable.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/large.h>
#include <pentago/utility/memory.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/process.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <getopt.h>
using namespace pentago;
using namespace pentago::mpi;

using Log::cout;
using Log::cerr;
using std::endl;
using std::flush;

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

static void report(const char* name) {
  cout << "memory "<<name<<": "<<memory_report()<<endl;
}

int main(int argc, char** argv) {
  // Initialize MPI
  report("start");
  mpi_world_t world(argc,argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);
  set_verbose(!rank);
  report("mpi");

  // Parse command line options
  int save = -100;
  int threads = 0;
  int block_size = 8;
  int level = 26;
  uint64_t memory_limit = 0;
  int samples = 256;
  int specified_ranks = -1;
  string dir;
  string test;
  int meaningless = 0;
  static const option options[] = {
    {"help",no_argument,0,'h'},
    {"threads",required_argument,0,'t'},
    {"block-size",required_argument,0,'b'},
    {"save",required_argument,0,'s'},
    {"dir",required_argument,0,'d'},
    {"level",required_argument,0,'l'},
    {"memory",required_argument,0,'m'},
    {"samples",required_argument,0,'p'},
    {"ranks",required_argument,0,'r'},
    {"test",required_argument,0,'u'},
    {"meaningless",required_argument,0,'n'},
    {0,0,0,0}};
  for (;;) {
    int option = 0;
    int c = getopt_long(argc,argv,"ht:b:s:d:l:m:",options,&option);
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
                  "  -l, --level <n>            Compression level: 1-9 is zlib, 20-29 is xz (default "<<level<<")\n"
                  "  -m, --memory <n>           Approximate memory usage limit per *rank* (required)\n"
                  "      --samples <n>          Number of sparse samples to save per section (default "<<samples<<")\n"
                  "      --ranks <n>            Allowed for compatibility with predict, but must match mpirun --np\n"
                  "      --test <name>          Run the MPI side of one of the unit tests\n"
                  "      --meaningless <n>      Use meaningless values the given slice\n"
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
      INT_ARG('b',block-size,block_size)
      INT_ARG('s',save,save)
      INT_ARG('l',level,level)
      INT_ARG('p',samples,samples)
      INT_ARG('r',ranks,specified_ranks)
      INT_ARG('n',meaningless,meaningless)
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
  if (block_size < 2 || (block_size&1))
    error("invalid block size %d",block_size);
  if (block_size != 8)
    error("for now, block size is hard coded to 8 (see compute.cpp)");
  if (samples < 0)
    error("must specify positive value for --samples, not %d",samples);
  if (save == -100 && !test.size())
    error("must specify how many slices to save with --save");
  if (specified_ranks>=0 && specified_ranks!=ranks)
    error("--ranks %d doesn't match actual number of ranks %d",specified_ranks,ranks);
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
  Log::configure("endgame",rank!=0,false,rank?0:1<<30);
  if (!rank)
    Log::copy_to_file(format("%s/log",dir),false);

  // Allocate thread pool
  const int workers = threads-1;
  init_threads(workers,0);
  report("threads");

  // Make sure the compression level is valid
  if (!rank) {
    Array<uint8_t> data(37,false);
    for (int i=0;i<data.size();i++)
      data[i] = i;
    OTHER_ASSERT(decompress(compress(data,level),data.size())==data);
  }

  // Run unit test if requested.  See test_mpi.py for the Python side of these tests.
  if (test.size()) {
    if (test=="write-3" || test=="write-4") {
      check_directory(comm,dir);
      const int slice = test[6]-'0';
      const auto sections = all_boards_sections(slice,8);
      const auto partition = new_<partition_t>(ranks,block_size,slice,sections);
      const auto blocks = meaningless_block_store(partition,rank);
      write_counts(comm,format("%s/counts-%d.npy",dir,slice),blocks);
      write_sparse_samples(comm,format("%s/sparse-%d.npy",dir,slice),blocks,samples);
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
         << "\nthreads / rank = "<<threads
         << "\nsection = "<<section
         << "\nblock size = "<<block_size
         << "\nsaved slices = "<<save
         << "\nlevel = "<<level
         << "\nmemory limit = "<<large(memory_limit)
         << endl;
#ifdef MPI_DEBUG
    cout << "WARNING: EXPENSIVE DEBUGGING CODE ENABLED!" << endl; 
#endif
  }

#ifdef MPI_DEBUG
  init_supertable(12);
#endif

  // For paranoia's sake, generate dummy file to make sure the directory works.
  // It would be very sad to have the computation run for an hour and then choke.
  check_directory(comm,dir);

  // Compute one list of sections per slice
  const vector<Array<const section_t>> slices = descendent_sections(section,meaningless?:35);

  // Allocate communicators
  flow_comms_t comms(comm);
  report("base");

  // Compute each slice in turn
  {
    Ptr<partition_t> prev_partition;
    Ptr<block_store_t> prev_blocks;
    const int first_slice = meaningless?meaningless-1:(int)slices.size()-1;
    for (int slice=first_slice;slice>=0;slice--) {
      if (!slices[slice].size())
        break;
      Log::Scope scope(format("slice %d",slice));

      // Allocate meaningless data if necessary
      if (slice+1==meaningless) {
        prev_partition = new_<partition_t>(ranks,block_size,slice+1,slices[slice+1]);
        prev_blocks = meaningless_block_store(*prev_partition,rank);
#ifdef MPI_DEBUG
        set_block_cache(store_block_cache(ref(prev_blocks)));
#endif
      }

      // Partition work among processors
      const auto partition = new_<partition_t>(ranks,block_size,slice,slices[slice]);

      // Allocate memory for all the blocks we own
      auto lines = partition->rank_lines(rank,true);
      const auto blocks = new_<block_store_t>(partition,rank,lines);
      lines.append_elements(partition->rank_lines(rank,false));

      // Measure current memory usage
      const auto partition_memory = memory_usage(prev_partition)+memory_usage(partition),
                 block_memory = memory_usage(prev_blocks)+memory_usage(blocks),
                 line_memory = memory_usage(lines)+base_compute_memory_usage(lines.size()),
                 base_memory = partition_memory+block_memory+line_memory;
      if (memory_limit < base_memory)
        die(format("memory limit exceeded: base = %s, limit = %s",large(base_memory),large(memory_limit)));
      const auto free_memory = memory_limit-base_memory;
      cout << "memory usage: partitions = "<<partition_memory<<", blocks = "<<large(block_memory)<<", lines = "<<line_memory<<", total = "<<large(base_memory)<<endl;
      cout << "maximum line parallelism = "<<free_memory/(2*13762560)<<endl;
      report("compute");

      // Compute (and communicate)
      {
        Log::Scope scope("compute");
        compute_lines(comms,prev_blocks,blocks,lines,free_memory);
      }

      // Deallocate obsolete slice
      prev_partition = partition;
      prev_blocks = blocks;
      lines.clean_memory();
      report("free");

      // Write various information to disk
      {
        Log::Scope scope("write");
        write_counts(comm,format("%s/counts-%d.npy",dir,slice),blocks);
        write_sparse_samples(comm,format("%s/sparse-%d.npy",dir,slice),blocks,samples);
        if (slice <= save)
          write_sections(comm,format("%s/slice-%d.pentago",dir,slice),blocks,level);
      }

      // Dump timing
      // TODO: synchronize all processes
      report_thread_times(false);
    }
  }

  // Dump total timing
  // TODO: synchronize all processes
  report_thread_times(true);
  report("final");
  return 0;
}
