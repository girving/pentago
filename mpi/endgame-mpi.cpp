// Massively parallel in-core endgame database computation

#include <pentago/mpi/flow.h>
#include <pentago/mpi/partition.h>
#include <pentago/mpi/io.h>
#include <pentago/mpi/utility.h>
#include <pentago/thread.h>
#include <pentago/compress.h>
#include <pentago/utility/large.h>
#include <pentago/utility/memory.h>
#include <other/core/utility/Log.h>
#include <sys/stat.h>
#include <getopt.h>
#include <mpi.h>
using namespace pentago;
using namespace pentago::mpi;

using Log::cout;
using std::endl;
using std::flush;

static section_t parse_section(const string& s) {
  section_t r;
  if (s.size() != 8)
    goto fail;
  for (int i=0;i<4;i++)
    for (int j=0;j<2;j++) {
      const char c = s[2*i+j];
      if (c<'0' || c>'9')
        goto fail;
      r.counts[i][j] = c-'0';
    }
  if (r.valid())
    return r;
fail:
  die_master(format("invalid section '%s'",s));
}

int main(int argc, char** argv) {
  // Initialize MPI
  mpi_world_t world(argc,argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  const int ranks = comm_size(comm),
            rank = comm_rank(comm);
  set_verbose(!rank);

  // Parse command line options
  int save = -100;
  int threads = 0;
  int block_size = 8;
  int level = 26;
  uint64_t memory_limit = 0;
  string dir;
  static const option options[] = {
    {"help",no_argument,0,'h'},
    {"threads",required_argument,0,'t'},
    {"block-size",required_argument,0,'b'},
    {"save",required_argument,0,'s'},
    {"dir",required_argument,0,'d'},
    {"level",required_argument,0,'l'},
    {"memory",required_argument,0,'m'},
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
                  "  -b, --block-size <size>    4D block size for each section (default 8)\n"
                  "  -s, --save <n>             Save all slices with n stones for fewer (required)\n"
                  "  -d, --dir <dir>            Save and log to given new directory (required)\n"
                  "  -l, --level <n>            Compression level: 1-9 is zlib, 20-29 is xz (default 26)\n"
                  "  -m, --memory <n>           Approximate memory usage limit per *rank* (required)\n"
               << flush;
          MPI_Abort(comm,0);
        }
        break;
      #define INT_ARG(short_opt,long_opt,var) \
        case short_opt: { \
          var = strtol(optarg,&end,0); \
          if (!*optarg || *end) \
            die_master(format("error: --" #long_opt " expected int, got '%s'",optarg)); \
          break; }
      INT_ARG('t',threads,threads)
      INT_ARG('b',block-size,block_size)
      INT_ARG('s',save,save)
      INT_ARG('l',level,level)
      case 'm': {
        double memory = strtod(optarg,&end);
        if (!strcmp(end,"MB") || !strcmp(end,"M"))
          memory_limit = uint64_t(memory*pow(2.,20));
        else if (!strcmp(end,"GB") || !strcmp(end,"G"))
          memory_limit = uint64_t(memory*pow(2.,30));
        else
          die(format("error: don't understand memory limit \"%s\", use e.g. 1.5GB",optarg));
        break; }
      case 'd':
        dir = optarg;
        break;
    }
  }
  if (argc-optind != 1)
    die_master("error: expected exactly one argument (section)");
  if (!threads)
    die_master("error: must specify --threads n (or -t n) with n > 1");
  if (threads < 1)
    die_master("error: need at least two threads for communication vs. compute");
  if (block_size < 2 || (block_size&1))
    die_master(format("error: invalid block size %d",block_size));
  if (block_size != 8)
    die_master(format("error: for now, block size is hard coded to 8 (see compute.cpp)"));
  if (save == -100)
    die_master("error: must specify how many slices to save with --save");
  section_t section = parse_section(argv[optind++]);
  if (section.sum()==36)
    die_master("error: refusing to compute a 36 stone section");

  // Check tag space
  OTHER_NOT_IMPLEMENTED(); 

  // Make sure the compression level is valid
  {
    Array<uint8_t> data(37,false);
    for (int i=0;i<data.size();i++)
      data[i] = i;
    OTHER_ASSERT(decompress(compress(data,level),data.size())==data);
  }

  // Make directory, insisting that it's new
  int r = mkdir(dir.c_str(),0777);
  if (r < 0)
    die_master(format("failed to make new directory '%s': %s",dir,strerror(errno)));

  // Configure logging
  Log::configure("endgame",rank!=0,rank!=0,rank?0:1<<30);
  if (!rank)
    Log::copy_to_file(format("%s/log",dir),false);

  // Report
  if (!rank) {
    cout << "command =";
    for (int i=0;i<argc;i++)
      cout << ' '<<argv[i];
    cout << "\nprocesses = "<<ranks
         << "\nthreads / process = "<<threads
         << "\nsection = "<<section
         << "\nblock size = "<<block_size
         << "\nsaved slices = "<<save
         << "\nlevel = "<<level
         << "\nmemory limit = "<<large(memory_limit)
         << endl;
  }

  // For paranoia's sake, generate dummy file to make sure the directory works.
  // It would be very sad to have the computation run for an hour and then choke.
  check_directory(comm,dir);

  // Allocate thread pool
  const int workers = threads-1;
  init_threads(workers,0);

  // Compute one list of sections per slice
  const vector<Array<const section_t>> slices = descendent_sections(section);

  // Allocate communicators
  flow_comms_t comms(comm);

  // Compute each slice in turn
  {
    Ptr<partition_t> prev_partition;
    Ptr<block_store_t> prev_blocks;
    for (int slice=(int)slices.size()-1;slice>=0;slice--) {
      Log::Scope scope(format("slice %d",slice));

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
                 base_memory = partition_memory+block_memory+line_memory,
                 free_memory = memory_limit-base_memory;
      cout << "memory usage: partitions = "<<partition_memory<<", blocks = "<<block_memory<<", lines = "<<line_memory<<", total = "<<large(base_memory)<<endl;
      cout << "maximum line parallelism = "<<free_memory/(2*13762560)<<endl;

      // Compute (and communicate)
      {
        Log::Scope scope("compute");
        compute_lines(comms,prev_blocks,blocks,lines,free_memory);
      }

      // Deallocate obsolete slice
      prev_partition = partition;
      prev_blocks = blocks;
      lines.clean_memory();

      // Write to disk if desired
      if (slice <= save) {
        Log::Scope scope("write");
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
  return 0;
}
