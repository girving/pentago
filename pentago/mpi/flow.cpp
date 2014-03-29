// Endgame computation structure code with interleaved communication and compute

#include <pentago/mpi/flow.h>
#include <pentago/end/compute.h>
#include <pentago/mpi/ibarrier.h>
#include <pentago/mpi/requests.h>
#include <pentago/mpi/trace.h>
#include <pentago/mpi/utility.h>
#include <pentago/end/fast_compress.h>
#include <pentago/utility/thread.h>
#include <pentago/utility/char_view.h>
#include <pentago/utility/large.h>
#include <pentago/utility/memory.h>
#include <geode/array/Array4d.h>
#include <geode/array/IndirectArray.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/curry.h>
#include <geode/utility/ProgressIndicator.h>
#include <geode/utility/Hasher.h>
#include <geode/utility/Log.h>
#include <geode/utility/tr1.h>
#include <boost/noncopyable.hpp>
namespace pentago {
namespace mpi {

using Log::cout;
using std::endl;
using std::make_pair;

static inline int request_id(local_id_t owner_block_id, uint8_t dimension) {
  return (owner_block_id.id<<2) | dimension;
}

static inline local_id_t request_block_id(int request_id) {
  return local_id_t(request_id>>2);
}

static inline uint8_t request_dimension(int request_id) {
  return request_id&3;
}

static inline dimensions_t request_dimensions(const Vector<int,2>* buffer) {
  return dimensions_t::raw(buffer->x);
}

static inline int request_response_tag(const Vector<int,2>* buffer) {
  return buffer->y;
}

namespace {
struct block_request_t : public boost::noncopyable {
  // The section is determined by the dependent line, so we don't need to store it
  const section_t section; // child section
  const Vector<uint8_t,4> block;
  const dimensions_t dimensions;
  const Vector<int,2> request_buffer; // dimensions, response_tag
  vector<line_details_t*> dependent_lines;

  block_request_t(const section_t section, const Vector<uint8_t,4> block, const dimensions_t dimensions, const int response_tag)
    : section(section), block(block), dimensions(dimensions), request_buffer(dimensions.data,response_tag) {}

  event_t block_lines_event() const {
    return pentago::mpi::block_lines_event(section,dimensions,block);
  }
};
}

static int total_blocks(RawArray<const line_t> lines) {
  int count = 0;
  for (const auto& line : lines)
    count += line.length;
  return count;
}

flow_comms_t::flow_comms_t(MPI_Comm comm)
  : rank(comm_rank(comm))
  , barrier_comm(comm_dup(comm))
  , request_comm(comm_dup(comm))
  , response_comm(comm_dup(comm))
  , output_comm(comm_dup(comm))
#if !PENTAGO_MPI_FUNNEL
  , wakeup_comm(comm_dup(MPI_COMM_SELF))
#endif
{}

flow_comms_t::~flow_comms_t() {
  CHECK(MPI_Comm_free(&barrier_comm));
  CHECK(MPI_Comm_free(&request_comm));
  CHECK(MPI_Comm_free(&response_comm));
  CHECK(MPI_Comm_free(&output_comm));
#if !PENTAGO_MPI_FUNNEL
  CHECK(MPI_Comm_free(&wakeup_comm));
#endif
}

// Leave flow_t outside an anonymous namespace to reduce backtrace sizes

struct flow_t {
  const flow_comms_t& comms;
  const Ptr<const readable_block_store_t> input_blocks;
  accumulating_block_store_t& output_blocks;

  // Pending requests, including wildcard requests for responding to block requests and output messages
  requests_t requests;

  // Lines which haven't yet been allocated and unscheduled
  Array<line_data_t*> unscheduled_lines;

  // Pending block requests, with internal references to the lines that depend on them.
  // There will be few enough of these that linear search is fine.
  Array<block_request_t*> block_requests;
  int next_response_tag;

  // Keep track of how much more stuff has to happen.  This includes both blocks we need to send and those we need to receive.
  ibarrier_countdown_t countdown;
  ProgressIndicator progress;

  // Current free memory, line gathers (lines whose input requests have been sent out), and allocated lines
  uint64_t free_memory;
  int free_line_gathers;
  int free_lines;

  // Space for persistent message requests
  Vector<Vector<int,2>,wildcard_recv_count> request_buffers;
  Vector<Array<Vector<super_t,2>>,wildcard_recv_count> output_buffers;

  // Wakeup support
#if !PENTAGO_MPI_FUNNEL
  uint64_t wakeup_buffer;
#endif

  flow_t(const flow_comms_t& comms, const Ptr<const readable_block_store_t> input_blocks, accumulating_block_store_t& output_blocks, RawArray<const line_t> lines, const uint64_t memory_limit, const int line_gather_limit, const int line_limit);
  ~flow_t();

  void schedule_lines();
  void post_barrier_recv();
  void post_request_recv(Vector<int,2>* buffer);
  void post_output_recv(Array<Vector<super_t,2>>* buffer);
  void process_barrier(MPI_Status* status);
  void process_request(Vector<int,2>* buffer, MPI_Status* status);
  void process_output(Array<Vector<super_t,2>>* buffer, MPI_Status* status);
  void process_response(block_request_t* request, MPI_Status* status);
  void send_output(line_details_t* const line, const int b);
  void finish_output_send(line_details_t* const line, MPI_Status* status);

  // Wakeup support
  typedef line_details_t::wakeup_block_t wakeup_block_t;
  void wakeup(line_details_t* const line, const wakeup_block_t b); // Call only from communication thread
  void post_wakeup(line_details_t& line, const wakeup_block_t b); // Wrapped in a wakeup_t and called from a compute thread
#if !PENTAGO_MPI_FUNNEL
  void post_wakeup_recv();
  void process_wakeup(MPI_Status* status);
#endif
};

flow_t::flow_t(const flow_comms_t& comms, const Ptr<const readable_block_store_t> input_blocks, accumulating_block_store_t& output_blocks, RawArray<const line_t> lines, const uint64_t memory_limit, const int line_gather_limit, const int line_limit)
  : comms(comms)
  , input_blocks(input_blocks)
  , output_blocks(output_blocks)
  , next_response_tag(0)
  , countdown(comms.barrier_comm,requests,barrier_tag,total_blocks(lines)+output_blocks.required_contributions)
  , progress(countdown.remaining(),false)
  , free_memory(memory_limit)
  , free_line_gathers(line_gather_limit)
  , free_lines(line_limit)
{
  GEODE_ASSERT(line_gather_limit<=32); // Make sure linear search through block_requests is okay
  GEODE_ASSERT(free_line_gathers>=1);

  // Compute information about each line
  unscheduled_lines.preallocate(lines.size());
  for (int i=lines.size()-1;i>=0;i--) {
    const auto line = new line_data_t(lines[i]);
    GEODE_ASSERT(line->memory_usage<=memory_limit/2);
    unscheduled_lines.append_assuming_enough_space(line);
  }

  // Start wildcard receives
  post_barrier_recv();
  for (auto& buffer : request_buffers)
    post_request_recv(&buffer);
  for (auto& buffer : output_buffers)
    post_output_recv(&buffer);
#if !PENTAGO_MPI_FUNNEL
  post_wakeup_recv();
#endif

  // Schedule some lines
  schedule_lines();

  // Enter communication loop
  while (!countdown.barrier.done())
    requests.waitsome();

  // Cancel the wildcard receives
  requests.cancel_and_waitall();

  // Finish up
  GEODE_ASSERT(!unscheduled_lines.size());
  GEODE_ASSERT(!block_requests.size());
  threads_wait_all_help();
}

flow_t::~flow_t() {}

void flow_t::schedule_lines() {
  // If there are unscheduled lines, try to schedule them
  while (free_lines && free_line_gathers && unscheduled_lines.size()) {
    line_details_t* line;
    {
      const line_data_t* preline = unscheduled_lines.back();
      const auto line_memory = preline->memory_usage;
      if (free_memory < line_memory)
        return;
      // Schedule the line
      thread_time_t time(allocate_line_kind,preline->line.line_event());
      unscheduled_lines.pop();
      free_memory -= line_memory;
      free_lines--;
      line = new line_details_t(*preline,curry(&flow_t::post_wakeup,this));
      PENTAGO_MPI_TRACE("allocate line %p: %s",line,str(line->pre.line));
    }
    // Request all input blocks
    if (!line->input_blocks)
      schedule_compute_line(*line);
    else {
      GEODE_ASSERT(input_blocks);
      free_line_gathers--;
      const auto child_section = line->standard_child_section;
      for (int b : range((int)line->input_blocks)) {
        const auto block = line->input_block(b);
        // Check for an existing block request if desired
        block_request_t* block_request = 0;
        static const bool merge_block_requests = !thread_history_enabled();
        if (merge_block_requests)
          for (auto request : block_requests)
            if (request->section==child_section && request->block==block) {
              block_request = request;
              break;
            }
        // Send a new request if necessary
        if (!block_request) {
          // Send a block request message
          const dimensions_t dimensions(line->section_transform,line->child_dimension);
          thread_time_t time(request_send_kind,block_lines_event(child_section,dimensions,block));
          const auto owner_and_id = input_blocks->partition->find_block(child_section,block);
          const int owner = owner_and_id.x;
          const local_id_t owner_block_id = owner_and_id.y;
          const int response_tag = next_response_tag++;
          GEODE_ASSERT(response_tag<(1<<23));
          block_request = new block_request_t(child_section,block,dimensions,response_tag);
          block_requests.append(block_request);
          // Post receives for all responses, feeding directly into the input memory for the first line.
          // We can still do this in compressed mode since the input buffer has an extra entry to account for expansion.
          const RawArray<Vector<super_t,2>> block_data = line->input_block_data(b);
          MPI_Request response_request;
          if (PENTAGO_MPI_COMPRESS)
            CHECK(MPI_Irecv(block_data.data(),CHECK_CAST_INT(memory_usage(block_data)),MPI_BYTE,owner,response_tag,comms.response_comm,&response_request));
          else
            CHECK(MPI_Irecv((uint64_t*)block_data.data(),8*block_data.size(),datatype<uint64_t>(),owner,response_tag,comms.response_comm,&response_request));
          // Send request
          MPI_Request request_request;
          const auto request_tag = request_id(owner_block_id,line->child_dimension);
          CHECK(MPI_Isend((void*)&block_request->request_buffer,2,MPI_INT,owner,request_tag,comms.request_comm,&request_request));
          requests.free(request_request);
          PENTAGO_MPI_TRACE("block request: owner %d, request_tag %d, response_tag %d",owner,request_tag,response_tag);
          requests.add(response_request,curry(&flow_t::process_response,this,block_request));
        }
        block_request->dependent_lines.push_back(line);
      }
    }
  }
}

void flow_t::post_barrier_recv() {
  PENTAGO_MPI_TRACE("post barrier recv");
  requests.add(countdown.barrier.irecv(),curry(&flow_t::process_barrier,this),true);
}

void flow_t::process_barrier(MPI_Status* status) {
  PENTAGO_MPI_TRACE("process barrier");
  countdown.barrier.process(*status);
  post_barrier_recv();
}

void flow_t::post_request_recv(Vector<int,2>* buffer) {
  PENTAGO_MPI_TRACE("post request recv");
  MPI_Request request;
  CHECK(MPI_Irecv((int*)buffer,2,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,comms.request_comm,&request));
  requests.add(request,curry(&flow_t::process_request,this,buffer),true);
}

void flow_t::process_request(Vector<int,2>* buffer, MPI_Status* status) {
  GEODE_ASSERT(input_blocks);
  const local_id_t local_block_id = request_block_id(status->MPI_TAG);
  const auto dimensions = request_dimensions(buffer);
  const int response_tag = request_response_tag(buffer);
  thread_time_t time(response_send_kind,input_blocks->local_block_lines_event(local_block_id,dimensions));
  PENTAGO_MPI_TRACE("process request: local block %d, dimensions %d",local_block_id.id,dimensions.data);
  // Send block data
  MPI_Request request;
#if PENTAGO_MPI_COMPRESS
  const auto compressed_data = input_blocks->get_compressed(local_block_id);
  CHECK(MPI_Isend((void*)compressed_data.data(),compressed_data.size(),MPI_BYTE,status->MPI_SOURCE,response_tag,comms.response_comm,&request));
#else
  const auto block_data = input_blocks->get_raw_flat(local_block_id);
  CHECK(MPI_Isend((void*)block_data.data(),8*block_data.size(),MPI_LONG_LONG_INT,status->MPI_SOURCE,response_tag,comms.response_comm,&request));
#endif
  PENTAGO_MPI_TRACE("block response: source %d, local block id %d, dimensions %d",status->MPI_SOURCE,local_block_id.id,dimensions.data);
  // The barrier tells us when all messages are finished, so we don't need this request
  requests.free(request);
  // Repost wildcard receive
  post_request_recv(buffer);
}

void flow_t::post_output_recv(Array<Vector<super_t,2>>* buffer) {
  PENTAGO_MPI_TRACE("post output recv");
  MPI_Request request;
  if (!buffer->size())
    *buffer = large_buffer<Vector<super_t,2>>(sqr(sqr(block_size))+PENTAGO_MPI_COMPRESS_OUTPUTS,false);
  GEODE_ASSERT(buffer->size()==sqr(sqr(block_size))+PENTAGO_MPI_COMPRESS_OUTPUTS);
  {
    thread_time_t time(mpi_kind,unevent);
    if (PENTAGO_MPI_COMPRESS_OUTPUTS)
      CHECK(MPI_Irecv(buffer->data(),memory_usage(*buffer),MPI_BYTE,MPI_ANY_SOURCE,MPI_ANY_TAG,comms.output_comm,&request));
    else
      CHECK(MPI_Irecv((uint64_t*)buffer->data(),8*buffer->size(),datatype<uint64_t>(),MPI_ANY_SOURCE,MPI_ANY_TAG,comms.output_comm,&request));
  }
  requests.add(request,curry(&flow_t::process_output,this,buffer),true);
}

// Absorb a compressed output block
static void absorb_compressed_output(accumulating_block_store_t* output_blocks, const local_id_t local_block_id, const uint8_t dimension, Array<const uint8_t> compressed, Array<Vector<super_t,2>> buffer) {
  // Uncompress block into temporary buffer, then copy back to buffer so that accumulate can use the temporary.
  const auto local_buffer = local_fast_uncompress(compressed,output_blocks->local_block_line_event(local_block_id,dimension));
  memcpy(buffer.data(),local_buffer.data(),memory_usage(local_buffer));
  const auto block_data = buffer.slice(0,local_buffer.size());
  // Send to block store
  output_blocks->accumulate(local_block_id,dimension,block_data);
}

// Incoming output data for a block
void flow_t::process_output(Array<Vector<super_t,2>>* buffer, MPI_Status* status) {
  {
    // How many elements did we receive?
    const int tag = status->MPI_TAG;
    const local_id_t local_block_id = request_block_id(tag);
    const uint8_t dimension = request_dimension(tag);
    const auto event = output_blocks.local_block_line_event(local_block_id,dimension);
    thread_time_t time(output_recv_kind,event);
    if (PENTAGO_MPI_COMPRESS_OUTPUTS) {
      const int count = get_count(status,MPI_BYTE);
      PENTAGO_MPI_TRACE("process output block: source %d, local block id %d, dimension %d, count %d, tag %d, event 0x%llx",status->MPI_SOURCE,local_block_id.id,dimension,count,tag,event);
      const auto compressed = char_view_own(*buffer).slice_own(0,count);
      // Schedule an accumulate as soon as possible to conserve memory
      threads_schedule(CPU,curry(absorb_compressed_output,&output_blocks,local_block_id,dimension,compressed,*buffer),true);
    } else {
      const int count = get_count(status,MPI_LONG_LONG_INT);
      PENTAGO_MPI_TRACE("process output block: source %d, local block id %d, dimension %d, count %g, tag %d, event 0x%llx",status->MPI_SOURCE,local_block_id.id,dimension,count/8.,tag,event);
      GEODE_ASSERT(!(count&7));
      const auto block_data = buffer->slice_own(0,count/8);
      // Schedule an accumulate as soon as possible to conserve memory
      threads_schedule(CPU,curry(&accumulating_block_store_t::accumulate,&output_blocks,local_block_id,dimension,block_data),true);
    }
    buffer->clean_memory();
  }
  // One step closer...
  progress.progress();
  countdown.decrement();
  post_output_recv(buffer);
}

#if !PENTAGO_MPI_FUNNEL
void flow_t::post_wakeup_recv() {
  PENTAGO_MPI_TRACE("post wakeup recv");
  MPI_Request request;
  CHECK(MPI_Irecv(&wakeup_buffer,1,MPI_LONG_LONG_INT,0,PENTAGO_MPI_COMPRESS_OUTPUTS?MPI_ANY_TAG:0,comms.wakeup_comm,&request));
  requests.add(request,curry(&flow_t::process_wakeup,this),true);
}
#endif

// Optionally compress an output block and send it
void flow_t::send_output(line_details_t* const line, const int b) {
  const auto block = line->pre.line.block(b);
  const auto owner_and_id = output_blocks.partition->find_block(line->pre.line.section,block);
  const int owner = owner_and_id.x;
  const local_id_t owner_block_id = owner_and_id.y;
  const int tag = request_id(owner_block_id,line->pre.line.dimension);
  const auto event = line->pre.line.block_line_event(b);
  MPI_Request request;
#if PENTAGO_MPI_COMPRESS_OUTPUTS
  // Send compressed block
  thread_time_t time(output_send_kind,event);
  const auto compressed = line->compressed_output_block_data(b);
  CHECK(MPI_Isend((void*)compressed.data(),compressed.size(),MPI_BYTE,owner,tag,comms.output_comm,&request));
  PENTAGO_MPI_TRACE("send output %p: owner %d, owner block id %d, dimension %d, count %d, tag %d, event 0x%llx",line,owner,owner_block_id.id,line->pre.line.dimension,compressed.size(),tag,event);
#else
  // Send without compression
  thread_time_t time(output_send_kind,event);
  const auto block_data = line->output_block_data(b);
  CHECK(MPI_Isend((void*)block_data.data(),8*block_data.size(),MPI_LONG_LONG_INT,owner,tag,comms.output_comm,&request));
  PENTAGO_MPI_TRACE("send output %p: owner %d, owner block id %d, dimension %d, count %d, tag %d, event 0x%llx",line,owner,owner_block_id.id,line->pre.line.dimension,block_data.size(),tag,event);
#endif
  requests.add(request,curry(&flow_t::finish_output_send,this,line));
}

// Line line has finished; post sends for all output blocks.  In compressed output mode, each wakeup corresponds to a single block.
void flow_t::wakeup(line_details_t* const line, const wakeup_block_t b) {
#if PENTAGO_MPI_COMPRESS_OUTPUTS
  PENTAGO_MPI_TRACE("process wakeup %p: %s, block %d",line,str(line->pre.line),b);
  GEODE_ASSERT(b<line->pre.line.length);
  // Send one compressed block
  send_output(line,b);
#else
  PENTAGO_MPI_TRACE("process wakeup %p: %s",line,str(line->pre.line));
  // Send all uncompressed output blocks
  for (int b=0;b<line->pre.line.length;b++)
    send_output(line,b);
#endif
}

static inline int wakeup_tag(int b)  { return b; }
static inline int wakeup_tag(unit b) { return 0; }

// Register a wakeup callback for the communication thread
void flow_t::post_wakeup(line_details_t& line, const wakeup_block_t b) {
#if PENTAGO_MPI_FUNNEL
  requests.add_immediate(curry(&flow_t::wakeup,this,&line,b));
#else
  BOOST_STATIC_ASSERT(sizeof(line_details_t*)==sizeof(long long int));
  // Send a pointer to ourselves to the communication thread
  MPI_Request request;
  CHECK(MPI_Isend((void*)&line.self,1,MPI_LONG_LONG_INT,0,wakeup_tag(b),comms.wakeup_comm,&request));
  // Since requests_t::free is not thread safe, we're forced to use MPI_Request_free here.
  // This is bad, because http://blogs.cisco.com/performance/mpi_request_free-is-evil.
  CHECK(MPI_Request_free(&request));
#endif
}

#if !PENTAGO_MPI_FUNNEL
// Process a wakeup message from a worker thread
void flow_t::process_wakeup(MPI_Status* status) {
  BOOST_STATIC_ASSERT(sizeof(line_details_t*)==sizeof(uint64_t) && sizeof(uint64_t)==sizeof(long long int));
  GEODE_ASSERT(get_count(status,MPI_LONG_LONG_INT)==1);
  line_details_t* const line = (line_details_t*)wakeup_buffer;
  wakeup(line,BOOST_PP_IF(PENTAGO_MPI_COMPRESS_OUTPUTS,status->MPI_TAG,unit()));
  post_wakeup_recv();
}
#endif

void flow_t::finish_output_send(line_details_t* const line, MPI_Status* status) {
  const int remaining = line->decrement_unsent_output_blocks();
  PENTAGO_MPI_TRACE("finish output send %p: %s: remaining %d",line,str(line->pre.line),remaining);
  if (!remaining) {
    PENTAGO_MPI_TRACE("deallocate line %p: %s",line,str(line->pre.line));
    const auto line_memory = line->pre.memory_usage;
    delete line;
    free_lines++;
    free_memory += line_memory;
    schedule_lines();
  }
  // One step closer...
  progress.progress();
  countdown.decrement();
}

static void absorb_response(block_request_t* request, const int recv_size) {
  const int lines = CHECK_CAST_INT(request->dependent_lines.size());
  GEODE_ASSERT(lines);
  const auto first_line = request->dependent_lines[0];
  const auto first_block_data = first_line->input_block_data(request->block);
  const int nodes = first_block_data.size()-PENTAGO_MPI_COMPRESS;
  const auto event = request->block_lines_event();

#if PENTAGO_MPI_COMPRESS
  // Uncompress data into a temporary buffer, then copy back to first dependent line
  const auto buffer = local_fast_uncompress(char_view(first_block_data).slice(0,recv_size),event);
  memcpy(first_block_data.data(),buffer.data(),memory_usage(buffer));
#endif

  // Copy data to dependent lines other than the first
  if (lines>1)
    for (int i=1;i<lines;i++) {
      const auto line = request->dependent_lines[i];
      const auto block_data = line->input_block_data(request->block);
      GEODE_ASSERT(block_data.size()==nodes+PENTAGO_MPI_COMPRESS);
      memcpy(block_data.data(),first_block_data.data(),sizeof(Vector<super_t,2>)*nodes);
      line->decrement_missing_input_blocks();
    }

  // Decrement here so that the line doesn't deallocate itself before we copy the data to other lines
  first_line->decrement_missing_input_blocks();

  // Deallocate request
  delete request;
}

void flow_t::process_response(block_request_t* request, MPI_Status* status) {
  {
    thread_time_t time(response_recv_kind,request->block_lines_event());
    PENTAGO_MPI_TRACE("process response: owner %d, owner block id %d, dimension %d",status->MPI_SOURCE,request_block_id(status->MPI_TAG).id,request->dimensions.data);

    // Erase block request
    const int index = block_requests.find(request);
    GEODE_ASSERT(block_requests.valid(index));
    block_requests.remove_index_lazy(index);

    // Decrement input response counters
    for (auto line : request->dependent_lines)
      if (!line->decrement_input_responses())
        free_line_gathers++;

    // Data has already been received into the first dependent line, but may be compressed.
    // Schedule a decompression and/or copying job.  No need to put this at the front of the queue,
    // since there's no memory to be deallocated.
    const int recv_size = get_count(status,MPI_BYTE);
    threads_schedule(CPU,curry(absorb_response,request,recv_size));
  }

  // We may be able to schedule more lines if any line gathers completed
  schedule_lines();
}

void compute_lines(const flow_comms_t& comms, const Ptr<const readable_block_store_t> input_blocks, accumulating_block_store_t& output_blocks, RawArray<const line_t> lines, const uint64_t memory_limit, const int line_gather_limit, const int line_limit) {
  // Everything happens in this helper class
  flow_t(comms,input_blocks,output_blocks,lines,memory_limit,line_gather_limit,line_limit);
}

}
}
