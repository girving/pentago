// Endgame computation structure code with interleaved communication and compute

#include <pentago/mpi/flow.h>
#include <pentago/mpi/compute.h>
#include <pentago/mpi/ibarrier.h>
#include <pentago/mpi/requests.h>
#include <pentago/mpi/utility.h>
#include <pentago/thread.h>
#include <pentago/utility/aligned.h>
#include <other/core/array/Array4d.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/Hasher.h>
#include <boost/noncopyable.hpp>
#include <boost/bind.hpp>
#include <tr1/unordered_map>
namespace pentago {
namespace mpi {

using std::make_pair;
using std::tr1::unordered_map;

namespace {
struct block_request_t : public boost::noncopyable {
  // The section is determined by the dependent line, so we don't need to store it
  const Vector<int,4> block;
  vector<line_data_t*> dependent_lines;

  block_request_t(Vector<int,4> block)
    : block(block) {}
};
}

static int total_blocks(RawArray<const line_t> lines) {
  int count = 0;
  for (const auto& line : lines)
    count += line.length;
  return count;
}

uint64_t base_compute_memory_usage(const int lines) {
  return (2*sizeof(void*)+sizeof(line_data_t))*lines;
}

flow_comms_t::flow_comms_t(MPI_Comm comm)
  : rank(comm_rank(comm))
  , barrier_comm(comm_dup(comm))
  , request_comm(comm_dup(comm))
  , response_comm(comm_dup(comm))
  , output_comm(comm_dup(comm))
  , wakeup_comm(comm_dup(comm)) {}

flow_comms_t::~flow_comms_t() {
  CHECK(MPI_Comm_free(&barrier_comm));
  CHECK(MPI_Comm_free(&request_comm));
  CHECK(MPI_Comm_free(&response_comm));
  CHECK(MPI_Comm_free(&output_comm));
  CHECK(MPI_Comm_free(&wakeup_comm));
}

namespace {

struct flow_t {
  const flow_comms_t& comms;
  const Ptr<const block_store_t> input_blocks;
  block_store_t& output_blocks;
  const int block_size;

  // Pending requests, including wildcard requests for responding to block requests and output messages
  requests_t requests;

  // Lines which haven't yet been allocated and unscheduled
  Array<line_data_t*> unscheduled_lines;

  // Pending block requests, with internal references to the lines that depend on them.
  unordered_map<uint64_t,block_request_t*> block_requests;

  // Keep track out how much more stuff has to happen.  This includes both blocks we need to send and those we need to receive.
  ibarrier_countdown_t countdown;

  // Current free memory
  uint64_t free_memory;

  // Space for persistent message requests
  const Array<Vector<super_t,2>> output_buffer;
  uint64_t wakeup_buffer;

  // Wildcard callbacks
  function<void(MPI_Status*)> barrier_callback, request_callback, output_callback, wakeup_callback, response_callback;

  flow_t(const flow_comms_t& comms, const Ptr<const block_store_t> input_blocks, block_store_t& output_blocks, RawArray<const line_t> lines, const uint64_t memory_limit);
  ~flow_t();

  void schedule_lines();
  void post_barrier_recv();
  void post_request_recv();
  void post_output_recv();
  void post_wakeup_recv();
  void process_barrier(MPI_Status* status);
  void process_request(MPI_Status* status);
  void process_output(MPI_Status* status);
  void process_wakeup(MPI_Status* status);
  void process_response(MPI_Status* status);
  void finish_output_send(line_data_t* const line, MPI_Status* status);
};

flow_t::flow_t(const flow_comms_t& comms, const Ptr<const block_store_t> input_blocks, block_store_t& output_blocks, RawArray<const line_t> lines, const uint64_t memory_limit)
  : comms(comms)
  , input_blocks(input_blocks)
  , output_blocks(output_blocks)
  , block_size(output_blocks.partition->block_size)
  , countdown(comms.barrier_comm,barrier_tag,total_blocks(lines)+output_blocks.required_contributions)
  , free_memory(memory_limit)
  , output_buffer(aligned_buffer<Vector<super_t,2>>(sqr(sqr(block_size))))
  , barrier_callback(boost::bind(&flow_t::process_barrier,this,_1))
  , request_callback(boost::bind(&flow_t::process_request,this,_1))
  , output_callback(boost::bind(&flow_t::process_output,this,_1))
  , wakeup_callback(boost::bind(&flow_t::process_wakeup,this,_1))
  , response_callback(boost::bind(&flow_t::process_response,this,_1)) {

  // Compute information about each line
  unscheduled_lines.preallocate(lines.size());
  for (int i=lines.size()-1;i>=0;i--) {
    const auto line = new line_data_t(lines[i],block_size);
    OTHER_ASSERT(line->memory_usage()<=memory_limit/2);
    unscheduled_lines.append_assuming_enough_space(line);
  }

  // Schedule some lines
  schedule_lines();

  // Start wildcard receives
  post_barrier_recv();
  post_request_recv();
  post_output_recv();
  post_wakeup_recv();

  // Enter communication loop
  while (!countdown.barrier.done())
    requests.waitsome();

  // Cancel the wildcard receives
  requests.cancel_and_waitall();

  // Finish up
  OTHER_ASSERT(!unscheduled_lines.size());
  OTHER_ASSERT(!block_requests.size());
  threads_wait_all();
  output_blocks.set_complete();
}

flow_t::~flow_t() {}

void flow_t::schedule_lines() {
  // If there are unscheduled lines, try to schedule them
  while (unscheduled_lines.size()) {
    const auto line = unscheduled_lines.last();
    const auto line_memory = line->memory_usage();
    if (free_memory < line_memory)
      break;
    // Schedule the line
    unscheduled_lines.pop();
    line->allocate(comms.wakeup_comm);
    free_memory -= line_memory;
    // Request all input blocks
    const int input_count = line->input_blocks();
    if (!input_count)
      schedule_compute_line(*line);
    else {
      OTHER_ASSERT(input_blocks);
      const auto child_section = line->standard_child_section();
      for (int b : range(input_count)) {
        const auto block = line->input_block(b);
        const auto block_id = input_blocks->partition->block_offsets(child_section,block).x;
        auto it = block_requests.find(block_id);
        if (it == block_requests.end()) {
          // Send a block request message
          const auto block_request = new block_request_t(block);
          const int owner = input_blocks->partition->block_to_rank(child_section,block);
          const auto owner_offsets = input_blocks->partition->rank_offsets(owner);
          const int owner_block_id = block_id-owner_offsets.x;
          CHECK(MPI_Send(0,0,MPI_INT,owner,owner_block_id,comms.request_comm));
          it = block_requests.insert(make_pair(block_id,block_request)).first;
          // Post a receive for the request
          const auto block_data = line->input_block_data(b);
          MPI_Request request;
          CHECK(MPI_Irecv(block_data.data(),8*block_data.size(),MPI_LONG_LONG_INT,owner,owner_block_id,comms.response_comm,&request));
          requests.add(request,response_callback);
        }
        it->second->dependent_lines.push_back(line);
      }
    }
  }
}

void flow_t::post_barrier_recv() {
  requests.add(countdown.barrier.irecv(),barrier_callback,true);
}

void flow_t::process_barrier(MPI_Status* status) {
  countdown.barrier.process(*status);
  post_barrier_recv();
}

void flow_t::post_request_recv() {
  MPI_Request request;
  CHECK(MPI_Irecv(0,0,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,comms.request_comm,&request));
  requests.add(request,request_callback,true);
}

void flow_t::process_request(MPI_Status* status) {
  OTHER_ASSERT(input_blocks);
  const int local_block_id = status->MPI_TAG;
  // Send block data
  const auto block_data = input_blocks->get(local_block_id);
  MPI_Request request;
  CHECK(MPI_Isend((void*)block_data.data(),8*block_data.size(),MPI_LONG_LONG_INT,status->MPI_SOURCE,local_block_id,comms.response_comm,&request));
  // The barrier tells us when all messages are finished, so we don't need this request
  CHECK(MPI_Request_free(&request));
  // Repost wildcard receive
  post_request_recv();
}

void flow_t::post_output_recv() {
  MPI_Request request;
  CHECK(MPI_Irecv(output_buffer.data(),8*output_buffer.size(),MPI_LONG_LONG_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,comms.output_comm,&request));
  requests.add(request,output_callback,true);
}

// Incoming output data for a block
void flow_t::process_output(MPI_Status* status) {
  // How many elements did we receive?
  int count = get_count(*status,MPI_LONG_LONG_INT);
  OTHER_ASSERT(!(count&7));
  const auto block_data = output_buffer.slice(0,count/8);
  // For now, do the accumulate in this thread
  output_blocks.accumulate(status->MPI_TAG,block_data);
  // One step closer...
  countdown.decrement();
  post_output_recv();
}

void flow_t::post_wakeup_recv() {
  MPI_Request request;
  CHECK(MPI_Irecv(&wakeup_buffer,1,MPI_LONG_LONG_INT,comms.rank,wakeup_tag,comms.wakeup_comm,&request));
  requests.add(request,wakeup_callback,true);
}

// Line line has finished; post sends for all output blocks
void flow_t::process_wakeup(MPI_Status* status) {
  BOOST_STATIC_ASSERT(sizeof(line_data_t*)==sizeof(uint64_t) && sizeof(uint64_t)==sizeof(long long int));
  OTHER_ASSERT(get_count(*status,MPI_LONG_LONG_INT)==1);
  line_data_t* const line = (line_data_t*)wakeup_buffer; 
  const function<void(MPI_Status*)> callback(boost::bind(&flow_t::finish_output_send,this,line,_1));
  for (int b=0;b<line->line.length;b++) {
    const auto block = line->line.block(b);
    const auto block_id = output_blocks.partition->block_offsets(line->line.section,block).x;
    const auto block_data = line->output_block_data(b);
    const int owner = output_blocks.partition->block_to_rank(line->line.section,block);
    const auto owner_offsets = output_blocks.partition->rank_offsets(owner);
    const int owner_block_id = block_id-owner_offsets.x;
    MPI_Request request;
    CHECK(MPI_Isend((void*)block_data.data(),8*block_data.size(),MPI_LONG_LONG_INT,owner,owner_block_id,comms.output_comm,&request));
    requests.add(request,callback);
  }
}

void flow_t::finish_output_send(line_data_t* const line, MPI_Status* status) {
  if (!line->decrement_unsent_output_blocks()) {
    free_memory -= line->memory_usage();
    delete line;
    schedule_lines();
  }
  // One step closer...
  countdown.decrement();
}

void flow_t::process_response(MPI_Status* status) {
  // Look up block request
  OTHER_ASSERT(input_blocks);
  const auto block_id = input_blocks->partition->rank_offsets(status->MPI_SOURCE).x+status->MPI_TAG;
  const auto it = block_requests.find(block_id);
  if (it == block_requests.end())
    die(format("other rank %d sent an unrequested block %lld",status->MPI_SOURCE,block_id));
  const auto request = it->second;
  block_requests.erase(it);

  // Data has already been received into the first dependent line.  Copy data to other dependent lines
  OTHER_ASSERT(request->dependent_lines.size());
  const auto first_line = request->dependent_lines[0];
  const auto first_block_data = first_line->input_block_data(request->block);
  for (int i=1;i<(int)request->dependent_lines.size();i++) {
    const auto& line = request->dependent_lines[i];
    const auto block_data = line->input_block_data(request->block);
    OTHER_ASSERT(block_data.size()==first_block_data.size());
    memcpy(block_data.data(),first_block_data.data(),sizeof(Vector<super_t,2>)*first_block_data.size());
    line->decrement_missing_input_blocks();
  }

  // Decrement here so that the line doesn't deallocate itself before we copy the data to other lines
  first_line->decrement_missing_input_blocks();

  // Deallocate request
  delete request;
}

}

void compute_lines(const flow_comms_t& comms, const Ptr<const block_store_t> input_blocks, block_store_t& output_blocks, Array<const line_t> lines, const uint64_t memory_limit) {
  // Everything happens in this helper class
  flow_t(comms,input_blocks,output_blocks,lines,memory_limit);
}

}
}