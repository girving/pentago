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

finished_lines_t::finished_lines_t() {}

finished_lines_t::~finished_lines_t() {
  if (lines.size())
    die("finished_lines_t destructed before all lines were complete");
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

void compute_lines(const MPI_Comm comm, const Ptr<const block_store_t> input_blocks, block_store_t& output_blocks, Array<const line_t> lines, const uint64_t memory_limit) {
  const int rank = comm_rank(comm);
  const int block_size = output_blocks.partition->block_size;

  /* The communication loop involves the following kind of messages:
   *
   * 1. Barrier messages: passed down to our ibarrier_t
   * 2. Block requests: one rank asking another for an input block
   * 3. Block responses: response to a block request, containing input block data
   * 4. Output block: one rank sending output block data to the owner of the block
   *
   * Each tag has the form kind_base*kind + (local block id on owner rank)
   * For barrier messages, the block id is zero.
   */
  const int kind_base = max(input_blocks?input_blocks->partition->max_rank_blocks:0,output_blocks.partition->max_rank_blocks)+1,
            barrier_kind = 1,
            request_kind = 2,
            response_kind = 3,
            output_kind = 4;

  // Keep track out how much more stuff has to happen.  This includes both blocks we need to send and those we need to receive.
  ibarrier_countdown_t countdown(comm,barrier_kind*kind_base,total_blocks(lines)+output_blocks.required_contributions);

  // Compute information about each line
  Array<line_data_t*> unscheduled_lines;
  unscheduled_lines.preallocate(lines.size());
  for (int i=lines.size()-1;i>=0;i--) {
    const auto line = new line_data_t(lines[i],block_size);
    OTHER_ASSERT(line->memory_usage()<=memory_limit/2);
    unscheduled_lines.append_assuming_enough_space(line);
  }

  // List of pending block requests, with internal references to the lines that depend on them.
  unordered_map<uint64_t,block_request_t*> block_requests;

  // List of pending input receives and output sends
  requests_t requests;

  // List of completed lines ready to be sent out
  finished_lines_t finished_lines;
  vector<line_data_t*> to_send;

  // Enter communication loop
  uint64_t memory_usage = 0;
  for (;;) {
    // If there are unscheduled lines, try to schedule them
    while (unscheduled_lines.size()) {
      const auto line = unscheduled_lines.last();
      const auto line_memory = line->memory_usage();
      if (memory_limit < memory_usage+line_memory)
        break;
      // Schedule the line
      unscheduled_lines.pop();
      line->allocate(finished_lines);
      memory_usage += line_memory;
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
            CHECK(MPI_Send(0,0,MPI_INT,owner,request_kind*kind_base+owner_block_id,comm));
            it = block_requests.insert(make_pair(block_id,block_request)).first;
          }
          it->second->dependent_lines.push_back(line);
        }
      }
    }

    // If there are finished lines, send them all out
    {
      spin_t spin(finished_lines.lock); 
      swap(to_send,finished_lines.lines);
    }
    for (const auto line : to_send) {
      // Note: We don't need to protect memory_usage with a spin lock since decrement_unsent is only called from this thread.
      const function<void()> decrement_unsent(boost::bind(&line_data_t::decrement_unsent_output_blocks,line,&memory_usage));
      for (int b=0;b<line->line.length;b++) {
        // Post output send
        const auto block = line->line.block(b);
        const auto block_id = output_blocks.partition->block_offsets(line->line.section,block).x;
        const auto block_data = line->output_block_data(b);
        const int owner = output_blocks.partition->block_to_rank(line->line.section,block);
        const auto owner_offsets = output_blocks.partition->rank_offsets(owner);
        const int owner_block_id = block_id-owner_offsets.x;
        MPI_Request request;
        CHECK(MPI_Isend((void*)block_data.data(),(size_t)8*block_data.size(),MPI_LONG_LONG_INT,owner,output_kind*kind_base+owner_block_id,comm,&request));
        requests.add(request,decrement_unsent);
      }
    }
    to_send.clear();

    // Finish as many requests as possible
    requests.test();

    // Otherwise, check for an incoming message
    int flag;
    MPI_Status status;
    CHECK(MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,comm,&flag,&status));
    if (flag) {
      const int kind = status.MPI_TAG/kind_base,
                local_block_id = status.MPI_TAG-kind*kind_base;
      switch (kind) {
        // Barrier synchronization message
        case barrier_kind:
          countdown.barrier.recv();
          if (countdown.barrier.done())
            goto done;
          break;

        // Request for input block data
        case request_kind: {
          OTHER_ASSERT(input_blocks);
          // Receive empty message
          CHECK(MPI_Recv(0,0,MPI_INT,status.MPI_SOURCE,status.MPI_TAG,comm,MPI_STATUS_IGNORE));
          // Send block data
          const auto block_data = input_blocks->get(local_block_id);
          MPI_Request request;
          const int tag = response_kind*kind_base+local_block_id;
          CHECK(MPI_Isend((void*)block_data.data(),(size_t)8*block_data.size(),MPI_LONG_LONG_INT,status.MPI_SOURCE,tag,comm,&request));
          // The barrier tells us when all messages are finished, so we don't need this request
          CHECK(MPI_Request_free(&request));
          break; }

        // Response to block request, containing input data
        case response_kind: {
          OTHER_ASSERT(input_blocks);
          // Look up block request
          const auto source_offsets = input_blocks->partition->rank_offsets(status.MPI_SOURCE);
          const auto block_id = source_offsets.x+local_block_id;
          const auto it = block_requests.find(block_id);
          if (it == block_requests.end())
            die(format("rank %d: other rank %d sent an unrequested block %lld",rank,status.MPI_SOURCE,block_id));
          const auto request = it->second;
          block_requests.erase(it);
          // Receive block data into first dependent line
          OTHER_ASSERT(request->dependent_lines.size());
          const auto first_line = request->dependent_lines[0]; // Hold a reference to first_line to make sure it lives until we copy the input block to other lines
          const auto first_block_data = first_line->input_block_data(request->block);
          CHECK(MPI_Recv(first_block_data.data(),(size_t)8*first_block_data.size(),MPI_LONG_LONG_INT,status.MPI_SOURCE,status.MPI_TAG,comm,MPI_STATUS_IGNORE));
          countdown.decrement();
          // Copy data to other dependent lines
          for (int i=1;i<(int)request->dependent_lines.size();i++) {
            const auto& line = request->dependent_lines[i];
            const auto block_data = line->input_block_data(request->block);
            OTHER_ASSERT(block_data.size()==first_block_data.size());
            memcpy(block_data.data(),first_block_data.data(),sizeof(Vector<super_t,2>)*first_block_data.size());
            line->decrement_missing_input_blocks();
            countdown.decrement();
          }
          // Decrement here so that the line doesn't deallocate itself before we copy the data to other lines
          first_line->decrement_missing_input_blocks();
          // Deallocate request
          delete request;
          break; }

        // Incoming output data for a block
        case output_kind: {
          // Receive new data
          OTHER_ASSERT(local_block_id<output_blocks.blocks());
          const auto& info = output_blocks.block_info[local_block_id];
          const auto block_data = aligned_buffer<Vector<super_t,2>>(block_shape(info.section.shape(),info.block,block_size).product());
          CHECK(MPI_Recv(block_data.data(),(size_t)8*block_data.size(),MPI_LONG_LONG_INT,status.MPI_SOURCE,status.MPI_TAG,comm,MPI_STATUS_IGNORE));
          // Schedule accumulate
          threads_schedule(CPU,boost::bind(&block_store_t::accumulate,&output_blocks,local_block_id,block_data));
          // One step closer...
          countdown.decrement();
          break; }

        // Unknown message kind
        default:
          die(format("compute_lines: strange message with tag = %d, kind = %d",status.MPI_TAG,kind));
      }
    }
  }

  // Finish up
  done:
  OTHER_ASSERT(!unscheduled_lines.size());
  OTHER_ASSERT(!block_requests.size());
  threads_wait_all();
  output_blocks.set_complete();
}

}
}
