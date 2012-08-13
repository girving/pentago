// Core compute kernels for the MPI code

#include <pentago/mpi/compute.h>
#include <pentago/mpi/utility.h>
#include <pentago/mpi/flow.h>
#include <pentago/mpi/trace.h>
#include <pentago/endgame.h>
#include <pentago/utility/ceil_div.h>
#include <pentago/utility/counter.h>
#include <pentago/utility/index.h>
#include <pentago/utility/memory.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/math/integer_log.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/Log.h>
#include <boost/bind.hpp>
namespace pentago {
namespace mpi {

using Log::cout;
using std::endl;

/*********************** line_data_t ************************/

line_data_t::line_data_t(const line_t& line)
  : line(line) {

  // Compute input and output shapes
  const int dim = line.dimension;
  const auto section_shape = line.section.shape();
  const auto child_section = line.section.child(dim);
  const auto child_section_shape = child_section.shape();
  OTHER_ASSERT(section_shape.remove_index(dim)==child_section_shape.remove_index(dim));
  auto output_shape = block_shape(section_shape,line.block(0));
  output_shape[dim] = section_shape[dim];
  auto input_shape = output_shape;
  input_shape[dim] = child_section_shape[dim];
  if (line.section.sum()==35) // We never need to compute the 36 stone slice
    input_shape = Vector<int,4>();
  const_cast_(this->input_shape) = input_shape;
  const_cast_(this->output_shape) = output_shape;
}

line_data_t::~line_data_t() {}

uint64_t line_data_t::memory_usage() const {
  return sizeof(Vector<super_t,2>)*(input_shape.product()+output_shape.product());
}

struct allocated_t : public boost::noncopyable {
  // Standardization
  const section_t standard_child_section;
  const uint8_t section_transform; // Maps output space to input (child) space
  const Vector<int,4> permutation; // Child quadrant i maps to quadrant permutation[i]
  const int child_dimension;
  const int child_length;
  const Vector<int,4> first_child_block; // First child block
  const symmetry_t inverse_transform;

  // Rotation minimal quadrants
  const Vector<RawArray<const quadrant_t>,4> rmin;

  // Symmetries needed to restore minimality after reflection
  const Array<const local_symmetry_t> all_reflection_symmetries;
  const Vector<int,5> reflection_symmetry_offsets;

  // Number of blocks we need before input is ready, microlines left to compute, and output blocks left to send.
  counter_t missing_input_blocks;
  counter_t missing_microlines;
  counter_t unsent_output_blocks;

  // Information needed to account for reflections in input block data
  const Vector<int,4> reflection_moves;

  // Input and output data.  Both are stored in 5D order where the first dimension
  // is the block, to avoid copying before and after compute.
  const Array<Vector<super_t,2>> input, output;

  // When computation is complete, send a wakeup message here
  const MPI_Comm wakeup_comm;
  const line_data_t* const self; // Pointer back to containing line for use as a send buffer

  allocated_t(const line_data_t& self, const MPI_Comm wakeup_comm);
};

// Mark this const so that the two identical calls in allocated_t's constructor can be CSE'ed
OTHER_CONST static inline Tuple<section_t,uint8_t> standardize_child_section(section_t section, int dimension) {
  return section.child(dimension).standardize<8>();
}

allocated_t::allocated_t(const line_data_t& self, const MPI_Comm wakeup_comm)
  // Standardize
  : standard_child_section(standardize_child_section(self.line.section,self.line.dimension).x)
  , section_transform(     standardize_child_section(self.line.section,self.line.dimension).y)
  , permutation(section_t::quadrant_permutation(symmetry_t::invert_global(section_transform)))
  , child_dimension(permutation.find(self.line.dimension&3))
  , child_length(ceil_div(self.input_shape[self.line.dimension],block_size))
  , first_child_block(self.line.block_base.insert(0,self.line.dimension&3).subset(permutation))

  // Prepare a transform that rotates quadrants globally while preserving their orientation, reflecting first if necessary.
  , inverse_transform(symmetry_t((4-(section_transform&3))&3,(1+4+16+64)*(section_transform&3))*symmetry_t(section_transform&4,0))

  // Rotation minimal quadrants
  , rmin(rotation_minimal_quadrants(self.line.section.counts[0]).x,
         rotation_minimal_quadrants(self.line.section.counts[1]).x,
         rotation_minimal_quadrants(self.line.section.counts[2]).x,
         rotation_minimal_quadrants(self.line.section.counts[3]).x)

  // Numbers of steps to complete
  , missing_input_blocks(child_length)
  , missing_microlines(self.output_shape.remove_index(self.line.dimension).product())
  , unsent_output_blocks(self.line.length)

  // Allocate memory for both input and output in a single buffer
  , input(large_buffer<Vector<super_t,2>>(self.input_shape.product()+self.output_shape.product(),false))

  // When computation is complete, send a wakeup message here
  , wakeup_comm(wakeup_comm)
  , self(&self) {

  // Split buffer into two pieces
  const int split = self.input_shape.product();
  const_cast_(output) = input.slice_own(split,input.size());
  const_cast_(input) = input.slice_own(0,split);

  // If there's a reflection, prepare to rearrange data accordingly
  const auto child_rmin = vec(rotation_minimal_quadrants(standard_child_section.counts[0]),
                              rotation_minimal_quadrants(standard_child_section.counts[1]),
                              rotation_minimal_quadrants(standard_child_section.counts[2]),
                              rotation_minimal_quadrants(standard_child_section.counts[3]));
  if (section_transform & 4)
    const_cast_(reflection_moves) = vec(child_rmin[0].y,child_rmin[1].y,child_rmin[2].y,child_rmin[3].y);

  // Compute symmetries needed to restore minimality after reflection
  const Vector<int,4> standard_input_shape(self.input_shape.subset(permutation));
  if (standard_input_shape != Vector<int,4>()) {
    const_cast_(all_reflection_symmetries) = Array<local_symmetry_t>(standard_input_shape.sum());
    const_cast_(reflection_symmetry_offsets[0]) = 0;
    int offset = 0;
    for (int i=0;i<4;i++) {
      RawArray<local_symmetry_t> symmetries = all_reflection_symmetries.const_cast_().slice(offset+range(standard_input_shape[i]));
      offset += standard_input_shape[i];
      const_cast_(reflection_symmetry_offsets[i+1]) = offset;
      if (section_transform & 4)
        for (int j=0;j<standard_input_shape[i];j++) {
          const quadrant_t q = child_rmin[i].x[block_size*first_child_block[i]+j];
          symmetries[j].local = (rotation_minimal_quadrants_inverse[pack(reflections[unpack(q,0)],reflections[unpack(q,1)])]&3)<<2*i;
        }
    }
  }
}

void line_data_t::allocate(const MPI_Comm wakeup_comm) {
  rest.reset(new allocated_t(*this,wakeup_comm));
}

section_t line_data_t::standard_child_section() const {
  return rest->standard_child_section;
}

int line_data_t::input_blocks() const {
  return rest->child_length;
}

Vector<int,4> line_data_t::input_block(int k) const {
  OTHER_ASSERT((unsigned)k<(unsigned)rest->child_length);
  auto block = rest->first_child_block;
  block[rest->child_dimension&3] = k;
  return block;
}

RawArray<Vector<super_t,2>> line_data_t::input_block_data(int k) const {
  OTHER_ASSERT((unsigned)k<(unsigned)rest->child_length);
  const int start = line.node_step*k;
  return rest->input.slice(start,min(start+line.node_step,rest->input.size()));
}

RawArray<Vector<super_t,2>> line_data_t::input_block_data(Vector<int,4> block) const {
  const int child_dim = rest->child_dimension;
  OTHER_ASSERT(block.remove_index(child_dim)==rest->first_child_block.remove_index(child_dim));
  return input_block_data(block[child_dim]);
}

RawArray<const Vector<super_t,2>> line_data_t::output_block_data(int k) const {
  OTHER_ASSERT(0<=k && k<line.length);
  const int start = line.node_step*k;
  return rest->output.slice(start,min(start+line.node_step,rest->output.size()));
}

void line_data_t::decrement_missing_input_blocks() {
  // Are we ready to compute?
  if (!--rest->missing_input_blocks)
    schedule_compute_line(*this);
}

int line_data_t::decrement_unsent_output_blocks() {
  return --rest->unsent_output_blocks;
}

/*********************** slow_verify ************************/

static spinlock_t slow_verify_lock;

static void slow_verify(const char* prefix, const bool turn, const side_t side0, const side_t side1, const Vector<super_t,2>& result, bool verbose=false) {
  // Switch from (we-win,we-win-or-tie) format to (black-wins,white-wins) format
  auto data = result; 
  data.y = ~data.y;
  if (turn)
    swap(data.x,data.y);
  const auto board = turn?pack(side1,side0):pack(side0,side1);
  spin_t spin(slow_verify_lock);
  endgame_verify_board(prefix,board,data,verbose);
}

static OTHER_UNUSED void slow_verify(const char* prefix, const board_t board, const Vector<super_t,2>& result, bool verbose=false) {
  const bool turn = count(board).sum()&1;
  slow_verify(prefix,turn,unpack(board,turn),unpack(board,1-turn),result,verbose);
}

/*********************** compute_microline ************************/

// Compute a single 1D line through a section (a 1D component of a block line)
template<bool slice_35> static void compute_microline(line_data_t* const line, const Vector<int,3> base) {
  // Prepare
  thread_time_t time("compute");
  const int dim = line->line.dimension & 3; // Tell compiler that 0<=dim<4
  const int length = line->line.length;
  const int child_dim = line->rest->child_dimension & 3;
  const auto full_base = base.insert(0,dim);
  auto& rest = *line->rest;
  const int child_length = rest.child_length;
  const auto first_block = line->line.block_base.insert(0,dim);
  const auto first_node = block_size*first_block+full_base;
  Vector<quadrant_t,4> base_quadrants(rest.rmin[0][first_node[0]],
                                      rest.rmin[1][first_node[1]],
                                      rest.rmin[2][first_node[2]],
                                      rest.rmin[3][first_node[3]]);
  base_quadrants[dim] = 0;
  const auto base_board = quadrants(base_quadrants[0],base_quadrants[1],base_quadrants[2],base_quadrants[3]);
  const bool turn = line->line.section.sum()&1;
  const auto base_side0 = unpack(base_board,turn),
             base_side1 = unpack(base_board,1-turn);
  const auto rmin = rest.rmin[dim];
  OTHER_ASSERT(rmin.size()==line->output_shape[dim]);
  const int dim_shift = 16*dim;

  // Prepare to index into output array
  const int block_stride = line->line.node_step;
  auto output_block_shape = line->output_shape;
  output_block_shape[dim] = block_size;
  const auto output_block_strides = strides(output_block_shape);
  const int output_stride = output_block_strides[dim];
  const int output_base = dot(output_block_strides,full_base); // Base if we're before the last block
  const RawArray<Vector<super_t,2>> output = rest.output;

  // Prepare to index into input array
  Vector<int,4> input_block_shape(output_block_shape.subset(rest.permutation&3));
  const auto input_block_strides = strides(input_block_shape);
  const int input_stride = input_block_strides[child_dim];
  const Vector<int,4> full_child_base(full_base.subset(rest.permutation&3));
  const auto first_child_node = block_size*rest.first_child_block+full_child_base;
  const auto reflected_child_base = full_child_base^Vector<int,4>(child_dim!=0 && first_child_node[0]<rest.reflection_moves[0],
                                                                  child_dim!=1 && first_child_node[1]<rest.reflection_moves[1],
                                                                  child_dim!=2 && first_child_node[2]<rest.reflection_moves[2],
                                                                  child_dim!=3 && first_child_node[3]<rest.reflection_moves[3]);
  const int input_base = dot(input_block_strides,reflected_child_base); // Base if we're before the last block
  const RawArray<const Vector<super_t,2>> input = rest.input;
  const int line_moves = rest.reflection_moves[child_dim];

  // If we hit the last block in either input or output, the base indices change
  output_block_shape[dim] = rmin.size()&(block_size-1)?:block_size;
  const int last_output_base = dot(strides(output_block_shape),full_base);
  input_block_shape[child_dim] = line->input_shape[dim]&(block_size-1)?:block_size;
  const int last_input_base = dot(strides(input_block_shape),reflected_child_base); // Base if we're in the last block

  // Account for symmetries
  const auto rs_offsets = rest.reflection_symmetry_offsets;
  const auto symmetries = rest.all_reflection_symmetries.slice(rs_offsets[child_dim],rs_offsets[child_dim+1]);
  const local_symmetry_t local_symmetry(rest.all_reflection_symmetries.size()? (child_dim!=0)*rest.all_reflection_symmetries[rs_offsets[0]+full_child_base[0]].local
                                                                              +(child_dim!=1)*rest.all_reflection_symmetries[rs_offsets[1]+full_child_base[1]].local
                                                                              +(child_dim!=2)*rest.all_reflection_symmetries[rs_offsets[2]+full_child_base[2]].local
                                                                              +(child_dim!=3)*rest.all_reflection_symmetries[rs_offsets[3]+full_child_base[3]].local
                                                                             :0);
  const symmetry_t base_transform = rest.inverse_transform*local_symmetry;

#ifdef PENTAGO_MPI_DEBUG
  // Check that child data is consistent before and after transformation
  const auto child_rmin = vec(rotation_minimal_quadrants(rest.standard_child_section.counts[0]),
                              rotation_minimal_quadrants(rest.standard_child_section.counts[1]),
                              rotation_minimal_quadrants(rest.standard_child_section.counts[2]),
                              rotation_minimal_quadrants(rest.standard_child_section.counts[3]));
  const auto reflected_first_child_node = block_size*rest.first_child_block+reflected_child_base;
  const auto child_board_base = quadrants((child_dim!=0)*child_rmin[0].x[reflected_first_child_node[0]],
                                          (child_dim!=1)*child_rmin[1].x[reflected_first_child_node[1]],
                                          (child_dim!=2)*child_rmin[2].x[reflected_first_child_node[2]],
                                          (child_dim!=3)*child_rmin[3].x[reflected_first_child_node[3]]);
  for (int j : range(line->input_shape[dim])) {
    const auto& child = input[((j>>block_shift)==child_length-1?last_input_base:input_base)+block_stride*(j>>block_shift)+input_stride*((j^(j<line_moves))&(block_size-1))];
    auto child_index = reflected_child_base;
    child_index[child_dim] = j^(j<line_moves);
    const auto child_node = block_size*rest.first_child_block+child_index;
    const auto board = quadrants(child_rmin[0].x[child_node[0]],
                                 child_rmin[1].x[child_node[1]],
                                 child_rmin[2].x[child_node[2]],
                                 child_rmin[3].x[child_node[3]]);
    slow_verify("child",board,child,true);
  }
#endif

  // Compute
  for (int i=0;i<rmin.size();i++) {
    const quadrant_t quadrant = rmin[i],
                     side_q0 = unpack(quadrant,turn),
                     side_q1 = unpack(quadrant,1-turn);
    const auto side0 = base_side0|(side_t)side_q0<<dim_shift,
               side1 = base_side1|(side_t)side_q1<<dim_shift;
    const auto wins0 = super_wins(side0),
               wins1 = super_wins(side1),
               immediate = wins0|wins1;
    super_t wins = 0,     // Known wins, ignoring immediate
            not_loss = 0; // Known wins or ties, ignoring immediate
    // Process moves
    quadrant_t moves = 511&~(side_q0|side_q1);
    while (moves) {
      const quadrant_t move = min_bit(moves);
      moves ^= move;
      // Can we win without rotating?
      const side_t new_side0 = side0|(side_t)move<<dim_shift;
      const auto new_wins = super_wins(new_side0);
      wins |= new_wins;
      // If not, look up child position
      if (slice_35) {
        // The game ends after this move, so the child result is immediate.
        wins |= rmax(new_wins&~wins1);
        not_loss |= rmax(new_wins|~wins1);
      } else {
        // Slice < 35, so we have to do some real work
        const quadrant_t new_q = quadrant+(1+turn)*pack_table[move];
        const uint16_t ir = rotation_minimal_quadrants_inverse[new_q];
        const int j = ir/4;
        const symmetry_t symmetry = local_symmetry_t((ir&3)<<2*dim)*base_transform*local_symmetry_t(symmetries[j]);
        const auto& child = input[((j>>block_shift)==child_length-1?last_input_base:input_base)+block_stride*(j>>block_shift)+input_stride*((j^(j<line_moves))&(block_size-1))];
#ifdef PENTAGO_MPI_DEBUG
        const auto child_board = child_board_base|(board_t)child_rmin[child_dim].x[j^(j<line_moves)]<<16*child_dim;
        slow_verify("child inline",child_board,child,true);
        const auto after_board = flip_board(pack(new_side0,side1),turn);
        OTHER_ASSERT(after_board==transform_board(symmetry,child_board));
        const auto after_super = vec(transform_super(symmetry,child.x),transform_super(symmetry,child.y));
        slow_verify("after",after_board,after_super,true);
#endif
        wins |= rmax(transform_super(symmetry,~child.y));
        not_loss |= rmax(transform_super(symmetry,~child.x));
      }
    }
    // Finish up
    auto& result = output[((i>>block_shift)==length-1?last_output_base:output_base)+block_stride*(i>>block_shift)+output_stride*(i&(block_size-1))];
    result.x = (wins&~immediate)|(wins0&~wins1);
    result.y = (not_loss&~immediate)|wins0;
  }

  // Are we done with this line?
  if (!--rest.missing_microlines) {
    BOOST_STATIC_ASSERT(sizeof(line_data_t*)==sizeof(uint64_t) && sizeof(uint64_t)==sizeof(long long int));
    // Send a pointer to ourselves to the communication thread
    MPI_Request request;
    CHECK(MPI_Isend((void*)&rest.self,1,MPI_LONG_LONG_INT,0,wakeup_tag,rest.wakeup_comm,&request));
    CHECK(MPI_Request_free(&request));
    PENTAGO_MPI_TRACE("sent wakeup for %s",str(line->line));
  }
}

template void compute_microline<true>(line_data_t* const,const Vector<int,3>);
template void compute_microline<false>(line_data_t* const,const Vector<int,3>);

/*********************** schedule_compute_line ************************/

/* This routine involves a key performance decision: how finely do we partition the computation
 * of each line?  We're shooting for at least 64 way parallelism in order to match one BlueGene/Q
 * node, and overshooting by a significant amount would be best.  The cost is the additional time
 * taken by locking.  Additionally, parallelizing within the critical dimension would remove cache
 * coherence: since different elements of one microline share input nodes.  Thus, the natural choice
 * is to use thread job per microline.
 *
 * I estimated the compute speed of the code on my laptop at 12.5 MB/s per thread.  At 64 bytes per node,
 * the time it would take to compute a size 20 microline is
 *
 *     64*20 / 12.5e6 ~= 100 us
 *
 * According to Jeff Dean, a lock/unlock takes around 100 ns, which gives us a safety margin of 1000.
 * Most microlines are significantly larger than 20, so this seems like plenty of room.  One microline
 * per job it is.  However, I may need (want?) to write a thread pool class that scales to a large
 * number of threads.  Yes, I clearly want to do that.  TODO: Say whether I did this.
 */
void schedule_compute_line(line_data_t& line) {
  PENTAGO_MPI_TRACE("schedule compute line %s",str(line.line));
  // Schedule each microline
  const auto cross_section = line.output_shape.remove_index(line.line.dimension);
  const auto compute = line.line.section.sum()==35?compute_microline<true>:compute_microline<false>;
  for (int i=0;i<cross_section[0];i++)
    for (int j=0;j<cross_section[1];j++)
      for (int k=0;k<cross_section[2];k++)
        threads_schedule(CPU,boost::bind(compute,&line,vec(i,j,k)));
}

}
}
