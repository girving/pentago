// Core compute kernels for the MPI code

#include <pentago/mpi/compute.h>
#include <pentago/mpi/utility.h>
#include <pentago/mpi/flow.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/counter.h>
#include <pentago/utility/index.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/math/integer_log.h>
#include <other/core/utility/const_cast.h>
#include <boost/bind.hpp>
namespace pentago {
namespace mpi {

/*********************** line_data_t ************************/

// We're going to be dividing by block size, so take advantage of the fact that it's 8
static const int block_size = 8;
static const int block_shift = 3;

line_data_t::line_data_t(const line_t& line, const int block_size_)
  : line(line) {
  OTHER_ASSERT(block_size_==8);

  // Compute input and output shapes
  const int dim = line.dimension;
  const auto section_shape = line.section.shape();
  const auto child_section = line.section.child(dim);
  const auto child_section_shape = child_section.shape();
  OTHER_ASSERT(section_shape.remove_index(dim)==child_section_shape.remove_index(dim));
  auto output_shape = block_shape(section_shape,line.block(0),block_size);
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
  const uint8_t section_transform;
  const Vector<int,4> permutation; // Child quadrant i maps to quadrant permutation[i]
  const int child_dimension;
  const int child_length;
  const Vector<int,4> first_child_block; // First child block
  const symmetry_t inverse_transform;

  // Symmetries needed to restore minimality after reflection
  const Array<const uint8_t> all_reflection_symmetries;
  const Vector<int,5> reflection_symmetry_offsets;

  // Number of blocks we need before input is ready, microlines left to compute, and output blocks left to send.
  counter_t missing_input_blocks;
  counter_t missing_microlines;
  counter_t unsent_output_blocks; 

  // Information needed to account for reflections in input block data
  const Vector<Tuple<RawArray<const quadrant_t>,int>,4> rmin;
  const Vector<int,4> reflection_moves;

  // Input and output data.  Both are stored in 5D order where the first dimension
  // is the block, to avoid copying before and after compute.
  const Array<Vector<super_t,2>> input, output;

  // When computation is complete, add self here
  finished_lines_t& finished;

  allocated_t(const line_data_t& self, finished_lines_t& finished);
};

// Mark this const so that the two identical calls in allocated_t's constructor can be CSE'ed
OTHER_CONST static Tuple<section_t,uint8_t> standardize_child_section(section_t section, int dimension) {
  return section.child(dimension).standardize<8>();
}

allocated_t::allocated_t(const line_data_t& self, finished_lines_t& finished)
  // Standardize
  : standard_child_section(standardize_child_section(self.line.section,self.line.dimension).x)
  , section_transform(     standardize_child_section(self.line.section,self.line.dimension).y)
  , permutation(section_t::quadrant_permutation(section_transform))
  , child_dimension(permutation.find(self.line.dimension))
  , child_length((self.input_shape[self.line.dimension]+block_size-1)/block_size)
  , first_child_block(self.line.block_base.insert(self.line.dimension,0).subset(permutation))

  // Prepare a transform that rotates quadrants locally while preserving their orientation
  , inverse_transform((4-(section_transform&3))&3,(1+4+16+64)*(section_transform&3))

  // Numbers of steps to complete
  , missing_input_blocks(child_length)
  , missing_microlines(self.output_shape.remove_index(self.line.dimension).product())
  , unsent_output_blocks(self.line.length)

  // If there's a reflection, prepare to rearrange data accordingly
  , rmin(rotation_minimal_quadrants(standard_child_section.counts[0]),
         rotation_minimal_quadrants(standard_child_section.counts[1]),
         rotation_minimal_quadrants(standard_child_section.counts[2]),
         rotation_minimal_quadrants(standard_child_section.counts[3]))
  , reflection_moves(section_transform<4?Vector<int,4>():vec(rmin[0].y,rmin[1].y,rmin[2].y,rmin[3].y))

  // Allocate memory
  , input(aligned_buffer<Vector<super_t,2>>(self.input_shape.product()))
  , output(aligned_buffer<Vector<super_t,2>>(self.output_shape.product()))

  // When computation is complete, add self here
  , finished(finished) {

  // Compute symmetries needed to restore minimality after reflection
  const Vector<int,4> standard_input_shape(self.input_shape.subset(permutation));
  const_cast_(all_reflection_symmetries) = Array<uint8_t>(standard_input_shape.sum());
  const_cast_(reflection_symmetry_offsets[0]) = 0;
  int offset = 0;
  for (int i=0;i<4;i++) {
    RawArray<uint8_t> symmetries = all_reflection_symmetries.const_cast_().slice(offset,offset+standard_input_shape[i]);
    if (section_transform>=4)
      for (int j=0;j<standard_input_shape[i];j++) {
        const quadrant_t q = rmin[i].x[block_size*first_child_block[i]+j];
        symmetries[j] = (rotation_minimal_quadrants_inverse[pack(reflections[unpack(q,0)],reflections[unpack(q,1)])]&3)<<2*i;
      }
    const_cast_(reflection_symmetry_offsets[i+1]) = reflection_symmetry_offsets[i]+symmetries.size();
  }
}

void line_data_t::allocate(finished_lines_t& finished) {
  rest.reset(new allocated_t(*this,finished));
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
  block[rest->child_dimension] = k;
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

void line_data_t::decrement_unsent_output_blocks(uint64_t* total_memory_usage) {
  // If all sends are complete, deallocate.
  if (!--rest->unsent_output_blocks) {
    *total_memory_usage -= memory_usage();
    delete this;
  }
}

/*********************** compute_microline ************************/

// Compute a single 1D line through a section (a 1D component of a block line)
template<bool slice_35> static void compute_microline(line_data_t* const line, const Vector<int,3> base) {
  // Prepare
  thread_time_t time("compute");
  const int dim = line->line.dimension;
  const int child_dim = line->rest->child_dimension;
  const auto full_base = base.insert(0,dim);
  auto& rest = *line->rest;
  const auto first_block = line->line.block_base.insert(0,dim);
  const auto first_node = block_size*first_block+full_base;
  Vector<quadrant_t,4> base_quadrants(rest.rmin[0].x[first_node[0]],
                                      rest.rmin[1].x[first_node[1]],
                                      rest.rmin[2].x[first_node[2]],
                                      rest.rmin[3].x[first_node[3]]);
  base_quadrants[dim] = 0;
  const auto base_board = quadrants(base_quadrants[0],base_quadrants[1],base_quadrants[2],base_quadrants[3]);
  const bool turn = line->line.section.sum()&1;
  const auto base_side0 = unpack(base_board,turn),
             base_side1 = unpack(base_board,1-turn);
  const auto rmin = rest.rmin[dim].x;
  OTHER_ASSERT(rmin.size()==line->output_shape[dim]);
  const int dim_shift = 16*dim;

  // Prepare to index into output array
  const int block_stride = line->line.node_step;
  auto output_block_shape = line->output_shape;
  output_block_shape[dim] = block_size;
  const auto output_block_strides = strides(output_block_shape);
  const int output_stride = output_block_strides[dim];
  const int output_base = dot(output_block_strides,full_base);
  const auto output = rest.output;

  // Prepare to index into input array
  const Vector<int,4> input_block_strides(output_block_strides.subset(rest.permutation));
  const int input_stride = input_block_strides[child_dim];
  const Vector<int,4> full_child_base(full_base.subset(rest.permutation));
  const auto first_child_node = block_size*rest.first_child_block+full_child_base;
  const int input_base = dot(input_block_strides,full_child_base^Vector<int,4>(child_dim!=0 && first_child_node[0]<rest.reflection_moves[0],
                                                                               child_dim!=1 && first_child_node[1]<rest.reflection_moves[1],
                                                                               child_dim!=2 && first_child_node[2]<rest.reflection_moves[2],
                                                                               child_dim!=3 && first_child_node[3]<rest.reflection_moves[3]));
  const auto input = rest.input.const_();
  const int line_moves = rest.reflection_moves[child_dim];

  // Account for symmetries
  const auto section_transform = rest.section_transform;
  const auto rs_offsets = rest.reflection_symmetry_offsets;
  const auto symmetries = rest.all_reflection_symmetries.slice(rs_offsets[child_dim],rs_offsets[child_dim+1]);
  const symmetry_t base_transform = rest.inverse_transform*symmetry_t(section_transform&4, (child_dim!=0)*rest.all_reflection_symmetries[rs_offsets[0]+full_child_base[0]]
                                                                                          +(child_dim!=1)*rest.all_reflection_symmetries[rs_offsets[1]+full_child_base[1]]
                                                                                          +(child_dim!=2)*rest.all_reflection_symmetries[rs_offsets[2]+full_child_base[2]]
                                                                                          +(child_dim!=3)*rest.all_reflection_symmetries[rs_offsets[3]+full_child_base[3]]);

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
        // The game ends after this move, so the child result is just new_wins.
        wins |= rmax(new_wins&~wins1); 
        not_loss |= rmax(new_wins|~wins1);
      } else {
        // Slice < 35, so we have to do some real work
        const quadrant_t new_q = quadrant+(1+turn)*pack_table[move];
        const uint16_t ir = rotation_minimal_quadrants_inverse[new_q];
        const int j = ir/4;
        const symmetry_t symmetry = symmetry_t(0,(ir&3)<<2*dim)*symmetry_t(base_transform.global,base_transform.local+symmetries[j]);
        const auto& child = input[input_base+block_stride*(j>>block_shift)+input_stride*((j^(j<line_moves))&(block_size-1))];
        wins |= rmax(transform_super(symmetry,~child.y));
        not_loss |= rmax(transform_super(symmetry,~child.x));
      }
    }
    // Finish up
    auto& result = output[output_base+block_stride*(i>>block_shift)+output_stride*(i&(block_size-1))];
    result.x = (wins&~immediate)|(wins0&~wins1);
    result.y = (not_loss&~immediate)|wins0;
  }

  // Are we done with this line?
  if (!--rest.missing_microlines) {
    spin_t spin(rest.finished.lock);
    rest.finished.lines.push_back(line);
  }
}

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
  // Schedule each microline
  const auto cross_section = line.output_shape.remove_index(line.line.dimension);
  const auto compute = line.line.section.sum()==3?compute_microline<true>:compute_microline<false>;
  for (int i=0;i<cross_section[0];i++)
    for (int j=0;j<cross_section[1];j++)
      for (int k=0;k<cross_section[2];k++)
        threads_schedule(CPU,boost::bind(compute,&line,vec(i,j,k)));
}

}
}
