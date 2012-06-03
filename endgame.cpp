// Endgame database computation

#include "symmetry.h"
#include "superscore.h"
#include "all_boards.h"
#include "supertensor.h"
#include "superengine.h"
#include <other/core/array/Array4d.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/math/integer_log.h>
#include <other/core/python/module.h>
#include <other/core/python/ExceptionValue.h>
#include <other/core/random/Random.h>
#include <other/core/utility/interrupts.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/openmp.h>
#include <other/core/utility/ProgressIndicator.h>
namespace pentago {

using Log::cout;
using std::endl;
using std::vector;

static RawArray<const quadrant_t> safe_rmin_slice(Vector<uint8_t,2> counts, int lo, int hi) {
  RawArray<const quadrant_t> rmin = rotation_minimal_quadrants(counts);
  OTHER_ASSERT(0<=lo && lo<=hi && (unsigned)hi<=(unsigned)rmin.size());
  return rmin.slice(lo,hi);
}

static RawArray<const quadrant_t> safe_rmin_slice(Vector<uint8_t,2> counts, Range<int> range) {
  return safe_rmin_slice(counts,range.lo,range.hi);
}

static void final_endgame_slice(section_t section, Vector<int,4> offset, RawArray<Vector<super_t,2>,4> results) {
  // Check input consistency
  OTHER_ASSERT(section.valid() && section.sum()==36);
  Vector<int,4> shape = results.shape;
  RawArray<const quadrant_t> rmin[4] = {safe_rmin_slice(section.counts[0],offset[0]+range(shape[0])),
                                        safe_rmin_slice(section.counts[1],offset[1]+range(shape[1])),
                                        safe_rmin_slice(section.counts[2],offset[2]+range(shape[2])),
                                        safe_rmin_slice(section.counts[3],offset[3]+range(shape[3]))};
  if (!shape.product())
    return;

  // Run the outer three dimensional loops in parallel
  const int first_three = shape[0]*shape[1]*shape[2];
  #pragma omp parallel for
  for (int ijk=0;ijk<first_three;ijk++) {
    if (interrupted())
      continue;
    const int ij = ijk/shape[2],
              k = ijk-ij*shape[2],
              i = ij/shape[1],
              j = ij-i*shape[1];
    const quadrant_t q0 = rmin[0][i],
                     q1 = rmin[1][j],
                     q2 = rmin[2][k],
                     s0b = unpack(q0,0),
                     s0w = unpack(q0,1),
                     s1b = unpack(q1,0),
                     s1w = unpack(q1,1),
                     s2b = unpack(q2,0),
                     s2w = unpack(q2,1);
    const side_t s012b = quadrants(s0b,s1b,s2b,0),
                 s012w = quadrants(s0w,s1w,s2w,0);
    RawArray<Vector<super_t,2>> slice(shape[3],&results(i,j,k,0));
    // Do the inner loop in serial to avoid overhead
    for (int l=0;l<shape[3];l++) {
      const quadrant_t q3 = rmin[3][l],
                       s3b = unpack(q3,0),
                       s3w = unpack(q3,1);
      const side_t side_black = s012b|(side_t)s3b<<3*16,
                   side_white = s012w|(side_t)s3w<<3*16;
      const super_t wins_black = super_wins(side_black),
                    wins_white = super_wins(side_white);
      slice[l][0] = wins_black&~wins_white;
      slice[l][1] = wins_white&~wins_black;
    }
  }
}

static Vector<int,4> in_order(Vector<int,4> order, Vector<int,4> I) {
  Vector<int,4> block;
  for (int i=0;i<4;i++)
    block[order[i]] = I[i];
  return block;
}

template<int d> static Vector<int,d> strides(Vector<int,d> shape) {
  Vector<int,d> strides;
  strides[d-1] = 1;
  for (int i=d-2;i>=0;i--)
    strides[i] = strides[i+1]*shape[i+1];
  return strides;
}

static Vector<int,4> quadrant_permutation(uint8_t global) {
  typedef Vector<uint8_t,2> CV;
  section_t s = section_t(vec(CV(0,0),CV(1,0),CV(2,0),CV(3,0))).transform(global);
  Vector<int,4> p;
  for (int i=0;i<4;i++)
    p[i] = s.counts.find(CV(i,0));
  return p;
}

static void verify(const char* prefix, const board_t board, const Vector<super_t,2>& result, bool verbose=false) {
  const bool turn = popcount(unpack(board,0))!=popcount(unpack(board,1));
  const board_t flipped = turn?pack(unpack(board,1),unpack(board,0)):board;
  const super_t win = super_evaluate_all(true,36,flipped);
  const super_t loss = ~super_evaluate_all(false,36,flipped);
  if (result[turn]!=win || result[1-turn]!=loss)
    for (int r=0;r<256;r++) {
      int fast = result[turn](r)-result[1-turn](r);
      int slow = win(r)-loss(r);
      if (fast!=slow) {
        if (verbose)
          cout << str_board(transform_board(symmetry_t(0,r),board)) << endl;
        section_t section;
        for (int i=0;i<4;i++)
          section.counts[i] = count(quadrant(board,i));
        throw RuntimeError(format("%s failed: section %s, board %lld, rotation %d, fast %d, slow %d",prefix,str(section),board,r,fast,slow));
      }
    }
}

static void endgame_verify(const supertensor_reader_t& reader, Random& random, const int sample_count) {
  // Generate the given number of samples, with replacement for simplicity, organized by block
  const section_t section = reader.header.section;
  const int block_size = reader.header.block_size;
  const Vector<int,4> blocks(reader.header.blocks);
  vector<Array<Vector<int,4>>> samples(blocks.product());
  for (int i=0;i<sample_count;i++) {
    const Vector<int,4> s = random.uniform(Vector<int,4>(),Vector<int,4>(reader.header.shape)),
                        b = s/block_size;
    samples[((b[0]*blocks[1]+b[1])*blocks[2]+b[2])*blocks[3]+b[3]].append(s);
  }

  // Look up minimal quadrants
  const RawArray<const quadrant_t> rmin[4] = {rotation_minimal_quadrants(section.counts[0]),
                                              rotation_minimal_quadrants(section.counts[1]),
                                              rotation_minimal_quadrants(section.counts[2]),
                                              rotation_minimal_quadrants(section.counts[3])};

  // Loop over the set of blocks
  ProgressIndicator progress(sample_count,true);
  Array<Vector<super_t,2>> buffer(reader.header.block_shape(Vector<int,4>()).product(),false);
  for (int i0 : range(blocks[0]))
    for (int i1 : range(blocks[1]))
      for (int i2 : range(blocks[2]))
        for (int i3 : range(blocks[3])) {
          // Learn about block
          const Vector<int,4> block(i0,i1,i2,i3);
          const Vector<int,4> shape = reader.header.block_shape(block);
          // Read block
          OTHER_ASSERT(shape.product()<=buffer.size());
          RawArray<Vector<super_t,2>,4> data(shape,buffer.data());
          reader.read_block(block,data);
          // Verify our chosen set of random samples
          for (Vector<int,4> sample : samples[((block[0]*blocks[1]+block[1])*blocks[2]+block[2])*blocks[3]+block[3]]) {
            const Vector<super_t,2>& result = data[sample-block_size*block];
            const board_t board = quadrants(rmin[0][sample[0]],rmin[1][sample[1]],rmin[2][sample[2]],rmin[3][sample[3]]);
            verify("endgame verify",board,result,true);
            progress.progress();
          }
        }
}

// Given a section s and order o, we define the 4-tensor A(s,o,I) = f(q[s[0],I[oi[0]]],...), where f(q0,q1,q2,q3) is the superoutcome for the board with
// quadrants q0,...,q3, q[c,i] is the ith quadrant with counts c, and oi is the inverse of the permutation o.  Hmm, we may have a serious problem.  If we
// apply a reflection transform to standardize a section, the resulting per quadrant reflections will screw up the ordering of the quadrants and break the
// block structure of our file format.  This doesn't apply to a global rotation standardization because we can cancel the global rotation with local rotations.
// This may force us to not standardize via reflections.  Let's see how bad that is: yeah, we lose almost a factor of two if we ignore reflections.  Extremely
// lame.  There is a two pass method of generating a reflected file from an unreflected file, by applying the reflection permutation two quadrant dimensions
// at a time, but this may not be all that much better than simply computing reflected files from scratch.

// Well, so be it.  I don't see any easy way around this, so for now we lose a factor of two.  Let's proceed.  We need a block section of a section A(s,o),
// but all we have is A(gs) for some global rotation g.  Adjust g to a global+local symmetry so that each quadrant moves without rotating.  Let g map quadrant
// k to p[k].  Then we have
//   A(s,o,I) = f(... q[s[k],I[oi[k]]] ...) = gi f(g(... q[s[k],I[oi[k]]] ...)) = gi f(... q[gs[k],I[oi[pi[k]]]] ...)

static void endgame_read_block_slice(section_t desired_section, const supertensor_reader_t& reader, Vector<int,4> order, int i, int j, RawArray<Vector<super_t,2>,4> data) {
  auto standard = desired_section.standardize<4>();
  OTHER_ASSERT((unsigned)standard.y<4); // Disallow reflections for now
  OTHER_ASSERT(standard.x==reader.header.section);

  // Prepare a transform that rotates quadrants locally while preserving their orientation
  const symmetry_t symmetry((4-standard.y)&3,(1+4+16+64)*standard.y);

  // Determine which quadrants are mapped to where, and compose with order
  const Vector<int,4> reorder(quadrant_permutation(standard.y).subset(order));

  const int block_size = reader.header.block_size;
  const Vector<int,4> first_block = in_order(reorder,vec(i,j,0,0));
  const Vector<int,4> first_block_shape = reader.header.block_shape(first_block);
  const Vector<int,4> blocks(reader.header.blocks);
  const Vector<int,2> slice_blocks(blocks[reorder[2]],blocks[reorder[3]]);
  const Vector<int,4> strides = in_order(reorder,pentago::strides(data.shape));
  for (int a=0;a<4;a++)
    OTHER_ASSERT(data.shape[a]==(a<2?first_block_shape[reorder[a]]:reader.header.shape[reorder[a]]));

  // Read blocks in parallel, mainly to accelerate zlib decompression
  ExceptionValue error;
  #pragma omp parallel
  {
    try {
      Array<Vector<super_t,2>> buffer(first_block_shape.product(),false);
      for (const int kl : partition_loop(slice_blocks.product())) {
        if (error)
          break;
        const int k = kl/slice_blocks[1],
                  l = kl-k*slice_blocks[1];
        const Vector<int,4> block = in_order(reorder,vec(i,j,k,l));
        const Vector<int,4> block_shape = reader.header.block_shape(block);
        OTHER_ASSERT(block_shape.product()<=buffer.size());
        // Read, uncompress, and unfilter the block
        RawArray<Vector<super_t,2>,4> block_data(block_shape,buffer.data());
        reader.read_block(block,block_data);
        // Move block data into place
        Vector<super_t,2>* start = &data(0,0,block_size*k,block_size*l);
        for (int ii=0;ii<block_shape[0];ii++)
          for (int jj=0;jj<block_shape[1];jj++)
            for (int kk=0;kk<block_shape[2];kk++)
              for (int ll=0;ll<block_shape[3];ll++) {
                const Vector<super_t,2>& src = block_data(ii,jj,kk,ll);
                Vector<super_t,2>& dst = start[strides[0]*ii+strides[1]*jj+strides[2]*kk+strides[3]*ll];
                for (int a=0;a<2;a++)
                  dst[a] = transform_super(symmetry,src[a]);
              }
      }
    } catch (const exception& e) {
      #pragma omp critical
      if (!error)
        error = ExceptionValue(e);
    }
  }
  if (error)
    error.throw_();
}

template<bool turn> static void in_place_extreme(RawArray<Vector<super_t,2>> dst, RawArray<const Vector<super_t,2>> src) {
  OTHER_ASSERT(dst.size()==src.size());
  for (const int i : range(dst.size())) {
    auto& d = dst[i];
    const auto& s = src[i];
    if (turn==0) // black to play
      d.set(d[0]|s[0],d[1]&s[1]);
    else // white to play
      d.set(d[0]&s[0],d[1]|s[1]);
  }
}

static void in_place_extreme(bool turn, RawArray<Vector<super_t,2>> dst, RawArray<const Vector<super_t,2>> src) {
  if (turn==0)
    in_place_extreme<0>(dst,src);
  else
    in_place_extreme<1>(dst,src);
}

// Do not use in performance critical code
static board_t section_board(const section_t& section, const Vector<int,4>& I) {
  return quadrants(rotation_minimal_quadrants(section.counts[0])[I[0]],
                   rotation_minimal_quadrants(section.counts[1])[I[1]],
                   rotation_minimal_quadrants(section.counts[2])[I[2]],
                   rotation_minimal_quadrants(section.counts[3])[I[3]]);
}

static void endgame_write_block_slice(supertensor_writer_t& writer, Ptr<supertensor_reader_t> first_pass, Vector<int,4> order, int i, int j, RawArray<const Vector<super_t,2>,4> data) {
  OTHER_ASSERT(!first_pass || writer.header.section==first_pass->header.section);
  const int block_size = writer.header.block_size;
  const Vector<int,4> first_block = in_order(order,vec(i,j,0,0));
  const Vector<int,4> first_block_shape = writer.header.block_shape(first_block);
  const Vector<int,4> blocks(writer.header.blocks);
  const Vector<int,2> slice_blocks(blocks[order[2]],blocks[order[3]]);
  const Vector<int,4> strides = in_order(order,pentago::strides(data.shape));
  for (int a=0;a<4;a++)
    OTHER_ASSERT(data.shape[a]==(a<2?first_block_shape[order[a]]:writer.header.shape[order[a]]));

  // Write blocks in parallel, mainly to accelerate zlib compression
  ExceptionValue error;
  #pragma omp parallel
  {
    try {
      Array<Vector<super_t,2>> buffer(first_block_shape.product(),false),
                               first_buffer(first_pass?first_block_shape.product():0,false);
      for (const int kl : partition_loop(slice_blocks.product())) {
        if (error)
          break;
        const int k = kl/slice_blocks[1],
                  l = kl-k*slice_blocks[1];
        const Vector<int,4> block = in_order(order,vec(i,j,k,l));
        const Vector<int,4> block_shape = writer.header.block_shape(block);
        OTHER_ASSERT(block_shape.product()<=buffer.size());
        // Move data into place
        RawArray<Vector<super_t,2>,4> block_data(block_shape,buffer.data());
        const Vector<super_t,2>* start = &data(0,0,block_size*k,block_size*l);
        for (int ii=0;ii<block_shape[0];ii++)
          for (int jj=0;jj<block_shape[1];jj++)
            for (int kk=0;kk<block_shape[2];kk++)
              for (int ll=0;ll<block_shape[3];ll++)
                block_data(ii,jj,kk,ll) = start[strides[0]*ii+strides[1]*jj+strides[2]*kk+strides[3]*ll];
        // Combine with data from the first pass if applicable
        if (first_pass) {
          RawArray<Vector<super_t,2>,4> first_block_data(block_shape,first_buffer.data());
          first_pass->read_block(block,first_block_data);
          in_place_extreme(writer.header.section.sum()&1,block_data.flat,first_block_data.flat);
        }
        // Filter, compress, and write the block
        writer.write_block(block,block_data,false);
      }
    } catch (const exception& e) {
      #pragma omp critical
      if (!error)
        error = ExceptionValue(e);
    }
  }
  if (error)
    error.throw_();
}

template<bool turn,bool final> static void endgame_compute_block_slice_helper(const supertensor_writer_t& writer, Vector<int,4> order, int i, int j, RawArray<Vector<super_t,2>,4> dest, RawArray<const Vector<super_t,2>,4> src2, RawArray<const Vector<super_t,2>,4> src3) {
  for (int a=0;a<4;a++)
    OTHER_ASSERT((a==2 || dest.shape[a]==src2.shape[a]) && (a==3 || dest.shape[a]==src3.shape[a]));
  const section_t section = writer.header.section;
  const Vector<uint8_t,2> move = Vector<uint8_t,2>::axis_vector(turn);
  const RawArray<const quadrant_t> rmin0 = safe_rmin_slice(section.counts[order[0]],writer.header.block_size*i+range(dest.shape[0])),
                                   rmin1 = safe_rmin_slice(section.counts[order[1]],writer.header.block_size*j+range(dest.shape[1])),
                                   rmin2 = rotation_minimal_quadrants(section.counts[order[2]]),
                                   rmin3 = rotation_minimal_quadrants(section.counts[order[3]]);
  OTHER_ASSERT(dest.shape[2]==rmin2.size());
  OTHER_ASSERT(dest.shape[3]==rmin3.size());
  OTHER_ASSERT(section.counts[order[2]].sum()==9 || src2.shape[2]==rotation_minimal_quadrants(section.counts[order[2]]+move).size());
  OTHER_ASSERT(section.counts[order[3]].sum()==9 || src3.shape[3]==rotation_minimal_quadrants(section.counts[order[3]]+move).size());

  // Run outer two dimensions in parallel
  const int outer_size = dest.shape[0]*dest.shape[1];
  #pragma omp parallel for
  for (int outer=0;outer<outer_size;outer++) {
    const int ii = outer/dest.shape[1],
              jj = outer-ii*dest.shape[1];
    // Run inner two dimensions sequentially.  Hopefully this produces nice cache behavior.
    for (int kk=0;kk<dest.shape[2];kk++)
      for (int ll=0;ll<dest.shape[3];ll++) {
        const quadrant_t q0 = rmin0[ii],
                         q1 = rmin1[jj],
                         q2 = rmin2[kk],
                         q3 = rmin3[ll];
        const board_t board =  (board_t)q0<<16*order[0]
                              |(board_t)q1<<16*order[1]
                              |(board_t)q2<<16*order[2]
                              |(board_t)q3<<16*order[3];
        const side_t side0 = unpack(board,turn),
                     side1 = unpack(board,1-turn);
        const super_t immediate_wins0 = super_wins(side0),
                      immediate_wins1 = super_wins(side1),
                      immediate = final?~super_t(0):immediate_wins0|immediate_wins1;
        super_t wins = 0, // known wins, ignoring immediate
                not_loss = 0; // known wins or ties, ignoring immediate
        // Process moves in each quadrant
        #define PROCESS_QUADRANT_MOVES(a) { \
          quadrant_t moves = 511&~unpack(q##a,0)&~unpack(q##a,1); \
          while (!final && moves) { \
            const quadrant_t move = min_bit(moves); \
            moves ^= move; \
            /* Can we win without rotating? */ \
            const side_t new_side0 = side0|(side_t)move<<16*order[a]; \
            wins |= super_wins(new_side0)&~immediate; \
            /* If not, look up child position */ \
            const quadrant_t new_q = q##a+(1+turn)*pack_table[move]; \
            const uint16_t ir = rotation_minimal_quadrants_inverse[new_q]; \
            const symmetry_t s = symmetry_t(0,(ir&3)<<2*order[a]); \
            Vector<int,4> I(ii,jj,kk,ll); \
            I[a] = ir/4; \
            const Vector<super_t,2>& child = src##a[I]; \
            wins |= rmax(transform_super(s,child[turn])); \
            not_loss |= rmax(transform_super(s,~child[1-turn])); \
          }}
        PROCESS_QUADRANT_MOVES(2)
        PROCESS_QUADRANT_MOVES(3)
        // Finish up
        wins = (wins&~immediate)|(immediate_wins0&~immediate_wins1);
        not_loss = final?immediate_wins0|~immediate_wins1:(not_loss&~immediate)|immediate_wins0|wins;
        Vector<super_t,2>& d = dest(ii,jj,kk,ll);
        d[turn] = wins;
        d[1-turn] = ~not_loss;
      }
  }
}

static void endgame_compute_block_slice(const supertensor_writer_t& writer, Vector<int,4> order, int i, int j, RawArray<Vector<super_t,2>,4> dest, RawArray<const Vector<super_t,2>,4> src2, RawArray<const Vector<super_t,2>,4> src3) {
  const int stones = writer.header.section.sum();
  if (stones&1)
    endgame_compute_block_slice_helper<1,false>(writer,order,i,j,dest,src2,src3);
  else if (stones<36)
    endgame_compute_block_slice_helper<0,false>(writer,order,i,j,dest,src2,src3);
  else
    endgame_compute_block_slice_helper<0,true >(writer,order,i,j,dest,src2,src3);
}

}
using namespace pentago;

void wrap_endgame() {
  OTHER_FUNCTION(final_endgame_slice)
  OTHER_FUNCTION(endgame_read_block_slice)
  OTHER_FUNCTION(endgame_write_block_slice)
  OTHER_FUNCTION(endgame_compute_block_slice)
  OTHER_FUNCTION(endgame_verify)
}
