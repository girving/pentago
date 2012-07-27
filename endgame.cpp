// Endgame database computation

#include <pentago/symmetry.h>
#include <pentago/superscore.h>
#include <pentago/section.h>
#include <pentago/supertensor.h>
#include <pentago/superengine.h>
#include <pentago/count.h>
#include <pentago/utility/aligned.h>
#include <other/core/array/Array2d.h>
#include <other/core/array/Array4d.h>
#include <other/core/array/NestedArray.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/math/integer_log.h>
#include <other/core/python/module.h>
#include <other/core/python/Class.h>
#include <other/core/python/ExceptionValue.h>
#include <other/core/random/Random.h>
#include <other/core/utility/interrupts.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/ProgressIndicator.h>
#include <boost/bind.hpp>
namespace pentago {

using Log::cout;
using std::endl;
using std::vector;

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
  const RawArray<const quadrant_t> rmin[4] = {rotation_minimal_quadrants(section.counts[0]).x,
                                              rotation_minimal_quadrants(section.counts[1]).x,
                                              rotation_minimal_quadrants(section.counts[2]).x,
                                              rotation_minimal_quadrants(section.counts[3]).x};

  // Loop over the set of blocks
  ProgressIndicator progress(sample_count,true);
  for (int i0 : range(blocks[0]))
    for (int i1 : range(blocks[1]))
      for (int i2 : range(blocks[2]))
        for (int i3 : range(blocks[3])) {
          // Read block
          const Vector<int,4> block(i0,i1,i2,i3);
          Array<Vector<super_t,2>,4> data = reader.read_block(block);
          // Verify our chosen set of random samples
          for (Vector<int,4> sample : samples[((block[0]*blocks[1]+block[1])*blocks[2]+block[2])*blocks[3]+block[3]]) {
            const Vector<super_t,2>& result = data[sample-block_size*block];
            const board_t board = quadrants(rmin[0][sample[0]],rmin[1][sample[1]],rmin[2][sample[2]],rmin[3][sample[3]]);
            verify("endgame verify",board,result,true);
            progress.progress();
          }
        }
}

static void endgame_sparse_verify(const section_t section, RawArray<const board_t> boards, RawArray<const Vector<super_t,2>> wins, Random& random, int samples) {
  OTHER_ASSERT(boards.size()==wins.size());
  OTHER_ASSERT((unsigned)samples<=(unsigned)boards.size());
  // Verify that all boards come from the section
  for (auto board : boards)
    OTHER_ASSERT(count(board)==section);
  // Check samples in random order
  Array<int> permutation = IdentityMap(boards.size()).copy();
  ProgressIndicator progress(samples,true);
  for (int i=0;i<samples;i++) {
    swap(permutation[i],permutation[random.uniform<int>(i,boards.size())]);
    verify("endgame sparse verify",boards[permutation[i]],wins[permutation[i]],true);
    progress.progress();
  }
}

// Given a section s and order o, we define the 4-tensor A(s,o,I) = f(... q[s[k],I[oi[k]]] ...), where f(q0,q1,q2,q3) is the superoutcome for the board with
// quadrants q0,...,q3, q[c,i] is the ith quadrant with counts c, and oi is the inverse of the permutation o.  Hmm, we may have a serious problem.  If we
// apply a reflection transform to standardize a section, the resulting per quadrant reflections will screw up the ordering of the quadrants and break the
// block structure of our file format.  This doesn't apply to a global rotation standardization because we can cancel the global rotation with local rotations.
// This may force us to not standardize via reflections.  Let's see how bad that is: yeah, we lose almost a factor of two if we ignore reflections.  Extremely
// lame.  There is a two pass method of generating a reflected file from an unreflected file, by applying the reflection permutation two quadrant dimensions
// at a time, but this may not be all that much better than simply computing reflected files from scratch.

// Well, so be it.  I don't see any easy way around this, so for now we lose a factor of two.  Let's proceed.  We need a block section of a section A(s,o),
// but all we have is A(gs) for some global rotation g.  Adjust g to a global+local symmetry so that each quadrant moves without rotating.  Let g map quadrant
// k to p[k].  Then we have
//   A(s,o,I) = f(... q[s[k],I[oi[k]]] ...) = gi f(g(... q[s[k],I[oi[k]]] ...)) = gi f(... q[gs[k],I[oi[pi[k]]]] ...) = gi A(gs, 1, I.oi.pi)

// Update: I figured out the easy way around this.  Rotation minimal quadrants are now ordered so that reflected pairs are adjacent in the ordering, and all
// quadrants which aren't fixed by reflection come first.  By enforcing an even block size, we ensure that reflections never cross blocks.  Unfortunately,
// after a board is reflected each quadrant must be rotated to restore minimality, and this rotation can be different per quadrant.  Thus we lose the
// simplicity of applying the same symmetry to each element.  This is a bit complicated (I hate Z_2).  Again, we seek A(s,o), and we have A(gs) for a global
// symmetry g factored as g = rw for a reflection r and rotation w, where w moves each quadrant without rotating.  Factor the reflection r as r = rl rq = rq rl,
// where rl reflects each quadrant locally and rq swaps quadrants 0 and 3.  We have
//   A(s,o,I) = f(... q[s[k],I[oi[k]]] ...) = gi f(g(... q[s[k],I[oi[k]]] ...)) = gi f(r(... q[ws[k],I[oi[wpi[k]]]] ...)) = gi f(rl(... q[gs[k],I[oi[pi[k]]]] ...))
// rl does not preserve rotation minimality of each quadrant, so define maps rs and ri s.t. rl[q[c,i]] = rs[c,i]q[c,ri[c,i]].  Let I.oi.pi = Iop.  Then
//   A(s,o,I) = gi f(... rs[gs[k],Iop[k]] q[gs[k],ri[gs[k],Iop[k]]] ...) = wi rq (... rs[gs[k],Iop[k]] ...) f(... q[gs[k],ri[gs[k],Iop[k]]] ...)

namespace {

struct read_helper_t : public boost::noncopyable {
  const supertensor_reader_t& reader;
  const int i, j;
  RawArray<Vector<super_t,2>,4> data;
  const Tuple<section_t,uint8_t> standard;
  const int rotation; 
  const symmetry_t inverse_rotation;
  const Vector<int,4> reorder;
  const int block_size;
  const Vector<int,4> first_block, first_block_shape, blocks;
  const Vector<int,2> slice_blocks;
  const Vector<int,4> strides;
  const Vector<Tuple<RawArray<const quadrant_t>,int>,4> rmin;
  const Vector<int,4> moves;

  read_helper_t(section_t desired_section, const supertensor_reader_t& reader, Vector<int,4> order, int i, int j, RawArray<Vector<super_t,2>,4> data)
    : reader(reader)
    , i(i), j(j)
    , data(data)
    , standard(desired_section.standardize<8>())

    // Prepare a transform that rotates quadrants locally while preserving their orientation
    , rotation(standard.y&3)
    , inverse_rotation((4-rotation)&3,(1+4+16+64)*rotation)

    // Determine which quadrants are mapped to where, and compose with order
    , reorder(section_t::quadrant_permutation(standard.y).subset(order))

    // Gather shape and stride information
    , block_size(reader.header.block_size)
    , first_block(in_order(reorder,vec(i,j,0,0)))
    , first_block_shape(reader.header.block_shape(first_block))
    , blocks(reader.header.blocks)
    , slice_blocks(blocks[reorder[2]],blocks[reorder[3]])
    , strides(in_order(reorder,pentago::strides(data.shape)))

    // If there's a reflection, prepare to rearrange data accordingly
    , rmin(rotation_minimal_quadrants(reader.header.section.counts[0]),
           rotation_minimal_quadrants(reader.header.section.counts[1]),
           rotation_minimal_quadrants(reader.header.section.counts[2]),
           rotation_minimal_quadrants(reader.header.section.counts[3]))
    , moves(standard.y<4?Vector<int,4>():vec(rmin[0].y,rmin[1].y,rmin[2].y,rmin[3].y))
  {
    OTHER_ASSERT(standard.x==reader.header.section);
    for (int a=0;a<4;a++)
      OTHER_ASSERT(data.shape[a]==(a<2?first_block_shape[reorder[a]]:reader.header.shape[reorder[a]]));
  }

  void process_block(Vector<int,4> block, RawArray<Vector<super_t,2>,4> block_data) {
    thread_time_t time("copy");
    OTHER_ASSERT(block[reorder[0]]==i && block[reorder[1]]==j);
    const int k = block[reorder[2]], l = block[reorder[3]];
    OTHER_ASSERT(block_data.shape==reader.header.block_shape(block));
    const Vector<int,4> block_moves = clamp_min(moves-block_size*block,0);
    // Compute symmetries needed to restore minimality after reflection (rs in the above derivation)
    uint8_t symmetries[block_size][4];
    if (standard.y>=4)
      for (int a=0;a<4;a++)
        for (int b=0;b<block_data.shape[a];b++) {
          const quadrant_t q = rmin[a].x[block_size*block[a]+b];
          symmetries[b][a] = rotation_minimal_quadrants_inverse[pack(reflections[unpack(q,0)],reflections[unpack(q,1)])]&3;
        }
    // Move block data into place
    Vector<super_t,2>* start = &data(0,0,block_size*k,block_size*l);
    for (int ii=0;ii<block_data.shape[0];ii++)
      for (int jj=0;jj<block_data.shape[1];jj++)
        for (int kk=0;kk<block_data.shape[2];kk++)
          for (int ll=0;ll<block_data.shape[3];ll++) {
            const Vector<super_t,2>& src = block_data(ii^(ii<block_moves[0]),jj^(jj<block_moves[1]),kk^(kk<block_moves[2]),ll^(ll<block_moves[3]));
            Vector<super_t,2>& dst = start[strides[0]*ii+strides[1]*jj+strides[2]*kk+strides[3]*ll];
            const symmetry_t symmetry = standard.y<4?inverse_rotation:inverse_rotation*symmetry_t(4,symmetries[ii][0]+4*symmetries[jj][1]+16*symmetries[kk][2]+64*symmetries[ll][3]);
            for (int a=0;a<2;a++)
              dst[a] = transform_super(symmetry,src[a]);
          }
  }
};

}

static void endgame_read_block_slice(section_t desired_section, const supertensor_reader_t& reader, Vector<int,4> order, int i, int j, RawArray<Vector<super_t,2>,4> data) {
  read_helper_t helper(desired_section,reader,order,i,j,data);
  Array<Vector<int,4>> slice;
  for (const int k : range(helper.slice_blocks[0]))
    for (const int l : range(helper.slice_blocks[1]))
      slice.append(in_order(helper.reorder,vec(i,j,k,l)));
  reader.schedule_read_blocks(slice,boost::bind(&read_helper_t::process_block,&helper,_1,_2));
  threads_wait_all();
}

// Do not use in performance critical code
OTHER_UNUSED static board_t section_board(const section_t& section, const Vector<int,4>& I) {
  return quadrants(rotation_minimal_quadrants(section.counts[0]).x[I[0]],
                   rotation_minimal_quadrants(section.counts[1]).x[I[1]],
                   rotation_minimal_quadrants(section.counts[2]).x[I[2]],
                   rotation_minimal_quadrants(section.counts[3]).x[I[3]]);
}

namespace {

struct sparse_sample_t : public Object {
  OTHER_DECLARE_TYPE

  const supertensor_header_t header;
  const Vector<int,4> blocks;
  const NestedArray<const Vector<int,4>> samples;
  const NestedArray<board_t> block_boards;
  const NestedArray<Vector<super_t,2>> block_wins;

  sparse_sample_t(const supertensor_writer_t& writer, int count)
    : header(writer.header)
    , blocks(header.blocks)
    , samples(random_samples(count))
    , block_boards(NestedArray<board_t>::zeros_like(samples))
    , block_wins(NestedArray<Vector<super_t,2>>::zeros_like(samples)) {}

  int block_index(const Vector<int,4>& block) const {
    OTHER_ASSERT(   (unsigned)block[0]<(unsigned)blocks[0]
                 && (unsigned)block[1]<(unsigned)blocks[1]
                 && (unsigned)block[2]<(unsigned)blocks[2]
                 && (unsigned)block[3]<(unsigned)blocks[3]);
    return ((block[0]*blocks[1]+block[1])*blocks[2]+block[2])*blocks[3]+block[3];
  }

  NestedArray<const Vector<int,4>> random_samples(int count) const {
    OTHER_ASSERT(count>=0);
    const int block_size = header.block_size;
    const Vector<int,4> shape = header.section.shape();
    Array<int,4> counts(blocks);
    Array<Vector<int,4>> flat_samples(count,false);
    const Ref<Random> random = new_<Random>(hash(header.section.sig()));
    for (Vector<int,4>& I : flat_samples) {
      for (int a=0;a<4;a++)
        I[a] = random->uniform<int>(0,shape[a]);
      counts[I/block_size]++; 
    }
    NestedArray<Vector<int,4>> samples(counts.flat,false);
    for (const Vector<int,4>& I : flat_samples) {
      const auto block = I/block_size;
      const int b = block_index(block);
      samples(b,--counts.flat[b]) = I-block_size*block;
    }
    return samples;
  }

  Array<const board_t> boards() const {
    return block_boards.flat;
  }

  Array<const Vector<super_t,2>> wins() const {
    return block_wins.flat;
  }
};

OTHER_DEFINE_TYPE(sparse_sample_t)

struct write_helper_t : public boost::noncopyable {
  supertensor_writer_t& writer;
  const Ptr<const supertensor_reader_t> first_pass;
  const Vector<int,4> order;
  const int i, j;
  const RawArray<const Vector<super_t,2>,4> data;
  const Ptr<sparse_sample_t> sparse;
  const int block_size;
  const Vector<int,4> first_block, first_block_shape, blocks;
  const Vector<int,2> slice_blocks;
  const Vector<int,4> strides;
  const Vector<RawArray<const quadrant_t>,4> rmin;

  write_helper_t(supertensor_writer_t& writer, Ptr<const supertensor_reader_t> first_pass, Vector<int,4> order, int i, int j, RawArray<const Vector<super_t,2>,4> data, const Ptr<sparse_sample_t> sparse)
    : writer(writer)
    , first_pass(first_pass)
    , order(order)
    , i(i), j(j)
    , data(data)
    , sparse(sparse)
    , block_size(writer.header.block_size)
    , first_block(in_order(order,vec(i,j,0,0)))
    , first_block_shape(writer.header.block_shape(first_block))
    , blocks(writer.header.blocks)
    , slice_blocks(blocks[order[2]],blocks[order[3]])
    , strides(in_order(order,pentago::strides(data.shape)))
    , rmin(rotation_minimal_quadrants(writer.header.section.counts[0]).x,
           rotation_minimal_quadrants(writer.header.section.counts[1]).x,
           rotation_minimal_quadrants(writer.header.section.counts[2]).x,
           rotation_minimal_quadrants(writer.header.section.counts[3]).x)
  {
    OTHER_ASSERT(!first_pass || writer.header.section==first_pass->header.section);
    for (int a=0;a<4;a++)
      OTHER_ASSERT(data.shape[a]==(a<2?first_block_shape[order[a]]:writer.header.shape[order[a]]));
  }

  void process_block(Vector<int,4> block, Array<Vector<super_t,2>,4> first_pass_data) {
    thread_time_t time("copy");
    OTHER_ASSERT(block[order[0]]==i && block[order[1]]==j);
    const int k = block[order[2]], l = block[order[3]];
    const Vector<int,4> block_shape = writer.header.block_shape(block);
    Array<Vector<super_t,2>,4> block_data;
    if (first_pass) {
      OTHER_ASSERT(first_pass_data.shape==block_shape);
      block_data = first_pass_data;
    } else
      block_data = aligned_buffer<Vector<super_t,2>>(block_shape);
    // Move data into place, combining with first pass data if applicable
    const Vector<super_t,2>* start = &data(0,0,block_size*k,block_size*l);
    if (!first_pass)
      for (int ii=0;ii<block_data.shape[0];ii++)
        for (int jj=0;jj<block_data.shape[1];jj++)
          for (int kk=0;kk<block_data.shape[2];kk++)
            for (int ll=0;ll<block_data.shape[3];ll++)
              block_data(ii,jj,kk,ll) = start[strides[0]*ii+strides[1]*jj+strides[2]*kk+strides[3]*ll];
    else if (!(writer.header.section.sum()&1)) // second pass, with black to move
      for (int ii=0;ii<block_data.shape[0];ii++)
        for (int jj=0;jj<block_data.shape[1];jj++)
          for (int kk=0;kk<block_data.shape[2];kk++)
            for (int ll=0;ll<block_data.shape[3];ll++) {
              auto& dst = block_data(ii,jj,kk,ll);
              const auto& src = start[strides[0]*ii+strides[1]*jj+strides[2]*kk+strides[3]*ll];
              dst.set(dst[0]|src[0],dst[1]&src[1]);
            }
    else // second pass, with white to move
      for (int ii=0;ii<block_data.shape[0];ii++)
        for (int jj=0;jj<block_data.shape[1];jj++)
          for (int kk=0;kk<block_data.shape[2];kk++)
            for (int ll=0;ll<block_data.shape[3];ll++) {
              auto& dst = block_data(ii,jj,kk,ll);
              const auto& src = start[strides[0]*ii+strides[1]*jj+strides[2]*kk+strides[3]*ll];
              dst.set(dst[0]&src[0],dst[1]|src[1]);
            }
    // Write
    writer.schedule_write_block(block,block_data);
    // Collect sparse samples
    if (sparse) {
      const int b = sparse->block_index(block);
      RawArray<const Vector<int,4>> samples = sparse->samples[b];
      RawArray<board_t> boards = sparse->block_boards[b];
      RawArray<Vector<super_t,2>> wins = sparse->block_wins[b];
      for (int i : range(samples.size())) {
        const auto I = samples[i];
        OTHER_ASSERT(block_data.valid(I));
        boards[i] = quadrants(rmin[0][block_size*block[0]+I[0]],
                              rmin[1][block_size*block[1]+I[1]],
                              rmin[2][block_size*block[2]+I[2]],
                              rmin[3][block_size*block[3]+I[3]]);
        wins[i] = block_data[I];
      }
    }
  }
};

}

static void endgame_write_block_slice(supertensor_writer_t& writer, Ptr<supertensor_reader_t> first_pass, Vector<int,4> order, int i, int j, RawArray<const Vector<super_t,2>,4> data, Ptr<sparse_sample_t> sparse) {
  write_helper_t helper(writer,first_pass,order,i,j,data,sparse);
  if (!first_pass) {
    vector<function<void()>> jobs;
    for (const int k : range(helper.slice_blocks[0]))
      for (const int l : range(helper.slice_blocks[1])) {
        const Vector<int,4> block = in_order(order,vec(i,j,k,l));
        jobs.push_back(boost::bind(&write_helper_t::process_block,&helper,block,Array<Vector<super_t,2>,4>()));
      }
    threads_schedule(CPU,jobs);
  } else {
    Array<Vector<int,4>> slice;
    for (const int k : range(helper.slice_blocks[0]))
      for (const int l : range(helper.slice_blocks[1]))
        slice.append(in_order(order,vec(i,j,k,l)));
    first_pass->schedule_read_blocks(slice,boost::bind(&write_helper_t::process_block,&helper,_1,_2));
  }
  threads_wait_all();
}

namespace {

template<bool turn,bool final> struct compute_helper_t {
  const Vector<int,4> order;
  const RawArray<Vector<super_t,2>,4> dest;
  const RawArray<const Vector<super_t,2>,4> src2, src3;
  const section_t section;
  const Vector<uint8_t,2> move;
  const Vector<RawArray<const quadrant_t>,4> rmin;
  const bool count;
  mutex_t win_counts_mutex;
  Vector<uint64_t,3> win_counts;

  compute_helper_t(const supertensor_writer_t& writer, Vector<int,4> order, int i, int j, RawArray<Vector<super_t,2>,4> dest, RawArray<const Vector<super_t,2>,4> src2, RawArray<const Vector<super_t,2>,4> src3, bool count)
    : order(order)
    , dest(dest), src2(src2), src3(src3)
    , section(writer.header.section)
    , move(Vector<uint8_t,2>::axis_vector(turn))
    , rmin(safe_rmin_slice(section.counts[order[0]],writer.header.block_size*i+range(dest.shape[0])),
           safe_rmin_slice(section.counts[order[1]],writer.header.block_size*j+range(dest.shape[1])),
           rotation_minimal_quadrants(section.counts[order[2]]).x,
           rotation_minimal_quadrants(section.counts[order[3]]).x)
    , count(count)
  {
    for (int a=0;a<4;a++)
      OTHER_ASSERT((a==2 || dest.shape[a]==src2.shape[a]) && (a==3 || dest.shape[a]==src3.shape[a]));
    OTHER_ASSERT(section==section.standardize<8>().x); // Prevent accidental computation of unnecessary data
    OTHER_ASSERT(dest.shape[2]==rmin[2].size());
    OTHER_ASSERT(dest.shape[3]==rmin[3].size());
    OTHER_ASSERT(section.counts[order[2]].sum()==9 || src2.shape[2]==rotation_minimal_quadrants(section.counts[order[2]]+move).x.size());
    OTHER_ASSERT(section.counts[order[3]].sum()==9 || src3.shape[3]==rotation_minimal_quadrants(section.counts[order[3]]+move).x.size());
  }

  void compute_slice(int ii, int jj) {
    // Run inner two dimensions sequentially.  Hopefully this produces nice cache behavior.
    thread_time_t time("compute");
    Vector<uint64_t,3> win_counts;
    for (int kk=0;kk<dest.shape[2];kk++)
      for (int ll=0;ll<dest.shape[3];ll++) {
        const quadrant_t q0 = rmin[0][ii],
                         q1 = rmin[1][jj],
                         q2 = rmin[2][kk],
                         q3 = rmin[3][ll];
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
        if (count)
          win_counts += Vector<uint64_t,3>(popcounts_over_stabilizers(board,d));
      }
    // Contribute to counts
    if (count) {
      lock_t lock(win_counts_mutex);
      this->win_counts += win_counts;
    }
  }
};

}

template<bool turn,bool final> static Vector<uint64_t,3> endgame_compute_block_slice_helper(const supertensor_writer_t& writer, Vector<int,4> order, int i, int j,
                                                                                            RawArray<Vector<super_t,2>,4> dest, RawArray<const Vector<super_t,2>,4> src2, RawArray<const Vector<super_t,2>,4> src3, bool count) {
  compute_helper_t<turn,final> helper(writer,order,i,j,dest,src2,src3,count);
  vector<function<void()>> jobs;
  for (const int ii : range(dest.shape[0]))
    for (const int jj : range(dest.shape[1]))
      jobs.push_back(boost::bind(&compute_helper_t<turn,final>::compute_slice,&helper,ii,jj));
  threads_schedule(CPU,jobs);
  threads_wait_all();
  return helper.win_counts;
}

static Vector<uint64_t,3> endgame_compute_block_slice(const supertensor_writer_t& writer, Vector<int,4> order, int i, int j,
                                                      RawArray<Vector<super_t,2>,4> dest, RawArray<const Vector<super_t,2>,4> src2, RawArray<const Vector<super_t,2>,4> src3, bool count) {
  const int stones = writer.header.section.sum();
  if (stones&1)
    return endgame_compute_block_slice_helper<1,false>(writer,order,i,j,dest,src2,src3,count);
  else if (stones<36)
    return endgame_compute_block_slice_helper<0,false>(writer,order,i,j,dest,src2,src3,count);
  else
    return endgame_compute_block_slice_helper<0,true >(writer,order,i,j,dest,src2,src3,count);
}

}
using namespace pentago;

void wrap_endgame() {
  OTHER_FUNCTION(endgame_read_block_slice)
  OTHER_FUNCTION(endgame_write_block_slice)
  OTHER_FUNCTION(endgame_compute_block_slice)
  OTHER_FUNCTION(endgame_verify)
  OTHER_FUNCTION(endgame_sparse_verify)

  typedef sparse_sample_t Self;
  Class<Self>("sparse_sample_t")
    .OTHER_INIT(const supertensor_writer_t&,int)
    .OTHER_GET(boards)
    .OTHER_GET(wins)
    ;
}
