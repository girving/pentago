// Here's some pseudocode code that streams through a bunch of supers in random order

#include "pentago/base/all_boards.h"
#include "pentago/base/count.h"
#include "pentago/end/blocks.h"
#include "pentago/end/sections.h"
#include "pentago/high/index.h"
#include "pentago/utility/const_cast.h"
#include "pentago/utility/log.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <string_view>
namespace pentago {
namespace {

using namespace end;
using namespace tensorflow;
using std::atomic;
using std::make_shared;
using std::make_tuple;
using std::min;
using std::string_view;

// Index sizes:
//   14: 54.79 MB
//   15: 118.02 MB
//   16: 258.77 MB
//   17: 483.89 MB
//   18: 908.39 MB
//   14-18: 1823.86 MB
// Path:
//   gs://pentago/edison/slice-%d.pentago.index

const int max_slice = 18;

struct all_blocks_t {
  const vector<shared_ptr<const sections_t>> all_sections;
  const vector<shared_ptr<const supertensor_index_t>> indices;
  const Array<const int> section_offsets;
  const Array<const int> block_offsets;

  all_blocks_t()
    : all_sections(descendent_sections(section_t(), max_slice)) {
    vector<int> section_offsets = {0};
    vector<int> block_offsets = {0};
    for (const auto& sections : all_sections) {
      const_cast_(indices).push_back(make_shared<supertensor_index_t>(sections));
      section_offsets.push_back(section_offsets.back() + sections->sections.size());
      for (const auto& section : sections->sections)
        block_offsets.push_back(block_offsets.back() + section_blocks(section).product());
    }
    const_cast_(this->section_offsets) = asarray(section_offsets).copy();
    const_cast_(this->block_offsets) = asarray(block_offsets).copy();
  }

  // Blocks for slices [0, slice]
  int blocks(const int slice) const {
    GEODE_ASSERT(section_offsets.valid(slice + 1),
                 format("max slice = %d < slice %d", max_slice, slice));
    return block_offsets[section_offsets[slice + 1]];
  }

  supertensor_index_t::block_t block(const int block_i) const {
    const auto find = [](const auto& offsets, const auto n) {
      GEODE_ASSERT(0 <= n && n < offsets.back());
      const auto lo = offsets.begin() + 1;
      const int i = std::upper_bound(lo, offsets.end(), n) - lo;
      GEODE_ASSERT(0 <= i && i < offsets.size() - 1);
      return i;
    };
    const int section_i = find(block_offsets, block_i);
    const int slice = find(section_offsets, section_i);
    const auto section = all_sections[slice]->sections[section_i - section_offsets[slice]];
    const auto block = decompose(section_blocks(section), block_i - block_offsets[section_i]);
    return make_tuple(section, Vector<uint8_t,4>(block));
  }
};

const all_blocks_t& all_blocks() {
  static const all_blocks_t single;
  return single;
}

struct BlockCounts : public OpKernel {
 public:
  explicit BlockCounts(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    Tensor* counts_t;
    OP_REQUIRES_OK(c, c->allocate_output(0, {max_slice + 1}, &counts_t));
    auto counts = counts_t->vec<int>();
    for (const int slice : range(max_slice + 1))
      counts(slice) = all_blocks().blocks(slice);
  }
};

struct CountBoards : public OpKernel {
 public:
  explicit CountBoards(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    const Tensor& slice_t = c->input(0);
    const Tensor& symmetries_t = c->input(1);
    OP_REQUIRES(c, slice_t.dims() == 0,
                errors::InvalidArgument("slice.shape = ", slice_t.shape().DebugString(), " != []"));
    OP_REQUIRES(c, symmetries_t.dims() == 0,
                errors::InvalidArgument("symmetries.shape = ", symmetries_t.shape().DebugString(), " != []"));
    const int slice = internal::SubtleMustCopy(slice_t.scalar<int>()());
    const int symmetries = internal::SubtleMustCopy(symmetries_t.scalar<int>()());
    int64_t count = 0;
    try {
      if (symmetries == -8)
        for (const auto& s : all_boards_sections(slice, 8))
          count += s.size();
      else
        count = count_boards(slice, symmetries);
    } catch (const std::exception& e) {
      OP_REQUIRES(c, false, errors::Internal(e.what()));
    }
    Tensor* count_t;
    OP_REQUIRES_OK(c, c->allocate_output(0, {}, &count_t));
    count_t->scalar<int64_t>()() = count;
  }
};

struct BlockInfo : public OpKernel {
 public:
  explicit BlockInfo(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    try {
      // Check input
      const auto indices_t = c->input(0);
      const auto index_t = c->input(1);
      OP_REQUIRES(c, indices_t.dims() == 1 && indices_t.dim_size(0) <= max_slice + 1,
                  errors::InvalidArgument("Expected indices.shape = [0]..[", max_slice + 1, "], got ",
                                          indices_t.shape().DebugString()));
      OP_REQUIRES(c, index_t.dims() == 0,
                  errors::InvalidArgument("index.shape = ", index_t.shape().DebugString(), " != []"));
      const int max_slice = indices_t.dim_size(0) - 1;
      const auto index = internal::SubtleMustCopy(index_t.scalar<int>()());
      const auto& blocks = all_blocks();
      const int count = blocks.blocks(max_slice);
      OP_REQUIRES(c, 0 <= index && index < count,
                  errors::InvalidArgument("index = ", index, " not in [0,", count, ")"));

      // Gather information about the block
      const auto info = blocks.block(index);
      const auto& [section, block] = info;
      const int slice = section.sum();
      const auto index_blob = blocks.indices[slice]->blob_location(info);
      const string_view index_data = indices_t.vec<string>()(slice);
      GEODE_ASSERT(index_blob.offset() + index_blob.size <= index_data.size());
      const auto blob = blocks.indices[slice]->block_location(RawArray<const uint8_t>(
          index_blob.size, reinterpret_cast<const uint8_t*>(index_data.data()) + index_blob.offset()));

      // Store slice and compressed blob location as tensors
      Tensor *slice_t, *blob_offset_t, *blob_size_t;
      OP_REQUIRES_OK(c, c->allocate_output(0, {}, &slice_t));
      OP_REQUIRES_OK(c, c->allocate_output(1, {}, &blob_offset_t));
      OP_REQUIRES_OK(c, c->allocate_output(2, {}, &blob_size_t));
      slice_t->scalar<int>()() = slice;
      blob_offset_t->scalar<int64_t>()() = blob.offset();
      blob_size_t->scalar<int64_t>()() = blob.size;

      // Store board quadrants as tensors
      for (const int q : range(4)) {
        const auto all_rmins = get<0>(rotation_minimal_quadrants(section.counts[q]));
        const RawArray<const uint16_t> rmins = all_rmins.slice(
            block_size*block[q], min(block_size*(block[q]+1), all_rmins.size()));
        Tensor* quads_t;
        OP_REQUIRES_OK(c, c->allocate_output(3 + q, {rmins.size()}, &quads_t));
        auto quads = quads_t->vec<uint16_t>();
        memcpy(quads.data(), rmins.data(), sizeof(uint16_t)*rmins.size());
      }
    } catch (const std::exception& e) {
      OP_REQUIRES(c, false, errors::Internal(e.what()));
    }
  };
};

struct UnpackBlock : public OpKernel {
 public:
  explicit UnpackBlock(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    // Check input
    const auto index_t = c->input(0);
    const auto data_t = c->input(1);
    OP_REQUIRES(c, index_t.dims() == 0,
                errors::InvalidArgument("index.shape = ", index_t.shape().DebugString(), " != []"));
    OP_REQUIRES(c, data_t.dims() == 0,
                errors::InvalidArgument("data.shape = ", data_t.shape().DebugString(), " != []"));
    const auto index = internal::SubtleMustCopy(index_t.scalar<int>()());
    const string& data_s = data_t.scalar<string>()();
    const RawArray<const uint8_t> data(data_s.size(), reinterpret_cast<const uint8_t*>(data_s.data()));

    try {
      // Unpack data
      const auto supers = supertensor_index_t::unpack_block(all_blocks().block(index), data);

      // Store as output
      const auto& shape = supers.shape();
      Tensor* supers_t;
      OP_REQUIRES_OK(c, c->allocate_output(0, {shape[0], shape[1], shape[2], shape[3], 2, 32}, &supers_t));
      auto flat = supers_t->flat<uint8_t>();
      const auto bytes = sizeof(Vector<super_t,2>)*supers.total_size();
      GEODE_ASSERT(flat.dimension(0) == bytes);
      memcpy(flat.data(), supers.data(), bytes);
    } catch (const std::exception& e) {
      OP_REQUIRES(c, false, errors::Internal(e.what()));
    }
  };
};

struct Pread : public OpKernel {
 public:
  explicit Pread(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    // Check input
    const char* names[] = {"path", "offset", "size"};
    for (const int i : range(3))
      OP_REQUIRES(c, c->input(i).dims() == 0,
                  errors::InvalidArgument(names[i], ".shape = ", c->input(i).shape().DebugString(), " != []"));
    const string& path = c->input(0).scalar<string>()();
    const auto offset = internal::SubtleMustCopy(c->input(1).scalar<int64>()());
    const auto size = internal::SubtleMustCopy(c->input(2).scalar<int64>()());
    OP_REQUIRES(c, 0 <= size && size <= (1<<30),
                errors::InvalidArgument("size = ", size, " not in [0,1<<30]"));

    // Read data
    unique_ptr<RandomAccessFile> file;
    OP_REQUIRES_OK(c, c->env()->NewRandomAccessFile(path, &file));
    string data(size, 0);
    StringPiece result;
    OP_REQUIRES_OK(c, file->Read(offset, size, &result, data.data()));
    OP_REQUIRES(c, result.size() == size, errors::InvalidArgument("read ", result.size(), " != ", size, " bytes"));

    // Move to output
    Tensor* data_t;
    OP_REQUIRES_OK(c, c->allocate_output(0, {}, &data_t));
    data_t->scalar<string>()() = std::move(data);
  }
};

#define REGISTER(name) \
  REGISTER_KERNEL_BUILDER(Name(#name).Device(DEVICE_CPU), name); \
  REGISTER_OP(#name)

REGISTER(BlockCounts)
    .Output("counts: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      c->set_output(0, c->MakeShape({max_slice + 1}));
      return Status::OK();
    });

REGISTER(CountBoards)
    .Input("slice: int32")
    .Input("symmetries: int32")
    .Output("count: int64");

REGISTER(BlockInfo)
    .Input("indices: string")
    .Input("index: int32")
    .Output("slice: int32")
    .Output("offset: int64")
    .Output("size: int64")
    .Output("quads0: uint16")
    .Output("quads1: uint16")
    .Output("quads2: uint16")
    .Output("quads3: uint16")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      for (const int i : range(3))
        c->set_output(i, c->Scalar());
      for (const int i : range(3, 7))
        c->set_output(i, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER(Pread)
    .Input("path: string")
    .Input("offset: int64")
    .Input("size: int64")
    .Output("data: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      for (const int i : range(3))
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER(UnpackBlock)
    .Input("index: int32")
    .Input("data: string")
    .Output("supers: uint8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      for (const int i : range(2))
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
      c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), c->UnknownDim(), 2, 32}));
      return Status::OK();
    });

}  // namespace
}  // namespace pentago
