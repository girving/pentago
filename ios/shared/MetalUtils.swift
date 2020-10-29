// Metal utilities

import Metal

func ceilDiv(_ x: Int, _ y: Int) -> Int {
  (x + y - 1) / y
}

class MakeBuffer<T> {
  let device: MTLDevice

  init(_ device: MTLDevice) {
    self.device = device
  }

  private func length(_ count: Int) -> Int {
    assert(count >= 0)
    return max(count, 1) * MemoryLayout<T>.size
  }

  func make(count: Int) -> MTLBuffer {
    device.makeBuffer(length: length(count))!
  }

  func make(data: [T]) -> MTLBuffer {
    device.makeBuffer(bytes: data, length: length(data.count))!
  }

  // Buffer that exists only on the GPU
  func gpu(count: Int) -> MTLBuffer {
    device.makeBuffer(length: length(count), options: [.storageModePrivate])!
  }
}

func setVertexBytes<T>(_ encoder: MTLRenderCommandEncoder, _ data: T, index: Int) {
  encoder.setVertexBytes([data], length: MemoryLayout.size(ofValue: data), index: index)
}

func setFragmentBytes<T>(_ encoder: MTLRenderCommandEncoder, _ data: T, index: Int) {
  encoder.setFragmentBytes([data], length: MemoryLayout.size(ofValue: data), index: index)
}

func setBytes<T>(_ encoder: MTLComputeCommandEncoder, _ data: T, index: Int) {
  encoder.setBytes([data], length: MemoryLayout.size(ofValue: data), index: index)
}

func function(_ lib: MTLLibrary, _ name: String) -> MTLFunction {
  if let f = lib.makeFunction(name: name) {
    return f
  } else {
    assertionFailure("Shader '\(name)' not found")
    exit(1)
  }
}

protocol BufferLike {
  func set(_ compute: MTLComputeCommandEncoder, index: Int)
}
struct Small<T>: BufferLike {
  let x: T
  init(_ x: T) { self.x = x }
  func set(_ compute: MTLComputeCommandEncoder, index: Int) {
    setBytes(compute, x, index: index)
  }
}

// iPhone 7 and even recent MacBook Pros don't support varying threadgroups, unfortunately, so we
// needs bounds checking in each compute kernel.
func dispatch(_ compute: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, _ count: Int) {
  if count == 0 { return }
  let t = min(count, pipeline.threadExecutionWidth)
  compute.dispatchThreadgroups(MTLSizeMake(ceilDiv(count, t), 1, 1), threadsPerThreadgroup: MTLSizeMake(t, 1, 1))
  // When varying works, we can switch to
  // compute.dispatchThreads(MTLSizeMake(count, 1, 1), threadsPerThreadgroup: MTLSizeMake(t, 1, 1))
}

// Set pipeline state and then dispatch
func dispatchSet(_ compute: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, _ count: Int) {
  compute.setComputePipelineState(pipeline)
  dispatch(compute, pipeline, count)
}

class Compute {
  let pipeline: MTLComputePipelineState
  
  init(_ device: MTLDevice, _ lib: MTLLibrary, _ f: String) {
    pipeline = try! device.makeComputePipelineState(function: function(lib, f))
  }

  func run(_ compute: MTLComputeCommandEncoder, _ buffers: Array<BufferLike>, _ count: Int) {
    compute.setComputePipelineState(pipeline)
    for i in 0..<buffers.count {
      buffers[i].set(compute, index: i)
    }
    dispatch(compute, pipeline, count)
  }
}
