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

func dispatch(_ compute: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, _ count: Int) {
  if count == 0 { return }
  let threads = min(count, pipeline.threadExecutionWidth)
  compute.dispatchThreads(MTLSizeMake(count, 1, 1), threadsPerThreadgroup: MTLSizeMake(threads, 1, 1))
}

func set(_ compute: MTLComputeCommandEncoder, _ buffers: [BufferLike], _ r: Range<Int>) {
  assert(buffers.count == r.count)
  for i in r {
    buffers[i - r.lowerBound].set(compute, index: i)
  }
}

class Compute {
  let pipeline: MTLComputePipelineState
  
  init(_ device: MTLDevice, _ lib: MTLLibrary, _ f: String) {
    pipeline = try! device.makeComputePipelineState(function: function(lib, f))
  }

  func run(_ compute: MTLComputeCommandEncoder, _ buffers: [BufferLike], _ count: Int) {
    compute.setComputePipelineState(pipeline)
    set(compute, buffers, 0..<buffers.count)
    dispatch(compute, pipeline, count)
  }
}

protocol CaptureLike { func stop() }
struct Capture: CaptureLike {
  let c: MTLCaptureManager
  let once: Bool
  func stop() { c.stopCapture(); if once { exit(0) } }
}
class NoCapture: CaptureLike { func stop() {} }

// Write a gpu trace to Documents/pentago.gputrace.
// To get the trace, see https://stackoverflow.com/questions/15219511
func makeCapture(_ device: MTLDevice, on: Bool = true, xcode: Bool = false,
                 once: Bool = false) -> CaptureLike {
  if !on { return NoCapture() }
  let capture = MTLCaptureManager.shared()
  let desc = MTLCaptureDescriptor()
  desc.captureObject = device
  if !xcode {
    desc.destination = .gpuTraceDocument
    let path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    desc.outputURL = path.appendingPathComponent("pentago.gputrace")
  }
  try! capture.startCapture(with: desc)
  return Capture(c: capture, once: once)
}
