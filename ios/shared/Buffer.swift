// Typed MTLBuffer

import Metal

class Buffer<T>: BufferLike {
  let offset: Int
  let count: Int
  let data: MTLBuffer
  private let raw: UnsafeMutablePointer<T>

  init(_ device: MTLDevice, _ count: Int) {
    offset = 0
    self.count = count
    data = MakeBuffer<T>(device).make(count: count)
    raw = data.contents().bindMemory(to: T.self, capacity: count)
  }
  
  init(_ src: Buffer<T>, _ r: Range<Int>) {
    assert(0 <= r.lowerBound)
    assert(r.upperBound <= src.count)
    offset = src.offset + r.lowerBound
    count = r.count
    data = src.data
    raw = src.raw
  }
  
  subscript(i: Int) -> T {
    get { assert(UInt(i) < UInt(count)); return raw[offset + i] }
    set(new) { assert(UInt(i) < UInt(count)); raw[offset + i] = new }
  }
  
  subscript(r: Range<Int>) -> Buffer<T> {
    get { Buffer<T>(self, r) }
  }
  
  func array() -> [T] {
    (0..<count).map { i in self[i] }
  }
  
  func set(_ compute: MTLComputeCommandEncoder, index: Int) {
    compute.setBuffer(data, offset: offset * MemoryLayout<T>.size, index: index)
  }
}

class GPUBuffer<T>: BufferLike {
  let offset: Int
  let count: Int
  let data: MTLBuffer

  init(_ device: MTLDevice, _ count: Int) {
    offset = 0
    self.count = count
    data = MakeBuffer<T>(device).gpu(count: count)
  }

  init(_ src: GPUBuffer<T>, _ r: Range<Int>) {
    assert(0 <= r.lowerBound)
    assert(r.upperBound <= src.count)
    offset = src.offset + r.lowerBound
    count = r.count
    data = src.data
  }

  subscript(r: Range<Int>) -> GPUBuffer<T> {
    get { GPUBuffer<T>(self, r) }
  }

  func set(_ compute: MTLComputeCommandEncoder, index: Int) {
    compute.setBuffer(data, offset: offset * MemoryLayout<T>.size, index: index)
  }
}

// Work around the 256 MB buffer size limit
class BigGPUBuffer<T> {
  let chunks: [GPUBuffer<T>]
  
  init(_ device: MTLDevice, chunks: Int, count: Int) {
    let maxBufferLength = 256 << 20  // Use a constant so that .metal files can have the same constant
    let chunkSize = maxBufferLength / MemoryLayout<T>.size
    assert(maxBufferLength == chunkSize * MemoryLayout<T>.size)
    assert(count <= chunkSize * chunks)
    self.chunks = (0..<chunks).map { c in
      GPUBuffer<T>(device, max(0, min(chunkSize, count - c * chunkSize))) }
    assert(self.count == count)
  }
  
  var count: Int { chunks.map { $0.count }.reduce(0, +) }
}
