// Pentago board operations

// Utilities
const six = [0,1,2,3,4,5]
const big = BigInt
const flat_map2 = (zs, f) => zs.flatMap(x => zs.flatMap(y => f(x,y)))
const grid_map = f => flat_map2(six.map(big), (x,y) => [f(x, y, 65536n**(x / 3n * 2n + y / 3n), 3n**(x % 3n * 3n + y % 3n))])
const assert = (cond, name) => { if (!cond) throw Error('board '+name) }

// Lines that count as wins
const win_rays = flat_map2(six, (x,y) => [[0,4],[1,0],[1,4],[1,-4]].flatMap(([a,b]) =>
  x+4*a < 6 & 0 <= y+b & y+b < 6 ? [[0,1,2,3,4].map(i => 6*(x + i*a) + y + i*b/4)] : []))

// Usage: new board_t(stones,middle), parse_board(name)
function board_t(grid, middle) {
  // Count stones
  let stones = 0n
  let count = 0  // count0 + count1
  let shift = 0  // 2*count1
  grid_map((x,y,q,p) => {
    let s = grid[6n*x+y]
    stones += big(s) * p * q
    count += !!s
    shift += s & 2
  })
  const name = stones + (middle ? 'm' : '')
  const turn = count - shift != middle

  // Place a stone at the given location
  const place = (x, y) => {
    let g = grid.slice()
    g[6*x+y] = 1 + turn
    return new board_t(g, 1)
  }

  // Rotate the given quadrant left (d=1) or right (d=-1)
  const rotate = (qx, qy, d) => {
    let g = grid.slice()
    let q = 18*qx + 3*qy
    flat_map2([0,1,2], (x,y) => g[q + 6*x + y] = grid[q + 7 + d*(6*y-x-5)])
    return new board_t(g, 0)
  }

  // Does the given side have 5 in a row?
  const won = side => win_rays.some(ray => ray.every(p => grid[p] & side + 1))

  // Most fields
  const B = this
  assert(count - shift - turn == middle - 2*turn*middle, name)  // Assert count0 and count1 are consistent
  B.grid = grid
  B.middle = middle
  B.name = name
  B.raw = stones | big(middle) << 63n
  B.turn = turn
  B.count = count
  B.place = place
  B.rotate = rotate
  B.done = won(0) | won(1) | (count==36 & !middle)
  B.value = won(turn) - won(!turn)

  // List moves
  B.moves = () =>
    middle ? flat_map2([0,1], (x,y) => [-1,1].map(d => rotate(x,y,d)))
           : flat_map2(six, (x,y) => grid[6*x+y] ? [] : [place(x,y)])

  // List of coordinates for each five in a row
  B.fives = win_rays.flatMap(ray => ray.every(p => grid[p] * grid[ray[0]] & 5) ? [ray.map(p => [(p-p%6)/6, p%6])] : [])
}

// Convert from canonical board name or raw_t
export function parse_board(name) {
  let m = (name + '').match(/^(\d+)(m?)$/)
  assert(m, name)
  let stones = big(m[1])
  let grid = grid_map((x,y,q,p) => +('' + stones / q % 32768n / p % 3n))  // Convert to Number using few characters
  return new board_t(grid, !!m[2] || stones >> 63n != 0)  // middle if we have an 'm', or the raw_t middle bit is set
}
