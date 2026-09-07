// Pentago explorer page logic.
//
// Everything visible is built here at load: page chrome and the board svg
// come from template strings (loops make repetition free, so there are no
// <use>/<defs> tricks — just plain elements manipulated directly).  Board
// geometry is y-up per tex/transforms.tex, negated into screen coordinates.
// All state lives in the url hash as a comma-separated move history; render()
// rebuilds classes, links, and value dots from the localStorage cache, and
// kicks off a backend fetch or local wasm midsolve for missing values.
// Animation is pure css: rotating a quadrant sets a transitioned transform on
// the .q group, while an inner group counter-rotates instantly so the
// already-updated board starts visually at the old orientation.

import { parse_board } from './board.js'

const backend_url = 'https://us-central1-naml-148801.cloudfunctions.net/pentago/'
const colors = ['red', 'blue', 'lime']  // value + 1; blue and lime are exactly #0000ff and #00ff00

// Lazy localStorage cache of board values, cleared wholesale if full
const storage = localStorage
const cache_get = key => JSON.parse(storage.getItem(key))
const cache_set = (key, value) => {
  if (storage.length > 40000)  // ~2MB; each server response caches ~325 boards
    storage.clear()
  storage.setItem(key, value)
}

// Run midsolve in a lazily created web worker, so that most visitors never
// fetch the wasm at all.  The worker answers strictly in request order.
// Boards below 18 stones use the tiled solver, on as many threads as the
// browser allows: helper workers need SharedArrayBuffer, which needs the
// cross-origin isolation headers in firebase.json.
let worker = null
const worker_cbs = []
const local_threads = () =>
  typeof SharedArrayBuffer == 'undefined' ? 1 : Math.max(1, Math.min(8, navigator.hardwareConcurrency || 2))
const midsolve = board => new Promise((resolve, reject) => {
  if (!worker) {
    worker = new Worker(new URL('mid_worker.js', import.meta.url), {type: 'module'})
    worker.onmessage = e => (e.data instanceof Error ? worker_cbs.shift()[1] : worker_cbs.shift()[0])(e.data)
  }
  worker_cbs.push([resolve, reject])
  worker.postMessage({board: board.raw + '', threads: board.count < 18 ? local_threads() : 0})
})

// Values for the empty board and all of its children, baked in so the first
// page load paints fully with zero network requests.  First moves: corner
// placements tie, everything else wins.  Rotations of one-stone boards
// (value+1 as a bit per rotation, msb first in child order): rows and
// columns 1 and 4 all lose, so only the 4x4 subgrid at coordinates {0,2,3,5}
// is stored, one byte per spot.
if (cache_get('0') === null) {
  cache_set('0', 1)
  const table = 'ff7fefffbfc030dff70c03fefffbfdff'
  const idx = [0, 2, 3, 5]
  const b0 = parse_board('0')
  for (let s = 0; s < 36; s++) {
    const m = b0.place(s / 6 | 0, s % 6)
    cache_set(m.name, [0, 5, 30, 35].includes(s) ? 0 : 1)
    const x = idx.indexOf(s / 6 | 0), y = idx.indexOf(s % 6)
    const byte = x < 0 || y < 0 ? 0 : parseInt(table.substr(2 * (4 * x + y), 2), 16)
    let j = 0
    for (const qx of [0, 1])
      for (const qy of [0, 1])
        for (const d of [-1, 1])
          cache_set(m.rotate(qx, qy, d).name, (byte >> (7 - j++) & 1) - 1)
  }
}

// Page chrome
document.body.innerHTML =
  `<div class=all><header><h2>Pentago is a first player win</h2>` +
  `<h3>An interactive explorer for perfect pentago play</h3></header>` +
  `<div class=main><nav class=contents>` +
  ['/;Home', 'intro;Introduction', 'rules;Pentago rules', 'algorithms;Algorithms',
   'server;Server setup', 'open;Code and data', 'useful;Usefulness',
   'thanks;Acknowledgements', 'contact;Contact'].map(x => {
    const [h, t] = x.split(';')
    return `<a href=${h == '/' ? '/' : 'details.html#' + h}>${t}</a>`
  }).join('') +
  `<a class=back>Back</a></nav><div class=content>` +
  `<nav class=back><a class=back>Back</a></nav>` +
  `<svg id=board viewBox="-4.7 -5.5 9.4 12"> </svg>` +
  `<div id=status></div></div></div></div>`

// Board svg.  Construction order defines element meaning:
//   rotators: (q, d) for q in 0..3, d in (-1, 1) → q = i>>1, d = (i&1) ? 1 : -1
//   spots: quadrant loops then x,y within → s = 6x+y as below
const $ = id => document.getElementById(id)
// Set or remove an attribute, skipping no-op writes: setAttribute
// invalidates style and paint even when the value is unchanged, and
// render() re-derives every attribute on each pass
const set = (e, k, v) => {
  if (e.getAttribute(k) !== (v === null ? null : '' + v))
    v === null ? e.removeAttribute(k) : e.setAttribute(k, v)
}
const svg = $('board')
{
  // Base rotator for quadrant (0,0), d=1: hover wedge, arrow, value arc.
  // The other seven rotators are signed-permutation transforms of it.
  // Frozen data; unit.js test_rotator_paths regenerates these from the
  // geometry formulas (radius 2.5, thickness .2, arrow .4, wedge 4) and
  // asserts this file contains the results, so they can't rot.
  const SEL = 'M-1.55,1.55L-0.913,5.499A4,4 0 0 1 -4.19,4.555z'
  const ARROW = 'M-3.134,3.353A2.4,2.4 0 0 0 -1.55,3.95L-1.55,3.65L-1.152,4.018L-1.55,4.45' +
    'L-1.55,4.15A2.6,2.6 0 0 1 -3.266,3.503z'
  const RV = 'M-2.858,3.562A2.4,2.4 0 0 0 -1.895,3.925L-1.924,4.123A2.6,2.6 0 0 1 -2.968,3.73z'

  // Header (turn stone and label), footer (value legend), separator bars
  let s = `<circle id=tn class=c cy=-4.5 r=.4 /><circle id=hv class=v cy=-4.5 r=.15 />` +
    `<text class=tl id=hl y=-3.7>to play</text>` +
    [1, 0, -1].map(v => `<circle class=fv cx=${-1.5 * v} cy=5 r=.25 fill=${colors[v + 1]} />` +
      `<text class=tl x=${-1.5 * v} y=5.65>${['loss', 'tie', 'win'][v + 1]}</text>`).join('') +
    `<rect class=sep x=-.05 y=-3.15 width=.1 height=6.3 /><rect class=sep x=-3.15 y=-.05 width=6.3 height=.1 />`

  // Quadrants.  Rotators come before their quadrant so that on iPhone,
  // tapping a quadrant center can't cause a rotation followed by an errant
  // fake placed stone.  The outer group centers the quadrant, .q animates
  // via css transition, and the inner group counter-rotates instantly.
  for (const qx of [0, 1])
    for (const qy of [0, 1]) {
      const dx = 2 * qx - 1, dy = 2 * qy - 1
      for (const d of [-1, 1])
        s += `<a class=rot transform="matrix(${
          (d > 0) ^ (qx == qy) ? [0, dy, dx, 0] : [-dx, 0, 0, -dy]} 0 0)">` +
          `<path class=sel d="${SEL}"/><path class=arrow d="${ARROW}"/><path class=rv d="${RV}"/></a>`
      s += `<g class=q><g>` +
        `<rect class=board x=-1.5 y=-1.5 width=3 height=3 />`
      for (let x = 3 * qx; x < 3 * qx + 3; x++)
        for (let y = 3 * qy; y < 3 * qy + 3; y++)
          s += `<a><circle class=c cx=${x % 3 - 1} cy=${1 - y % 3} r=.4 /><circle class=v cx=${
            x % 3 - 1} cy=${1 - y % 3} r=.15 /></a>`
      s += `</g></g>`
    }
  svg.innerHTML = s + `<g id=fives> </g>`  // five-in-a-row markers land here
}
const status_el = $('status')
const backs = document.querySelectorAll('a.back')
const rots = [...svg.querySelectorAll('.rot')]
const spots = [...svg.querySelectorAll('g g a')]
const quads = [...svg.querySelectorAll('.q')]
const grids = quads.map(q => q.firstChild)
const rot_d = i => (i & 1) ? 1 : -1
const spot_s = i => {
  const qx = i / 18 | 0, qy = (i / 9 | 0) & 1, w = i % 9
  return 6 * (3 * qx + (w / 3 | 0)) + 3 * qy + w % 3
}

// Show a value dot (a .v circle or .rv arc, default-hidden in css), or hide it
const dot = (e, v) => {
  const d = v === null ? '' : 'inline'
  if (e.style.display != d)
    e.style.display = d
  if (v !== null)
    set(e, 'fill', colors[v + 1])
}

// Swivel state: how far each quadrant has visually rotated, in quarter turns.
// Swivel accumulates forever; the counter-rotation keeps net rotation zero
// once each transition finishes.
const swivel = [0, 0, 0, 0]
const spinning = [false, false, false, false]
const nospin = q => {
  if (spinning[q]) {
    spinning[q] = false
    render()
  }
}
for (const [q, e] of quads.entries())
  for (const ev of ['transitionend', 'transitioncancel'])
    e.addEventListener(ev, () => nospin(q))

// A rotator click only records the intended spin; render() applies it iff
// the hash really became the clicked link.  Mutating swivel here directly
// would race: a click that never produces its matching hashchange (e.g. a
// fast double click, whose second navigation is a no-op) would desync the
// accumulated rotation from the board by 90 degrees, invisible at rest but
// animating a spurious quadrant swing on the next render.
let pending_spin = null
for (const [i, a] of rots.entries())
  a.addEventListener('click', () => pending_spin = [i >> 1, rot_d(i), a.getAttribute('href')])

// The current board
let board = parse_board('0')

const history = () => (location.hash || '#0').slice(1).split(',')

const error = msg => {
  status_el.textContent = ''
  const d = document.createElement('div')
  d.id = 'error'
  d.textContent = msg
  status_el.append(d)
}

const loading = msg => {
  status_el.innerHTML = msg.split('').map((c, i) =>
    `<span class=load style=animation-delay:${(1.7 * i / msg.length).toFixed(2)}s>${c}</span>`).join('')
}

// Value of a board if we know it, else null.  Like the server, done boards
// know their own value.
const known = b => b.done ? b.value : cache_get(b.name)

// svg path for a five-in-a-row marker, in screen coordinates
const tweak = c => c - 2.5 + .05 * (c > 2 ? 1 : -1)
function five_path(f) {
  const x0 = tweak(f[0][0]), y0 = -tweak(f[0][1])
  const x1 = tweak(f[4][0]), y1 = -tweak(f[4][1])
  const dx = x1 - x0, dy = y1 - y0
  const s = .15 / 2 / Math.sqrt(dx * dx + dy * dy)
  const nx = s * dy, ny = -s * dx
  return `M${x0 + nx},${y0 + ny}L${x1 + nx},${y1 + ny}L${x1 - nx},${y1 - ny}L${x0 - nx},${y0 - ny}z`
}

function draw_fives(b) {
  const g = $('fives')
  const html = b.fives.flatMap(f => {
    if (f.some(([x, y]) => spinning[2 * (x / 3 | 0) + (y / 3 | 0)]))
      return []
    const color = b.grid[6 * f[0][0] + f[0][1]]
    return `<path class=five fill=${color == 1 ? 'black' : 'white'} d="${five_path(f)}"/>` +
      // Restore the black outlines of white stones under a white stripe
      (color == 2 ? f.map(([x, y]) => `<circle class=mask cx=${tweak(x)} cy=${-tweak(y)} r=.39 />`).join('') : '')
  }).join('')
  if (g.innerHTML != html)
    g.innerHTML = html
}

function render() {
  const hist = history()
  try {
    board = parse_board(hist[hist.length - 1])
  } catch (err) {
    error('Invalid board ' + location.hash + ', error = ' + err.message)
    return
  }
  const b = board
  const base = '#' + hist.join(',') + ','
  const mid = b.middle && !b.done

  // Start the spin recorded by a rotator click, if this render is its result
  let spin_d = 0
  if (pending_spin) {
    const [q, d, href] = pending_spin
    pending_spin = null
    if (location.hash == href) {
      swivel[q] += d
      spinning[q] = true
      spin_d = d
      // If the browser skips the transition (e.g. hidden tab), no transition
      // event ever fires, so time out just past the .5s transition
      setTimeout(() => nospin(q), 600)
    }
  }

  // Turn-dependent css state, header stone and label
  set(svg, 'class', (b.turn ? 'wt' : 'bt') + (mid ? ' mid' : ''))
  $('tn').style.fill = b.turn ? '#fff' : '#000'
  const v = known(b)
  dot($('hv'), v)
  $('hl').textContent = b.done ? ['loses!', 'ties!', 'wins!'][b.value + 1]
                      : v === null ? 'to play' : ['to lose', 'to tie', 'to win'][v + 1]
  let complete = b.done || v !== null

  // Back links
  const back = hist.length > 1 ? '#' + hist.slice(0, -1).join(',') : null
  for (const a of backs)
    set(a, 'href', back)

  // Quadrant swivels, from tex/transforms.tex (negated into screen
  // coordinates): during a spin, each .q group carries a translate-rotate
  // chain that both places the quadrant and, because the old and new chains
  // share the same function structure, css-interpolates as a swing keeping
  // the quadrant's inner corner in rolling contact with the separator
  // cross.  The chain's net rotation is -90*swivel degrees; the inner group
  // counter-rotates instantly so the already-updated board starts visually
  // at the old orientation and settles at net zero.  At rest the quadrant
  // carries only the equivalent trivial transform: repaints of a quadrant
  // bearing the six-function chain can catch chrome mid-evaluation and
  // paint one garbled frame (seen as a brief flash on stone placement).
  for (const q of [0, 1, 2, 3]) {
    const dx = q & 2 ? 1 : -1, dy = q & 1 ? 1 : -1
    const t = swivel[q]
    const chain = t => {
      const j = t & 1
      const T = v => `translate(${v * dx}px,${-v * dy}px)`
      const R = a => `rotate(${a}turn)`
      return T(.05) + R(1/8 - j/4) + T(3 / 8 ** .5) + R(j/2 - 1/4) +
        T(3 / 8 ** .5) + R(1/8 - j/4 - t/4)
    }
    const e = quads[q]
    if (spinning[q]) {
      if (e.style.transform != chain(t)) {
        // Entering a spin: jump (no transition) to the chain form of the
        // old orientation, so the transition interpolates chain-to-chain
        e.style.transition = 'none'
        e.style.transform = chain(t - spin_d)
        getComputedStyle(e).transform  // flush so the jump isn't transitioned
        e.style.transition = ''
        e.style.transform = chain(t)
      }
    } else {
      // At rest the placement lives in the svg transform attribute and the
      // css transform is cleared entirely.  Attribute first (same matrix
      // the chain form shows), then clear inline css with the transition
      // disabled: the chain and rest forms are equal as matrices but not
      // as lists, and since translate-rotate is a type prefix of the
      // six-function chain, css would otherwise pad the shorter list with
      // identity functions and interpolate per-function -- an animated
      // wobble after every spin rather than the identity it looks like.
      set(e, 'transform', `translate(${1.55 * dx} ${-1.55 * dy}) rotate(${-90 * t})`)
      if (e.style.transform) {
        e.style.transition = 'none'
        e.style.transform = ''
        getComputedStyle(e).transform  // flush so the collapse isn't transitioned
        e.style.transition = ''
      }
    }
    set(grids[q], 'transform', `rotate(${90 * t})`)
  }

  // Stones, placement links, and child values
  for (const [i, a] of spots.entries()) {
    const s = spot_s(i)
    const stone = b.grid[s]
    const q = 2 * (s / 18 | 0) + (s % 6 / 3 | 0)
    const open = !stone && !b.middle && !b.done
    set(a, 'class', stone ? stone == 1 ? 'b' : 'w' : open && !spinning[q] ? 'p' : '')
    let val = null
    if (open) {
      const child = b.place(s / 6 | 0, s % 6)
      set(a, 'href', base + child.name)
      val = known(child)
      complete &&= val !== null
    } else
      set(a, 'href', null)
    dot(a.children[1], val)
  }

  // Rotation links and values
  for (const [i, a] of rots.entries()) {
    let val = null
    if (mid) {
      const child = b.rotate(i >> 2, i >> 1 & 1, rot_d(i))
      set(a, 'href', base + child.name)
      const w = known(child)
      val = w === null ? null : -w
      complete &&= w !== null
    } else
      set(a, 'href', null)
    dot(a.children[2], val)
  }

  draw_fives(b)

  // Status, and a lookup if values are missing
  if (b.done)
    status_el.innerHTML = 'Game complete<br>' +
      (b.value ? (b.value > 0) == b.turn ? 'White wins!' : 'Black wins!' : 'Tie!')
  else if (complete) {
    if (status_el.firstChild)
      status_el.textContent = ''
  }
  else {
    const start = Date.now()
    const remote = b.count <= 15  // 16 and 17 stone boards are solved locally by the tiled solver
    // Show the loading animation only once the lookup has proven slow, so
    // fast lookups (warm server, refilling after a cache wipe) don't flash
    // text under the board
    const slow = setTimeout(() => {
      if (board.name == b.name)
        loading(remote ? 'Looking up ' + b.count + ' stone board...'
                       : 'Computing ' + b.count + ' stone board locally' +
                         (b.count < 18 && local_threads() > 1 ? ' on ' + local_threads() + ' threads...' : '...'))
    }, 250)
    const absorb = (op, values) => {
      clearTimeout(slow)
      for (const [raw, value] of Object.entries(values))
        cache_set(parse_board(raw).name, value)
      if (board.name == b.name) {
        render()
        status_el.innerHTML = op + ' ' + b.count + ' stone board<br>elapsed = ' +
          (Date.now() - start) / 1000 + ' s'
      }
    }
    const fail = err => {
      clearTimeout(slow)
      if (board.name == b.name)
        error(err.message)
    }
    if (remote)
      fetch(backend_url + b.name).then(async res => {
        if (!res.ok) throw Error('Server request failed, https status = ' + res.status)
        absorb('Received', await res.json())
      }).catch(fail)
    else
      midsolve(b).then(values => absorb('Computed', values)).catch(fail)
  }
}

// Favicon: the 18 stone test board 274440791932540184 with its true values
// as win/tie dots, drawn with a tiny canvas renderer.  The same decorative
// board appears in the standalone favicon.svg used by details.html.
{
  const b = parse_board('274440791932540184')
  const dots = '521a0'  // win/tie bit per empty spot, in grid order, msb first
  const cv = document.createElement('canvas')
  cv.width = cv.height = 60
  const g = cv.getContext('2d')
  const circle = (x, y, r, fill, stroke) => {
    g.beginPath()
    g.arc(x, y, r, 0, 7)
    g.fillStyle = fill
    g.fill()
    if (stroke)
      g.stroke()
  }
  g.fillStyle = 'tan'
  g.fillRect(0, 0, 60, 60)
  g.lineWidth = .3
  let e = 0
  for (let s = 0; s < 36; s++) {
    const x = 10 * (s / 6 | 0) + 5, y = 55 - 10 * (s % 6), v = b.grid[s]
    circle(x, y, 4, v == 1 ? 'black' : v == 2 ? 'white' : 'tan', true)
    if (!v)
      circle(x, y, 2, parseInt(dots[e >> 2], 16) >> (3 - (e++ & 3)) & 1 ? 'green' : 'blue')
  }
  const link = document.createElement('link')
  link.rel = 'icon'
  link.href = cv.toDataURL()
  document.head.append(link)
}

window.onhashchange = render
render()

// Test boards:
//   base: #0
//   rotation and fives: #238128874881424344m
//   white wins: #3694640587299947153m
//   black wins: #3694640600154188633m
//   tie: #3005942238600111847
//   midsolve: #274440791932540184
