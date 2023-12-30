<svg id="board" width={width} height={scale*(6+header_size+footer_size)}>
  <g transform="{'translate('+scale*(3+margin_size)+','+scale*(3+header_size)+') scale('+scale+','+-scale+') '}">
    <!-- Header -->
    <circle class="{turncolor}" cx=0 cy={header_y} r={spot_radius}/>
    {#await child_value(board) then v}
      <circle class="tvalue" cx=0 cy={header_y} r={value_radius} style="fill: {value_colors[v]}"/>
    {/await}
    <text class="turnlabel" transform="scale(1,-1)" x=0 y={-(header_y-spot_radius-font_size)}
        style="font-size: {font_size}px">
      {#await turn_label}
        to play
      {:then label}
        {label}
      {/await}
    </text>

    <!-- Footer -->
    {#each [1, 0, -1] as v}
      <g class="footer">
        <circle class="fvalue" cx={-footer_sep*v} cy={footer_cy} r={footer_radius} fill={value_colors[v]}/>
        <text class="valuelabel" transform="scale(1,-1)" x={-footer_sep*v} y={-(footer_cy-footer_radius-font_size)}
            style="font-size: {font_size}px">
          {{'1':'win', '0':'tie', '-1':'loss'}[v]}
        </text>
      </g>
    {/each}

    <!-- Separators -->
    <rect class="separators" x={-bar_size/2} y={-(bar_size+6.2)/2} width={bar_size} height={bar_size+6.2}/>
    <rect class="separators" y={-bar_size/2} x={-(bar_size+6.2)/2} height={bar_size} width={bar_size+6.2}/>

    <!-- Board -->
    {#each quadrant_data as q}
      <!-- Rotators -->
      <!-- Important to put these prior to quadrants; otherwise on iPhone tapping on a quadrant center causes -->
      <!-- a rotation falled by an errant fake placed stone. -->
      {#if !done && board.middle}
        {#each q.rotators as r}
          <a href="{rotate_link(r)}" on:click={spin(r)}>
            <path class="rotateselect" d="{r.select}"/>
            <path class="rotate{turncolor}" d="{r.path}"/>
            {#if board.middle && !board.done}
              {#await child_value(board.rotate(r.qx, r.qy, r.d)) then v}
                <path class="rvalue" d="{r.value}" style="fill: {value_colors[-v]}"/>
              {/await}
            {/if}
          </a>
        {/each}
      {/if}

      <!-- Quadrants -->
      <g class="quadrant" style="transform: {transform(q, swivel[q.q])}" on:transitionend={nospin(q)}>
        <g transform="rotate({-90*swivel[q.q]})">
          <rect class="board" x=-1.5 y=-1.5 width=3 height=3/>

          <!-- Spots -->
          {#each q.grid as s}
            <g transform="translate({s.x%3-1},{s.y%3-1})">
              <a href="{spot_link(s)}">
                <circle class="{spot_class(spinning[q.q], board.grid[s.s])}" r={spot_radius}/>
                {#if !(board.middle || board.done || board.grid[s.s])}
                  {#await child_value(board.place(s.x, s.y)) then v}
                    <circle class="cvalue" r={value_radius} style="fill: {value_colors[v]}"/>
                  {/await}
                {/if}
              </a>
            </g>
          {/each}
        </g>
      </g>
    {/each}

    <!-- Fives-in-a-row -->
    {#each fives as f}
      <path class="five" style="fill: {five_color(f) == 1 ? 'black' : 'white'}" d={five_path(f)}/>
      {#if five_color(f) == 2}
        {#each f as [x, y]}
          <circle class="mask" cx={five_tweak(x)} cy={five_tweak(y)} r={spot_radius - 0.01}/>
        {/each}
      {/if}
    {/each}
  </g>
</svg>

<div class="status">
  {#if error}
    <div id="error">{error}</div>
  {:else if done}
    Game complete<br>
    {board.value ? (board.value > 0) == board.turn ? 'White wins!' : 'Black wins!' : 'Tie!'}
  {:else}
    {#await status}
      {#each loading.split('') as c, i}
        <div class="load" style="animation-delay: {1.7*i/loading.length}s">{c}</div>
      {/each}
    {:then lines}
      {#each lines as line}
        {line}<br>
      {/each}
    {:catch e}
      <div id="error">{e.message}</div>
    {/await}
  {/if}
</div>

<svelte:window on:resize={resize}/>

<style>
  svg {
    width: 400px;
    max-width: 100%;
    display: block;
    margin-left: auto;
    margin-right: auto;
    /* Improve animations on iOS, following
     * https://chrissilich.com/blog/fix-css-animation-slow-or-choppy-in-mobile-browsers */
    -webkit-transform: translateZ(0);
  }

  .black,.white,.empty,.emptyblack,.emptywhite {
    stroke: black;
    stroke-width: 1;
    vector-effect: non-scaling-stroke;
  }
  .empty,.emptyblack,.emptywhite { fill: tan }
  .emptyblack:hover, .emptyblack:hover + .emptyblack { fill: black }
  .emptywhite:hover, .emptywhite:hover + .emptywhite { fill: white }
  .black { fill: black }
  .white { fill: white }
  .board { fill: tan }
  .separators { fill: darkgray }

  .rotateselect { opacity: 0 }
  .rotateselect:hover + .rotateblack { fill: black }
  .rotateselect:hover + .rotatewhite { fill: white }
  .norotate { visibility: hidden }
  .rotateblack,.rotatewhite {
    fill: tan;
    stroke: black;
    stroke-width: 1;
    vector-effect: non-scaling-stroke;
  }
  .rotateblack:hover { fill: black }
  .rotatewhite:hover { fill: white }

  .cvalue,.rvalue,.tvalue {
    stroke: gray;
    stroke-width: .5;
    vector-effect: non-scaling-stroke;
    pointer-events: none;
  }
  .fvalue {
    stroke: gray;
    stroke-width: .5;
    vector-effect: non-scaling-stroke;
  }
  .rvalue {
    stroke: black;
    stroke-width: 1;
  }

  .turnlabel,.valuelabel {
    text-anchor: middle;
  }

  .five {
    stroke: black;
    stroke-width: 1;
    vector-effect: non-scaling-stroke;
  }
  .mask {
    fill: white;
  }

  .quadrant {
    transition: transform 0.5s ease-in-out;
  }

  .status {
    text-align: center;
    width: 100%;
    height: 3em;
  }

  #error { color: red; }

  .load {
    display: inline;
    animation: spin 2s infinite ease;
  }
  @keyframes spin {
    0% { color: black; text-shadow: none; }
    20% { color: purple; text-shadow: 0px 0px 6px purple; }
    40% { color: black; text-shadow: none; }
  }
</style>

<script>
  import { parse_board } from './board.js'
  import pending from './pending.js'
  import { midsolve } from './mid_async.js'
  import { get as cache_get, set as cache_set } from './cache.js'
  import { onMount } from 'svelte'

  // Pull in math stuff
  const pi = Math.PI
  const cos = Math.cos
  const sin = Math.sin
  const sqrt = Math.sqrt
  const floor = Math.floor

  // Backend, with a bit of caching to avoid flicker on the back button
  const backend_url = 'https://us-central1-naml-148801.cloudfunctions.net/pentago/'

  // Track url hash
  let hash = window.location.hash || '#0'
  window.onhashchange = () => {
    const h = window.location.hash
    if (hash != h) hash = h
  }

  // Board and history
  let history, board, error
  export let back
  $: history = hash ? hash.slice(1).split(',') : []
  $: {
    board = parse_board('0')
    error = null
    if (history.length) {
      try {
        board = parse_board(history[history.length - 1])
      } catch (e) {
        const s = 'Invalid board '+hash+', error = '+e.message
        console.log(s, e)
        error = s
      }
    }
  }
  $: back = history.length > 1 ? '#'+history.slice(0, -1).join(',') : null

  // Derived board information
  let turncolor, spot_class, base, done, spot_link, rotate_link, loading, status, child_value, turn_label
  $: {
    // Basics
    const b = board
    turncolor = b.turn ? 'white' : 'black'
    done = b.done
    spot_class = (sp, v) => v ? v == 1 ? 'black' : 'white' : 'empty' + (done || b.middle || sp ? '' : turncolor)
    base = '#' + history.join(',') + ','
    spot_link = s => done || b.middle || b.grid[s.s] ? null : base + b.place(s.x, s.y).name
    rotate_link = r => base + board.rotate(r.qx, r.qy, r.d).name

    // Start asynchronous lookup / local computation as required
    status = ''
    loading = null
    const has = c => cache_get(c.name) !== null
    if (!b.done && !(has(b) && b.moves().every(has))) {
      const start = Date.now()
      function absorb(op, values) {
        const elapsed = (Date.now() - start) / 1000
        for (const [raw, value] of Object.entries(values))
          cache_set(parse_board(raw).name, value)
        return [op + ' ' + board.count + ' stone board', 'elapsed = ' + elapsed + ' s']
      }
      if (board.count <= 17) {  // Look up via server
        loading = 'Looking up ' + board.count + ' stone board...'
        status = fetch(backend_url + board.name).then(async res => {
          if (res.ok)
            return absorb('Received', await res.json())
          else {
            const s = 'Server request failed, https status = ' + res.status
            console.log(s, res)
            throw Error(s)
          }
        })
      } else {  // Compute locally via WebAssembly
        loading = 'Computing ' + board.count + ' stone board locally...'
        status = midsolve(board).then(values => absorb('Computed', values))
      }
    }

    // Value promises
    child_value = async child => {
      if (child.done)
        return child.value
      const v = cache_get(child.name)
      if (v !== null)
        return v
      await status
      return cache_get(child.name)
    }
    turn_label = b.done ? {'1': 'wins!', '0': 'ties!', '-1': 'loses!'}[b.value]
                        : child_value(b).then(v => ({'1': 'to win', '0': 'to tie', '-1': 'to lose'}[v]))
  }

  // Swivel state (how far we've rotated each quadrant)
  let swivel = [0, 0, 0, 0]
  let spinning = [false, false, false, false]
  const spin = r => () => {
    hash = rotate_link(r)
    swivel[r.q] += r.d
    spinning[r.q] = true
  }
  const nospin = q => () => spinning[q.q] = false
  const transform = (q, t) => {
    // See transforms.tex for details
    const j = t & 1
    const d = [q.x - 1/2, q.y - 1/2]
    const w0 = d.map(c => bar_size * c)
    const w = d.map(c => 3 / sqrt(2) * c)
    const T = ([x,y]) => `translate(${x}px,${y}px)`
    const R = t => `rotate(${t}turn)`
    return [T(w0), R(j/4-1/8), T(w), R(1/4-j/2), T(w), R(t/4+j/4-1/8)].join(' ')
  }

  // Fives
  let fives
  $: fives = board.fives.filter(f => !f.some(([x,y]) => spinning[2*floor(x/3)+floor(y/3)]))

  // Drawing parameters
  const bar_size = .1
  const spot_radius = .4
  const header_size = 2.5
  const footer_size = 3.5
  const margin_size = 1.7
  const value_radius = .15
  const rotator_radius = 2.5
  const rotator_thickness = .2
  const rotator_arrow = .4
  const select_radius = 4
  const font_size = .4
  const header_y = 4.5
  const footer_sep = 1.5
  const footer_cy = -5
  const footer_radius = .25

  // Colors for each board value, taking care to be nice to colorblind folk.
  const value_colors = {'1': '#00ff00', '0': '#0000ff', '-1': '#ff0000', 'undefined': null}

  // Track svg width
  let width = 0
  const resize = () => {
    const w = window.getComputedStyle(document.getElementById('board'))['width']
    width = parseInt(w.match(/^(\d+)px$/)[1])
  }
  onMount(resize)
  $: scale = width / (6 + 2*margin_size)

  // Quadrant data
  const quadrant_data = []
  for (const qx of [0, 1]) {
    for (const qy of [0, 1]) {
      // Spot data
      const grid = []
      for (let x = 0; x < 3; x++) {
        for (let y = 0; y < 3; y++) {
          const sx = 3*qx+x
          const sy = 3*qy+y
          grid.push({x: sx, y: sy, s: 6*sx+sy})
        }
      }

      // Rotator data
      const rotators = []
      for (const d of [-1, 1]) {
        const dx = qx ? 1 : -1
        const dy = qy ? 1 : -1
        const cx = 3*qx+1-2.5+bar_size/2*dx
        const cy = 3*qy+1-2.5+bar_size/2*dy
        const r = rotator_radius
        let xa, ya, xb, yb
        if ((d>0)^(qx==qy))
          xa = 0, ya = dy, xb = dx, yb = 0
        else
          xa = dx, ya = 0, xb = 0, yb = dy
        const point = (r, t) => {
          const c = cos(t)
          const s = sin(t)
          return [cx+r*(c*xa+s*xb),cy+r*(c*ya+s*yb)]
        }
        const a = rotator_arrow
        const h = rotator_thickness/2
        const t0 = .85, t1 = pi/2, t2 = t1+a/r
        const sa = select_radius
        const select =   'm'+point(0,0)+' L'+point(sa,t2)
                       +' A'+[sa,sa]+' 0 0 '+(d>0?0:1)+' '+point(sa,t0)
                       +' z'
        const path =   'm'+point(r-h,t0)
                     +' A'+[r-h,r-h]+' 0 0 '+(d>0?1:0)+' '+point(r-h,t1)
                     +' L'+point(r-a,t1)
                     +' L'+point(r,t2)
                     +' L'+point(r+a,t1)
                     +' L'+point(r+h,t1)
                     +' A'+[r+h,r+h]+' 0 0 '+(d>0?0:1)+' '+point(r+h,t0)
                     +' z'
        const v0 = t0+.2*(t1-t0),
              v1 = t0+.8*(t1-t0)
        const value =   'm'+point(r-h,v0)
                      +' A'+[r-h,r-h]+' 0 0 '+(d>0?1:0)+' '+point(r-h,v1)
                      +' L'+point(r+h,v1)
                      +' A'+[r+h,r+h]+' 0 0 '+(d>0?0:1)+' '+point(r+h,v0)
                      +' z'
        rotators.push({path: path, select: select, value: value, q: 2*qx+qy, qx: qx, qy: qy, d: d})
      }

      // Done with this quadrant!
      quadrant_data.push({q: 2*qx+qy, x: qx, y: qy, grid: grid, rotators: rotators})
    }
  }

  // Path generation for five-in-a-rows
  const five_color = f => board.grid[6*f[0][0]+f[0][1]]
  const five_tweak = c => c-2.5+bar_size/2*(c>2?1:-1)
  function five_path(f) {
    const x0 = five_tweak(f[0][0]),
          y0 = five_tweak(f[0][1]),
          x1 = five_tweak(f[4][0]),
          y1 = five_tweak(f[4][1]),
          dx = x1-x0,
          dy = y1-y0,
          s = .15/2/sqrt(dx*dx+dy*dy),
          nx =  s*dy,
          ny = -s*dx
    const point = (x,y) => x+','+y
    return 'm'+point(x0+nx,y0+ny)
         +' L'+point(x1+nx,y1+ny)
         +' L'+point(x1-nx,y1-ny)
         +' L'+point(x0-nx,y0-ny)
         +' z'
  }

  // Test boards:
  //   base: #0
  //   rotation and fives: #238128874881424344m
  //   white wins: #3694640587299947153m
  //   black wins: #3694640600154188633m
  //   tie: #3005942238600111847
  //   midsolve: #274440791932540184
</script>
