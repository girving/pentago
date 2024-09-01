// Generate website svgs

#include "pentago/base/count.h"
#include "pentago/utility/box.h"
#include "pentago/utility/format.h"
#include "pentago/utility/log.h"
#include "pentago/utility/sqr.h"
#include "pentago/utility/vector.h"
#include <fstream>
#include <functional>
#include <map>
namespace pentago {
namespace {

typedef double T;
typedef Vector<double,2> TV;
typedef Vector<int,2> IV;
const T pi = M_PI;
using namespace std;

bool endswith(const string& x, const string& y) {
  return x.size() >= y.size() && x.substr(x.size() - y.size()) == y;
}

// Lazy global variable for string outputs
vector<string> outputs;

// Optional separation and indentation
const bool verbose = true;  // DO NOT SUBMIT
int indent = 0;

string indentation() {
  return format("%*s", verbose ? 2*indent : 0, "");
}

// Run a function that outputs into 'output'
string run(const function<void()>& generate) {
  outputs.emplace_back();
  const int depth = outputs.size();
  generate();
  GEODE_ASSERT(outputs.size() == depth);
  const string result = std::move(outputs.back());
  outputs.pop_back();
  return result;
}

void add_output(const string& s) {
  GEODE_ASSERT(outputs.size());
  outputs.back() += s;
}

struct Value {
  string value;
  Value(const string& value) : value(value) {}
  Value(const char* value) : value(value) {}
  Value(const T& value) : value(format("%.2g", value)) {}  // DO NOT SUBMIT: Think about precision
};

string start_tag(const string& name, const vector<tuple<string,Value>>& attrs, const bool close = false) {
  string tag = "<" + name;
  for (const auto& [k,v] : attrs)
    if (v.value.size())
      tag += format(" %s=\"%s\"", k, v.value);
  tag += close ? "/>" : ">";
  tag += close && verbose ? "\n" : "";
  return tag;
}

string close_tag(const string& name) {
  return format("</%s>%s", name, verbose ? "\n" : "");
}

void tag(const string& name, const vector<tuple<string,Value>>& attrs, const string& body = "") {
  add_output(indentation() + start_tag(name, attrs, body.empty()));
  if (body.size()) add_output(body + close_tag(name));
}

struct Indent {
  const int delta;
  Indent(const int delta) : delta(delta) { indent += delta; }
  ~Indent() { indent -= delta; }
};

// Scoped tags
struct Scope {
  const string name;
  Scope(const string& name, const vector<tuple<string,Value>>& attrs)
    : name(name) {
    add_output(indentation() + start_tag(name, attrs) + (verbose ? "\n" : ""));
    indent++;
  }
  ~Scope() {
    indent--;
    add_output(indentation() + close_tag(name));
  }
};

struct SVG : public Scope {
  SVG(const int width, const int height)
    : Scope("svg", {{"viewBox", format("0 0 %d %d", width, height)},
                    {"xmlns", "http://www.w3.org/2000/svg"}}) {}
};

void counts() {
  // Preliminaries
  const Box<TV> box = {{161, 50}, {1163, 449}};
  const TV axes(36, 15);
  const auto snap_x = [=](const T x) { return int(rint(box.min[0] + box.shape()[0]*x/axes[0])); };
  const auto snap_y = [=](const T y) { return int(rint(box.max[1] - box.shape()[1]*y/axes[1])); };
  const auto snap = [=](const TV X) { return IV(snap_x(X[0]), snap_y(X[1])); };

  // Draw polygonal lines
  const auto lines = [&](const vector<TV>& Xs, const string& style, const string& marker = "") {
    GEODE_ASSERT(Xs.size());
    IV prev = snap(Xs[0]);
    string path = format("M%d %d", prev[0], prev[1]);
    for (const TV X : asarray(Xs).slice(1, Xs.size())) {
      const IV next = snap(X);
      const IV diff = next - prev;
      path += !diff[1] ? format("h%d", diff[0])
            : !diff[0] ? format("v%d", diff[1])
                       : format("l%d %d", diff[0], diff[1]);
      prev = next;
    }
    vector<tuple<string,Value>> attrs = {{"d", path}, {"style", style}};
    if (marker.size())
      for (const string& m : {"marker-start", "marker-mid", "marker-end"})
        attrs.emplace_back(m, format("url(#%s)", marker));
    tag("path", attrs);
  };

  // Markers
  const auto marker = [=](const string& name, const int size, const string& element) {
    const auto s = format("%d", size);
    tag("marker", {{"id", name}, {"viewBox", "0 0 10 10"}, {"refX", "5"}, {"refY", "5"},
                   {"markerWidth", s}, {"markerHeight", s}}, element);
  };
  const auto circles = [=](const string& name, const string& color) {
    marker(name, 8, format(R"(<circle cx="5" cy="5" r="4" fill="%s"/>)", color));
  };

  // Move (0,0) to the desired coordinates
  const auto translate = [=](const TV X) {
    const IV I = snap(X);
    return format("translate(%d,%d)", I[0], I[1]);
  };

  // Rotate by degrees
  const auto degrees = [=](const int degrees) {
    return format("rotate(%d)", degrees);
  };

  // Text
  const auto text = [&](const IV I, const string& text, const string& style = "") {
    vector<tuple<string,Value>> attrs;
    for (const int d : range(2))
      if (I[d])
        attrs.emplace_back(d ? "y" : "x", format("%d", I[d]));
    attrs.emplace_back("style", style);
    tag("text", attrs, text);
  };

  // Start our SVG!
  SVG svg(1292, 498);

  // CSS
  tag("style", {}, string() +
    "path{fill:none}" +
    "text{text-anchor:middle}" +
    ".rest{font-size:30px}" +
    ".ax{font-size:20px}" +
    ".ax tspan{baseline-shift:super;font-size:15px}"
  );

  // Defs section
  const string blue = "#0000ff";
  const string green = "#008000";
  const string tick = "stroke-width:1;stroke:#000000";
  {
    Scope defs("defs", {});
    circles("bl", blue);
    circles("gr", green);
    marker("xtick", 8, format(R"(<path d="M5 5v-7" style="%s"/>)", tick));
    marker("ytick", 8, format(R"(<path d="M5 5h7" style="%s"/>)", tick));
  }

  // Axes
  {
    Scope axes_g("g", {{"class", "ax"}});
    {
      // y-axis
      vector<TV> Xs;
      for (int y = 15; y >= 0; y--)
        Xs.emplace_back(0, y);
      lines(Xs, "stroke:#000000", "ytick");
      Scope left("g", {{"transform", format("translate(%d,0)", snap_x(-.85))}});
      for (const int y : range(15+1))
        text(IV(0, snap_y(y-.17)), format(R"(10<tspan>%d</tspan>)", y));
    } {
      // x-axis
      vector<TV> Xs;
      for (const int x : range(36+1))
        Xs.emplace_back(x, 0);
      lines(Xs, ";stroke:#000000", "xtick");
      Scope left("g", {{"transform", format("translate(0,%d)", snap_y(-.77))}});
      for (const int x : range(36+1))
        text(IV(snap_x(x), 0), format("%d", x));
    }
  }

  {
    Scope rest_g("g", {{"class", "rest"}});
    {
      Scope g("g", {{"transform", translate(TV(-1.7, 15./2)) + degrees(-90)}});
      tag("text", {{"x", "0"}, {"y", "0"}}, "positions");
    }
    text(snap(TV(18, -1.8)), "stones on the board");

    // All bound counts
    {
      vector<TV> all;
      for (const int n : range(36+1))
        all.emplace_back(n, log10(T(count_boards(n, 8))));
      lines(all, "stroke:" + blue, "bl");
      text(snap(all[24] - TV(0, 1.2)), "all boards");
    }

    // Midsolve counts
    {
      vector<TV> mid;
      for (const int n : range(18+1))
        mid.emplace_back(n+18, log10(T(choose(18, n) * choose(n, n/2))));
      lines(mid, "stroke:" + green, "gr");
      text(snap(mid[12] - TV(0, 1.6)), R"(descendents of<tspan dx="-6.5em" dy="1.2em">an 18 stone board</tspan>)");
    }
  }
}

void favicon() {
  // Sizes
  const T spot_size = 1;
  const T spot_radius = 0.4;
  const T value_radius = 0.2;
  const T quad_size = 3*spot_size;
  const T bar_width = 0.1 * 0;
  const T bar_extra = 0.1 * 0;
  const T bar_length = 2*bar_extra + 2*quad_size + bar_width;
  const T scale = 10;

  // Start our svg
  const T total = scale * (2*bar_extra + 2*quad_size + bar_width);
  SVG svg(total, total);

  // Utilities
  const auto pos = [=](const T z) { return format("%g", scale*z); };
  const auto rect = [&](const string& fill, const T x, const T y, const T width, const T height) {
    tag("rect", {{"fill", fill}, {"x", pos(x)}, {"y", pos(y)}, {"width", pos(width)}, {"height", pos(height)}});
  };

  // CSS
  tag("style", {}, string() +
    ".b,.w,.e{stroke:black;stroke-width:0.3}" +
    ".b{fill:black}" +
    ".w{fill:white}" +
    ".e{fill:tan}" +
    ".v{fill:green}" +
    ".t{fill:blue}"
  );

  // Quadrants, as a single rectangle
  rect("tan", bar_extra, bar_extra, 2*quad_size + bar_width, 2*quad_size + bar_width);

  // Separators
  if (bar_width) {
    rect("darkgray", 0, bar_extra + quad_size, bar_length, bar_width);
    rect("darkgray", bar_extra + quad_size, 0, bar_width, bar_length);
  }

  // Circles: b = black, w = white, t = tie (blue), v = win (green)
  const string board[6][6] = {
    {"b", "w", "w", "t", "b", "t"},
    {"w", "t", "t", "b", "v", "t"},
    {"v", "b", "w", "t", "v", "b"},
    {"t", "b", "v", "t", "w", "t"},
    {"v", "t", "w", "b", "b", "v"},
    {"t", "w", "b", "w", "w", "t"}};
  const auto sx = [=](const int z) { return pos(bar_extra + spot_size*(z + 0.5) + bar_width*(z > 2)); };
  for (const int x : range(6)) {
    for (const int y : range(6)) {
      const auto cls = board[x][y];
      const auto outer = cls == "b" || cls == "w" ? cls : "e";
      tag("circle", {{"class", outer}, {"cx", sx(y)}, {"cy", sx(x)}, {"r", pos(spot_radius)}});
      if (outer == "e")
        tag("circle", {{"class", cls}, {"cx", sx(y)}, {"cy", sx(x)}, {"r", pos(value_radius)}});
    }
  }
}

void board() {
  // Drawing parameters
  // DO NOT SUBMIT: unused
  const T bar_size = .1;
  const T spot_radius = .4;
  const T header_size = 2.5;
  const T footer_size = 3.5;
  const T margin_size = 1.7;
  const T value_radius = .15;
  const T rotator_radius = 2.5;
  const T rotator_thickness = .2;
  const T rotator_arrow = .4;
  const T select_radius = 4;
  const T font_size = .4;
  const T header_y = 4.5;
  const T footer_sep = 1.5;
  const T footer_cy = -5;
  const T footer_radius = .25;

  // DO NOT SUBMIT: No "style", all classes + css

  // Scale-invariant units
  const auto xp = [=](const T x) { return format("%g%%", 100 * x / (6 + 2*margin_size)); };
  const auto yp = [=](const T y) { return format("%g%%", 100 * y / (6 + header_size + footer_size)); };
  const T normalized_diagonal = sqrt(sqr(6 + 2*margin_size) + sqr(6 + header_size + footer_size)) / sqrt(2);
  const auto dp = [=](const T d) { return format("%g%%", 100 * d / normalized_diagonal); };

  // DO NOT SUBMIT
  // child_value(board)
  //   // Colors for each board value, taking care to be nice to colorblind folk.
  //   const value_colors = {'1': '#00ff00', '0': '#0000ff', '-1': '#ff0000', 'undefined': null}
  //   {#await child_value(board) then v}
  const map<int, string> value_colors = {{1, "#00ff00"}, {0, "#0000ff"}, {-1, "#ff0000"}};
  const int child_value = 1;
  const string child_value_color = value_colors.at(child_value);

  // DO NOT SUBMIT
  // These are actualy variable
  const string turncolor = "turncolor";
  const string turn_label = "to win";
  const string spot_link = "spot link";

  // Start our svg
  Indent indent(3);
  Scope svg("svg", {{"id", "board"}});

  // DO NOT SUBMIT: Use defs for rotators

  // Overall coordinate transform
  //Scope g("g", {{"transform", format("translate(%g,%g) scale(%g,%g)", scale*(3+margin_size), scale*(3+header_size), scale, -scale)}});
  Scope g("g", {{"transform", format("translate(%s,%s) scale(1,-1)", xp(3+margin_size), yp(3+header_size))}});

  if (0) {
  // Header
  tag("circle", {{"class", turncolor}, {"cx", 0}, {"cy", yp(header_y)}, {"r", dp(spot_radius)}});
  tag("circle", {{"class", "tvalue"}, {"cx", 0}, {"cy", yp(header_y)}, {"r", dp(value_radius)},
                 {"style", format("fill:%s", child_value_color)}});
  tag("text", {{"class", "turnlabel"}, {"transform", "scale(1,-1)"},
               {"x", 0}, {"y", yp(-(header_y-spot_radius-font_size))},
               {"style", format("font-size:%dpx", font_size)}}, turn_label);

  // Footer
  for (const int v : {1, 0, -1}) {
    Scope g("g", {{"class", "footer"}});
    tag("circle", {{"class", "fvalue"}, {"cx", xp(-footer_sep*v)}, {"cy", yp(footer_cy)},
                   {"r", dp(footer_radius)}, {"fill", value_colors.at(v)}});
    tag("text", {{"class", "valuelabel"}, {"transform", "scale(1,-1)"}, {"x", xp(-footer_sep*v)},
                 {"y", yp(-(footer_cy-footer_radius-font_size))},
                 {"style", format("font-size:%dpx", font_size)}}, v == 1 ? "win" : v == 0 ? "tie" : "loss");
  }
  }  // if (0)

  // Separators
  tag("rect", {{"class", "separators"}, {"x", xp(-bar_size/2)}, {"y", yp(-(bar_size+6.2)/2)},
               {"width", xp(bar_size)}, {"height", yp(bar_size+6.2)}});
  tag("rect", {{"class", "separators"}, {"y", yp(-bar_size/2)}, {"x", xp(-(bar_size+6.2)/2)},
               {"height", yp(bar_size)}, {"width", xp(bar_size+6.2)}});

  if (0) {
  // Board
  for (const int qx : {0, 1}) {
    for (const int qy : {0, 1}) {
      // Rotators.
      // Important to put these prior to quadrants; otherwise an iPhone tap on a quadrant center causes
      // a rotation followed by an errant fake placed stone.
      // DO NOT SUBMIT: {#if !done && board.middle}
      for (const int d : {-1, 1}) {
        // Rotator paths
        const T dx = qx ? 1 : -1;
        const T dy = qy ? 1 : -1;
        const T cx = 3*qx+1-2.5+bar_size/2*dx;
        const T cy = 3*qy+1-2.5+bar_size/2*dy;
        const T r = rotator_radius;
        T xa, ya, xb, yb;
        if ((d>0)^(qx==qy)) {
          xa = 0; ya = dy; xb = dx; yb = 0;
        } else {
          xa = dx; ya = 0; xb = 0; yb = dy;
        }
        const auto point = [=](const T r, const T t) {
          const T c = cos(t);
          const T s = sin(t);
          return TV(cx+r*(c*xa+s*xb), cy+r*(c*ya+s*yb));
        };
        const auto node = [=](const string& s) { return [=](const T r, const T t) {
          const auto p = point(r, t);
          return format("%s%s,%s", s, xp(p[0]), yp(p[1]));
        }; };
        const auto m = node("m");
        const auto L = node("L");
        const auto A = [=](const T r, const T pr, const T pt) {
          const auto p = point(pr, pt);
          return format(" A%g,%g 0 0 %d %s,%s", r, r, d>0 ? 0 : 1, xp(p[0]), yp(p[1]));
        };
        const T a = rotator_arrow;
        const T h = rotator_thickness/2;
        const T t0 = .85, t1 = pi/2, t2 = t1+a/r;
        const T sa = select_radius;
        const T v0 = t0+.2*(t1-t0);
        const T v1 = t0+.8*(t1-t0);
        const string select = m(0,0) + L(sa,t2) + A(sa,sa,t0) + "z";
        const string path = m(r-h,t0) + A(r-h,r-h,t1) + L(r-a,t1) + L(r,t2) + L(r+a,t1) +
            L(r+h,t1) + A(r+h,r+h,t0) + "z";
        const string value = m(r-h,v0) + A(r-h,r-h,v1) + L(r+h,v1) + A(r+h,r+h,v0) + "z";

        // Rotator elements
        Scope a1("a", {{"href", "rotate_link"}});
        tag("path", {{"class", "rotateselect"}, {"d", select}});
        tag("path", {{"class", "rotate" + turncolor}, {"d", path}});
        // {#if board.middle && !board.done}  // DO NOT SUBMIT
        //   {#await child_value(board.rotate(r.qx, r.qy, r.d)) then v}
        const int v = 1;  // DO NOT SUBMIT
        tag("path", {{"class", "rvalue"}, {"d", value},
                     {"style", format("fill:%s", value_colors.at(-v))}});
      }

      {
        // Quadrants
        const string transform = "blah";
        // DO NOT SUBMIT, about the following g: on:transitionend={nospin(q)}
        Scope g("g", {{"class", "quadrant"}, {"style", "transform:" + transform}});
        tag("rect", {{"class", "board"}, {"x", xp(-1.5)}, {"y", yp(-1.5)}, {"width", xp(3)}, {"height", yp(3)}});

        // Spots
        for (const int x : range(3)) {
          for (const int y : range(3)) {
            Scope g("g", {{"transform", format("translate(%d,%d)", xp(x-1), yp(y-1))}});
            Scope a("a", {{"href", spot_link}});
            tag("circle", {{"id", format("s%d%d", 3*qx+x, 3*qy+y)}, {"r", dp(spot_radius)}});
            tag("circle", {{"id", format("v%d%d", 3*qx+x, 3*qy+y)}, {"r", dp(value_radius)},
                           {"style", "fill:{value_colors[v]}"}});
          }
        }
      }
    }
  }

  // Fives-in-a-row
  for (const int x0 : range(6)) {
    for (const int y0 : range(6)) {
      for (const auto& [dx, dy] : vector<tuple<int,int>>{{0,1}, {1,0}, {1,1}, {1,-1}}) {
        const int x1 = x0 + 4*dx;
        const int y1 = y0 + 4*dy;
        if (x1 < 6 && 0 <= y1 && y1 < 6) {
          // Path generation for five-in-a-rows
          const string five_color = "black";  // DO NOT SUBMIT
          const auto five_tweak = [=](const T c) { return c-2.5+bar_size/2*(c>2?1:-1); };
          const T tx0 = five_tweak(x0),
                  ty0 = five_tweak(y0),
                  tx1 = five_tweak(x1),
                  ty1 = five_tweak(y1),
                  dx = tx1-tx0,
                  dy = ty1-ty0,
                  s = .15/2/sqrt(dx*dx+dy*dy),
                  nx =  s*dy,
                  ny = -s*dx;
          const string path = format("m%s,%sL%s,%sL%s,%sL%s,%sz",
              xp(tx0+nx), yp(ty0+ny), xp(tx1+nx), yp(ty1+ny), xp(tx1-nx), yp(ty1-ny), xp(tx0-nx), yp(ty0-ny));

          tag("path", {{"class", "five"}, {"id", format("f%d%d%d%d", x0, y0, x1, y1)}, {"d", path}});
          // DO NOT SUBMIT:
          //   {#if five_color(f) == 2}
          const T fr = spot_radius - .01;
          tag("circle", {{"class", "mask"}, {"cx", xp(tx0)}, {"cy", yp(ty0)}, {"r", dp(fr)}});
          tag("circle", {{"class", "mask"}, {"cx", xp(tx1)}, {"cy", yp(ty1)}, {"r", dp(fr)}});
        }
      }
    }
  }
  }  // weird if (0)
/*
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

  // Test boards:
  //   base: #0
  //   rotation and fives: #238128874881424344m
  //   white wins: #3694640587299947153m
  //   black wins: #3694640600154188633m
  //   tie: #3005942238600111847
  //   midsolve: #274440791932540184
</script>
*/
}

string replace(string src, const string& pat, const string& sub) {
  size_t n = src.find(pat);
  if (n == string::npos)
    throw ValueError("Couldn't find " + pat + " in the string");
  src.replace(n, pat.size(), sub);
  return src;
}

void index_html(const string& src) {
  // Read the template index.html
  GEODE_ASSERT(endswith(src, "index.html"));
  ifstream file(src);
  GEODE_ASSERT(file, format("Failed to open %s", src));
  const string input((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());

  // Insert the board svg
  add_output(replace(input, "      %board", run(board)));
}

void save(const string& suffix, const string& path, const function<void()>& generate) {
  if (!endswith(path, suffix))
    throw ValueError(format("%s does not end with %s", path, suffix));
  ofstream out(path);
  GEODE_ASSERT(out);
  out << run(generate);
}

void toplevel(int argc, char** argv) {
  const vector<string> paths(argv + 1, argv + argc);
  GEODE_ASSERT(paths.size() == 4);
  save("counts.svg", paths[0], counts);
  save("favicon.svg", paths[1], favicon);
  save("index.html", paths[2], [&]() { index_html(paths[3]); });
}

}  // namespace
}  // namespace pentago

int main(int argc, char** argv) {
  try {
    pentago::toplevel(argc, argv);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
