// Generate the static index.html for the pentago explorer client
//
// All board geometry is computed here at build time and emitted as inline
// svg, and all animation is expressed as css transitions on that svg.
// The runtime (src/app.js) only toggles classes and attributes.
//
// The svg uses screen coordinates (y down).  The geometry formulas are
// inherited from the old Board.svelte and tex/transforms.tex, which worked
// in y-up coordinates, so points are computed y-up and negated on emission
// (arc sweep flags flip accordingly).

#include "web/client/cc/tags.h"
#include <cmath>
#include <iostream>
namespace pentago {
namespace {

// Drawing parameters
const double bar = .1;            // separator bar size
const double spot_radius = .4;
const double value_radius = .15;
const double header_size = 2.5;
const double footer_size = 3.5;
const double margin_size = 1.7;
const double rotator_radius = 2.5;
const double rotator_thickness = .2;
const double rotator_arrow = .4;
const double select_radius = 4;
const double header_y = 4.5;      // y-up
const double footer_sep = 1.5;
const double footer_cy = -5;      // y-up
const double footer_radius = .25;
const double font_size = .4;
const double quad_center = 1.5 + bar/2;

struct P { double x, y; };  // a y-up point

string fmt(const double x) {
  return tfm::format("%g", round(1000*x)/1000);
}

// Emit a y-up point in screen coordinates
string xy(const P p) {
  return fmt(p.x) + "," + fmt(-p.y);
}

// Path commands, taking y-up geometry
string move(const P p) { return "M" + xy(p); }
string line(const P p) { return "L" + xy(p); }
string arc(const double r, const int sweep, const P p) {  // sweep in the y-up sense
  return tfm::format("A%s,%s 0 0 %d %s", fmt(r), fmt(r), 1-sweep, xy(p));
}

string text(const string& cls, const string& id, const P at, const string& body) {
  return tag("text", {{"class", cls}, {"id", id}, {"x", fmt(at.x)}, {"y", fmt(-at.y)}}, body);
}

// The base rotator (hidden hover region, arrow, value arc) for quadrant
// (0,0), d=1; the other seven are signed-permutation images of it (see
// rotator() below).  All shapes live in one <g> referenced via <use>.
string base_rotator() {
  const double pi = M_PI;
  const double cx = -1.5 - bar/2, cy = cx;  // quadrant (0,0) center of rotation
  // Axes for qx=qy=0, d=1: A=(-1,0), B=(0,-1)
  const auto point = [=](const double r, const double t) {
    return P{cx - r*cos(t), cy - r*sin(t)};
  };
  const double r = rotator_radius;
  const double a = rotator_arrow;
  const double h = rotator_thickness/2;
  const double sa = select_radius;
  const double t0 = .85, t1 = pi/2, t2 = t1 + a/r;
  const string select = move(point(0,0)) + line(point(sa,t2))
                      + arc(sa, 0, point(sa,t0)) + "z";
  const string path = move(point(r-h,t0))
                    + arc(r-h, 1, point(r-h,t1))
                    + line(point(r-a,t1)) + line(point(r,t2)) + line(point(r+a,t1))
                    + line(point(r+h,t1))
                    + arc(r+h, 0, point(r+h,t0)) + "z";
  const double v0 = t0 + .2*(t1-t0), v1 = t0 + .8*(t1-t0);
  const string value = move(point(r-h,v0))
                     + arc(r-h, 1, point(r-h,v1))
                     + line(point(r+h,v1))
                     + arc(r+h, 0, point(r+h,v0)) + "z";
  return tag("g", {{"id", "r"}},
             tag("path", {{"class", "sel"}, {"d", select}})
           + tag("path", {{"class", "arrow"}, {"d", path}})
           + tag("path", {{"class", "rv"}, {"d", value}}));
}

// One rotator link: a signed-permutation transform of the base rotator.
// In y-up coordinates the base maps to (qx,qy,d) by diag(-dx,-dy) when
// (d>0)^(qx==qy)==0 and by the antidiagonal [[0,-dx],[-dy,0]] otherwise;
// conjugating by the y-flip into screen coordinates negates the
// off-diagonal entries.
string rotator(const int qx, const int qy, const int d) {
  const int dx = qx ? 1 : -1;
  const int dy = qy ? 1 : -1;
  const string m = (d > 0) ^ (qx == qy)
    ? tfm::format("matrix(0,%d,%d,0,0,0)", dy, dx)
    : tfm::format("matrix(%d,0,0,%d,0,0)", -dx, -dy);
  // app.js recovers (q, d) from document order, so no data attributes
  return tag("a", {{"class", "rot"}},
             tag("use", {{"href", "#r"}, {"transform", m}}));
}

string board_svg() {
  string s;

  // Shared geometry: a spot (stone circle + optional value dot) used 37
  // times.  The value dot's visibility and fill come from css variables set
  // on the referencing element by app.js.
  s += tag("defs", {},
           tag("g", {{"id", "s"}},
               tag("circle", {{"class", "c"}, {"r", fmt(spot_radius)}})
             + tag("circle", {{"class", "v"}, {"r", fmt(value_radius)}}))
         + base_rotator());

  // Header: whose turn, their value, and a label
  s += tag("use", {{"id", "turn"}, {"href", "#s"}, {"y", fmt(-header_y)}});
  s += text("tl", "hl", {0, header_y - spot_radius - font_size}, "to play");

  // Footer: legend of value colors
  const std::pair<int,string> legend[3] = {{1, "win"}, {0, "tie"}, {-1, "loss"}};
  const string value_colors[3] = {"#00ff00", "#0000ff", "#ff0000"};  // win, tie, loss
  for (const int i : {0, 1, 2}) {
    const auto& [v, label] = legend[i];
    s += tag("circle", {{"class", "fv"}, {"cx", fmt(-footer_sep*v)}, {"cy", fmt(-footer_cy)},
                        {"r", fmt(footer_radius)}, {"fill", value_colors[i]}});
    s += text("tl", "", {-footer_sep*v, footer_cy - footer_radius - font_size}, label);
  }

  // Separator bars
  const double blen = bar + 6.2;
  for (const bool flip : {false, true}) {
    const string a = fmt(-bar/2), b = fmt(-blen/2), w = fmt(bar), l = fmt(blen);
    s += tag("rect", {{"class", "sep"}, {"x", flip ? b : a}, {"y", flip ? a : b},
                      {"width", flip ? l : w}, {"height", flip ? w : l}});
  }

  // Quadrants.  Rotators come before their quadrant so that on iPhone,
  // tapping a quadrant center can't cause a rotation followed by an errant
  // fake placed stone (see the old Board.svelte for history).
  for (const int qx : {0, 1}) {
    for (const int qy : {0, 1}) {
      const int q = 2*qx + qy;
      for (const int d : {-1, 1})
        s += rotator(qx, qy, d);

      // The outer group centers the quadrant, .quadrant animates via css
      // transition, and the inner group counter-rotates instantly so the
      // freshly updated board starts visually at the old orientation.
      string spots;
      spots += tag("rect", {{"class", "board"}, {"x", "-1.5"}, {"y", "-1.5"},
                            {"width", "3"}, {"height", "3"}});
      // app.js recovers the board position from document order
      for (int x = 3*qx; x < 3*qx+3; x++)
        for (int y = 3*qy; y < 3*qy+3; y++)
          spots += tag("a", {},
                       tag("use", {{"href", "#s"}, {"x", fmt(x%3-1)}, {"y", fmt(-(y%3-1))}}));
      const P center = {(qx ? 1 : -1) * quad_center, (qy ? 1 : -1) * quad_center};
      s += tag("g", {{"transform", tfm::format("translate(%s,%s)", fmt(center.x), fmt(-center.y))}},
               tag("g", {{"class", "quadrant"}, {"id", tfm::format("q%d", q)}},
                   tag("g", {{"id", tfm::format("g%d", q)}}, spots)));
    }
  }

  // Five-in-a-row markers are added dynamically
  s += tag("g", {{"id", "fives"}}, " ");

  const double width = 6 + 2*margin_size;
  const double height = 6 + header_size + footer_size;
  // No xmlns: inline svg in html needs none
  return tag("svg", {{"id", "board"},
                     {"viewBox", tfm::format("%s %s %s %s", fmt(-width/2), fmt(-(3+header_size)),
                                             fmt(width), fmt(height))}}, s);
}

// Styles for the board svg and status area.  Shared page layout lives in
// main.css (also used by details.html); everything board-specific is here.
//
// Stones and value dots are <use> clones of #s, so per-spot state flows in
// through inherited css variables: --f is the stone fill (set by the .b/.w/.p
// classes on the containing link, or inline on #turn), and --vd/--v are the
// value dot's visibility and fill (set inline by app.js).
const char* const style = R"(
#board{width:400px;max-width:100%;display:block;margin:0 auto;-webkit-transform:translateZ(0)}
.c,.arrow{stroke:black;stroke-width:1;vector-effect:non-scaling-stroke}
.c{fill:var(--f,tan)}
.b{--f:black}
.w{--f:white}
.bturn .p:hover{--f:black}
.wturn .p:hover{--f:white}
.board{fill:tan}
.sep{fill:darkgray}
.rot{display:none}
.mid .rot{display:inline}
.sel{opacity:0}
.arrow{fill:var(--af,tan)}
.bturn .rot:hover{--af:black}
.wturn .rot:hover{--af:white}
.v,.rv{display:var(--vd,none);fill:var(--v);stroke:gray;stroke-width:.5;vector-effect:non-scaling-stroke;pointer-events:none}
.fv{stroke:gray;stroke-width:.5;vector-effect:non-scaling-stroke}
.rv{stroke:black;stroke-width:1}
.tl{text-anchor:middle;font-size:.4px}
.five{stroke:black;stroke-width:1;vector-effect:non-scaling-stroke}
.mask{fill:white}
.quadrant{transition:transform .5s ease-in-out}
.status{text-align:center;width:100%;height:3em}
#error{color:red}
.load{display:inline;animation:spin 2s infinite ease}
@keyframes spin{0%{color:black;text-shadow:none}20%{color:purple;text-shadow:0 0 6px purple}40%{color:black;text-shadow:none}}
)";

const char* const nav_links = R"(
<a href="/">Home</a>
<a href="details.html#intro">Introduction</a>
<a href="details.html#rules">Pentago rules</a>
<a href="details.html#algorithms">Algorithms</a>
<a href="details.html#server">Server setup</a>
<a href="details.html#open">Code and data</a>
<a href="details.html#useful">Usefulness</a>
<a href="details.html#thanks">Acknowledgements</a>
<a href="details.html#contact">Contact</a>
)";

void site() {
  string html;
  html += "<!DOCTYPE html>\n<html>\n<head>\n";
  html += "<meta charset=\"utf8\">\n";
  html += "<title>Pentago is a first player win</title>\n";
  html += "<link rel=\"stylesheet\" href=\"/main.css\">\n";
  // No favicon link: app.js installs one as a data uri
  html += "<meta name=\"description\" content=\"pentago strongly solved\">\n";
  html += "<style>" + string(style) + "</style>\n";
  html += "</head>\n<body>\n";
  html += "<div class=\"all\">\n<header>\n";
  html += "<h2>Pentago is a first player win</h2>\n";
  html += "<h3>An interactive explorer for perfect pentago play</h3>\n";
  html += "</header>\n<div class=\"main\">\n";
  html += "<nav class=\"contents\">" + string(nav_links) + "<a class=\"back\">Back</a></nav>\n";
  html += "<div class=\"content\">\n";
  html += "<nav class=\"back\"><a class=\"back\">Back</a></nav>\n";
  html += board_svg() + "\n";
  html += "<div class=\"status\" id=\"status\"></div>\n";
  html += "</div>\n</div>\n</div>\n";
  html += "<script type=\"module\" src=\"app.js\"></script>\n";
  html += "</body>\n</html>";
  std::cout << html << std::endl;
}

}  // namespace
}  // namespace pentago

int main() {
  try {
    pentago::site();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
