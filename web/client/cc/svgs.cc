// Generate website svgs

#include "pentago/base/count.h"
#include "pentago/utility/box.h"
#include "pentago/utility/format.h"
#include "pentago/utility/log.h"
#include "pentago/utility/vector.h"
namespace pentago {
namespace {

typedef double T;
typedef Vector<double,2> TV;
typedef Vector<int,2> IV;
using std::string;

string tag(const string& name, const vector<tuple<string,string>>& attrs, const string& body = "") {
  string start = name;
  for (const auto& [k,v] : attrs)
    if (v.size())
      start += format(" %s=\"%s\"", k, v);
  return format("<%s%s", start, body.size() ? format(">%s</%s>", body, name) : "/>");
}

string svg(const int width, const int height, const string& body) {
  return tag("svg", {{"viewBox", format("0 0 %d %d", width, height)},
                     {"xmlns", "http://www.w3.org/2000/svg"}}, body);
}

void counts() {
  // Preliminaries
  const Box<TV> box = {{161, 50}, {1163, 449}};
  const TV axes(36, 15);
  const auto snap = [=](const TV X) {
    const auto size = box.shape();
    return IV(rint(box.min[0] + size[0]*X[0]/axes[0]), rint(box.max[1] - size[1]*X[1]/axes[1]));
  };
  const string sep = "";  // Change to "\n" for easier reading

  // Draw polygonal lines
  const auto lines = [=](const vector<TV>& Xs, const string& style, const string& marker = "") {
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
    vector<tuple<string,string>> attrs = {{"d", path}, {"style", style}};
    if (marker.size())
      for (const string& m : {"marker-start", "marker-mid", "marker-end"})
        attrs.emplace_back(m, format("url(#%s)", marker));
    return sep + tag("path", attrs);
  };

  // Markers
  const auto marker = [=](const string& name, const int size, const string& element) {
    const auto s = format("%d", size);
    return sep + tag("marker", {{"id", name}, {"viewBox", "0 0 10 10"}, {"refX", "5"}, {"refY", "5"},
                                {"markerWidth", s}, {"markerHeight", s}}, element);
  };
  const auto circles = [=](const string& name, const string& color) {
    return marker(name, 8, format(R"(<circle cx="5" cy="5" r="4" fill="%s"/>)", color));
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
  const auto text = [=](const TV X, const string& text, const string& style = "") {
    const auto I = snap(X);
    return sep + tag("text", {{"x", format("%d", I[0])}, {"y", format("%d", I[1])}, {"style", style}}, text);
  };

  // Start our SVG!
  string body;

  // CSS
  body += sep + tag("style", {}, string() +
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
  body += sep + tag("defs", {},
    circles("bl", blue) +
    circles("gr", green) +
    marker("xtick", 8, format(R"(<path d="M5 5v-7" style="%s"/>)", tick)) +
    marker("ytick", 8, format(R"(<path d="M5 5h7" style="%s"/>)", tick))
  );

  // Axes
  body += sep + format(R"(<g class="ax">)");
  {
    // y-axis
    vector<TV> Xs;
    for (int y = 15; y >= 0; y--)
      Xs.emplace_back(0, y);
    body += lines(Xs, "stroke:#000000", "ytick");
    for (const int y : range(15+1))
      body += text(TV(-.85, y-.17), format(R"(10<tspan>%d</tspan>)", y));
  } {
    // x-axis
    vector<TV> Xs;
    for (const int x : range(36+1))
      Xs.emplace_back(x, 0);
    body += lines(Xs, ";stroke:#000000", "xtick");
    for (const int x : range(36+1))
      body += text(TV(x, -.77), format("%d", x));
  }
  body += sep + "</g>";
  body += sep + format(R"(<g class="rest">)");
  body += sep + tag("g", {{"transform", translate(TV(-1.7, 15./2)) + degrees(-90)}},
    tag("text", {{"x", "0"}, {"y", "0"}}, "positions"));
  body += text(TV(18, -1.8), "stones on the board");
    
  // All bound counts
  {
    vector<TV> all;
    for (const int n : range(36+1))
      all.emplace_back(n, log10(T(count_boards(n, 8))));
    body += lines(all, "stroke:" + blue, "bl");
    body += text(all[24] - TV(0, 1.2), "all boards");
  }

  // Midsolve counts
  {
    vector<TV> mid;
    for (const int n : range(18+1))
      mid.emplace_back(n+18, log10(T(choose(18, n) * choose(n, n/2))));
    body += lines(mid, "stroke:" + green, "gr");
    body += text(mid[12] - TV(0, 1.6), R"(descendents of<tspan dx="-6.5em" dy="1.2em">an 18 stone board</tspan>)");
  }
  body += sep + "</g>";

  // Print the result
  slog(svg(1292, 498, body));
}

}  // namespace
}  // namespace pentago

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) != "counts") {
    std::cerr << "Usage: svgs counts" << std::endl;
    return 1;
  }
  try {
    pentago::counts();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
