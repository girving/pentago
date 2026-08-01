// Generate the static index.html shell for the pentago explorer client
//
// The board svg itself is built at page load by src/app.js from the same
// geometry formulas this file used to emit statically (see tex/transforms.tex
// for the y-up derivation): 36 spots and 8 rotators cost far fewer bytes as
// javascript loops than as markup.  This file emits only the page shell and
// all css, including the layout rules shared conceptually with details.html
// (which carries its own copy, since it sits outside the size budget).

#include <iostream>
#include <string>
namespace pentago {
namespace {

using std::string;

// All page css, golfed.  Kept in one inline block so the page needs no
// stylesheet request.  Layout section mirrors details.html's copy.
const char* const style =
  // Layout: a grid sidebar (nav column plus content column, natively equal
  // height), folding to a top nav bar on narrow screens
  "html{overflow-y:scroll}"
  "body{background:#d5d6d7;margin:8px;line-height:1;-webkit-tap-highlight-color:transparent}"
  ".all{max-width:60em;margin:0 auto;background:#fff;border:1px solid;border-radius:1em;overflow:hidden}"
  "header{text-align:center;background:silver;border-bottom:1px solid;display:flow-root}"
  "h2{font-size:1.5em;font-weight:bold;margin:.83em 0}"
  "h3{font-size:1.17em;font-weight:bold;margin:1em 0}"
  ".main{display:grid;grid-template-columns:11em 1fr}"
  "nav{background:silver}"
  "nav a{display:block;height:3em;line-height:3em;text-align:center;text-decoration:none;"
    "border-bottom:1px solid #000;color:#000}"
  "nav a:hover{background:#fff}"
  "nav.back{position:absolute;top:0;right:0;width:7em}"
  "nav.back a{border-left:1px solid #000}"
  ".contents .back{visibility:hidden}"
  ".content{border-left:1px solid #000;position:relative}"
  "@media(max-width:800px){"
    ".main{display:block}"
    "nav.contents{text-align:center;border-bottom:1px solid #000}"
    "nav a{display:inline-block;height:1.5em;line-height:1.5em;padding:0 .5em;border:none;"
      "text-decoration:underline}"
    "nav.back{visibility:hidden}"
    ".contents .back{visibility:visible}}"
  // Board.  Value dots (.v circles, .rv arcs) are default-hidden; app.js
  // shows them with inline display and fill.
  "#board{width:400px;max-width:100%;display:block;margin:0 auto;-webkit-transform:translateZ(0)}"
  ".c,.arrow,.five{stroke:#000;stroke-width:1;vector-effect:non-scaling-stroke}"
  ".c,.arrow,.board{fill:tan}"
  ".v,.fv,.rv{stroke:gray;stroke-width:.5;vector-effect:non-scaling-stroke}"
  ".v,.rv{display:none;pointer-events:none}"
  ".rv{stroke:#000;stroke-width:1}"
  ".b .c{fill:#000}"
  ".w .c{fill:#fff}"
  ".bt .p:hover .c,.bt .rot:hover .arrow{fill:#000}"
  ".wt .p:hover .c,.wt .rot:hover .arrow{fill:#fff}"
  ".sep{fill:darkgray}"
  ".rot{display:none}"
  ".mid .rot{display:inline}"
  ".sel{opacity:0}"
  ".tl{text-anchor:middle;font-size:.4px}"
  ".mask{fill:#fff}"
  ".q{transition:transform .5s ease-in-out}"
  "#status{text-align:center;width:100%;height:3em}"
  "#error{color:red}"
  ".load{display:inline;animation:l 2s infinite ease}"
  "@keyframes l{0%{color:#000;text-shadow:none}20%{color:purple;text-shadow:0 0 6px purple}"
    "40%{color:#000;text-shadow:none}}";

void site() {
  string html;
  html += "<!DOCTYPE html>\n<html>\n<head>\n";
  html += "<meta charset=utf8>\n";
  html += "<title>Pentago is a first player win</title>\n";
  // No favicon link: app.js installs one as a data uri
  html += "<meta name=description content=\"pentago strongly solved\">\n";
  html += "<style>" + string(style) + "</style>\n";
  html += "</head>\n<body>\n";
  // The page itself is built by app.js
  html += "<script type=module src=app.js></script>\n";
  html += "</body>\n</html>";
  std::cout << html << std::endl;
}

}  // namespace
}  // namespace pentago

int main() {
  pentago::site();
  return 0;
}
