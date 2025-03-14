// Pentago lookup table precomputation:
//
// In order to avoid needless work, a good chunk of the logic for basic pentago
// operations is baked into lookup tables computed by this script.  This includes
// state transformation (rotations, base-2 to/from base-3, etc.), win computation,
// symmetry helpers, etc.  These tables also show up in the file formats, since
// the ordering of boards within a section is defined by rotation_minimal_quadrants.

#include "pentago/utility/array.h"
#include "pentago/utility/char_view.h"
#include "pentago/utility/endian.h"
#include "pentago/utility/format.h"
#include "pentago/utility/join.h"
#include "pentago/utility/log.h"
#include "pentago/utility/nested.h"
#include "pentago/utility/popcount.h"
#include "pentago/utility/portable_hash.h"
#include "pentago/utility/scalar_view.h"
#include "pentago/utility/sqr.h"
#include <fstream>
#include <map>
#include <unordered_map>
namespace pentago {
namespace {

using std::function;
using std::make_tuple;
using std::map;
using std::min;
using std::ofstream;
using std::ostringstream;
using std::swap;
using std::tuple;
using std::unordered_map;
string halfsuper_wins();

struct unusable {};
#define REMEMBER(name, ...) const auto name = []() { __VA_ARGS__; return unusable(); }();

const string autogenerated = "// Autogenerated by precompute: do not edit directly\n\n";

// type, name, c++ sizes, c++ initializer, flags
const int for_js = 1, for_wasm = 2;
vector<tuple<string,string,string,string,int>> tables;

template<class Data> void
cpp_init(ostream& out, const string& fmt, const Data& data, const int axis, const int offset) {
  if (axis == data.rank())
    tfm::format(out, fmt.c_str(), data.data()[offset]);
  else {
    const int n = data.shape()[axis];
    out << '{';
    for (const int i : range(n)) {
      if (i) out << ',';
      cpp_init(out, fmt, data, axis+1, offset * n + i);
    }
    out << '}';
  }
}

template<class Data> string cpp_init(const string& fmt, const Data& data) {
  ostringstream init;
  cpp_init(init, fmt, data.raw(), 0, 0);
  return init.str();
}

// Remember a table we want to save
template<class Data> void
remember(const string& type, const string& name, const string& fmt, const Data& data, const int flags = 0) {
  const auto scalars = scalar_view(asarray(data));
  string sizes;
  for (const auto n : scalars.shape())
    sizes += tfm::format("[%d]", n);
  tables.emplace_back(type, name, sizes, cpp_init(fmt, scalars), flags);
}

// Reformat an initializer for Javascript
string js_init(const string& init) {
  string js;
  for (const char c : init)
    js.push_back(c == '{' ? '[' : c == '}' ? ']' : c);
  return js;
}

// Generate file contents
unordered_map<string,string> save() {
  unordered_map<string,string> files;
  files["halfsuper_wins.h"] = halfsuper_wins();
  {
    string h, cc;
    h += autogenerated;
    h += "#pragma once\n\n#include <cstdint>\n";
    h += "namespace pentago {\n\n";

    cc += autogenerated;
    cc += "#include \"pentago/base/gen/tables.h\"\n";
    cc += "namespace pentago {\n\n";

    for (const int phase : {0, 1}) {
      for (const auto& [type, name, sizes, init, flags] : tables) {
        if (flags & for_wasm ? !phase : phase) {
          h += tfm::format("extern const %s %s%s;\n", type, name, sizes);
          cc += tfm::format("const %s %s%s = %s;\n", type, name, sizes, init);
        }
      }
      const auto pre = phase ? "#endif  // !defined(__wasm__)\n" : "#ifndef __wasm__\n";
      h += pre;
      cc += pre;
    }
    h += "\n}\n";
    cc += "\n}\n";
    files["tables.h"] = h;
    files["tables.cc"] = cc;
  }
  {
    string js;
    js += autogenerated;
    for (const auto& [type, name, sizes, init, flags] : tables)
      if (flags & for_js)
        js += tfm::format("exports.%s = %s\n", name, js_init(init));
    files["tables.js"] = js;
  }
  return files;
}

uint64_t ipow(uint64_t a, uint64_t b) {
  uint64_t p = 1;
  while (b) {
    if (b & 1) p *= a;
    b >>= 1;
    a *= a;
  }
  return p;
}

uint16_t qbit(int x, int y) {
  return 1 << (3*x + y);
}

// Check that an array has given hash
template<class A> void check(const A& data, const string& expected) {
  const auto actual = portable_hash(data);
  if (actual != expected)
    throw ValueError(tfm::format("hash mismatch: expected '%s', got '%s'", expected, actual));
}

// Check, but cast to different type first for backwards compatibility
template<class T,class A> void check(const A& data, const string& expected) {
  const auto flat = scalar_view(asarray(data)).flat();
  Array<T> cast(flat.size());
  for (const int i : range(flat.size()))
    cast[i] = flat[i];
  check(cast, expected);
}

// There are 3**9 = 19683 possible states in each quadrant.  3**9 < 2**16, so we can store
// a quadrant state in 16 bits using radix 3.  However, radix 3 is inconvenient for computation,
// so we need lookup tables to go to and from packed form.
const Array<const uint16_t> pack = []() {
  const Array<uint16_t> pack(1 << 9);
  for (const int v : range(512)) {
    uint16_t pv = 0;
    for (const int i : range(9))
      if (v & 1 << i)
        pv += ipow(3, i);
    pack[v] = pv;
  }
  check(pack, "b86e92ca7f525bd398ba376616219831e3f4f1a5");
  remember("uint16_t", "pack_table", "%d", pack);
  return pack;
}();

const Array<const Vector<uint16_t,2>> unpack = []() {
  const Array<Vector<uint16_t,2>> unpack(ipow(3, 9));
  for (const int v : range(ipow(3, 9))) {
    auto vv = v;
    uint16_t p0 = 0, p1 = 0;
    for (const int i : range(9)) {
      const auto c = vv % 3;
      if (c == 1)
        p0 += 1 << i;
      else if (c == 2)
        p1 += 1 << i;
      vv /= 3;
    }
    unpack[v] = vec(p0, p1);
  }

  // pack and unpack should be inverses
  for (const int i : range(ipow(3, 9)))
    GEODE_ASSERT(pack[unpack[i][0]] + 2*pack[unpack[i][1]] == i);

  check(unpack, "99e742106ae60b62c0bb71dee15789ef1eb761a0");
  remember("uint16_t", "unpack_table", "0x%x", unpack);
  return unpack;
}();

const Array<const uint64_t> win_patterns = []() {
  const auto bit = [](int x, int y, const bool flip, const bool rotate) {
    GEODE_ASSERT(0 <= x && x < 6);
    GEODE_ASSERT(0 <= y && y < 6);
    if (flip)
      swap(x, y);
    if (rotate) {
      const auto t = 5 - y;
      y = x;
      x = t;
    }
    return uint64_t(1) << (16*(2*(x/3)+y/3)+3*(x%3)+y%3);
  };

  // List the various ways of winning
  vector<uint64_t> wins;
  // Off-centered diagonal 5 in a row
  for (const int x : range(2)) {
    for (const int y : range(2)) {
      uint64_t p = 0;
      for (const int i : range(-2,3))
        p |= bit(2+x+i, 2+y+i*(x==y ? -1 : 1), false, false);
      wins.push_back(p);
    }
  }
  // Axis aligned 5 in a row
  for (const bool flip : {false, true}) {
    for (const int x : range(2)) {
      for (const int y : range(6)) {
        uint64_t p = 0;
        for (const int i : range(5))
          p |= bit(x+i, y, flip, false);
        wins.push_back(p);
      }
    }
  }
  // Centered diagonal 5 in a row
  for (const bool rotate : {false, true}) {
    for (const int x : range(2)) {
      uint64_t p = 0;
      for (const int i : range(5))
        p |= bit(i+x, i+x, false, rotate);
      wins.push_back(p);
    }
  }

  // Check that 5 bits are set in each
  for (const auto w : wins)
    GEODE_ASSERT(popcount(w) == 5);

  // There should be 4*3*2+4+4 == 32 ways to win.
  // The first four of these are special: they require contributions from three quadrants.
  // The remaining 28 require contributions from only two quadrants.
  GEODE_ASSERT(wins.size() == 32 && 32 == 4*3*2+4+4);
  return asarray(wins).copy();
}();

const Array<const uint64_t,2> win_contributions = []() {
  // For each possible value of each quadrant, determine all the ways that quadrant can
  // contribute to victory.
  const Array<uint64_t,2> table(4, 512);
  for (const int qx : range(2)) {
    for (const int qy : range(2)) {
      const auto q = 2*qx+qy;
      const auto qb = uint64_t(0x1ff) << 16*q;
      for (const uint64_t v : range(512)) {
        const auto b = v << 16*q;
        for (const int i : range(win_patterns.size())) {
          const auto w = win_patterns[i];
          if (w&qb && !(~b&(w&qb)))
            table(q, v) |= uint64_t(1) << 2*i;
        }
      }
    }
  }
  check(table, "4e5cf35e82fceecd464d73c3de35e6af4f75ee34");
  remember("uint64_t", "win_contributions", "0x%xL", table);
  return table;
}();

const Array<const Vector<uint16_t,2>> rotations = []() {
  const Array<Vector<uint16_t,2>> table(512);
  for (const uint16_t v : range(512)) {
    uint16_t left = 0, right = 0;
    for (const int x : range(3))
      for (const int y : range(3))
        if (v & qbit(x,y)) {
          left |= qbit(2-y, x);
          right |= qbit(y, 2-x);
        }
    GEODE_ASSERT(popcount(v) == popcount(left));
    GEODE_ASSERT(popcount(v) == popcount(right));
    table[v] = vec(left, right);
  }
  check(table, "195f19d49311f82139a18ae681592de02b9954bc");
  remember("uint16_t", "rotations", "0x%x", table);
  return table;
}();

// Precompute table of multistep rotations
const Array<const uint16_t,2> all_rotations = []() {
  const Array<uint16_t,2> rotate(512,4);
  for (const int v : range(512)) {
    rotate(v,0) = v;
    for (const int i : range(3))
      rotate(v,i+1) = rotations[rotate(v,i)][0];
  }
  return rotate;
}();

template<class F> string joins(const string& sep, const int count, const F& f) {
  vector<string> values;
  for (const int i : range(count))
    values.push_back(f(i));
  return join(sep, values);
}

// Sort patterns into quadrant sets
const map<vector<int>,vector<uint64_t>> quadrant_set_win_patterns = []() {
  map<vector<int>,vector<uint64_t>> patterns;
  for (const auto pat : win_patterns) {
    vector<int> quads;
    for (const int q : range(4))
      if (pat & uint64_t(0x1ff) << 16*q)
        quads.push_back(q);
    patterns[quads].push_back(pat);
  }
  GEODE_ASSERT(patterns.size() == 4 + 4 + 2);
  return patterns;
}();

string halfsuper_wins() {
  string file;
  const auto line = [&](auto... args) { file += tfm::format(args...); file += '\n'; };

  // Start ourselves off
  file += autogenerated;
  line("#pragma once\n");
  line("NAMESPACE_PENTAGO\n");

  // Routines to expand single-quadrant supers into halfsuper_t
  for (const int q : range(4)) {
    line("METAL_INLINE halfsuper_t only%d(bool a0, bool a1, bool a2, bool a3, bool parity) {", q);
    Array<uint64_t> masks(4);
    for (const int i : range(4))
      for (const int j : range(64))
        if (i == (q ? j>>(2*q-1)&3 : 2*(j&1)+((j>>1&1)^(j>>3&1)^(j>>5&1))))
          masks[i] |= uint64_t(1)<<j;
    if (q == 3)
      for (const int i : range(2))
        masks[2+i] = masks[i];
    const auto c = [=](int i) { return tfm::format("(a%d ? 0 : 0x%xull)", i, masks[i]); };
    switch (q) {
      case 0: case 1: case 2:
        if (q == 0) {
          for (const int k : range(2)) {
            line("  const bool t%d = parity & (a%d ^ a%d);", k, 2*k, 2*k+1);
            line("  a%d ^= t%d; a%d ^= t%d;", 2*k, k, 2*k+1, k);
          }
        }
        line("  const auto x = %s;", joins("\n               | ", 4, c));
        line("  return halfsuper_t(x, x);");
        break;
      case 3:
        line("  return halfsuper_t(%s | %s,", c(0), c(1));
        line("                     %s | %s);", c(2), c(3));
        break;
    }
    line("}\n");
  }

  line("halfsuper_t halfsuper_wins(const side_t side, const bool parity) {");
  line("  auto wins = halfsuper_t(0);\n");
  for (const int q : range(4))
    line("  const auto q%d = ~quadrant(side, %d);", q, q);
  line("  #define M(m,a) only##a(%s, parity)", joins(", ", 4,
      [](int r) { return tfm::format("q##a & m[%d]", r); }));

  // Process each pattern set
  int counter = 0;
  for (const auto& p : quadrant_set_win_patterns) {
    const auto quads = get<0>(p);
    const auto pats = get<1>(p);
    line("\n  // Quadrants %s", asarray(quads));
    Array<uint16_t,3> masks(pats.size(), quads.size(), 4);
    for (const int ip : range(pats.size())) {
      for (const int iq : range(quads.size())) {
        const int q = quads[iq];
        const auto pat = uint16_t(pats[ip] >> 16*q);
        for (const int r : range(4))
          masks(ip, iq, r) = all_rotations(pat, (4-r)%4);
      }
    }
    const auto mask = tfm::format("m%d", counter);
    const auto fmt = "0%o";
    string leading, nounroll, loop, ith, init;
    if (pats.size() > 1) {
      leading = tfm::format("[%d]", pats.size());
      nounroll = "WASM_NOUNROLL\n  ";
      loop = tfm::format("for (int i = 0; i < %d; i++) ", pats.size());
      ith = tfm::format("%s[i]", mask);
      init = cpp_init(fmt, masks);
    } else {
      ith = mask;
      init = cpp_init(fmt, masks[0]);
    }
    line("  const quadrant_t %s%s[%d][4] = %s;", mask, leading, quads.size(), init);
    line("  %s%swins |= %s;", nounroll, loop, joins(" & ", quads.size(),
        [&](int iq) { return tfm::format("M(%s[%d],%d)", ith, iq, quads[iq]); }));
    counter++;
  }
  line("\n  #undef M");
  line("  return wins;");
  line("}\n\nEND_NAMESPACE_PENTAGO");
  return file;
}

REMEMBER(rotated_win_contributions,
  const auto wins = win_contributions;
  const Array<uint64_t,2> deltas(4, 512);
  for (const int q : range(4)) {
    for (const int v : range(512)) {
      const auto [r0, r1] = rotations[v];
      deltas(q,v) = (wins(q,v)|wins(q,r0)|wins(q,r1)) - wins(q,v);
    }
  }
  remember("uint64_t", "rotated_win_contribution_deltas", "0x%xL", deltas);
);

REMEMBER(unrotated_win_distances,
  // Work out the various ways of winning
  const auto patterns = win_patterns.reshape(vec(2, 16));

  // Generate a distance table giving the total number of moves required for player 0 (black) to
  // reach the pattern, or 4 for unreachable (a white stone is in the way).  Each pattern distance
  // has 3 bits (for 0,1,2,3 or 4=infinity) plus an extra 0 bit to stop carries at runtime, so we
  // fit 16 patterns into each of 2 64-bit ints
  const Array<uint64_t,3> table(4, ipow(3,9), 2);  // Indexed by quadrant,position,word
  for (const int i : range(patterns.shape()[0])) {
    for (const int j : range(patterns.shape()[1])) {
      const auto pat = patterns(i,j);
      for (const int q : range(4)) {
        for (const int b : range(ipow(3,9))) {
          const auto [s0, s1] = unpack[b];
          const auto qp = (pat>>16*q)&0x1ff;
          const uint64_t d = s1&qp ? 4 : popcount(qp&~s0);
          table(q,b,i) |= d << 4*j;
        }
      }
    }
  }
  check(table, "02b780e3172e11b861dd3106fc068ccb59cebc1c");
  remember("uint64_t", "unrotated_win_distances", "0x%xL", table);
)

REMEMBER(arbitrarily_rotated_win_distances,
  // Work out the various ways of winning, allowing arbitrarily many rotations
  const auto patterns = win_patterns.reshape(vec(2,16));

  // Generate a distance table giving the total number of moves required for player 0 (black) to
  // reach the pattern, or 4 for unreachable (a white stone is in the way).  Each pattern distance
  // has 3 bits (for 0,1,2,3 or 4=infinity) plus an extra 0 bit to stop carries at runtime, so we
  // fit 16 patterns into each of 2 64-bit ints
  const Array<uint64_t,3> table(4, ipow(3,9), 2);  // Indexed by quadrant,position,word
  for (const int i : range(patterns.shape()[0])) {
    for (const int j : range(patterns.shape()[1])) {
      const auto pat = patterns(i,j);
      for (const int q : range(4)) {
        for (const int b : range(ipow(3,9))) {
          const auto [s0, s1] = unpack[b];
          int d = 1000;
          for (const int r : range(4)) {
            const auto qp = (pat>>16*q)&0x1ff;
            d = min(d, all_rotations(s1,r)&qp ? 4 : popcount(qp&~all_rotations(s0,r)));
          }
          table(q,b,i) |= uint64_t(d) << 4*j;
        }
      }
    }
  }
  check(table, "b1bd000ba42513ee696f065503d68f62b98ac85e");
  remember("uint64_t", "arbitrarily_rotated_win_distances", "0x%xL", table);
)

REMEMBER(rotated_win_distances,
  // Work out the various ways of winning, allowing at most one quadrant to rotate
  const auto patterns = win_patterns;
  GEODE_ASSERT(patterns.size() == 32);
  vector<tuple<uint64_t,int>> rotated_patterns;  // List of pattern,rotated_quadrant pairs
  for (const auto pat : patterns)
    for (const int q : range(4))
      if (pat & uint64_t(0x1ff)<<16*q)
        rotated_patterns.push_back(make_tuple(pat, q));
  GEODE_ASSERT(rotated_patterns.size() == 68);  // We'll treat this as shape (4,17) below

  // Generate a distance table giving the total number of moves required for player 0 (black) to
  // reach the pattern, or 4 for unreachable (a white stone is in the way).  Each pattern distance
  // has 3 bits (for 0,1,2,3 or 4=infinity), so we fit 17 patterns into each of 4 64-bit ints
  const Array<uint64_t,3> table(4, ipow(3,9), 4);  // Indexed by quadrant,position,word
  for (const int i : range(rotated_patterns.size())) {
    const auto [pat, qr] = rotated_patterns[i];
    for (const int q : range(4)) {
      for (const int b : range(ipow(3,9))) {
        const auto [s0, s1] = unpack[b];
        const auto qp = (pat>>16*q)&0x1ff;
        auto d = s1&qp ? 4 : popcount(qp&~s0);
        if (q == qr)
          for (const int r : range(2))
            d = min(d, rotations[s1][r]&qp ? 4 : popcount(qp&~rotations[s0][r]));
        table(q,b,i/17) |= uint64_t(d) << 3*(i%17);
      }
    }
  }
  check(table, "6fc4ae84c574d330f38e3f07b37ece103fa80c45");
  remember("uint64_t", "rotated_win_distances", "0x%xL", table);
)

const Array<const uint16_t> reflections = []() {
  const Array<uint16_t> table(512);
  for (const uint16_t v : range(512)) {
    uint16_t r = 0;
    for (const int x : range(3))
      for (const int y : range(3))
        if (v & qbit(x,y))
          r |= qbit(2-y, 2-x);  // Reflect about x = y line
    GEODE_ASSERT(popcount(v) == popcount(r));
    table[v] = r;
  }
  check(table, "2b23dc37f4bc1008eba3df0ee1b7815675b658bf");
  remember("uint16_t", "reflections", "0x%x", table);
  return table;
}();

REMEMBER(moves,
  // Given a quadrant, compute all possible moves by player 0 as a function of the set of filled spaces,
  // stored as a nested array.
  vector<uint16_t> offsets;
  vector<uint16_t> flat;
  offsets.push_back(0);
  for (const uint16_t filled : range(1 << 9)) {
    vector<uint16_t> mv;
    const auto free = ~filled;
    for (const int i : range(9)) {
      const auto b = 1 << i;
      if (free & b)
        mv.push_back(b);
    }
    GEODE_ASSERT(int(mv.size()) == 9-popcount(filled));
    extend(flat, mv);
    offsets.push_back(flat.size());
  }
  GEODE_ASSERT(flat.size() < (1 << 16));
  check<uint64_t>(offsets, "15b71d6e563787b098860ae0afb1d1aede6e91c2");
  check<uint64_t>(flat, "1aef3a03571fe13f0ab71173d79da107c65436e0");
  remember("uint16_t", "move_offsets", "%d", offsets);
  remember("uint16_t", "move_flat", "%d", flat);
);

const int newaxis = -1;
const int fullaxis = -2;

template<class T> struct View {
  T* data;
  vector<int> shape, strides;

  template<int d> View(const Array<T,d>& array)
    : data(array.data()) {
    extend(shape, array.shape());
    extend(strides, pentago::strides(array.shape()));
  }

  View(T* data, const vector<int>& shape, const vector<int>& strides)
    : data(data), shape(shape), strides(strides) {}

  int rank() const { return shape.size(); }

  bool singleton(const int axis) const {
    GEODE_ASSERT(unsigned(axis) < unsigned(rank()));
    return shape[axis] == 1 && strides[axis] == 0;
  }

  View operator()(const vector<int>& indices) const {
    T* new_data = data;
    vector<int> new_shape, new_strides;
    int j = 0;
    for (const auto i : indices) {
      if (i == fullaxis) {
        GEODE_ASSERT(j < rank());
        new_shape.push_back(shape[j]);
        new_strides.push_back(strides[j]);
        j++;
      } else if (i == newaxis) {
        new_shape.push_back(1);
        new_strides.push_back(0);
      } else {
        GEODE_ASSERT(j < rank() && 0 <= i && i < shape[j]);
        new_data += strides[j] * i;
        j++;
      }
    }
    GEODE_ASSERT(j <= rank());
    while (j < rank()) {
      new_shape.push_back(shape[j]);
      new_strides.push_back(strides[j]);
      j++;
    }
    return View(new_data, new_shape, new_strides);
  }

  View swapaxes(const int a0, const int a1) const {
    GEODE_ASSERT(unsigned(a0) < unsigned(rank()));
    GEODE_ASSERT(unsigned(a1) < unsigned(rank()));
    auto new_shape = shape;
    auto new_strides = strides;
    swap(new_shape[a0], new_shape[a1]);
    swap(new_strides[a0], new_strides[a1]);
    return View(data, new_shape, new_strides);
  }

  void operator|=(const View v) const {
    GEODE_ASSERT(rank() == v.rank());
    vector<int> full_shape;
    for (const int i : range(rank())) {
      GEODE_ASSERT(shape[i] == v.shape[i] || singleton(i) || v.singleton(i));
      const auto size = singleton(i) ? v.shape[i] : shape[i];
      if (!size) return;
      full_shape.push_back(size);
    }
    vector<int> indices(rank());
    T* dst = data;
    const T* src = v.data;
    for (;;) {
      *dst |= *src;
      for (int a = rank()-1; a >= 0; a--) {
        dst += strides[a];
        src += v.strides[a];
        if (++indices[a] < full_shape[a]) {
          goto next;
        } else {
          dst -= full_shape[a] * strides[a];
          src -= full_shape[a] * v.strides[a];
          indices[a] = 0;
        }
      }
      break;
      next:;
    }
  }

  vector<bool> flat() const {
    vector<bool> flat;
    for (const auto s : shape)
      if (!s) return flat;
    vector<int> indices(rank());
    const T* p = data;
    for (;;) {
      flat.push_back(*p);
      for (int a = rank()-1; a >= 0; a--) {
        p += strides[a];
        if (++indices[a] < shape[a]) {
          goto next;
        } else {
          p -= shape[a] * strides[a];
          indices[a] = 0;
        }
      }
      break;
      next:;
    }
    return flat;
  }
};

REMEMBER(superwin_info,
  const Array<Vector<uint16_t,4>> ways(32);
  char_view(ways).copy(char_view(win_patterns));
  for (const int i : range(32))
    for (const int j : range(4))
      ways[i][j] = win_patterns[i]>>16*j&0x1ff;

  const string types = "hvlua";  // Horizontal, vertical, diagonal lo/hi/assist
  const string patterns[] = {"vv--", "h-h-", "-h-h", "--vv", "l-al", "ua-u", "all-", "-uua"};
  const Array<bool,7> info(vec(4,512,5,4,4,4,4));  // Indexed by quadrants, quadrant state, superwin_info field, r0-3
  for (const auto& pattern : patterns) {
    GEODE_ASSERT(pattern.size() == 4);
    vector<int> used, unused;
    for (const int i : range(4))
      (pattern[i] != '-' ? used : unused).push_back(i);
    GEODE_ASSERT(used.size() == 2 || used.size() == 3);
    vector<Vector<uint16_t,4>> ws;
    for (const auto& w : ways) {
      for (const int i : range(4))
        if (!(bool(w[i]) == (pattern[i]!='-') || pattern[i]=='a'))
          goto skip;
      ws.push_back(w);
      skip:;
    }
    GEODE_ASSERT(ws.size() == 6 || ws.size() == 3);
    GEODE_ASSERT(ws.size() <= ipow(4,4-used.size()));
    for (const auto q : used) {
      for (const int i : range(ws.size())) {
        const auto w = ws[i][q];
        auto s = View<bool>(info)({q,fullaxis,newaxis,newaxis,newaxis,newaxis,int(types.find(pattern[q]))});
        s = s.swapaxes(4, 5+q);
        for (const int j : range(unused.size()))
          s = s.swapaxes(1+j, 5+unused[j]);
        const Array<bool,2> x(512,4);
        for (const int v : range(512))
          for (const int r : range(4))
            x(v,r) = (all_rotations(v,r)&w) == w;
        if (unused.size() == 1)
          s({fullaxis,i}) |= View<bool>(x)({fullaxis,newaxis,newaxis,fullaxis,newaxis,newaxis,newaxis,newaxis});
        else
          s({fullaxis,i/4,i%4}) |= View<bool>(x)({fullaxis,newaxis,fullaxis,newaxis,newaxis,newaxis,newaxis});
      }
    }
  }

  // Switch to r3, r2, r1, r0 major order to match super_t
  const auto flat = View<bool>(info).swapaxes(3,6).swapaxes(4,5).flat();
  // Pack into 64 bit chunks
  const Array<uint64_t> bits(flat.size() / 64);
  for (const int i : range(flat.size()))
    bits[i/64] |= uint64_t(flat[i]) << i%64;
  // Check hash before converting to final endianness
  check(bits, "668eb0a940489f434f804d994698a4fc85f5b576");
  // Account for big endianness if necessary.  Note that the byte order for the entire 256 bit super_t is reversed
  if (std::endian::native == std::endian::big)
    for (const int i : range(bits.size() / (256/8)))
      std::reverse(bits.data()+256/8*i, bits.data()+256/8*(i+1));
  remember("uint64_t", "superwin_info", "0x%xL", bits);
)

// An element of S_{6*6}
struct Sym {
  Vector<uint8_t,6*6> p;

  Sym() {  // Identity
    for (const int i : range(6*6))
      p[i] = i;
  }

  static Sym reflect() {
    Sym g;
    for (const int x : range(6))
      for (const int y : range(6))
        g.p[6*x+y] = 6*(5-y)+(5-x);
    GEODE_ASSERT(g(0,0) == vec<uint8_t>(5,5));
    GEODE_ASSERT(g(5,0) == vec<uint8_t>(5,0));
    return g;
  }

  static Sym gr() {
    Sym g;
    for (const int x : range(6))
      for (const int y : range(6))
        g.p[6*x+y] = 6*(5-y)+x;
    GEODE_ASSERT(g(0,0) == vec<uint8_t>(5,0));
    GEODE_ASSERT(g(5,0) == vec<uint8_t>(5,5));
    return g;
  }

  static Sym lr(const int q) {
    Sym g;
    const int d = 3 * (6*(q/2) + q%2);
    for (const int x : range(3))
      for (const int y : range(3)) {
        g.p[d + 6*x + y] = d + 6*(2-y) + x;
      }
    return g;
  }

  bool operator==(const Sym& h) const { return p == h.p; }

  Vector<uint8_t,2> operator()(const int x, const int y) const {
    const auto i = p[6*x+y];
    return vec<uint8_t>(i/6, i%6);
  }

  Sym operator*(const Sym& h) const {
    Sym gh;
    for (const int i : range(6*6))
      gh.p[i] = p[h.p[i]];
    return gh;
  }

  Sym inv() const {
    Sym inv;
    for (const int i : range(6*6))
      inv.p[p[i]] = i;
    return inv;
  }
};

REMEMBER(commute_global_local_symmetries,
  // Represent the various symmetries as subgroups of S_{6*6}
  const auto identity = Sym();
  const auto reflect = Sym::reflect();
  for (const auto& f : {identity, reflect})
    GEODE_ASSERT(sqr(f) == identity);

  const auto gr = Sym::gr();
  GEODE_ASSERT(sqr(sqr(gr)) == identity);

  Sym lr[4], plr[4][4];
  for (const int q : range(4)) {
    lr[q] = plr[q][1] = Sym::lr(q);
    for (const int i : range(3))
      plr[q][i+1] = lr[q] * plr[q][i];
    GEODE_ASSERT(sqr(plr[q][2]) == identity);
  }

  // Construct local rotation group
  const Array<Sym,4> local_4d(4,4,4,4);
  for (const int i0 : range(4))
    for (const int i1 : range(4))
      for (const int i2 : range(4))
        for (const int i3 : range(4))
          local_4d(i0,i1,i2,i3) = plr[0][i0] * plr[1][i1] * plr[2][i2] * plr[3][i3];
  const auto local = local_4d.flat();
  // Verify commutativity
  for (const auto& g : local)
    for (const auto& h : local)
      GEODE_ASSERT(g*h == h*g);

  // Construct global rotation group
  const Array<Sym> global(8);
  for (const int i : range(3))
    global[i+1] = gr * global[i];
  for (const int i : range(4))
    global[4+i] = reflect * global[i];

  // Determine where each global/local conjugation appears in local
  unordered_map<Vector<uint8_t,6*6>,uint8_t> local_id;
  for (const int i : range(256))
    local_id[local[i].p] = i;
  const Array<uint8_t,2> table(8,256);
  for (const int i : range(8))
    for (const int j : range(256))
      table(i,j) = check_get(local_id, (global[i].inv() * local[j] * global[i]).p);
  check<uint64_t>(table, "e051d034c07bfa79fa62273b05839aedf446d499");
  remember("uint8_t", "commute_global_local_symmetries", "%d", table);
)

REMEMBER(superstandardize_table,
  // Given the relative rankings of the four quadrants, determine the global rotation that minimizes the board value
  const Array<int,2> rotate(4,4);
  for (const int i : range(4))
    rotate(0,i) = i;
  rotate[1].copy(asarray(vec(2,0,3,1)));
  for (const int p : range(2))
    for (const int i : range(4))
      rotate(p+2,i) = rotate(1,rotate(p+1,i));
  const Array<int> table(256);
  for (const int p0 : range(4)) {
    for (const int p1 : range(4)) {
      for (const int p2 : range(4)) {
        for (const int p3 : range(4)) {
          const int p[4] = {p0,p1,p2,p3};
          int min_value = 1 << 16, min_r = -1;
          for (const int r : range(4)) {
            int value = 0;
            for (const int i : range(4))
              value += p[i] << 2*rotate(r,i);
            if (min_value > value) {
              min_value = value;
              min_r = r;
            }
          }
          table[p0+4*(p1+4*(p2+4*p3))] = min_r;
        }
      }
    }
  }
  check<uint64_t>(table, "dd4f59fea3135a860e76ed397b8f1863b23cc17b");
  remember("uint8_t", "superstandardize_table", "%d", table);
)

REMEMBER(rotation_minimal_quadrants,
  // Find quadrants minimal w.r.t. rotations but not necessarily reflections
  const Array<uint16_t> minq(ipow(3,9));
  vector<uint16_t> all_rmins;
  for (const uint16_t q : range(ipow(3,9))) {
    const auto [s0, s1] = unpack[q];
    minq[q] = q;
    for (const int r : range(1,4))
      minq[q] = min<uint16_t>(minq[q], pack[all_rotations(s0,r)] + 2*pack[all_rotations(s1,r)]);
    if (minq[q] == q)
      all_rmins.push_back(q);
  }

  // Sort quadrants so that reflected versions are consecutive (after rotation minimizing), and all pairs come first
  const auto min_reflect = [&](const uint16_t q) {
    const auto [s0, s1] = unpack[q];
    return minq[pack[reflections[s0]] + 2*pack[reflections[s1]]];
  };
  const auto sig = [&](const uint16_t q) {
    const auto r = min_reflect(q);
    return min(q,r) + ipow(3,9)*(q == r);
  };
  std::stable_sort(all_rmins.begin(), all_rmins.end(), [&](uint16_t q0, uint16_t q1) { return sig(q0) < sig(q1); });
  const auto ordered = [&](RawArray<const uint16_t> qs) {
    for (const int i : range(qs.size())) {
      const auto q = qs[i], r = min_reflect(q);
      const int j = i ^ (q != r);
      GEODE_ASSERT(qs.valid(j) && r == qs[j]);
    }
  };
  ordered(all_rmins);

  // Partition quadrants by stone counts
  vector<vector<uint16_t>> rmins(10*(10+1)/2);
  for (const auto q : all_rmins) {
    const auto [s0, s1] = unpack[q];
    const auto b = popcount(s0), w = popcount(s1);
    rmins[b*(21-b)/2+w].push_back(q);
  }
  for (const auto& qs: rmins)
    ordered(qs);

  // Count the number of elements in each bucket not fixed by reflection
  vector<int> moved;
  for (const auto& qs : rmins) {
    int m = 0;
    for (const auto q : qs)
      m += min_reflect(q) != q;
    GEODE_ASSERT(m % 2 == 0);
    moved.push_back(m);
  }

  // Compute inverse.  inverse[q] = 4*i+r if rmins[?][i] rotated left 90*r degrees is q
  const Array<int> inverse(ipow(3, 9));
  inverse.fill(1 << 20);
  for (const auto& qs : rmins) {
    for (const int j : range(qs.size())) {
      const auto [s0, s1] = unpack[qs[j]];
      for (const int r : range(4)) {
        const auto q = pack[all_rotations(s0,r)]+2*pack[all_rotations(s1,r)];
        inverse[q] = min(inverse[q], 4*j+r);
      }
    }
  }
  GEODE_ASSERT(inverse.max() < 420*4);

  // Save as a nested array
  const auto nest = asnested(rmins);
  check<uint64_t>(nest.offsets, "7e450e73e0d54bd3591710e10f4aa76dbcbbd715");
  check<uint64_t>(nest.flat, "8f48bb94ad675de569b07cca98a2e930b06b45ac");
  check<uint64_t>(inverse, "339369694f78d4a197db8dc41a1f41300ba4f46c");
  check<uint64_t>(moved, "dce212878aaebbcd995a8a0308335972bd1d5ef7");
  remember("uint16_t", "rotation_minimal_quadrants_offsets", "%d", nest.offsets, for_js);
  remember("uint16_t", "rotation_minimal_quadrants_flat", "%d", nest.flat);
  remember("uint16_t", "rotation_minimal_quadrants_inverse", "%d", inverse, for_js);
  remember("uint16_t", "rotation_minimal_quadrants_reflect_moved", "%d", moved);
)

bool endswith(const string& x, const string& y) {
  return x.size() >= y.size() && x.substr(x.size() - y.size()) == y;
}

}  // namespace
}  // namespace pentago
using namespace pentago;

int main(int argc, char** argv) {
  try {
    const vector<string> paths(argv+1, argv+argc);
    auto left = paths;
    for (const auto& pair : save()) {
      const auto& suffix = get<0>(pair);
      const auto& data = get<1>(pair);
      const auto it = std::find_if(left.begin(), left.end(), [=](const string& p) { return endswith(p, suffix); });
      if (it == left.end())
        throw ValueError(tfm::format("No path in %s ending with %s (all paths = %s)",
                                     asarray(left), suffix, asarray(paths)));
      ofstream f(it->c_str());
      f << data;
      left.erase(it);
    }
    if (left.size())
      throw ValueError(tfm::format("Don't know how to write paths %s", asarray(left)));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
