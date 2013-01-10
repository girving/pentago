// Symmetry-aware board counting via Polya's enumeration theorem

#include <pentago/convert.h>
#include <pentago/section.h>
#include <pentago/symmetry.h>
#include <other/core/array/sort.h>
#include <other/core/math/uint128.h>
#include <other/core/python/wrap.h>
#include <other/core/random/Random.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/utility/Hasher.h>
#include <other/core/utility/remove_const_reference.h>
#include <tr1/unordered_map>
#include <vector>
namespace pentago {

using std::cout;
using std::endl;
using std::vector;
using std::tr1::unordered_map;

namespace {

// We want to count all board positions with the given number of white and black stones,
// ignoring all supersymmetries.  To do this, we apply Polya's theorem with |G| = 2048
// and color weight generating function f(b,w) = 1+b+w.  For details and notation, see
//     http://en.wikipedia.org/wiki/Polya_enumeration_theorem

template<int d_> struct Polynomial {
  static const int d = d_;
  unordered_map<Vector<int,d>,uint128_t,Hasher> x;

  Polynomial() {}

  Polynomial(uint128_t a) {
    x[Vector<int,d>()] = a;
  }
};

template<int d> Polynomial<d> monomial(const Vector<int,d>& m) {
  Polynomial<d> p;
  p.x[m]++;
  return p;
}

template<int d> Polynomial<d> operator*(const Polynomial<d>& f, const Polynomial<d>& g) {
  Polynomial<d> fg;
  for (auto& fm : f.x)
    for (auto& gm : g.x)
      fg.x[fm.first+gm.first] += fm.second*gm.second;
  return fg;
}

template<int d> Polynomial<d>& operator*=(Polynomial<d>& f, const Polynomial<d>& g) {
  f = f*g;
  return f;
}

template<int d> Polynomial<d>& operator+=(Polynomial<d>& f, const Polynomial<d>& g) {
  for (auto& m : g.x)
    f.x[m.first] += m.second;
  return f;
}

template<int fd,class Gs> auto compose(const Polynomial<fd>& f, const Gs& g)
  -> Polynomial<remove_const_reference<decltype(g[0])>::type::d> {
  const int gd = remove_const_reference<decltype(g[0])>::type::d;
  OTHER_ASSERT(fd==(int)g.size());
  vector<vector<Polynomial<gd>>> gp(fd); // indexed by i, power of g[i]
  Polynomial<gd> fg;
  for (auto& fm : f.x) {
    Polynomial<gd> m = fm.second;
    for (int i=0;i<fd;i++)
      if (fm.first[i]) {
        while ((int)gp[i].size()<fm.first[i])
          gp[i].push_back(gp[i].size()?g[i]*gp[i].back():g[i]);
        m *= gp[i][fm.first[i]-1];
      }
    fg += m;
  }
  return fg;
}

string OTHER_UNUSED str(uint128_t n) {
  uint64_t lo(n);
  OTHER_ASSERT(lo==n);
  return format("%lld",lo);
}

template<int d> string str(const Polynomial<d>& f, const char* names) {
  OTHER_ASSERT(d==strlen(names));
  Array<Vector<int,d> > ms;
  for (auto& m : f.x)
    ms.append(m.first.reversed());
  sort(ms,LexicographicCompare());
  string s;
  if (!ms.size())
    s = '0';
  for (int i=0;i<ms.size();i++) {
    auto m = ms[i].reversed();
    if (i)
      s += " + ";
    uint128_t c = f.x.find(m)->second;
    if (c!=1 || m==Vector<int,d>())
      s += str(c);
    bool need_space = false;
    for (int j=0;j<d;j++)
      if (m[j]) {
        if (need_space)
          s += ' ';
        need_space = false;
        s += names[j];
        if (m[j]>1) {
          s += '^';
          s += str(m[j]);
          need_space = true;
        }
      }
  }
  return s;
}

Polynomial<2> count_generator(int symmetries) {
  // Make the color generating function
  Polynomial<2> f;
  f.x[vec(0,0)]++;
  f.x[vec(1,0)]++;
  f.x[vec(0,1)]++;

  // List symmetries
  Array<symmetry_t> group;
  if (symmetries==1)
    group.append(symmetry_t(0)); 
  else if (symmetries==8)
    for (int g=0;g<8;g++)
      group.append(symmetry_t(g,0));
  else if (symmetries==2048)
    for (symmetry_t s : pentago::symmetries)
      group.append(s);
  else
    OTHER_ASSERT(false);

  // Generate the cycle index
  Polynomial<36> Z;
  for (symmetry_t s : group) {
    Vector<int,36> cycles;
    for (int i=0;i<4;i++) for (int j=0;j<9;j++) {
      side_t start = (side_t)1<<(16*i+j);
      side_t side = start;
      for (int o=1;;o++) {
        side = transform_side(s,side);
        if (side==start) {
          cycles[o-1]++;
          break;
        }
      }
    }
    for (int i=0;i<36;i++) {
      OTHER_ASSERT(cycles[i]%(i+1)==0);
      cycles[i] /= i+1;
    }
    Z.x[cycles]++;
  }

  // Compute f(b^k,w^k) for k = 1 to 36
  vector<Polynomial<2>> fk(36);
  for (int k=1;k<=36;k++)
    fk[k-1] = compose(f,vec(monomial(vec(k,0)),monomial(vec(0,k))));

  // Compose Zg and fk to get F
  auto F = compose(Z,fk);
  for (auto& m : F.x) {
    OTHER_ASSERT(m.second%symmetries==0);
    m.second /= symmetries;
  }
  return F;
}

}

static uint64_t safe_mul(uint64_t a, uint64_t b) {
  uint128_t ab = uint128_t(a)*b;
  OTHER_ASSERT(!(ab>>64));
  return uint64_t(ab);
}

uint64_t choose(int n, int k) {
  if (n<0 || k<0 || k>n)
    return 0;
  k = max(k,n-k);
  uint64_t result = 1;
  for (int i=n;i>k;i--)
    result = safe_mul(result,i);
  for (int i=2;i<=n-k;i++)
    result /= i;
  return result;
}

uint64_t count_boards(int n, int symmetries) {
  OTHER_ASSERT(0<=n && n<=36);
  OTHER_ASSERT(symmetries==1 || symmetries==8 || symmetries==2048);
  static Polynomial<2> Fs[3]; // 1, 8, and 2048 symmetries
  Polynomial<2>& F = Fs[symmetries==1?0:symmetries==8?1:2]; 
  if (!F.x.size())
    F = count_generator(symmetries);
  const int b = (n+1)/2;
  auto it = F.x.find(vec(b,n-b));
  return it!=F.x.end()?it->second:0;
}

double estimate_choose(int n, int k) {
  return exp(lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1));
}

double estimate_count_boards(int n) {
  OTHER_ASSERT(0<=n && n<=36);
  return estimate_choose(36,n)*estimate_choose(n,n/2);
}

double estimate_supercount_boards(int n, double tol) {
  OTHER_ASSERT(0<=n && n<=36);
  int steps = ceil(2048/sqr(tol));
  OTHER_ASSERT(steps*sqr(tol)>=2048);
  int hits = 0;
  Ref<Random> random = new_<Random>(283111);
  for (int s=0;s<steps;s++) {
    const board_t board = random_board(random,n);
    if (board==superstandardize(board).x)
      hits++;
  }
  return (double)hits/steps*estimate_count_boards(n);
}

static inline int log_count_local_stabilizers(quadrant_t q) {
  const quadrant_t s0 = unpack(q,0),
                   s1 = unpack(q,1),
                   s0r = rotations[s0][0],
                   s1r = rotations[s1][0];
  if (s0==s0r && s1==s1r)
    return 2;
  const quadrant_t s0rr = rotations[s0r][0],
                   s1rr = rotations[s1r][0];
  return s0==s0rr && s1==s1rr ? 1 : 0;
 }

static int log_count_local_stabilizers(board_t board) {
  return log_count_local_stabilizers(quadrant(board,0))
        +log_count_local_stabilizers(quadrant(board,1))
        +log_count_local_stabilizers(quadrant(board,2))
        +log_count_local_stabilizers(quadrant(board,3));
}

Vector<uint16_t,3> popcounts_over_stabilizers(board_t board, const Vector<super_t,2>& wins) {
  const int shift = log_count_local_stabilizers(board),
            mask = (1<<shift)-1,
            w0 = popcount(wins.x),
            w1 = popcount(wins.y);
  OTHER_ASSERT(!(w0&mask) && !(w1&mask));
  return Vector<uint16_t,3>(w0>>shift,w1>>shift,256>>shift);
}

static void popcounts_over_stabilizers_test(int steps) {
  Ref<Random> random = new_<Random>(83191941);
  Hashtable<board_t> seen;
  for (int step=0;step<steps;step++) {
    // Generate a board with significant probability of duplicate quadrants
    const int stones = random->uniform<int>(0,37);
    board_t board = random_board(random,stones);
    const int qs = random->uniform<int>(0,256);
    board = quadrants(quadrant(board,qs>>0&3),
                      quadrant(board,qs>>2&3),
                      quadrant(board,qs>>4&3),
                      quadrant(board,qs>>6&3));
    // Compute meaningless data
    Vector<super_t,2> wins;
    wins.x = super_meaningless(board);
    wins.y = ~wins.x;
    // Count the fast way
    const Vector<uint16_t,3> fast = popcounts_over_stabilizers(board,wins);
    OTHER_ASSERT(fast.x+fast.y==fast.z);
    // Count the slow way 
    Vector<uint16_t,3> slow;
    seen.clear(); 
    for (int g=0;g<8;g++)
      for (int l=0;l<256;l++) {
        const board_t b = transform_board(symmetry_t(g,l),board);
        for (int q=0;q<4;q++)
          if (rotation_standardize_quadrant(quadrant(board,q)).x != rotation_standardize_quadrant(quadrant(b,q)).x)
            goto skip;
        if (seen.set(b))
          slow.x += meaningless(b);
        skip:;
      }
    slow.z = seen.size();
    slow.y = slow.z-slow.x;
    OTHER_ASSERT(slow==fast);
  }
}

Vector<uint64_t,3> sum_section_counts(RawArray<const section_t> sections, RawArray<const Vector<uint64_t,3>> counts) {
  OTHER_ASSERT(sections.size()==counts.size());
  Vector<uint64_t,3> total;
  for (int i=0;i<sections.size();i++) {
    // Count how many different unstandardized sections we can make from this one
    section_t transforms[8];
    transforms[0] = sections[i];
    int multiplier = 1;
    for (int s=1;s<8;s++) {
      transforms[s] = sections[i].transform(s);
      for (int t=0;t<s;t++)
        if (transforms[t]==transforms[s])
          goto found;
      multiplier++;
      found:;
    }
    total += (uint64_t)multiplier*counts[i];
  }
  return total;
}

}
using namespace pentago;

void wrap_count() {
  OTHER_FUNCTION(count_boards)
  OTHER_FUNCTION(estimate_supercount_boards)
  OTHER_FUNCTION(popcounts_over_stabilizers_test)
  OTHER_FUNCTION(sum_section_counts)
}
