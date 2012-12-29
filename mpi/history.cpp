// History-related utilities

#include <pentago/mpi/utility.h>
#include <pentago/convert.h>
#include <pentago/section.h>
#include <pentago/symmetry.h>
#include <pentago/thread.h>
#include <other/core/array/Array3d.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/geometry/BoxScalar.h>
#include <other/core/math/constants.h>
#include <other/core/python/module.h>
#include <other/core/python/stl.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/interrupts.h>
#include <other/core/utility/Log.h>
#include <other/core/utility/str.h>
namespace pentago {
namespace mpi {

using Log::cout;
using std::endl;

static inline int64_t subtime(RawArray<const history_t> events, const int i) {
  if (!events.valid(i>>1))
    return numeric_limits<int64_t>::max();
  const auto& event = events[i>>1];
  return i&1?event.end.us:event.start.us;
}

// Find the first instance of the given event if any exist
static int find_event(RawArray<const history_t> event_sorted_events, const event_t event) {
  int lo = 0, hi = event_sorted_events.size();
  while (lo<hi) {
    const int mid = (lo+hi)/2;
    if (event <= event_sorted_events[mid].event)
      hi = mid;
    else
      lo = mid+1;
  }
  return lo;
}

static int find_time(RawArray<const history_t> events, const double time) {
  // Find the endpoint immediately after the given time
  int lo = 0, hi = 2*events.size();
  while (lo<hi) {
    const int mid = (lo+hi)/2;
    if (time < subtime(events,mid))
      hi = mid;
    else
      lo = mid+1;
  }
  // Turn the endpoint into an event
  if (!lo || lo==2*events.size())
    return -1;
  return lo>>1;
}

// Find the endpoint immediately after the given time.  Returns kind,event,side.
Vector<int,2> search_thread(const vector<Array<const history_t>>& thread, double time) {
  OTHER_ASSERT(thread.size()>=master_idle_kind);
  Vector<int,2> best(-1,-1);
  double distance = inf;
  for (int k : range((int)master_idle_kind)) {
    if (!thread[k].size())
      continue;
    const int e = find_time(thread[k],time);
    if (e < 0)
      continue;
    for (int ee : range(max(0,e-1),min(e+2,thread[k].size()))) {
      const auto& event = thread[k][ee];
      const auto d = Box<double>(event.start.us,event.end.us).signed_distance(time);
      if (distance > d) {
        distance = d;
        best = vec(k,ee);
      }
    }
  }
  return best;
}

static section_t parse_section(const event_t event) {
  const uint32_t microsig(event>>29);
  uint8_t counts[8];
  for (int i=0;i<8;i++)
    counts[i] = microsig>>4*i&0xf;
  section_t section;
  memcpy(&section,counts,sizeof(section));
  return section;
}

static Vector<uint8_t,4> parse_block(const event_t event) {
  Vector<uint8_t,4> block;
  for (int i=0;i<4;i++)
    block[i] = event>>6*i&0x3f;
  return block;
}

static inline uint8_t parse_dimensions(const event_t event) {
  return event>>24&31;
}

string str_event(const event_t event) {
  // Parse event
  const section_t section = parse_section(event);
  const auto block = parse_block(event);
  const uint8_t dimensions = parse_dimensions(event);

  // Parse kind
  switch (event&ekind_mask) {
    case unevent:
      return "unevent";
    case block_ekind:
      return format("s%s b%d,%d,%d,%d",str(section),block[0],block[1],block[2],block[3]);
    case line_ekind: {
      string b[4];
      for (int i=0;i<4;i++)
        b[i] = i==dimensions?"_":str(int(block[i-(i>=dimensions)]));
      return format("s%s d%d b%s,%s,%s,%s",str(section),dimensions,b[0],b[1],b[2],b[3]); }
    case block_line_ekind:
      return format("s%s d%d b%d,%d,%d,%d",str(section),dimensions,block[0],block[1],block[2],block[3]);
    case block_lines_ekind:
      return format("s%s ss%d cd%d b%d,%d,%d,%d",str(section),dimensions>>2,dimensions&3,block[0],block[1],block[2],block[3]);
    default:
      return "<error>";
  }
}

static Array<Tuple<time_kind_t,event_t>> dependencies(const time_kind_t kind, event_t event) {
  // Parse event
  const section_t section = parse_section(event);
  const auto block = parse_block(event);
  const uint8_t dimensions = parse_dimensions(event),
                parent_to_child_symmetry = dimensions>>2,
                dimension = dimensions&3;
  const auto ekind = event&ekind_mask;

  // See mpi/graph for summarized explanation
  Array<Tuple<time_kind_t,event_t>> deps;
  switch (kind) {
    case allocate_line_kind: {
      OTHER_ASSERT(ekind==line_ekind);
      break; }
    case request_send_kind: {
      OTHER_ASSERT(ekind==block_lines_ekind);
      const auto parent_section = section.parent(dimension).transform(symmetry_t::invert_global(parent_to_child_symmetry));
      const auto permutation = section_t::quadrant_permutation(parent_to_child_symmetry);
      const uint8_t parent_dimension = permutation.find(dimension);
      const auto block_base = Vector<uint8_t,4>(block.subset(permutation)).remove_index(parent_dimension);
      deps.append(tuple(allocate_line_kind,line_event(parent_section,parent_dimension,block_base)));
      break; }
    case response_send_kind: {
      OTHER_ASSERT(ekind==block_lines_ekind);
      deps.append(tuple(request_send_kind,event));
      break; }
    case response_recv_kind: {
      OTHER_ASSERT(ekind==block_lines_ekind);
      deps.append(tuple(response_send_kind,event));
      break; }
    case schedule_kind: {
      OTHER_ASSERT(ekind==line_ekind);
      if (section.sum()!=35) {
        const auto child_section = section.child(dimension).standardize<8>();
        const auto permutation = section_t::quadrant_permutation(symmetry_t::invert_global(child_section.y));
        const uint8_t child_dimension = permutation.find(dimension);
        const dimensions_t dimensions(child_section.y,child_dimension);
        auto child_block = Vector<uint8_t,4>(block.slice<0,3>().insert(0,dimension).subset(permutation));
        for (uint8_t b : range(section_blocks(child_section.x)[child_dimension])) {
          child_block[child_dimension] = b;
          deps.append(tuple(response_recv_kind,block_lines_event(child_section.x,dimensions,child_block)));
        }
      }
      break; }
    case compute_kind: { // Note: all microline compute events have the same line event
      OTHER_ASSERT(ekind==line_ekind);
      deps.append(tuple(schedule_kind,event));
      break; }
    case wakeup_kind: {
      OTHER_ASSERT(ekind==line_ekind);
      deps.append(tuple(compute_kind,event)); // Corresponds to many different microline compute events
      break; }
    case output_send_kind: {
      OTHER_ASSERT(ekind==block_line_ekind);
      deps.append(tuple(wakeup_kind,line_event(section,dimension,block.remove_index(dimension))));
      break; }
    case output_recv_kind: {
      OTHER_ASSERT(ekind==block_line_ekind);
      deps.append(tuple(output_send_kind,event));
      break; }
    default:
      break;
  }
  return deps;
}

// Compute the dependencies of a given event.  Returns a list of (thread,kind,event) triples.
vector<Tuple<int,int,history_t>> event_dependencies(const vector<vector<Array<const history_t>>>& event_sorted_history, const int thread, const int kind, const history_t source) {
  vector<Tuple<int,int,history_t>> deps;
  for (const auto kind_event : dependencies(time_kind_t(kind),source.event)) {
    const int dep_kind = kind_event.x;
    const event_t dep_event = kind_event.y;
    // Search for event in each thread
    bool found = false;
    for (const int t : range((int)event_sorted_history.size())) {
      const auto& sorted_events = event_sorted_history.at(t).at(dep_kind);
      for (const int i : range(find_event(sorted_events,dep_event),sorted_events.size())) {
        const history_t& event = sorted_events[i];
        if (event.event != dep_event)
          break;
        OTHER_ASSERT(!found || dep_kind==compute_kind);
        found = true;
        deps.push_back(tuple(t,dep_kind,event));
      }
    }
    if (!found) {
      int count = 0;
      for (const int t : range((int)event_sorted_history.size())) {
        const auto& events = event_sorted_history[t].at(dep_kind);
        for (const int i : range(max(0,events.size()-1)))
          if (events[i].event > events[i+1].event)
            throw RuntimeError(format("event_dependencies: order failure: thread %d, kind %d, i %d (%d), events %lld %lld",t,dep_kind,i,events.size(),events[i].event,events[i+1].event));
        for (auto& e : events)
          if (e.event==dep_event)
            count++;
      }
      throw RuntimeError(format("event_dependencies: dependency not found, count %d, source = %d %s %s, dependency = %s %s",
        count,
        thread,time_kind_names().at(kind),str_event(source.event),
        time_kind_names().at(dep_kind),str_event(dep_event)));
    }
  }
  return deps;
}

// Find dependencies for all dvents as a consistency check
void check_dependencies(const vector<vector<Array<const history_t>>>& event_sorted_history) {
  for (const int thread : range((int)event_sorted_history.size()))
    for (const int kind : range((int)event_sorted_history[thread].size()))
      for (const auto& event : event_sorted_history[thread][kind])
        event_dependencies(event_sorted_history,thread,kind,event);
}

// Compute rank-to-rank bandwidth estimates localized in time (dimensions: epoch,src,dst)
Array<double,3> estimate_bandwidth(const vector<vector<Array<const history_t>>>& event_sorted_history, const int threads, const double dt_seconds) {
  Log::Scope scope("estimate bandwidth");
  OTHER_ASSERT(threads>1);
  const int ranks = event_sorted_history.size()/threads;
  OTHER_ASSERT((int)event_sorted_history.size()==ranks*threads);
  const double dt = 1e6*dt_seconds;
  // Count how many epochs we need
  int64_t elapsed = 0;
  for (auto& thread : event_sorted_history)
    for (auto& events : thread)
      if (events.size())
        elapsed = max(elapsed,events.back().end.us);
  const int epochs = int(ceil(elapsed/dt)); // Last epoch is incomplete
  // Statics: responses, outputs, total
  Vector<uint64_t,3> messages;
  Vector<double,3> total_data, total_time, max_time;
  int64_t max_time_travel = 0;
  const double compression_ratio = .35;
  // Traverse each large message, accumulating total data sent
  Array<double,3> bandwidths(epochs,ranks,ranks);
  for (const int target_rank : range(ranks))
    for (const int kind : vec(response_recv_kind,output_recv_kind))
      for (const history_t& target : event_sorted_history[threads*target_rank][kind]) {
        const auto deps = event_dependencies(event_sorted_history,threads*target_rank,kind,target);
        OTHER_ASSERT(deps.size()==1);
        const int source_thread = deps[0].x;
        const int source_rank = source_thread/threads;
        OTHER_ASSERT(source_thread==source_rank*threads);
        const history_t& source = deps[0].z;
        const bool which = kind==output_recv_kind;
        messages[which]++;
        // Estimate message size
        const section_t section = parse_section(source.event);
        const Vector<uint8_t,4> block = parse_block(source.event);
        const double data_size = sizeof(Vector<super_t,2>)*block_shape(section.shape(),block).product()*(kind==response_recv_kind?compression_ratio:1);
        total_data[which] += data_size;
        // Distribute data amongst all overlapped epochs
        const int64_t time_travel = source.start.us - target.end.us;
        max_time_travel = max(max_time_travel,time_travel);
        Box<double> box(source.start.us/dt,target.end.us/dt);
        if (box.size()<=1e-7)
          box = Box<double>(box.center()).thickened(.5e-7);
        total_time[which] += box.size();
        max_time[which] = max(max_time[which],box.size());
        const double rate = data_size/box.size();
        for (const int epoch : range(max(0,int(box.min)),min(epochs,int(box.max)+1)))
          bandwidths(epoch,source_rank,target_rank) += rate*Box<double>::intersect(box,Box<double>(epoch,epoch+1)).size();
      }

  // Rescale
  bandwidths /= dt_seconds;

  // Print statistics
  cout << "dt = "<<dt_seconds<<" s"<<endl;
  cout << "elapsed = "<<1e-6*elapsed<<" s"<<endl;
  cout << "ranks = "<<ranks<<endl;
  messages[2] = messages.sum();
  total_data[2] = total_data.sum();
  total_time[2] = total_time.sum();
  max_time[2] = max_time.max();
  for (int i=0;i<3;i++) {
    cout << (i==0?"responses:":i==1?"outputs:":"total:") << endl;
    cout << "  messages = "<<messages[i]<<endl;
    cout << "  total data = "<<total_data[i]<<endl;
    cout << "  total time = "<<dt_seconds*total_time[i]<<endl;
    cout << "  average time = "<<dt_seconds*total_time[i]/messages[i]<<endl;
    cout << "  max time = "<<dt_seconds*max_time[i]<<endl;
    cout << "  average bandwidth = "<<total_data[i]/(1e-6*elapsed)<<endl;
    cout << "  average bandwidth / ranks = "<<total_data[i]/(1e-6*elapsed*ranks)<<endl;
  }
  cout << "max time travel = "<<1e-6*max_time_travel<<endl;
  cout << "bandwidth array stats:"<<endl;
  const double sum = bandwidths.sum(); 
  cout << "  sum = "<<sum<<endl;
  cout << "  average rank bandwidth = "<<sum/epochs/ranks<<endl;
  cout << "  average rank-to-rank bandwidth = "<<sum/epochs/sqr(ranks)<<endl;

  // All done
  return bandwidths;
}

}
}
using namespace pentago;
using namespace mpi;

void wrap_history() {
  OTHER_FUNCTION(str_event)
  OTHER_FUNCTION(search_thread)
  OTHER_FUNCTION(event_dependencies)
  OTHER_FUNCTION(check_dependencies)
  OTHER_FUNCTION(estimate_bandwidth)
}
