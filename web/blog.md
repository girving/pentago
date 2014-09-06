# Serving Pentago
## Exploring HPC data using Node.js on Rackspace

[Pentago](https://en.wikipedia.org/wiki/Pentago) is a board game designed by
Tomas Flodén and developed and sold by [Mindtwister](http://mindtwisterusa.com).
Like chess and go, pentago is a two player game with no hidden cards or chance.
Unlike chess and go, pentago is small enough for a computer to play perfectly:
with symmetries removed, there are a mere 3,009,081,623,421,558 (3e15) possible
positions.

Iterating over all these positions took a bit of work, and hundreds of thousands
of processor-hours on a Cray at [NERSC](http://nersc.gov).
The details of this computation are described on [arxiv](http://arxiv.org/abs/1404.0743)
and [summarized here](http://perfect-pentago.net/details.html).  The result of this
computation is an 3.7 terabyte database of all pentago positions with up to 18 stones.

This database would be quite boring without a mechanism for exploring the data, playing through
pentago games interactively and visualizing which moves win or lose.  Unfortunately, I did all
of this work as a side project, and hosting 3.7 TB of data on the web is an expensive prospect.

### Rackspace to the rescue!

JESSE: Is there a page I should link to about the open source hosting program?

Here I am indebted to Rackspace and Jesse Noller in particular for their wonderful open source
hosting program.  Since the pentago project is all [open source](https://github.com/girving/pentago)
and [open data](https://github.com/girving/pentago#data), Rackspace donated enough free hosting
to cover both storage and server costs.  Finding a home for both data and visualization was essential
to the goals of the project: open data is meaningless without easy access.  The result can be
explored at

* http://perfect-pentago.net

The rest of this post describes my experience building this website, using [Node.js](http://nodejs.org),
[Rackspace Cloud Files](http://www.rackspace.com/cloud/files), and
[Cloud Servers](http://www.rackspace.com/cloud/servers).

### Cloud files: storing the data

JESSE: Let me know if the interface difficulties that I mention have been fixed since.

Step one was to move 3.7 TB of data from NERSC to Rackspace.  This was slightly more difficult than I
naively expected at first: the data consisted of a few dozen flat files, the largest of which was
1.8 TB in size.  Rackspace's Cloud Files requires files larger than 5 GB to be chunked, divided into
smaller pieces and them assembled into a unified whole via a
[manifest](http://docs.rackspace.com/files/api/v1/cf-devguide/content/Static_Large_Object-d1e2226.html).
Although Rackspace's [pyrax](https://github.com/rackspace/pyrax) python library can do this chunking
automatically, at the time I used it pyrax had no features for restarting a partially completed upload
or creating a manifest from a list of files.

The lack of restart was important, since my first try errored out 31 hours into the upload.
Happily, the pyrax folk were [very helpful](https://github.com/rackspace/pyrax/issues/266), indeed
probably as helpful as they could be given that I didn't want to burn another 31 hours to reproduce
the bug.  I ended up writing a little script to do my own chunking, use pyrax to upload chunks not
already uploaded (so that restart worked), and then follow Rackspace's REST documentation to upload
the manifest via HTTP PUT.

### Client/server architecture: stateless + caching

In the uploaded Cloud Files data, pentago positions with similar patterns of stones are organized into
blocks.  Each block is compressed using [LZMA](http://tukaani.org/xz), and all blocks with the same
number of stones are packed into a single gigantic file (ignoring the chunking described above).
Although client javascript could download these blocks directly, there are several problems with a
pure client + Cloud Files setup:

1. Uncompressed, each block is 256 KB.  It would a shame to download megabytes of data in order to
   extract only a few bits (who wins or loses for a given move).  If we have to transfer a bunch of
   data, it's better to stay inside Rackspace's network.

2. LZMA decompression is a performance intensive process.  Javascript is slow.

3. The Cloud Files database only extends up to boards with 18 stones.  If a pentago position has more
   than 18 stones, its result must be recomputed from scratch.  This takes 15 seconds even with heavily
   SSE optimized C++; doing it on the client is unreasonable.

Thus, we push as much complexity and resource use as possible to the server, leaving the client to
receive results only for the positions it needs to visualize.  Since we are serving only static
data, there are no connections or similar details to manage; the server appears completely stateless
from the outside.

As a client user explores the game, they are likely to request the same position multiple times or
access different positions in the same block.  Unrelated users are also likely to request similar
positions, especially near the beginning of the game.  Therefore, the server implements two different
caches: a "block cache" storing uncompressed blocks downloaded from Cloud Files and a "compute cache"
storing positions we've recomputed from scratch.

### Layered asynchronous caching in Node.js

A fully asynchronous server was a clear requirement from the beginning.  Even when only one client is
using the system, a request to visualize one pentago position creates up to four different block
requests.  To reduce overall latency if the blocks miss cache, all of these requests should be launched
at the same time, decompressed as soon as they arrive, and combined to form the server response.  A
block request is a high latency operation (up to 256 KB), and this latency should be hidden from other
users.  A request to a position with over 18 stones triggers expensive local computation which cannot
block other requests.

The pentago server was my first project in Node, and it turned out to be quite a pleasant experience.
In particular, Node's largely functional style makes it easy to layer different types of control flow
on top of each other, letting each piece of code do exactly one thing.  For example, if a user looks
at a position with more than 18 stones, it must be recomputed from scratch if it is not in the cache.
This behavior can be split into the following constraints:

1. Keep a cache of computed values.  If we hit cache, we're done.
2. Computing a value is expensive, so it should be done in a separate worker process.
3. If there's a request for a value already being computed, the request should be merged.

The Node code handles each of these constraints using a separate module.  For (1) and (2), I used
the npm modules [lru-cache](https://www.npmjs.org/package/lru-cache) and
[mule](https://www.npmjs.org/package/mule), respectively.  For (3), I wrote a little
[pending.js](https://github.com/girving/pentago/blob/master/web/pending.js) module which maintains
a map from input to continuations (callbacks) to call with the result.  New computations are launched,
and computations already active are added to the pending list without triggering more compute.

The beauty of this layered structure is that each piece doesn't need to know anything about the other
layers.  The pending module has no notion of caching: it forgets about each computation as soon as it
completes.  `lru-cache` is purely synchronous, with no notion of slow computations.  `mule`, which ships
computations off to worker threads, has no notion of caching or merging.  As a consequence, each piece
can be tested separately, then merged together with little fear that the different layers will interact
poorly.

Language-nerd aside: Although it's wonderful that Node makes this sort of layering easy, I can't help
but be a bit sad that it's Node.js instead of Node.haskell.  Haskell code written using exactly the same
programming paradigms would be a fraction of size and far more readable.  While Node forces the programmer
into manual [continuation passing style](http://en.wikipedia.org/wiki/Continuation-passing_style), Haskell's
syntactic sugar for monadic control flow lets you write normal looking straight line code
which expands into continuation passing or other fanciness automatically.  Strong typing is also a big help
as higher order programming (such as the layered caching discussed above) becomes more complex.  I'm only a
bit sad, though: mostly I am happy that more people are being exposed to interesting styles of functional
programming, Javascript or not.

### The best code either works or it doesn't

The happiest surprise with my little Node server was how easy it was to debug.  There were plenty of bugs
during the development process, as expected given my inexperience with Node, but nearly all of them manifested
as immediate failures or unconditional deadlocks.  I've written plenty of concurrent code which _almost_
works, with bugs triggering only occasionally as a consequence of race conditions or order dependencies.
Since Node is single threaded, a large class of race conditions vanishes.  More subtly, any nonblocking
operation in Node never completes until the next event loop clock tick, even if the answer would be available
immediately ([more background here](http://howtonode.org/understanding-process-next-tick)).  As a consequence,
bugs which manifest only due to concurrently tend to show up immediately, even in rapid fire unit tests.
The Node programming model tends to make broken code fail immediately; once it works, it has a good chance
of actually being correct.

My evidence for this is that once I got through my initial testing phase, the pentago server has been running
without a hitch for 8 months
[despite being slashdotted](http://tech.slashdot.org/story/14/01/23/1733250/pentago-is-a-first-player-win).
The server is admittedly simple, but it does combine fully asynchronous programming, several layers of caching,
and CPU and memory intensive computation (the 8 GB RAM server uses 7 worker threads each with a 1 GB buffer).

### Rackspace experience

JESSE: Is there an Akamai bug page to link to?

My experience with Rackspace has been great throughout (after the extremely minor difficulties uploading
terabytes of data).  All of my bug reports and support requests were responded to promptly and in detail,
including a rather subtle bug involving HTTP range requests into multiterabyte chunked files.  As with the
rest of the debugging experience, this bug manifested immediately and consistently: said range request would
simply hang forever.  I gave the Rackspace folks a reduced test case for the problem, and it turned out be
a bug in Akamai rather than Rackspace per se.  The bug has since been fixed, and my workaround code has been
happily stripped.

The main feature I missed in Rackspace compared to Amazon EC2 is the ability to suspend server instances
(and stop paying for them).  Suspend doesn't matter so much for a project with donated hosting, but for
other personal projects where I'd be paying out of pocket starting and stopped EC2 servers is a friendlier
process than explicitly saving and restoring instance images.

### On the client: d3 and static html

With all the complexity shielded away on the server, the client is easy.  Rackspace Cloud Files
can serve static websites
[directly from a container](http://www.rackspace.com/blog/rackspace-cloud-files-how-to-create-a-static-website).
The pentago interface was done with [d3](http://d3js.org), including simple animations when quadrants are rotated.

### Conclusion

Building the pentago website turned out to be quite a pleasant experience, and a good opportunity to play with
Rackspace, Node.js, and asynchronous servers.  Since the supercomputer computation that produced the original
3.7 TB database was also fully asynchronous, but written in C++ and MPI, the contrast with Javascript and the
Node environment was an interesting part of the project.  Indeed, now that C++11 provides language sugar for
lambda expressions and functional code, the Node model can be mapped reasonably easily over to C++ land.  Had
I written the asynchronous server first, the asynchronous MPI might have been a little bit cleaner.

Rackspace mostly faded into the background and just worked after the few initial hiccups, and the website has
been up for 8 months without a hitch.  My interaction with them for the minor problems I did have was good,
with everything resolved satisfactorily (even for the complicated Akamai bug).  And it was all free hosting
donated to an open source project!  Thanks so much!
