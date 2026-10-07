package topk

import "math"

// hashWithSeed is FNV-1a-64 seeded per sketch row, with a splitmix64
// finalizer for avalanche. Deterministic across runs (test-friendly).
func hashWithSeed(seed uint64, s string) uint64 {
	h := seed ^ 14695981039346656037
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	h ^= h >> 33
	h *= 0xff51afd7ed558ccd
	h ^= h >> 33
	h *= 0xc4ceb9fe1a85ec53
	h ^= h >> 33
	return h
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// cmsCore is a standard Count-Min Sketch. For a point query on item x with
// true count f(x) and total count N: est(x) >= f(x) always, and
// est(x) - f(x) <= eps * N with probability >= 1 - delta, where
// width = ceil(e/eps) and depth = ceil(ln(1/delta)).
type cmsCore struct {
	depth int
	width int
	seeds []uint64
	table [][]uint64
	total uint64
}

func newCMSCore(depth, width int) cmsCore {
	c := cmsCore{
		depth: depth,
		width: width,
		seeds: make([]uint64, depth),
		table: make([][]uint64, depth),
	}
	for i := range c.table {
		c.table[i] = make([]uint64, width)
		c.seeds[i] = splitmix64(uint64(i) * 0x2545f4914f6cdd1d)
	}
	return c
}

func (c *cmsCore) add(item string) {
	for i := 0; i < c.depth; i++ {
		c.table[i][hashWithSeed(c.seeds[i], item)%uint64(c.width)]++
	}
	c.total++
}

func (c *cmsCore) estimate(item string) uint64 {
	m := uint64(math.MaxUint64)
	for i := 0; i < c.depth; i++ {
		if v := c.table[i][hashWithSeed(c.seeds[i], item)%uint64(c.width)]; v < m {
			m = v
		}
	}
	return m
}

func (c *cmsCore) reset() {
	for i := range c.table {
		clear(c.table[i])
	}
	c.total = 0
}

// SlidingCMS is a sliding-window Count-Min Sketch built by time-bucketing:
// the window is split into B sub-windows of duration bucketDur = window/B,
// each sub-window keeps its own cmsCore, and buckets are reused in a ring as
// time advances. A point query sums the estimates of all live buckets (those
// whose time range intersects (now-window, now]).
//
// Error bound (union bound over at most L live buckets, each configured with
// per-bucket failure probability delta/L):
//
//	est(x) >= f(x)                                  always (never underestimates)
//	est(x) - f(x) <= eps * N_live + N_boundary      with probability >= 1 - delta
//
// where N_live is the total count stored in live buckets and N_boundary is
// the total of the single oldest live bucket. The boundary term exists
// because the oldest live bucket only partially overlaps the window: its
// expired events are still counted by the sketch. Only one bucket can
// straddle the window edge, and N_boundary ≈ N_live/B under bounded rate, so
// the effective relative error is about eps + 1/B. Memory stays
// O(B * depth * width) regardless of stream length.
type SlidingCMS struct {
	window    int64
	bucketDur int64
	epsilon   float64
	buckets   []cmsCore // ring, len = number of sub-windows + 2
	slots     []int64   // slot id currently held by each ring position
	now       int64
	hasNow    bool
}

// NewSlidingCMS builds a sliding-window sketch. subwindows is B (>= 1);
// epsilon and delta are the overall accuracy targets (delta is split evenly
// across live buckets via the union bound).
func NewSlidingCMS(window int64, subwindows int, epsilon, delta float64) *SlidingCMS {
	if window <= 0 {
		panic("topk: window must be positive")
	}
	if subwindows < 1 {
		subwindows = 1
	}
	if epsilon <= 0 || epsilon >= 1 {
		panic("topk: epsilon must be in (0,1)")
	}
	if delta <= 0 || delta >= 1 {
		panic("topk: delta must be in (0,1)")
	}
	bucketDur := window / int64(subwindows)
	if bucketDur < 1 {
		bucketDur = 1
	}
	ring := subwindows + 2 // max simultaneously live slots, plus margin
	width := int(math.Ceil(math.E / epsilon))
	depth := int(math.Ceil(math.Log(float64(ring) / delta)))
	s := &SlidingCMS{
		window:    window,
		bucketDur: bucketDur,
		epsilon:   epsilon,
		buckets:   make([]cmsCore, ring),
		slots:     make([]int64, ring),
	}
	for i := range s.buckets {
		s.buckets[i] = newCMSCore(depth, width)
		s.slots[i] = -1
	}
	return s
}

// Add records one occurrence of item at time ts (non-decreasing).
func (s *SlidingCMS) Add(item string, ts int64) {
	s.advanceNow(ts)
	slot := ts / s.bucketDur
	idx := int(slot % int64(len(s.buckets)))
	if s.slots[idx] != slot {
		s.buckets[idx].reset()
		s.slots[idx] = slot
	}
	s.buckets[idx].add(item)
}

// Advance moves the query horizon to ts without adding an event.
func (s *SlidingCMS) Advance(ts int64) { s.advanceNow(ts) }

func (s *SlidingCMS) advanceNow(ts int64) {
	if !s.hasNow || ts > s.now {
		s.now = ts
		s.hasNow = true
	}
}

// live reports whether the bucket holding slot id `slot` intersects the
// current window (now-window, now]. A bucket is live iff its time range
// [slot*d, (slot+1)*d) overlaps the window.
func (s *SlidingCMS) live(slot int64) bool {
	if slot < 0 || !s.hasNow {
		return false
	}
	return (slot+1)*s.bucketDur > s.now-s.window && slot*s.bucketDur <= s.now
}

// Estimate returns the approximate in-window count of item. It never
// underestimates the true in-window count: every non-expired event lives in a
// bucket whose slot intersects the window, hence in exactly one live bucket.
func (s *SlidingCMS) Estimate(item string) uint64 {
	var sum uint64
	for i := range s.buckets {
		if s.live(s.slots[i]) {
			sum += s.buckets[i].estimate(item)
		}
	}
	return sum
}

// LiveTotal returns N_live, the total count stored in live buckets.
func (s *SlidingCMS) LiveTotal() uint64 {
	var sum uint64
	for i := range s.buckets {
		if s.live(s.slots[i]) {
			sum += s.buckets[i].total
		}
	}
	return sum
}

// BoundarySlack returns the total count of the oldest live bucket — the only
// bucket that can contain expired-but-still-counted events. Every other live
// bucket lies entirely inside the window: live slots are consecutive, and any
// slot newer than the oldest starts after now-window.
func (s *SlidingCMS) BoundarySlack() uint64 {
	if !s.hasNow {
		return 0
	}
	oldest := int64(-1)
	var total uint64
	for i := range s.buckets {
		if s.live(s.slots[i]) && (oldest == -1 || s.slots[i] < oldest) {
			oldest = s.slots[i]
			total = s.buckets[i].total
		}
	}
	return total
}

// ErrorBound returns the current theoretical overestimation bound
// ceil(epsilon * N_live) + N_boundary, valid with probability >= 1 - delta.
func (s *SlidingCMS) ErrorBound() uint64 {
	return uint64(math.Ceil(s.epsilon*float64(s.LiveTotal()))) + s.BoundarySlack()
}

// MemoryCounters returns the number of uint64 counters held — a constant
// independent of stream length, evidencing bounded memory.
func (s *SlidingCMS) MemoryCounters() int {
	total := 0
	for i := range s.buckets {
		total += s.buckets[i].depth * s.buckets[i].width
	}
	return total
}

// Now returns the current query horizon.
func (s *SlidingCMS) Now() int64 { return s.now }

// BucketDur returns the sub-window duration in time units.
func (s *SlidingCMS) BucketDur() int64 { return s.bucketDur }
