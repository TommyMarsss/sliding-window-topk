package topk

import (
	"encoding/binary"
	"hash/fnv"
	"math"
)

// WindowCMS is a sliding-window Count-Min Sketch ("bucketed CMS").
//
// The time axis is divided into nBuckets consecutive buckets of equal span
// covering the window. Each bucket owns an independent Count-Min Sketch with
// depth d = ceil(ln(1/delta)) and width w = 2^ceil(log2(e/eps)). When time
// advances past a bucket, its sketch is zeroed and reused — eviction is an
// O(d*w) reset of one bucket, never a rescan of the event stream.
//
// Error bound (see README for the derivation):
//
//	For any item x with estimate est and true count a over the
//	bucket-aligned window, with N = total events in the window:
//
//	    a <= est <= a + eps * N     with probability >= 1 - nBuckets*delta
//
// The bound is deterministic in magnitude (eps*N); only its confidence is
// probabilistic. Error cannot diverge beyond eps*N unless one of the
// per-bucket sketches violates its own (eps, delta) guarantee.
type WindowCMS struct {
	span     int64 // timestamps per bucket
	nBuckets int
	depth    int
	width    uint32 // power of two
	mask     uint32
	seeds    []uint64
	slots    []cmsSlot
	cur      int64 // current (newest) bucket index; -1 before first Add
}

type cmsSlot struct {
	bucket int64 // bucket index this slot holds; -1 = empty
	table  []uint32
	total  int64
}

// NewWindowCMS builds approximate counting over a sliding window.
//
//	window:   window length in timestamp units
//	nBuckets: number of time buckets (window granularity)
//	eps, delta: per-sketch CMS parameters
func NewWindowCMS(window int64, nBuckets int, eps, delta float64) *WindowCMS {
	if window <= 0 || nBuckets <= 0 {
		panic("topk: window and nBuckets must be positive")
	}
	if eps <= 0 || delta <= 0 || delta >= 1 {
		panic("topk: bad eps/delta")
	}
	depth := int(math.Ceil(math.Log(1 / delta)))
	width := uint32(1)
	for float64(width) < math.E/eps {
		width <<= 1
	}
	span := window / int64(nBuckets)
	if span < 1 {
		span = 1
	}
	seeds := make([]uint64, depth)
	s := uint64(0x9e3779b97f4a7c15)
	for i := range seeds {
		s = splitmix64(s)
		seeds[i] = s
	}
	slots := make([]cmsSlot, nBuckets)
	for i := range slots {
		slots[i].bucket = -1
		slots[i].table = make([]uint32, depth*int(width))
	}
	return &WindowCMS{
		span:     span,
		nBuckets: nBuckets,
		depth:    depth,
		width:    width,
		mask:     width - 1,
		seeds:    seeds,
		slots:    slots,
		cur:      -1,
	}
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// Add records one occurrence of item at timestamp ts.
func (c *WindowCMS) Add(ts int64, item string) {
	b := ts / c.span
	if b < c.cur {
		panic("topk: WindowCMS requires non-decreasing timestamps")
	}
	c.cur = b
	s := &c.slots[b%int64(c.nBuckets)]
	if s.bucket != b { // stale or empty slot: reset (this is the eviction)
		clear(s.table)
		s.bucket = b
		s.total = 0
	}
	h := hash64(item)
	for i := 0; i < c.depth; i++ {
		s.table[i*int(c.width)+int(splitmix64(h^c.seeds[i])&uint64(c.mask))]++
	}
	s.total++
}

// Estimate returns the approximate in-window count of item.
func (c *WindowCMS) Estimate(item string) int64 {
	if c.cur < 0 {
		return 0
	}
	oldest := c.cur - int64(c.nBuckets) + 1
	h := hash64(item)
	var sum int64
	for i := range c.slots {
		s := &c.slots[i]
		if s.bucket < oldest { // expired slot (not yet overwritten)
			continue
		}
		var m uint32 = math.MaxUint32
		for d := 0; d < c.depth; d++ {
			v := s.table[d*int(c.width)+int(splitmix64(h^c.seeds[d])&uint64(c.mask))]
			if v < m {
				m = v
			}
		}
		sum += int64(m)
	}
	return sum
}

// Total returns the exact number of events in live buckets.
func (c *WindowCMS) Total() int64 {
	if c.cur < 0 {
		return 0
	}
	oldest := c.cur - int64(c.nBuckets) + 1
	var t int64
	for i := range c.slots {
		if c.slots[i].bucket >= oldest {
			t += c.slots[i].total
		}
	}
	return t
}

// ErrorBound returns the explicit error upper bound eps*N for the current
// window contents: with probability >= 1-nBuckets*delta every item's
// estimate exceeds its true count by at most this value.
func (c *WindowCMS) ErrorBound(eps float64) float64 {
	return eps * float64(c.Total())
}

// AlignedStart returns the inclusive lower timestamp bound of the
// bucket-aligned window: estimates cover events with Ts >= AlignedStart.
// This is the range over which the eps*N error bound is stated.
func (c *WindowCMS) AlignedStart() int64 {
	if c.cur < 0 {
		return 0
	}
	return (c.cur - int64(c.nBuckets) + 1) * c.span
}

// MemoryBytes returns the exact memory footprint of the sketch tables.
// It is constant regardless of how many events have been processed.
func (c *WindowCMS) MemoryBytes() int {
	return len(c.slots) * c.depth * int(c.width) * 4
}

func hash64(s string) uint64 {
	h := fnv.New64a()
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(len(s)))
	h.Write(b[:])
	h.Write([]byte(s))
	return h.Sum64()
}
