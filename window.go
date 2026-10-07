// Package topk implements sliding-time-window Top-K frequent-item tracking:
// an exact sliding-window counter with incremental eviction, a memory-bounded
// sliding-window Count-Min Sketch for approximate counts with an explicit
// error bound, and an incrementally maintained Top-K candidate structure.
package topk

import "sort"

// Delta describes a count change of one item caused by window eviction.
type Delta struct {
	Item string
	Ts   int64 // timestamp of the expired event
	Old  int64 // count before eviction
	New  int64 // count after eviction (0 means the item left the window)
}

// event is one in-window occurrence.
type event struct {
	item string
	ts   int64
}

// Window maintains exact per-item counts over a sliding time window
// (now-size, now]. Eviction is incremental: each expired event is
// decremented out of the count map in O(1) amortized, never rescanned.
type Window struct {
	size   int64
	queue  []event // FIFO of in-window events, ts non-decreasing
	head   int     // logical pop front; compacted lazily
	counts map[string]int64
	total  int64
}

// NewWindow returns a Window covering the last `size` time units.
func NewWindow(size int64) *Window {
	if size <= 0 {
		panic("topk: window size must be positive")
	}
	return &Window{size: size, counts: make(map[string]int64)}
}

// Add records one occurrence of item at time ts (ts must be non-decreasing
// across calls), evicts everything at or before ts-size, and returns the
// per-item count deltas caused by that eviction.
func (w *Window) Add(item string, ts int64) []Delta {
	if n := len(w.queue); n > 0 && ts < w.queue[n-1].ts {
		panic("topk: timestamps must be non-decreasing")
	}
	w.queue = append(w.queue, event{item, ts})
	w.counts[item]++
	w.total++
	return w.evict(ts)
}

// Advance slides the window boundary to ts without adding an event,
// returning the eviction deltas.
func (w *Window) Advance(ts int64) []Delta { return w.evict(ts) }

// evict removes all events with ts <= now-size, incrementally.
func (w *Window) evict(now int64) []Delta {
	cutoff := now - w.size
	var deltas []Delta
	for w.head < len(w.queue) && w.queue[w.head].ts <= cutoff {
		ev := w.queue[w.head]
		w.head++
		old := w.counts[ev.item]
		nc := old - 1
		if nc == 0 {
			delete(w.counts, ev.item)
		} else {
			w.counts[ev.item] = nc
		}
		w.total--
		deltas = append(deltas, Delta{Item: ev.item, Ts: ev.ts, Old: old, New: nc})
	}
	// Amortized compaction of the consumed prefix.
	if w.head > 1024 && w.head*2 > len(w.queue) {
		w.queue = append([]event(nil), w.queue[w.head:]...)
		w.head = 0
	}
	return deltas
}

// Count returns the exact in-window count of item.
func (w *Window) Count(item string) int64 { return w.counts[item] }

// Total returns the number of in-window events.
func (w *Window) Total() int64 { return w.total }

// Distinct returns the number of items with nonzero in-window count.
func (w *Window) Distinct() int { return len(w.counts) }

// Len returns the number of buffered in-window events.
func (w *Window) Len() int { return len(w.queue) - w.head }

// Size returns the window length in time units.
func (w *Window) Size() int64 { return w.size }

// Counts returns a snapshot copy of the exact in-window counts.
func (w *Window) Counts() map[string]int64 {
	m := make(map[string]int64, len(w.counts))
	for k, v := range w.counts {
		m[k] = v
	}
	return m
}

// ItemCount pairs an item with a count.
type ItemCount struct {
	Item  string `json:"item"`
	Count int64  `json:"count"`
}

// better is the canonical ranking order: higher count first, ties broken by
// lexicographically smaller item so results are deterministic.
func better(a, b ItemCount) bool {
	if a.Count != b.Count {
		return a.Count > b.Count
	}
	return a.Item < b.Item
}

// ExactTopK fully sorts all counts — O(D log D) over D distinct items.
// It is the ground-truth baseline used to verify incremental structures.
func ExactTopK(counts map[string]int64, k int) []ItemCount {
	ic := make([]ItemCount, 0, len(counts))
	for it, c := range counts {
		if c > 0 {
			ic = append(ic, ItemCount{Item: it, Count: c})
		}
	}
	sort.Slice(ic, func(i, j int) bool { return better(ic[i], ic[j]) })
	if len(ic) > k {
		ic = ic[:k]
	}
	return ic
}
