// Package topk implements sliding-window Top-K frequent-item statistics.
//
// The package provides three cooperating structures:
//
//   - ExactWindow: exact per-item counts over a sliding time window with
//     incremental eviction (O(1) amortized per event, no rescan).
//   - WindowCMS: approximate counts via a bucketed Count-Min Sketch with
//     an explicit, provable error bound.
//   - TopK: an incrementally maintained Top-K candidate set (two lazy
//     heaps), updated in O(log K + log R) per count change instead of
//     re-sorting all seen items.
package topk

// Event is a single occurrence of an item at a logical timestamp.
// Timestamps must be non-decreasing when fed to a window.
type Event struct {
	Ts   int64
	Item string
}

// ExactWindow maintains exact counts for events whose timestamps lie in
// the half-open interval (latestTs-window, latestTs].
//
// Eviction is incremental: expired events are popped from a ring buffer
// and their counts decremented in O(1) amortized time each. The window is
// never rescanned.
type ExactWindow struct {
	window int64
	q      []Event // ring buffer of in-window events, oldest at head
	head   int     // index of oldest element
	n      int     // number of live elements in the ring
	counts map[string]int64
	total  int64
	latest int64
}

// NewExactWindow creates a window covering the last `window` timestamp units.
func NewExactWindow(window int64) *ExactWindow {
	if window <= 0 {
		panic("topk: window must be positive")
	}
	return &ExactWindow{
		window: window,
		counts: make(map[string]int64),
	}
}

// Add inserts ev and evicts every event that has slid out of the window.
// Returns the evicted events so callers can maintain derived structures.
func (w *ExactWindow) Add(ev Event) []Event {
	if ev.Ts < w.latest {
		panic("topk: events must arrive with non-decreasing timestamps")
	}
	w.latest = ev.Ts
	w.push(ev)
	w.counts[ev.Item]++
	w.total++
	return w.evict()
}

// evict removes all events with Ts <= latest-window, decrementing counts.
func (w *ExactWindow) evict() []Event {
	cutoff := w.latest - w.window
	var out []Event
	for w.n > 0 && w.q[w.head].Ts <= cutoff {
		ev := w.q[w.head]
		w.head = (w.head + 1) % len(w.q)
		w.n--
		w.total--
		c := w.counts[ev.Item] - 1
		if c == 0 {
			delete(w.counts, ev.Item) // keep the map bounded by live distinct items
		} else {
			w.counts[ev.Item] = c
		}
		out = append(out, ev)
	}
	return out
}

func (w *ExactWindow) push(ev Event) {
	if w.n == len(w.q) { // grow: double capacity, linearize
		nq := make([]Event, max(2*len(w.q), 16))
		for i := 0; i < w.n; i++ {
			nq[i] = w.q[(w.head+i)%len(w.q)]
		}
		w.q, w.head = nq, 0
	}
	w.q[(w.head+w.n)%len(w.q)] = ev
	w.n++
}

// Count returns the exact in-window count of item.
func (w *ExactWindow) Count(item string) int64 { return w.counts[item] }

// Total returns the number of events currently in the window.
func (w *ExactWindow) Total() int64 { return w.total }

// Distinct returns the number of distinct items currently in the window.
func (w *ExactWindow) Distinct() int { return len(w.counts) }

// Len returns the number of buffered events (== Total).
func (w *ExactWindow) Len() int { return w.n }

// WindowStart returns the inclusive lower timestamp bound of the window.
func (w *ExactWindow) WindowStart() int64 { return w.latest - w.window + 1 }

// Latest returns the most recent timestamp seen.
func (w *ExactWindow) Latest() int64 { return w.latest }

// CountsSince returns exact counts of events with Ts >= since, computed by
// scanning the live ring buffer. Used for verification and reporting, never
// on the hot path.
func (w *ExactWindow) CountsSince(since int64) map[string]int64 {
	m := make(map[string]int64, len(w.counts))
	for i := 0; i < w.n; i++ {
		ev := w.q[(w.head+i)%len(w.q)]
		if ev.Ts >= since {
			m[ev.Item]++
		}
	}
	return m
}

// Snapshot returns a copy of the current exact counts.
func (w *ExactWindow) Snapshot() map[string]int64 {
	m := make(map[string]int64, len(w.counts))
	for k, v := range w.counts {
		m[k] = v
	}
	return m
}
