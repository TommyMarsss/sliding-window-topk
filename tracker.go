package topk

// Tracker composes the three structures:
//   - Window:     exact counts + incremental eviction (ground truth)
//   - SlidingCMS: memory-bounded approximate counts with error bound
//   - TopK:       incremental Top-K over the sketch estimates
//
// Per event the cost is O(1) amortized for the window, O(depth) for the
// sketch, and O(log D) per changed item for the Top-K — no full rescans and
// no global re-sorts anywhere on the hot path.
//
// Why the ghost queue: a sketch estimate est(x) changes only when
//  1. an x event is added (est += 1), or
//  2. a sketch bucket dies (stops being live) — then est drops for every
//     item that had events in that bucket.
//
// Exact-window eviction alone does NOT change est(x): the evicted event's
// bucket is still alive and still counted. So the Top-K structure must be
// refreshed on adds and on bucket deaths — not on exact evictions. Bucket
// deaths are detected via a bounded FIFO of "ghost" events: events already
// evicted from the exact window but potentially still counted by the sketch.
// A ghost's bucket is dead once (slot+1)*bucketDur <= now-window, so ghosts
// live for at most ~2 bucket durations and the queue stays O(bucketDur).
type Tracker struct {
	win *Window
	cms *SlidingCMS
	tk  *TopK

	ghosts []ghostEvent // FIFO, ts-ordered
	ghead  int
}

type ghostEvent struct {
	item string
	ts   int64
}

// NewTracker builds a Tracker with the given window size (time units),
// top-k size, sketch sub-window count and accuracy targets.
func NewTracker(windowSize int64, k, subwindows int, epsilon, delta float64) *Tracker {
	return &Tracker{
		win: NewWindow(windowSize),
		cms: NewSlidingCMS(windowSize, subwindows, epsilon, delta),
		tk:  NewTopK(k),
	}
}

// Add ingests one event. Only items whose sketch estimate actually changed
// (the added item, plus items in any bucket that just died) are pushed into
// the Top-K structure.
func (t *Tracker) Add(item string, ts int64) {
	deltas := t.win.Add(item, ts)
	t.cms.Add(item, ts)
	t.tk.Update(item, int64(t.cms.Estimate(item)))
	t.enqueueGhosts(deltas)
	t.flushGhosts(ts)
}

// Advance slides the window (and sketch horizon) without a new event.
func (t *Tracker) Advance(ts int64) {
	deltas := t.win.Advance(ts)
	t.cms.Advance(ts)
	t.enqueueGhosts(deltas)
	t.flushGhosts(ts)
}

func (t *Tracker) enqueueGhosts(deltas []Delta) {
	for _, d := range deltas {
		t.ghosts = append(t.ghosts, ghostEvent{item: d.Item, ts: d.Ts})
	}
}

// flushGhosts drops ghost events whose sketch bucket has died by `now` and
// refreshes the Top-K entries of every affected item with its current
// estimate. Ghosts are ts-ordered and bucket death is monotone in ts, so a
// FIFO scan suffices.
func (t *Tracker) flushGhosts(now int64) {
	if t.ghead == len(t.ghosts) {
		return
	}
	d := t.cms.BucketDur()
	window := t.win.Size()
	var affected map[string]struct{}
	for t.ghead < len(t.ghosts) {
		g := t.ghosts[t.ghead]
		slot := g.ts / d
		if (slot+1)*d > now-window {
			break // bucket still live: event still counted by the sketch
		}
		t.ghead++
		if affected == nil {
			affected = make(map[string]struct{})
		}
		affected[g.item] = struct{}{}
	}
	if t.ghead > 1024 && t.ghead*2 > len(t.ghosts) {
		t.ghosts = append([]ghostEvent(nil), t.ghosts[t.ghead:]...)
		t.ghead = 0
	}
	for item := range affected {
		t.tk.Update(item, int64(t.cms.Estimate(item)))
	}
}

// GhostLen returns the number of buffered ghost events (bounded by ~2
// bucket durations worth of events).
func (t *Tracker) GhostLen() int { return len(t.ghosts) - t.ghead }

// TopK returns the current approximate Top-K (counts are sketch estimates).
func (t *Tracker) TopK() []ItemCount { return t.tk.TopK() }

// Estimate returns the sketch estimate for item.
func (t *Tracker) Estimate(item string) int64 { return int64(t.cms.Estimate(item)) }

// ExactCount returns the exact in-window count for item.
func (t *Tracker) ExactCount(item string) int64 { return t.win.Count(item) }

// ExactCounts snapshots the exact in-window counts (ground truth).
func (t *Tracker) ExactCounts() map[string]int64 { return t.win.Counts() }

// ErrorBound returns the current sketch overestimation bound.
func (t *Tracker) ErrorBound() int64 { return int64(t.cms.ErrorBound()) }

// LiveTotal returns N_live, the total count in live sketch buckets.
func (t *Tracker) LiveTotal() int64 { return int64(t.cms.LiveTotal()) }

// WindowTotal returns the exact number of in-window events.
func (t *Tracker) WindowTotal() int64 { return t.win.Total() }

// Distinct returns the number of distinct in-window items.
func (t *Tracker) Distinct() int { return t.win.Distinct() }

// Window exposes the exact window (for verification and reporting).
func (t *Tracker) Window() *Window { return t.win }

// Sketch exposes the sliding sketch (for verification and reporting).
func (t *Tracker) Sketch() *SlidingCMS { return t.cms }
