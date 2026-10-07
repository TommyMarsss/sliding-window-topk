package topk

import (
	"fmt"
	"math/rand"
	"testing"
)

// End-to-end: the tracker must satisfy all three invariants simultaneously
// on a random stream — incremental eviction equals full rescan, sketch error
// stays within the bound, and incremental Top-K equals a full re-sort of all
// current estimates.
func TestTrackerEndToEnd(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const (
		nEvents = 60000
		window  = int64(2000)
		k       = 8
	)
	tr := NewTracker(window, k, 32, 0.001, 0.01)
	history := make([]string, 0, nEvents)

	for i := 0; i < nEvents; i++ {
		var item string
		if rng.Float64() < 0.5 {
			item = fmt.Sprintf("hot-%d", rng.Intn(20))
		} else {
			item = fmt.Sprintf("cold-%d", rng.Intn(400))
		}
		ts := int64(i)
		tr.Add(item, ts)
		history = append(history, item)

		if (i+1)%733 == 0 {
			truth := naiveCounts(history, ts, window)
			if got := tr.ExactCounts(); !equalCounts(got, truth) {
				t.Fatalf("step %d: window counts != full rescan", i)
			}
			bound := tr.ErrorBound()
			estimates := make(map[string]int64, len(truth))
			for it, c := range truth {
				est := tr.Estimate(it)
				estimates[it] = est
				if est < c || est-c > bound {
					t.Fatalf("step %d: item %s est=%d true=%d bound=%d", i, it, est, c, bound)
				}
			}
			if got, want := tr.TopK(), ExactTopK(estimates, k); !equalRank(got, want) {
				t.Fatalf("step %d: tracker top-k %v != full re-sort %v", i, got, want)
			}
		}
	}
}

// Advance (time passing with no events) must keep the tracker consistent.
func TestTrackerAdvance(t *testing.T) {
	tr := NewTracker(100, 5, 10, 0.001, 0.01)
	for i := 0; i < 100; i++ {
		tr.Add(fmt.Sprintf("item-%d", i%7), int64(i))
	}
	tr.Advance(500)
	if tr.WindowTotal() != 0 || tr.Distinct() != 0 {
		t.Fatalf("window not empty: total=%d distinct=%d", tr.WindowTotal(), tr.Distinct())
	}
	if got := tr.TopK(); len(got) != 0 {
		t.Fatalf("top-k not empty after full eviction: %v", got)
	}
	if got := tr.Estimate("item-0"); got != 0 {
		t.Fatalf("estimate after full eviction = %d", got)
	}
}
