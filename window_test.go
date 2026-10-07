package topk

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

// naiveCounts recomputes window counts from scratch over the in-window
// suffix of the full event history — the ground truth for eviction tests.
func naiveCounts(history []string, now, window int64) map[string]int64 {
	lo := sort.Search(len(history), func(i int) bool { return int64(i) > now-window })
	m := make(map[string]int64)
	for _, it := range history[lo:] {
		m[it]++
	}
	return m
}

func equalCounts(a, b map[string]int64) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func equalRank(a, b []ItemCount) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Requirement 1: incremental eviction must exactly match a full rescan of
// the remaining in-window events, at every checkpoint, including the Top-K
// derived from those counts.
func TestWindowIncrementalEvictionMatchesFullRecompute(t *testing.T) {
	for _, seed := range []int64{1, 7, 42} {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))
			const (
				nEvents    = 30000
				window     = 997
				nItems     = 150
				checkEvery = 37
			)
			w := NewWindow(window)
			history := make([]string, 0, nEvents)
			for i := 0; i < nEvents; i++ {
				item := fmt.Sprintf("item-%03d", rng.Intn(nItems))
				ts := int64(i)
				w.Add(item, ts)
				history = append(history, item)
				if (i+1)%checkEvery == 0 {
					want := naiveCounts(history, ts, window)
					if got := w.Counts(); !equalCounts(got, want) {
						t.Fatalf("step %d: incremental counts != full recompute", i)
					}
					if got, want2 := ExactTopK(w.Counts(), 10), ExactTopK(want, 10); !equalRank(got, want2) {
						t.Fatalf("step %d: top-k after eviction != recomputed top-k", i)
					}
					if w.Total() != sumCounts(want) {
						t.Fatalf("step %d: total mismatch", i)
					}
				}
			}
		})
	}
}

func sumCounts(m map[string]int64) int64 {
	var s int64
	for _, v := range m {
		s += v
	}
	return s
}

// Eviction deltas must name exactly the items and counts that expired.
func TestWindowEvictionDeltas(t *testing.T) {
	w := NewWindow(10)
	w.Add("a", 0)
	w.Add("a", 1)
	w.Add("b", 2)
	d := w.Add("c", 11) // window (1,11]: evicts a@0 and a@1? a@1 has ts=1 > 1, stays
	// cutoff = 11-10 = 1; events with ts <= 1 evicted: a@0, a@1.
	if len(d) != 2 || d[0].Item != "a" || d[1].Item != "a" {
		t.Fatalf("unexpected deltas: %+v", d)
	}
	if d[1].New != 0 {
		t.Fatalf("expected a to leave window, got %+v", d[1])
	}
	if w.Count("a") != 0 || w.Count("b") != 1 || w.Count("c") != 1 {
		t.Fatalf("bad counts: a=%d b=%d c=%d", w.Count("a"), w.Count("b"), w.Count("c"))
	}
	if w.Total() != 2 {
		t.Fatalf("total=%d want 2", w.Total())
	}
}

// Advance without new events must evict everything once the horizon passes.
func TestWindowAdvanceEvictsEverything(t *testing.T) {
	w := NewWindow(100)
	for i := 0; i < 50; i++ {
		w.Add(fmt.Sprintf("item-%d", i%5), int64(i))
	}
	deltas := w.Advance(1000)
	if w.Total() != 0 || w.Distinct() != 0 || w.Len() != 0 {
		t.Fatalf("window not empty after advance: total=%d distinct=%d len=%d",
			w.Total(), w.Distinct(), w.Len())
	}
	if len(deltas) != 50 {
		t.Fatalf("expected 50 eviction deltas, got %d", len(deltas))
	}
}

// The in-window event buffer must stay bounded by the window size no matter
// how many events stream through.
func TestWindowBufferBounded(t *testing.T) {
	w := NewWindow(500)
	for i := 0; i < 200000; i++ {
		w.Add(fmt.Sprintf("item-%d", i%97), int64(i))
		if w.Len() > 500 {
			t.Fatalf("step %d: buffered events %d exceed window", i, w.Len())
		}
		if w.Distinct() > 97 {
			t.Fatalf("step %d: distinct %d exceeds universe", i, w.Distinct())
		}
	}
}
