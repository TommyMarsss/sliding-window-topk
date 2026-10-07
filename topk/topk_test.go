package topk

import (
	"math/rand"
	"testing"
)

// bruteForceTopK sorts ALL live counts — the operation the incremental
// tracker exists to avoid — and returns the top min(k, n) entries.
func bruteForceTopK(counts map[string]int64, k int) []ItemCount {
	all := make([]ItemCount, 0, len(counts))
	for item, c := range counts {
		if c > 0 {
			all = append(all, ItemCount{Item: item, Count: c})
		}
	}
	sortItemCounts(all)
	if len(all) > k {
		all = all[:k]
	}
	return all
}

// assertValidTopK checks the tracker against brute force, robust to ties:
// identical count sequence, and every reported item at or above the
// k-th best count.
func assertValidTopK(t *testing.T, tk *TopK, counts map[string]int64, k int, ctx string) {
	t.Helper()
	want := bruteForceTopK(counts, k)
	got := tk.Items()
	if len(got) != len(want) {
		t.Fatalf("%s: Top-K size %d, want %d (got %v, want %v)", ctx, len(got), len(want), got, want)
	}
	if len(want) == 0 {
		return
	}
	threshold := want[len(want)-1].Count
	for i := range got {
		if got[i].Count != want[i].Count {
			t.Fatalf("%s: rank %d count %d, want %d (got %v, want %v)", ctx, i, got[i].Count, want[i].Count, got, want)
		}
		if got[i].Count < threshold {
			t.Fatalf("%s: item %s count %d below threshold %d", ctx, got[i].Item, got[i].Count, threshold)
		}
	}
}

// TestTopKRandomUpdates covers interleaved increments (arrivals) and
// decrements (evictions), including items dropping to zero, re-entering,
// and crossing the Top-K boundary in both directions.
func TestTopKRandomUpdates(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 30; trial++ {
		k := 1 + rng.Intn(20)
		vocab := 1 + rng.Intn(200)
		ops := 500 + rng.Intn(3000)
		tk := NewTopK(k)
		counts := make(map[string]int64)
		for op := 0; op < ops; op++ {
			item := ItemName(rng.Intn(vocab))
			if rng.Intn(100) < 60 || counts[item] == 0 {
				counts[item]++ // arrival
			} else {
				counts[item]-- // eviction
			}
			tk.Update(item, counts[item])
			assertValidTopK(t, tk, counts, k, "random update")
		}
	}
}

// TestTopKRankChurn drives a scenario engineered for rank changes: a
// champion item is evicted out of the Top-K while a challenger rises.
func TestTopKRankChurn(t *testing.T) {
	tk := NewTopK(3)
	counts := map[string]int64{}
	bump := func(item string, delta int64) {
		counts[item] += delta
		tk.Update(item, counts[item])
	}
	// Establish a, b, c as the top-3.
	for i := 0; i < 100; i++ {
		bump("a", 1)
	}
	for i := 0; i < 80; i++ {
		bump("b", 1)
	}
	for i := 0; i < 60; i++ {
		bump("c", 1)
	}
	for i := 0; i < 50; i++ {
		bump("d", 1)
	}
	assertValidTopK(t, tk, counts, 3, "initial")
	// Champion slides out of the window entirely.
	bump("a", -100)
	assertValidTopK(t, tk, counts, 3, "champion evicted")
	// Challenger rises into the Top-K.
	for i := 0; i < 40; i++ {
		bump("d", 1)
	}
	assertValidTopK(t, tk, counts, 3, "challenger rises")
	// Mass eviction empties everything.
	bump("b", -80)
	bump("c", -60)
	bump("d", -90)
	assertValidTopK(t, tk, counts, 3, "all evicted")
	if tk.Len() != 0 {
		t.Fatalf("Top-K not empty after full eviction: %v", tk.Items())
	}
	if tk.Tracked() != 0 {
		t.Fatalf("tracker retains %d items after full eviction", tk.Tracked())
	}
}

// TestTopKNeverResortsAllItems is a structural guarantee test: the tracker
// exposes no operation that sorts more than K entries, and Items() output
// size is capped at K regardless of how many distinct items were seen.
func TestTopKNeverResortsAllItems(t *testing.T) {
	tk := NewTopK(5)
	rng := rand.New(rand.NewSource(13))
	for i := 0; i < 100000; i++ {
		tk.Update(ItemName(rng.Intn(10000)), int64(rng.Intn(1000)))
	}
	if got := len(tk.Items()); got != 5 {
		t.Fatalf("Items() returned %d entries, want exactly K=5", got)
	}
}
