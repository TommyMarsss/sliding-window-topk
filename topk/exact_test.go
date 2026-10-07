package topk

import (
	"math/rand"
	"reflect"
	"testing"
)

// recompute counts events with Ts > latest-window from a retained list.
// This is the ground-truth full recomputation the incremental structure
// must match exactly.
func recompute(events []Event, window, latest int64) map[string]int64 {
	m := make(map[string]int64)
	for _, ev := range events {
		if ev.Ts > latest-window {
			m[ev.Item]++
		}
	}
	return m
}

// TestIncrementalEvictionMatchesFullRecompute feeds random streams and,
// after every event, compares the incrementally maintained counts against
// a full rescan of the events still inside the window.
func TestIncrementalEvictionMatchesFullRecompute(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 20; trial++ {
		window := int64(50 + rng.Intn(500))
		vocab := 1 + rng.Intn(50)
		n := 2000 + rng.Intn(8000)
		w := NewExactWindow(window)
		var history []Event
		ts := int64(0)
		for i := 0; i < n; i++ {
			ts += int64(rng.Intn(5)) // irregular arrival times
			ev := Event{Ts: ts, Item: ItemName(rng.Intn(vocab))}
			w.Add(ev)
			history = append(history, ev)
			want := recompute(history, window, ts)
			got := w.Snapshot()
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("trial %d event %d: incremental counts differ from full recompute\ngot  %v\nwant %v", trial, i, got, want)
			}
			var sum int64
			for _, c := range want {
				sum += c
			}
			if w.Total() != sum {
				t.Fatalf("trial %d event %d: Total()=%d, want %d", trial, i, w.Total(), sum)
			}
		}
	}
}

// TestEvictedItemsAreRemoved verifies counts drop to zero (and the map
// entry disappears) once all of an item's events slide out.
func TestEvictedItemsAreRemoved(t *testing.T) {
	w := NewExactWindow(10)
	w.Add(Event{Ts: 0, Item: "a"})
	w.Add(Event{Ts: 5, Item: "a"})
	w.Add(Event{Ts: 5, Item: "b"})
	if w.Count("a") != 2 || w.Count("b") != 1 {
		t.Fatalf("unexpected counts: a=%d b=%d", w.Count("a"), w.Count("b"))
	}
	// ts=20 slides the window to (10,20]: every prior event expires.
	w.Add(Event{Ts: 20, Item: "c"})
	if w.Count("a") != 0 || w.Count("b") != 0 {
		t.Fatalf("evicted items still counted: a=%d b=%d", w.Count("a"), w.Count("b"))
	}
	if w.Distinct() != 1 {
		t.Fatalf("Distinct()=%d, want 1 (evicted items must leave the map)", w.Distinct())
	}
	if w.Total() != 1 {
		t.Fatalf("Total()=%d, want 1", w.Total())
	}
}

// TestEngineTopKMatchesFullRecompute verifies the requirement end to end:
// the incrementally maintained exact Top-K equals the Top-K obtained by
// re-sorting a full recomputation over the remaining window events.
func TestEngineTopKMatchesFullRecompute(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const (
		window = 200
		vocab  = 60
		k      = 8
		n      = 20000
	)
	eng := NewEngine(k, window, 20, 0.001, 0.01)
	zipf := ZipfSampler(rng, vocab, 1.2)
	var history []Event
	for i := 0; i < n; i++ {
		ev := Event{Ts: int64(i), Item: ItemName(zipf())}
		eng.Add(ev)
		history = append(history, ev)
		if i%97 != 0 && i != n-1 {
			continue
		}
		// Ground truth: full recompute + full sort.
		counts := recompute(history, window, int64(i))
		all := make([]ItemCount, 0, len(counts))
		for item, c := range counts {
			all = append(all, ItemCount{Item: item, Count: c})
		}
		sortItemCounts(all)
		if len(all) > k {
			all = all[:k]
		}
		got := eng.ExactTopK().Items()
		if len(got) != len(all) {
			t.Fatalf("event %d: Top-K size %d, want %d", i, len(got), len(all))
		}
		// Count sequences must match exactly; under ties the specific
		// items may differ, so membership is checked via the threshold.
		threshold := all[len(all)-1].Count
		for j := range got {
			if got[j].Count != all[j].Count {
				t.Fatalf("event %d: rank %d count %d, want %d (got %v, want %v)", i, j, got[j].Count, all[j].Count, got, all)
			}
			if got[j].Count < threshold {
				t.Fatalf("event %d: item %s count %d below top-%d threshold %d", i, got[j].Item, got[j].Count, k, threshold)
			}
		}
	}
}
