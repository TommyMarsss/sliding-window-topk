package topk

import (
	"fmt"
	"math/rand"
	"testing"
)

// Requirement 3: incremental Top-K maintenance (two heaps, O(log D) per
// update) must always agree with a full re-sort of all tracked counts,
// under interleaved inserts, increases, decreases and removals.
func TestTopKMatchesFullResort(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const k = 10
	tk := NewTopK(k)
	ref := make(map[string]int64)

	for step := 0; step < 20000; step++ {
		item := fmt.Sprintf("item-%03d", rng.Intn(120))
		switch rng.Intn(4) {
		case 0: // increase
			ref[item] += int64(1 + rng.Intn(5))
		case 1: // decrease (eviction-like)
			if ref[item] > 0 {
				ref[item] -= int64(1 + rng.Intn(int(ref[item])))
			}
		case 2: // remove entirely
			ref[item] = 0
		default: // set to a fresh value
			ref[item] = int64(rng.Intn(50))
		}
		tk.Update(item, ref[item])
		if ref[item] == 0 {
			delete(ref, item)
		}

		if step%53 == 0 {
			want := ExactTopK(ref, k)
			if got := tk.TopK(); !equalRank(got, want) {
				t.Fatalf("step %d: incremental top-k %v != full re-sort %v", step, got, want)
			}
			// The full tracked set must also match the reference exactly.
			all := tk.All()
			if len(all) != len(ref) {
				t.Fatalf("step %d: tracked %d items, want %d", step, len(all), len(ref))
			}
			for _, ic := range all {
				if ref[ic.Item] != ic.Count {
					t.Fatalf("step %d: item %s count %d, want %d", step, ic.Item, ic.Count, ref[ic.Item])
				}
			}
		}
	}
}

// A top item whose count drops below the best outsider must be swapped out
// incrementally (root swap), without any global re-sort.
func TestTopKDecrementSwapsOut(t *testing.T) {
	tk := NewTopK(3)
	tk.Update("a", 100)
	tk.Update("b", 90)
	tk.Update("c", 80)
	tk.Update("d", 70) // outsider, best of rest
	tk.Update("a", 60) // a falls below d
	got := tk.TopK()
	want := []ItemCount{{"b", 90}, {"c", 80}, {"d", 70}}
	if !equalRank(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

// A new item rising from zero must enter the top-k as soon as it passes the
// current k-th best.
func TestTopKNewItemEnters(t *testing.T) {
	tk := NewTopK(2)
	tk.Update("a", 50)
	tk.Update("b", 40)
	tk.Update("c", 10)
	if got := tk.TopK(); !equalRank(got, []ItemCount{{"a", 50}, {"b", 40}}) {
		t.Fatalf("got %v", got)
	}
	tk.Update("c", 45)
	if got := tk.TopK(); !equalRank(got, []ItemCount{{"a", 50}, {"c", 45}}) {
		t.Fatalf("got %v", got)
	}
}

// Removing an item from the top must promote the best outsider.
func TestTopKRemovalPromotes(t *testing.T) {
	tk := NewTopK(2)
	tk.Update("a", 50)
	tk.Update("b", 40)
	tk.Update("c", 30)
	tk.Update("a", 0) // evicted entirely
	if got := tk.TopK(); !equalRank(got, []ItemCount{{"b", 40}, {"c", 30}}) {
		t.Fatalf("got %v", got)
	}
	if tk.Tracked() != 2 {
		t.Fatalf("tracked=%d want 2", tk.Tracked())
	}
}

// Ties must resolve deterministically by item name.
func TestTopKTieBreak(t *testing.T) {
	tk := NewTopK(2)
	tk.Update("b", 10)
	tk.Update("a", 10)
	tk.Update("c", 10)
	got := tk.TopK()
	want := []ItemCount{{"a", 10}, {"b", 10}}
	if !equalRank(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}
