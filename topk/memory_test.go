package topk

import (
	"math/rand"
	"runtime"
	"testing"
)

// TestMemoryBoundedLongRun verifies that memory stays bounded by the
// window (not by total events processed) over a long run:
//
//   - the exact window buffers at most (events in window) entries;
//   - the Top-K tracker maps hold at most (distinct items in window) entries;
//   - the sketch is constant-size by construction;
//   - process heap does not grow without bound as events accumulate.
func TestMemoryBoundedLongRun(t *testing.T) {
	const (
		window   = 10000
		nBuckets = 50
		vocab    = 100000 // much larger than the window: forces churn
		n        = 3000000
	)
	eng := NewEngine(10, window, nBuckets, 0.001, 0.01)
	rng := rand.New(rand.NewSource(17))

	// Warm up: fill the window completely.
	for i := 0; i < 2*window; i++ {
		eng.Add(Event{Ts: int64(i), Item: ItemName(rng.Intn(vocab))})
	}
	runtime.GC()
	var base runtime.MemStats
	runtime.ReadMemStats(&base)

	for i := 2 * window; i < n; i++ {
		eng.Add(Event{Ts: int64(i), Item: ItemName(rng.Intn(vocab))})
		if i%100000 == 0 {
			// Structural bounds must hold at all times.
			if got := eng.Exact().Len(); got > window {
				t.Fatalf("event %d: window buffers %d events > window %d", i, got, window)
			}
			if got := eng.ExactTopK().Tracked(); got > window {
				t.Fatalf("event %d: Top-K tracks %d items > window %d", i, got, window)
			}
		}
	}
	runtime.GC()
	var final runtime.MemStats
	runtime.ReadMemStats(&final)

	// Heap may fluctuate (lazy heap entries, ring growth), but must not
	// scale with the 3M events processed: bound it to a generous constant.
	const maxGrowth = 64 << 20 // 64 MiB
	growth := int64(final.HeapAlloc) - int64(base.HeapAlloc)
	if growth > maxGrowth {
		t.Fatalf("heap grew by %d bytes over %d events (limit %d): memory scales with stream length", growth, n, maxGrowth)
	}
	t.Logf("heap growth over %d events: %d bytes (base %d, final %d)", n, growth, base.HeapAlloc, final.HeapAlloc)
}
