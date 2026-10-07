package topk

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
)

// Requirement: memory stays bounded under a long-running stream — it must
// not grow with the total number of events processed. We check both the
// structural invariants (buffer sizes bounded by window/universe, sketch
// counters constant) and the process heap across a 3x increase in stream
// length.
func TestMemoryBoundedUnderLongStream(t *testing.T) {
	const (
		window  = int64(20000)
		nItems  = 1000
		phase1  = 500000
		phase2  = 1500000          // 3x the events of phase 1
		maxGrow = uint64(16 << 20) // 16 MiB slack for allocator noise
	)
	rng := rand.New(rand.NewSource(11))
	tr := NewTracker(window, 10, 32, 0.001, 0.01)
	counters0 := tr.Sketch().MemoryCounters()

	var ts int64
	heapAfter := func(n int) uint64 {
		for i := 0; i < n; i++ {
			tr.Add(fmt.Sprintf("item-%d", rng.Intn(nItems)), ts)
			ts++
		}
		runtime.GC()
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		return ms.HeapAlloc
	}

	h1 := heapAfter(phase1)

	// Structural bounds after half a million events.
	if got := tr.Window().Len(); got > int(window) {
		t.Fatalf("window buffer %d exceeds window size %d", got, window)
	}
	if got := tr.Distinct(); got > nItems {
		t.Fatalf("distinct %d exceeds universe %d", got, nItems)
	}
	if got := tr.Sketch().MemoryCounters(); got != counters0 {
		t.Fatalf("sketch counters not constant: %d -> %d", counters0, got)
	}
	if got := tr.tk.Tracked(); got > nItems {
		t.Fatalf("top-k tracked %d exceeds universe %d", got, nItems)
	}

	h2 := heapAfter(phase2)

	if h2 > h1+maxGrow {
		t.Fatalf("heap grew unboundedly: %d -> %d bytes (+%d)", h1, h2, h2-h1)
	}
	t.Logf("heap after %d events: %d bytes; after %d more: %d bytes", phase1, h1, phase2, h2)
}
