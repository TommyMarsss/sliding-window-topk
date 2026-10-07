package topk

import (
	"math"
	"math/rand"
	"testing"
)

// runDistribution feeds n events drawn from sampler into an Engine and, at
// regular checkpoints, verifies for EVERY distinct item in the
// bucket-aligned window that:
//
//	0 <= est - true <= eps * N
//
// i.e. the estimate never under-counts and never exceeds the explicit
// error bound. This is the exhaustive check; the report only samples.
func runDistribution(t *testing.T, name string, sampler func() int, vocab, n int, window int64, nBuckets int, eps, delta float64) {
	t.Helper()
	eng := NewEngine(10, window, nBuckets, eps, delta)
	for i := 0; i < n; i++ {
		ts := int64(i)
		eng.Add(Event{Ts: ts, Item: ItemName(sampler())})
		if int64(i) < window || i%1000 != 0 {
			continue
		}
		truth := eng.Exact().CountsSince(eng.CMS().AlignedStart())
		N := eng.CMS().Total()
		bound := eps * float64(N)
		for item, want := range truth {
			est := eng.CMS().Estimate(item)
			if est < want {
				t.Fatalf("%s event %d: item %s estimate %d < true %d (CMS must never under-count)", name, i, item, est, want)
			}
			if d := float64(est - want); d > bound {
				t.Fatalf("%s event %d: item %s error %v exceeds bound eps*N=%v (est=%d true=%d N=%d)", name, i, item, d, bound, est, want, N)
			}
		}
		// The bound must also hold for items NOT in the window (true=0).
		for _, ghost := range []string{"ghost-1", "ghost-2", "ghost-3"} {
			if est := eng.CMS().Estimate(ghost); float64(est) > bound {
				t.Fatalf("%s event %d: ghost item %s estimate %d exceeds bound %v", name, i, ghost, est, bound)
			}
		}
	}
}

// TestApproxErrorWithinBound checks the explicit error bound across
// qualitatively different frequency distributions.
func TestApproxErrorWithinBound(t *testing.T) {
	const (
		window   = 10000
		nBuckets = 50
		eps      = 0.001
		delta    = 0.01
		n        = 120000
		vocab    = 2000
	)
	t.Run("uniform", func(t *testing.T) {
		rng := rand.New(rand.NewSource(1))
		runDistribution(t, "uniform", UniformSampler(rng, vocab), vocab, n, window, nBuckets, eps, delta)
	})
	t.Run("zipf-1.1", func(t *testing.T) {
		rng := rand.New(rand.NewSource(2))
		runDistribution(t, "zipf-1.1", ZipfSampler(rng, vocab, 1.1), vocab, n, window, nBuckets, eps, delta)
	})
	t.Run("zipf-1.5", func(t *testing.T) {
		rng := rand.New(rand.NewSource(3))
		runDistribution(t, "zipf-1.5", ZipfSampler(rng, vocab, 1.5), vocab, n, window, nBuckets, eps, delta)
	})
	t.Run("bursty", func(t *testing.T) {
		// Two-regime stream: a small hot set alternating with a long tail,
		// stressing eviction when hot items slide out.
		rng := rand.New(rand.NewSource(4))
		phase := 0
		sampler := func() int {
			if rng.Intn(2000) == 0 {
				phase = 1 - phase
			}
			if phase == 0 {
				return rng.Intn(20) // hot head
			}
			return 20 + rng.Intn(vocab-20) // cold tail
		}
		runDistribution(t, "bursty", sampler, vocab, n, window, nBuckets, eps, delta)
	})
}

// TestErrorBoundIsTight verifies the bound is not vacuous: observed error
// stays a constant fraction of eps*N and does not grow with stream length
// (no unbounded divergence as events accumulate).
func TestErrorBoundIsTight(t *testing.T) {
	const (
		window   = 5000
		nBuckets = 25
		eps      = 0.002
		delta    = 0.01
		vocab    = 500
	)
	rng := rand.New(rand.NewSource(9))
	zipf := ZipfSampler(rng, vocab, 1.3)
	eng := NewEngine(10, window, nBuckets, eps, delta)
	var ratios []float64
	for i := 0; i < 200000; i++ {
		eng.Add(Event{Ts: int64(i), Item: ItemName(zipf())})
		if i < int(window) || i%5000 != 0 {
			continue
		}
		truth := eng.Exact().CountsSince(eng.CMS().AlignedStart())
		var maxErr int64
		for item, want := range truth {
			if d := eng.CMS().Estimate(item) - want; d > maxErr {
				maxErr = d
			}
		}
		ratios = append(ratios, float64(maxErr)/(eps*float64(eng.CMS().Total())))
	}
	for i, r := range ratios {
		if r > 1.0 {
			t.Fatalf("checkpoint %d: error/bound ratio %v > 1 (bound violated)", i, r)
		}
	}
	// No divergence: the last-quarter ratios must not exceed the
	// first-quarter maximum by a wide margin.
	first, last := ratios[:len(ratios)/4], ratios[3*len(ratios)/4:]
	maxFirst, maxLast := 0.0, 0.0
	for _, r := range first {
		maxFirst = math.Max(maxFirst, r)
	}
	for _, r := range last {
		maxLast = math.Max(maxLast, r)
	}
	if maxLast > 2*maxFirst+0.05 {
		t.Fatalf("error ratio grows over time: first-quarter max %v, last-quarter max %v", maxFirst, maxLast)
	}
}

// TestCMSMemoryConstant verifies sketch memory is fixed at construction.
func TestCMSMemoryConstant(t *testing.T) {
	c := NewWindowCMS(10000, 50, 0.001, 0.01)
	before := c.MemoryBytes()
	rng := rand.New(rand.NewSource(5))
	for i := 0; i < 1000000; i++ {
		c.Add(int64(i), ItemName(rng.Intn(100000)))
	}
	if got := c.MemoryBytes(); got != before {
		t.Fatalf("sketch memory changed: %d -> %d", before, got)
	}
}
