package topk

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// distribution is a sampler over item names with a known shape.
type distribution struct {
	name   string
	sample func(rng *rand.Rand) string
}

func testDistributions() []distribution {
	return []distribution{
		{
			name: "uniform-500",
			sample: func(rng *rand.Rand) string {
				return fmt.Sprintf("u-%d", rng.Intn(500))
			},
		},
		{
			name: "zipf-2000",
			sample: func(rng *rand.Rand) string {
				// P(rank r) ∝ 1/(r+1): harmonic sampler via exponential trick.
				u := rng.Float64()
				r := int(math.Exp(u*math.Log(2001))) - 1
				if r > 1999 {
					r = 1999
				}
				return fmt.Sprintf("z-%d", r)
			},
		},
		{
			name: "bursty-hot-cold",
			sample: func(rng *rand.Rand) string {
				if rng.Float64() < 0.6 {
					return "hot"
				}
				return fmt.Sprintf("c-%d", rng.Intn(300))
			},
		},
	}
}

// Requirement 2: for multiple known frequency distributions, the sketch
// error must satisfy 0 <= est-true <= eps*N_live at every checkpoint —
// bounded, never diverging.
func TestSlidingCMSErrorBound(t *testing.T) {
	const (
		window     = int64(5000)
		buckets    = 32
		epsilon    = 0.001
		delta      = 0.01
		nEvents    = 120000
		checkEvery = 997
	)
	for _, dist := range testDistributions() {
		t.Run(dist.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(99))
			s := NewSlidingCMS(window, buckets, epsilon, delta)
			w := NewWindow(window) // exact reference
			var maxRatio float64
			checks := 0
			for i := 0; i < nEvents; i++ {
				item := dist.sample(rng)
				ts := int64(i)
				s.Add(item, ts)
				w.Add(item, ts)
				if (i+1)%checkEvery == 0 {
					bound := int64(s.ErrorBound())
					truth := w.Counts()
					for it, c := range truth {
						est := int64(s.Estimate(it))
						if est < c {
							t.Fatalf("step %d: underestimate item %s est=%d true=%d", i, it, est, c)
						}
						if err := est - c; err > bound {
							t.Fatalf("step %d: error %d exceeds bound %d (item %s est=%d true=%d)",
								i, err, bound, it, est, c)
						}
						if bound > 0 {
							if r := float64(est-c) / float64(bound); r > maxRatio {
								maxRatio = r
							}
						}
					}
					checks++
				}
			}
			if checks == 0 {
				t.Fatal("no checkpoints ran")
			}
			if maxRatio > 1.0 {
				t.Fatalf("error/bound ratio %.3f exceeds 1", maxRatio)
			}
			t.Logf("max error/bound ratio = %.3f over %d checkpoints", maxRatio, checks)
		})
	}
}

// Expired events must leave the sketch: after the horizon passes the window,
// estimates for old items drop to zero.
func TestSlidingCMSExcludesExpired(t *testing.T) {
	s := NewSlidingCMS(100, 10, 0.001, 0.01)
	for i := 0; i < 100; i++ {
		s.Add("old", int64(i))
	}
	if got := s.Estimate("old"); got != 100 {
		t.Fatalf("est=%d want 100", got)
	}
	s.Advance(1000) // window (900,1000]: everything expired
	if got := s.Estimate("old"); got != 0 {
		t.Fatalf("est after expiry=%d want 0", got)
	}
	if got := s.LiveTotal(); got != 0 {
		t.Fatalf("live total after expiry=%d want 0", got)
	}
}

// Sketch memory must be a compile-time constant w.r.t. stream length.
func TestSlidingCMSMemoryConstant(t *testing.T) {
	s := NewSlidingCMS(1000, 16, 0.001, 0.01)
	before := s.MemoryCounters()
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 500000; i++ {
		s.Add(fmt.Sprintf("item-%d", rng.Intn(10000)), int64(i))
	}
	if got := s.MemoryCounters(); got != before {
		t.Fatalf("counters changed: %d -> %d", before, got)
	}
	// Sanity: bounded and small (ring * depth * width).
	ring := int64(16 + 2)
	width := int64(math.Ceil(math.E / 0.001))
	depth := int64(math.Ceil(math.Log(float64(ring) / 0.01)))
	if int64(before) != ring*depth*width {
		t.Fatalf("counters=%d, want ring*depth*width=%d", before, ring*depth*width)
	}
}

// Boundary semantics: an event exactly at the window edge (ts == now-window)
// is expired from the exact window, but it sits in the still-live oldest
// bucket, so the sketch conservatively counts it. That over-count is covered
// by the boundary term of the error bound, and disappears once the bucket
// dies.
func TestSlidingCMSBoundary(t *testing.T) {
	s := NewSlidingCMS(10, 5, 0.0001, 0.001)
	s.Add("x", 0)
	s.Add("x", 10)
	// now=10, exact window (0,10]: true count is 1, but the ts=0 event is in
	// the live boundary bucket -> sketch reports 2.
	if got := s.Estimate("x"); got != 2 {
		t.Fatalf("est=%d want 2 (boundary bucket still counted)", got)
	}
	if over := int64(s.Estimate("x")) - 1; over > int64(s.ErrorBound()) {
		t.Fatalf("overestimate %d exceeds bound %d", over, s.ErrorBound())
	}
	if s.BoundarySlack() != 1 {
		t.Fatalf("boundary slack=%d want 1 (the ts=0 event)", s.BoundarySlack())
	}
	// Once the boundary bucket dies, the slack moves on / disappears.
	s.Advance(20) // window (10,20]: both events expired, but ts=10 sits in
	// bucket [10,12) which is still live -> still counted as boundary slack.
	if got := s.Estimate("x"); got != 1 {
		t.Fatalf("est at now=20 = %d want 1 (slack from bucket [10,12))", got)
	}
	s.Advance(22) // bucket [10,12) now dead too
	if got := s.Estimate("x"); got != 0 {
		t.Fatalf("est after boundary bucket death=%d want 0", got)
	}
}
