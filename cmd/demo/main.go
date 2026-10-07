// Command demo generates a synthetic event stream with a known frequency
// distribution, runs the sliding-window Top-K tracker over it, verifies
// correctness invariants at checkpoints (incremental eviction vs full
// rescan, sketch error vs theoretical bound, incremental Top-K vs full
// re-sort), and writes a single self-contained HTML replay report.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"

	topk "github.com/TommyMarsss/sliding-window-topk"
)

func main() {
	var (
		events      = flag.Int("events", 200000, "number of events to generate")
		window      = flag.Int64("window", 10000, "window size in time units (1 event = 1 time unit)")
		k           = flag.Int("k", 10, "top-k size")
		subwindows  = flag.Int("b", 32, "sketch sub-windows (buckets)")
		epsilon     = flag.Float64("eps", 0.001, "sketch epsilon")
		delta       = flag.Float64("delta", 0.01, "sketch delta")
		nitems      = flag.Int("items", 800, "number of distinct items in the universe")
		seed        = flag.Int64("seed", 1, "random seed")
		out         = flag.String("out", "report.html", "output HTML file")
		verifyEvery = flag.Int("verify-every", 2000, "checkpoint interval for invariant verification")
	)
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))
	tr := topk.NewTracker(*window, *k, *subwindows, *epsilon, *delta)

	// Zipf-like popularity over the item universe, reshuffled at two regime
	// changes so the Top-K membership visibly churns during replay.
	perm := shuffledItems(rng, *nitems)
	regime1, regime2 := *events*2/5, *events*4/5

	sampleEvery := *events / 600
	if sampleEvery < 1 {
		sampleEvery = 1
	}

	history := make([]string, 0, *events) // ground truth for full-rescan checks
	var frames []topk.Frame
	var violations int
	maxErrSeen, maxBoundSeen := int64(0), int64(0)

	for i := 0; i < *events; i++ {
		if i == regime1 || i == regime2 {
			perm = shuffledItems(rng, *nitems) // popularity regime change
		}
		item := perm[zipfIndex(rng, *nitems, 1.2)]
		ts := int64(i)
		tr.Add(item, ts)
		history = append(history, item)

		if (i+1)%*verifyEvery == 0 {
			violations += verify(tr, history, ts, *k)
		}
		if (i+1)%sampleEvery == 0 || i == *events-1 {
			f := topk.BuildFrame(tr, ts, *k)
			if f.MaxErr > maxErrSeen {
				maxErrSeen = f.MaxErr
			}
			if f.Bound > maxBoundSeen {
				maxBoundSeen = f.Bound
			}
			frames = append(frames, f)
		}
	}

	fmt.Printf("events=%d window=%d k=%d buckets=%d eps=%g delta=%g\n",
		*events, *window, *k, *subwindows, *epsilon, *delta)
	fmt.Printf("sketch counters=%d (constant)  window distinct=%d  in-window events=%d\n",
		tr.Sketch().MemoryCounters(), tr.Distinct(), tr.WindowTotal())
	fmt.Printf("max error seen=%d  bound<=%d  invariant violations=%d\n",
		maxErrSeen, maxBoundSeen, violations)

	meta := topk.ReportMeta{
		Window: *window, K: *k, Subwindows: *subwindows,
		Epsilon: *epsilon, Delta: *delta, Events: *events,
		Counters: tr.Sketch().MemoryCounters(),
	}
	fh, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, "create:", err)
		os.Exit(1)
	}
	defer fh.Close()
	if err := topk.RenderHTML(fh, meta, frames); err != nil {
		fmt.Fprintln(os.Stderr, "render:", err)
		os.Exit(1)
	}
	fmt.Println("wrote", *out)
	if violations > 0 {
		os.Exit(1)
	}
}

// shuffledItems returns item names item-0000.. in a random popularity order:
// perm[0] is the most popular item of the current regime.
func shuffledItems(rng *rand.Rand, n int) []string {
	p := rng.Perm(n)
	out := make([]string, n)
	for rank, id := range p {
		out[rank] = fmt.Sprintf("item-%04d", id)
	}
	return out
}

// zipfIndex samples a rank in [0,n) with P(rank r) ∝ 1/(r+1)^s.
func zipfIndex(rng *rand.Rand, n int, s float64) int {
	// Inverse-CDF via precomputed-ish closed form is overkill; use the
	// standard rejection-free approximation: floor(n * u^(1/(1-s))) is wrong
	// for s>1, so do a direct CDF walk on a cached table instead.
	cdf := zipfCDF(n, s)
	u := rng.Float64()
	return sort.Search(len(cdf), func(i int) bool { return cdf[i] >= u })
}

var cdfCache = map[int][]float64{}

func zipfCDF(n int, s float64) []float64 {
	if c, ok := cdfCache[n]; ok {
		return c
	}
	cdf := make([]float64, n)
	var acc float64
	for r := 0; r < n; r++ {
		acc += 1 / math.Pow(float64(r+1), s)
		cdf[r] = acc
	}
	for r := range cdf {
		cdf[r] /= acc
	}
	cdfCache[n] = cdf
	return cdf
}

// verify checks the three core invariants against ground truth:
//  1. incremental eviction == full rescan of remaining in-window events
//  2. sketch error within the theoretical bound for every active item
//  3. incremental Top-K == full re-sort of all sketch estimates
func verify(tr *topk.Tracker, history []string, now int64, k int) int {
	violations := 0

	// Ground truth: full rescan of the in-window suffix of history.
	lo := sort.Search(len(history), func(i int) bool { return int64(i) > now-tr.Window().Size() })
	naive := make(map[string]int64)
	for _, it := range history[lo:] {
		naive[it]++
	}

	// (1) exact window counts must match the rescan exactly.
	exact := tr.ExactCounts()
	if !countsEqual(exact, naive) {
		fmt.Printf("VIOLATION @%d: incremental eviction != full rescan\n", now)
		violations++
	}

	// (2) sketch estimates: never underestimate, never exceed the bound.
	bound := tr.ErrorBound()
	estimates := make(map[string]int64, len(naive))
	for it, truth := range naive {
		est := tr.Estimate(it)
		estimates[it] = est
		if est < truth || est-truth > bound {
			fmt.Printf("VIOLATION @%d: item %s est=%d truth=%d bound=%d\n", now, it, est, truth, bound)
			violations++
		}
	}

	// (3) incremental Top-K must equal a full re-sort of all estimates.
	got := tr.TopK()
	want := topk.ExactTopK(estimates, k)
	if !rankEqual(got, want) {
		fmt.Printf("VIOLATION @%d: incremental top-k != full re-sort\n", now)
		violations++
	}
	return violations
}

func countsEqual(a, b map[string]int64) bool {
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

func rankEqual(a, b []topk.ItemCount) bool {
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
