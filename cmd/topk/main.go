// Command topk generates a synthetic event stream, runs the sliding-window
// Top-K pipeline (exact + approximate), and writes a single self-contained
// HTML report that replays the window sliding and plots the approximation
// error against its theoretical bound.
package main

import (
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"

	"github.com/TommyMarsss/sliding-window-topk/topk"
)

//go:embed report.html
var reportTemplate string

func main() {
	var (
		n       = flag.Int("n", 200000, "number of events")
		k       = flag.Int("k", 10, "Top-K size")
		window  = flag.Int64("window", 10000, "window length (timestamp units; 1 event per unit)")
		buckets = flag.Int("buckets", 10, "CMS time buckets per window")
		eps     = flag.Float64("eps", 0.03, "per-sketch CMS epsilon")
		delta   = flag.Float64("delta", 0.01, "per-sketch CMS delta")
		vocab   = flag.Int("vocab", 2000, "number of distinct items")
		zipfS   = flag.Float64("zipf", 1.3, "Zipf skew parameter")
		frames  = flag.Int("frames", 400, "number of replay frames")
		seed    = flag.Int64("seed", 42, "RNG seed")
		out     = flag.String("out", "report.html", "output HTML file")
	)
	flag.Parse()

	rng := rand.New(rand.NewSource(*seed))
	sampler := topk.ZipfSampler(rng, *vocab, *zipfS)
	eng := topk.NewEngine(*k, *window, *buckets, *eps, *delta)

	rep := topk.Report{
		K: *k, Window: *window, Buckets: *buckets, Eps: *eps, Delta: *delta,
		Events: int64(*n), Vocab: *vocab, ZipfS: *zipfS,
		CMSBytes: eng.CMS().MemoryBytes(),
	}
	stride := *n / *frames
	if stride < 1 {
		stride = 1
	}
	for i := 0; i < *n; i++ {
		eng.Add(topk.Event{Ts: int64(i), Item: topk.ItemName(sampler())})
		if (i+1)%stride == 0 {
			rep.Frames = append(rep.Frames, eng.Frame(len(rep.Frames)))
		}
	}

	data, err := json.Marshal(rep)
	if err != nil {
		fatal(err)
	}
	html := strings.Replace(reportTemplate, "/*__REPORT_DATA__*/", string(data), 1)
	if strings.Contains(html, "/*__REPORT_DATA__*/") {
		fatal(fmt.Errorf("data placeholder not substituted"))
	}
	if err := os.WriteFile(*out, []byte(html), 0o644); err != nil {
		fatal(err)
	}
	fmt.Printf("wrote %s: %d events, %d frames, K=%d, window=%d, CMS=%d bytes (eps=%g delta=%g buckets=%d)\n",
		*out, *n, len(rep.Frames), *k, *window, rep.CMSBytes, *eps, *delta, *buckets)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "topk:", err)
	os.Exit(1)
}
