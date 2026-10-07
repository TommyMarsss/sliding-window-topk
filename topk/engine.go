package topk

// Frame is one replay snapshot of the sliding-window computation.
type Frame struct {
	Index       int         `json:"index"`
	WindowStart int64       `json:"windowStart"` // exact window lower bound (inclusive)
	WindowEnd   int64       `json:"windowEnd"`   // latest timestamp
	N           int64       `json:"n"`           // events in exact window
	Exact       []ItemCount `json:"exact"`       // exact Top-K (incrementally maintained)
	Approx      []ItemCount `json:"approx"`      // approximate Top-K (CMS estimates at query time)
	MaxErr      int64       `json:"maxErr"`      // max |est-true| over candidate items
	Bound       float64     `json:"bound"`       // eps * N (explicit error upper bound)
}

// Report is the full replay payload embedded into the HTML report.
type Report struct {
	K        int     `json:"k"`
	Window   int64   `json:"window"`
	Buckets  int     `json:"buckets"`
	Eps      float64 `json:"eps"`
	Delta    float64 `json:"delta"`
	Events   int64   `json:"events"`
	Vocab    int     `json:"vocab"`
	ZipfS    float64 `json:"zipfS"`
	CMSBytes int     `json:"cmsBytes"`
	Frames   []Frame `json:"frames"`
}

// Engine drives an event stream through the exact window, the approximate
// sketch and the exact Top-K tracker, recording frames for replay.
//
// Maintenance strategy (see README):
//
//   - Exact Top-K: fully incremental. Every arrival and every evicted
//     event triggers one TopK.Update — O(log K + log R) per event, never a
//     re-sort of all seen items.
//   - Approximate Top-K: derived at query time. A bucket roll changes the
//     estimates of ALL items at once, so eagerly tracking per-item
//     estimates would cost O(distinct) updates per bucket roll for no
//     benefit between queries. Instead Frame() evaluates CMS estimates for
//     the live candidates (distinct items in the window, O(D log K)) —
//     bounded by the window, never by total stream history.
type Engine struct {
	exact    *ExactWindow
	cms      *WindowCMS
	exactTop *TopK
	eps      float64
	k        int
}

// NewEngine wires the full pipeline. k is the Top-K size; window/nBuckets/
// eps/delta configure the window and the sketch.
func NewEngine(k int, window int64, nBuckets int, eps, delta float64) *Engine {
	return &Engine{
		exact:    NewExactWindow(window),
		cms:      NewWindowCMS(window, nBuckets, eps, delta),
		exactTop: NewTopK(k),
		eps:      eps,
		k:        k,
	}
}

// Add processes one event: incremental eviction, sketch update, and
// incremental exact Top-K maintenance — including the counts of items
// whose events slid out of the window.
func (e *Engine) Add(ev Event) {
	evicted := e.exact.Add(ev)
	e.cms.Add(ev.Ts, ev.Item)
	e.exactTop.Update(ev.Item, e.exact.Count(ev.Item))
	for _, old := range evicted {
		if old.Item != ev.Item {
			e.exactTop.Update(old.Item, e.exact.Count(old.Item))
		}
	}
}

// ApproxTopK computes the approximate Top-K over the bucket-aligned window
// at query time: CMS estimates for every distinct live item, selected with
// a fresh incremental TopK in O(D log K), D = distinct items in window.
func (e *Engine) ApproxTopK() []ItemCount {
	tk := NewTopK(e.k)
	for item := range e.exact.Snapshot() {
		tk.Update(item, e.cms.Estimate(item))
	}
	return tk.Items()
}

// Frame snapshots the current state. maxErr/bound are evaluated over the
// union of the exact and approximate Top-K candidate sets (the items whose
// error matters for ranking); tests verify the bound exhaustively.
func (e *Engine) Frame(index int) Frame {
	exact := e.exactTop.Items()
	approx := e.ApproxTopK()
	truth := e.exact.CountsSince(e.cms.AlignedStart())
	var maxErr int64
	seen := make(map[string]bool, 2*e.k)
	for _, ic := range append(append([]ItemCount{}, exact...), approx...) {
		if seen[ic.Item] {
			continue
		}
		seen[ic.Item] = true
		d := e.cms.Estimate(ic.Item) - truth[ic.Item]
		if d < 0 {
			d = -d
		}
		if d > maxErr {
			maxErr = d
		}
	}
	return Frame{
		Index:       index,
		WindowStart: e.exact.WindowStart(),
		WindowEnd:   e.exact.Latest(),
		N:           e.exact.Total(),
		Exact:       exact,
		Approx:      approx,
		MaxErr:      maxErr,
		Bound:       e.cms.ErrorBound(e.eps),
	}
}

// Exact is the exact-window tracker (exposed for tests).
func (e *Engine) Exact() *ExactWindow { return e.exact }

// CMS is the sketch (exposed for tests).
func (e *Engine) CMS() *WindowCMS { return e.cms }

// ExactTopK is the exact Top-K tracker (exposed for tests).
func (e *Engine) ExactTopK() *TopK { return e.exactTop }
