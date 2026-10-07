package topk

import "container/heap"

// ItemCount pairs an item with its (exact or estimated) count.
type ItemCount struct {
	Item  string `json:"item"`
	Count int64  `json:"count"`
}

// TopK incrementally maintains the K items with the largest counts under a
// stream of count updates (increments from arrivals, decrements from window
// eviction). It never re-sorts all seen items.
//
// Data structure: two lazy-deletion heaps.
//
//	top:  min-heap holding the current Top-K candidates (size <= K)
//	rest: max-heap holding every other item with a nonzero count
//
// Each Update bumps a global sequence number and pushes a new entry for the
// item; stale entries (older seq) are discarded lazily when they reach a
// heap root. After each update the structure is rebalanced:
//
//   - if |top| < K, promote the best of rest;
//   - while max(rest) > min(top), swap the two roots.
//
// Complexity per Update: O(log K + log R) where R = items outside the Top-K
// (each stale entry is pushed once and popped once, so amortized cost stays
// logarithmic). Memory: O(distinct live items).
type TopK struct {
	k       int
	seq     uint64
	cur     map[string]uint64 // item -> latest seq (absent => count 0 / unknown)
	cnt     map[string]int64  // item -> current count
	inTop   map[string]bool
	top     minHeap
	rest    maxHeap
	topSize int
}

// NewTopK creates a tracker for the K most frequent items.
func NewTopK(k int) *TopK {
	if k <= 0 {
		panic("topk: k must be positive")
	}
	return &TopK{
		k:     k,
		cur:   make(map[string]uint64),
		cnt:   make(map[string]int64),
		inTop: make(map[string]bool),
	}
}

// Update sets the current count of item. count must be the item's new total
// (the tracker is source-agnostic: exact counts or sketch estimates).
func (t *TopK) Update(item string, count int64) {
	if count <= 0 {
		// Item leaves the candidate space entirely; its stale heap
		// entries become invalid via the seq maps and are popped lazily.
		delete(t.cur, item)
		delete(t.cnt, item)
		if t.inTop[item] {
			delete(t.inTop, item)
			t.topSize--
		}
		t.rebalance()
		return
	}
	t.seq++
	t.cur[item] = t.seq
	t.cnt[item] = count
	e := tkEntry{item: item, count: count, seq: t.seq}
	if t.inTop[item] {
		heap.Push(&t.top, e)
	} else {
		heap.Push(&t.rest, e)
	}
	t.rebalance()
}

func (t *TopK) valid(e tkEntry) bool {
	s, ok := t.cur[e.item]
	return ok && s == e.seq
}

func (t *TopK) cleanTop() {
	for t.top.Len() > 0 && !t.valid(t.top[0]) {
		heap.Pop(&t.top)
	}
}

func (t *TopK) cleanRest() {
	for t.rest.Len() > 0 && !t.valid(t.rest[0]) {
		heap.Pop(&t.rest)
	}
}

func (t *TopK) rebalance() {
	// Fill top up to K from the best of rest.
	for t.topSize < t.k {
		t.cleanRest()
		if t.rest.Len() == 0 {
			return
		}
		e := heap.Pop(&t.rest).(tkEntry)
		t.inTop[e.item] = true
		t.topSize++
		heap.Push(&t.top, e)
	}
	// Swap while the best of rest beats the worst of top.
	for {
		t.cleanTop()
		t.cleanRest()
		if t.top.Len() == 0 || t.rest.Len() == 0 {
			return
		}
		if t.rest[0].count <= t.top[0].count {
			return
		}
		out := heap.Pop(&t.top).(tkEntry)
		in := heap.Pop(&t.rest).(tkEntry)
		t.inTop[out.item] = false
		t.inTop[in.item] = true
		heap.Push(&t.rest, out)
		heap.Push(&t.top, in)
	}
}

// Items returns the current Top-K, sorted by count descending
// (ties broken by item name for determinism).
func (t *TopK) Items() []ItemCount {
	out := make([]ItemCount, 0, t.topSize)
	for _, e := range t.top {
		if t.valid(e) {
			out = append(out, ItemCount{Item: e.item, Count: e.count})
		}
	}
	sortItemCounts(out)
	return out
}

// Len returns the current number of items in the Top-K set.
func (t *TopK) Len() int { return t.topSize }

// Tracked returns the number of distinct items with nonzero count.
func (t *TopK) Tracked() int { return len(t.cnt) }

// sortItemCounts sorts by count desc, then item asc (insertion sort is fine
// for K-sized slices; K is small by definition).
func sortItemCounts(s []ItemCount) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0; j-- {
			a, b := s[j-1], s[j]
			if a.Count > b.Count || (a.Count == b.Count && a.Item <= b.Item) {
				break
			}
			s[j-1], s[j] = b, a
		}
	}
}

// --- heaps ---

type tkEntry struct {
	item  string
	count int64
	seq   uint64
}

type minHeap []tkEntry

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].count < h[j].count }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(tkEntry)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	e := old[n-1]
	*h = old[:n-1]
	return e
}

type maxHeap []tkEntry

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].count > h[j].count }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x interface{}) { *h = append(*h, x.(tkEntry)) }
func (h *maxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	e := old[n-1]
	*h = old[:n-1]
	return e
}
