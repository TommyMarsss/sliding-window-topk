package topk

import (
	"container/heap"
	"sort"
)

// TopK maintains the K highest-count items incrementally under both
// increments (new events) and decrements (window eviction).
//
// Data structure: two binary heaps plus a position map.
//   - top:  min-heap of the current best <= K items (worst-of-top at root)
//   - rest: max-heap of all other tracked items   (best-of-rest at root)
//   - pos:  item -> *tkItem (with heap id + index for O(log n) fix/remove)
//
// Every Update is O(log D), where D is the number of tracked distinct items
// (bounded by the window capacity, not by total stream length). This avoids
// the O(D log D) full re-sort per event of the naive approach. Decrements are
// handled correctly: when a top item's count drops below the best outsider,
// the two roots are swapped.
type TopK struct {
	k    int
	top  tkHeap // min-heap (worst at root)
	rest tkHeap // max-heap (best at root)
	pos  map[string]*tkItem
}

type tkItem struct {
	item  string
	count int64
	heap  int // 0 = top, 1 = rest
	index int
}

const (
	heapTop  = 0
	heapRest = 1
)

// NewTopK returns an incremental Top-K structure for the given k.
func NewTopK(k int) *TopK {
	if k < 1 {
		k = 1
	}
	t := &TopK{k: k, pos: make(map[string]*tkItem)}
	t.top = tkHeap{max: false, heapID: heapTop}
	t.rest = tkHeap{max: true, heapID: heapRest}
	return t
}

// betterItem ranks tkItems: higher count, then smaller item name.
func betterItem(a, b *tkItem) bool {
	if a.count != b.count {
		return a.count > b.count
	}
	return a.item < b.item
}

// tkHeap implements heap.Interface. max=true keeps the best item at the
// root; max=false keeps the worst at the root.
type tkHeap struct {
	items  []*tkItem
	max    bool
	heapID int
}

func (h tkHeap) Len() int { return len(h.items) }

func (h tkHeap) Less(i, j int) bool {
	if h.max {
		return betterItem(h.items[i], h.items[j])
	}
	return betterItem(h.items[j], h.items[i])
}

func (h tkHeap) Swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
	h.items[i].index = i
	h.items[j].index = j
}

func (h *tkHeap) Push(x any) {
	it := x.(*tkItem)
	it.heap = h.heapID
	it.index = len(h.items)
	h.items = append(h.items, it)
}

func (h *tkHeap) Pop() any {
	old := h.items
	n := len(old)
	it := old[n-1]
	old[n-1] = nil
	h.items = old[:n-1]
	it.index = -1
	return it
}

// Update sets the tracked count of item (an estimate or exact count,
// depending on the caller). count <= 0 removes the item. O(log D).
func (t *TopK) Update(item string, count int64) {
	it, ok := t.pos[item]
	if count <= 0 {
		if ok {
			t.remove(it)
		}
		t.rebalance()
		return
	}
	if !ok {
		it = &tkItem{item: item, count: count}
		t.pos[item] = it
		if t.top.Len() < t.k {
			heap.Push(&t.top, it)
		} else {
			heap.Push(&t.rest, it)
		}
	} else {
		it.count = count
		if it.heap == heapTop {
			heap.Fix(&t.top, it.index)
		} else {
			heap.Fix(&t.rest, it.index)
		}
	}
	t.rebalance()
}

func (t *TopK) remove(it *tkItem) {
	if it.heap == heapTop {
		heap.Remove(&t.top, it.index)
	} else {
		heap.Remove(&t.rest, it.index)
	}
	delete(t.pos, it.item)
}

// rebalance restores the invariant: top holds the min(k, D) best items.
func (t *TopK) rebalance() {
	for t.top.Len() < t.k && t.rest.Len() > 0 {
		it := heap.Pop(&t.rest).(*tkItem)
		heap.Push(&t.top, it)
	}
	for t.top.Len() > 0 && t.rest.Len() > 0 && betterItem(t.rest.items[0], t.top.items[0]) {
		in := heap.Pop(&t.rest).(*tkItem)
		out := heap.Pop(&t.top).(*tkItem)
		heap.Push(&t.top, in)
		heap.Push(&t.rest, out)
	}
}

// TopK returns the current top items in rank order. O(k log k).
func (t *TopK) TopK() []ItemCount {
	out := make([]ItemCount, 0, t.top.Len())
	for _, it := range t.top.items {
		out = append(out, ItemCount{Item: it.item, Count: it.count})
	}
	sort.Slice(out, func(i, j int) bool { return better(out[i], out[j]) })
	return out
}

// All returns every tracked item in rank order (top + rest). Used by tests
// to compare against a full re-sort.
func (t *TopK) All() []ItemCount {
	out := make([]ItemCount, 0, len(t.pos))
	for _, it := range t.pos {
		out = append(out, ItemCount{Item: it.item, Count: it.count})
	}
	sort.Slice(out, func(i, j int) bool { return better(out[i], out[j]) })
	return out
}

// Tracked returns the number of distinct items currently tracked.
func (t *TopK) Tracked() int { return len(t.pos) }
