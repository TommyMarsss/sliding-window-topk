package topk

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// ZipfSampler returns a function sampling item indices 0..n-1 with
// probability proportional to 1/rank^s (rank = index+1).
func ZipfSampler(rnd *rand.Rand, n int, s float64) func() int {
	cum := make([]float64, n)
	var acc float64
	for i := 0; i < n; i++ {
		acc += 1.0 / math.Pow(float64(i+1), s)
		cum[i] = acc
	}
	return func() int {
		x := rnd.Float64() * acc
		return sort.Search(n, func(i int) bool { return cum[i] >= x })
	}
}

// UniformSampler returns a function sampling item indices uniformly.
func UniformSampler(rnd *rand.Rand, n int) func() int {
	return func() int { return rnd.Intn(n) }
}

// ItemName formats a vocabulary item id.
func ItemName(i int) string { return fmt.Sprintf("item-%04d", i) }
