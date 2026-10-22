package sim

import (
	"math"
	"testing"
)

func gemmProfileFixture() (NativeGEMMProfile, NativeGEMMQuery) {
	p := NativeGEMMProfile{Schema: 1, Role: "warm_native_gemm_within_row_tile_surface", DType: "bfloat16", MaximumQueryRows: 384,
		Device:       NativeGEMMDevice{Name: "fixture", SMCount: 108, Capability: [2]int{8, 0}},
		KernelConfig: NativeGEMMKernelConfig{BlockM: 128, BlockN: 128, BlockK: 64, GroupM: 8, NumStages: 3, NumWarps: 8},
		Curves: map[string]NativeGEMMCurve{"qkv": {K: 16, N: 32, Bias: true, RowGroups: map[string][][]float64{
			"1": {{1, 2.5}, {128, 4.5}}, "2": {{1, 20}, {128, 30}},
		}}}}
	q := NativeGEMMQuery{Role: "qkv", M: 128, K: 16, N: 32, Bias: true, AStride: [2]int{16, 1}, BStride: [2]int{1, 16}, Device: p.Device, DType: p.DType, KernelConfig: p.KernelConfig, Warmed: true, Isolated: true}
	return p, q
}

func TestNativeGEMMPreservesTileDiscontinuityAndOwnsProfile(t *testing.T) {
	p, q := gemmProfileFixture()
	e, err := NewNativeGEMMEstimator(p)
	if err != nil {
		t.Fatal(err)
	}
	// The caller can reuse/edit its input without mutating this frozen model.
	p.Curves["qkv"].RowGroups["1"][1][1] = 999
	delete(p.Curves, "qkv")
	a, err := e.Estimate(q)
	if err != nil || a.GPUServiceUS != 4.5 || a.ClockDurationUS != 5 || a.Scope != NativeGEMMCostScope {
		t.Fatal(a, err)
	}
	q.M = 129
	b, err := e.Estimate(q)
	if err != nil || b.GPUServiceUS != 20 {
		t.Fatal("tile boundary was smoothed over", b, err)
	}
	q.M = 257
	if _, err := e.Estimate(q); err == nil {
		t.Fatal("missing tile group silently extrapolated")
	}
}

func TestNativeGEMMRejectsUnmeasuredContexts(t *testing.T) {
	p, q := gemmProfileFixture()
	e, err := NewNativeGEMMEstimator(p)
	if err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*NativeGEMMQuery){
		"first use":   func(q *NativeGEMMQuery) { q.Warmed = false },
		"concurrency": func(q *NativeGEMMQuery) { q.Isolated = false },
		"device":      func(q *NativeGEMMQuery) { q.Device.SMCount++ },
		"dtype":       func(q *NativeGEMMQuery) { q.DType = "float32" },
		"kernel":      func(q *NativeGEMMQuery) { q.KernelConfig.NumWarps = 4 },
		"matrix":      func(q *NativeGEMMQuery) { q.N++ },
		"bias":        func(q *NativeGEMMQuery) { q.Bias = false },
		"stride":      func(q *NativeGEMMQuery) { q.BStride = [2]int{32, 1} },
		"M":           func(q *NativeGEMMQuery) { q.M = 385 },
	} {
		t.Run(name, func(t *testing.T) {
			changed := q
			change(&changed)
			if _, err := e.Estimate(changed); err == nil {
				t.Fatal("unmeasured context silently accepted", changed)
			}
		})
	}
}

func TestNativeGEMMRejectsMalformedCalibration(t *testing.T) {
	for name, points := range map[string][][]float64{
		"extra field": {{1, 1, 1}}, "noninteger": {{1.5, 1}}, "NaN": {{1, math.NaN()}},
		"infinity": {{1, math.Inf(1)}}, "overflow": {{1, float64(math.MaxInt64)}},
		"unsorted": {{128, 1}, {1, 2}}, "duplicate": {{1, 1}, {1, 2}}, "zero": {{1, 0}},
	} {
		t.Run(name, func(t *testing.T) {
			p, _ := gemmProfileFixture()
			p.Curves["qkv"].RowGroups["1"] = points
			if _, err := NewNativeGEMMEstimator(p); err == nil {
				t.Fatal("malformed anchor accepted")
			}
		})
	}
	p, q := gemmProfileFixture()
	p.Curves["qkv"].RowGroups["1"] = [][]float64{{16, 1}, {96, 2}}
	e, err := NewNativeGEMMEstimator(p)
	if err != nil {
		t.Fatal(err)
	}
	q.M = 1
	if _, err := e.Estimate(q); err == nil {
		t.Fatal("unmeasured occupancy silently extrapolated")
	}
}
