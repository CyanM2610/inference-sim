package sim

import (
	"math"
	"testing"
)

func biasProfileFixture() (NativeBiasAddProfile, NativeBiasAddQuery) {
	p := NativeBiasAddProfile{Schema: 1, Role: "warm_native_producer_local_bias_add", DType: "bfloat16", Device: NativeGEMMDevice{Name: "fixture", SMCount: 108, Capability: [2]int{8, 0}}, N: 4608, MinimumM: 1, MaximumM: 128, Points: [][]float64{{1, 2.048}, {128, 3.072}}}
	q := NativeBiasAddQuery{M: 1, N: p.N, InputStride: [2]int{p.N, 1}, OutputStride: [2]int{p.N, 1}, BiasStride: [1]int{1}, DType: p.DType, Device: p.Device, Warmed: true, Isolated: true, ProducerLocal: true}
	return p, q
}

func TestNativeBiasAddRejectsDifferentOperationAndExecution(t *testing.T) {
	p, q := biasProfileFixture()
	e, err := NewNativeBiasAddEstimator(p)
	if err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*NativeBiasAddQuery){
		"in-place":     func(q *NativeBiasAddQuery) { q.InPlace = true },
		"cold input":   func(q *NativeBiasAddQuery) { q.ProducerLocal = false },
		"concurrent":   func(q *NativeBiasAddQuery) { q.Isolated = false },
		"first use":    func(q *NativeBiasAddQuery) { q.Warmed = false },
		"new width":    func(q *NativeBiasAddQuery) { q.N++ },
		"new device":   func(q *NativeBiasAddQuery) { q.Device.SMCount++ },
		"layout":       func(q *NativeBiasAddQuery) { q.OutputStride = [2]int{1, q.M} },
		"out of range": func(q *NativeBiasAddQuery) { q.M = 129 },
	} {
		t.Run(name, func(t *testing.T) {
			changed := q
			change(&changed)
			if _, err := e.Estimate(changed); err == nil {
				t.Fatal("accepted unmeasured bias operation", changed)
			}
		})
	}
}

func TestNativeBiasAddOwnsProfileAndPreservesNonzeroSmallCost(t *testing.T) {
	p, q := biasProfileFixture()
	e, err := NewNativeBiasAddEstimator(p)
	if err != nil {
		t.Fatal(err)
	}
	p.Points[0][1] = 0
	cost, err := e.Estimate(q)
	if err != nil || cost.GPUServiceUS != 2.048 || cost.ClockDurationUS != 3 || cost.Scope != NativeBiasAddCostScope {
		t.Fatal("small addition became free or lost its scope", cost, err)
	}
}

func TestNativeBiasAddRejectsUnresolvedOrInvalidCalibration(t *testing.T) {
	for _, points := range [][][]float64{{{1, 0}, {128, 2}}, {{2, 1}, {128, 2}}, {{1, math.NaN()}, {128, 2}}, {{1, 2}, {1, 2}, {128, 2}}, {{1, 2}, {128, float64(math.MaxInt64)}}} {
		p, _ := biasProfileFixture()
		p.Points = points
		if _, err := NewNativeBiasAddEstimator(p); err == nil {
			t.Fatal("accepted malformed calibration", points)
		}
	}
}
