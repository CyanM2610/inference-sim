package kv

import (
	"math"
	"testing"
)

func TestBenefitForecastUsesCompletedHistoryAndStableSmallMu(t *testing.T) {
	c := HotPrefixBenefitConfig{HorizonUS: 60_000_000, DecayUS: 60_000_000}
	var h hotPrefixReferenceHistory
	f := h.predict(0, c)
	if f.NextProbability != 0 || f.ExpectedCount != 0 || f.NextOffsetUS != 30_000_000 {
		t.Fatal("zero history invented observations", f)
	}
	h.completed(0, c)
	f = h.predict(0, c)
	if math.Abs(f.ExpectedCount-1) > 1e-12 || math.Abs(f.NextProbability-(1-math.Exp(-1))) > 1e-12 {
		t.Fatal("count/probability mismatch", f)
	}
	if f.NextOffsetUS >= 30_000_000 || f.ObservedReferences != 1 || f.ObservedThroughUS != 0 {
		t.Fatal("wrong conditional next reference", f)
	}
	h.completed(c.DecayUS, c)
	f = h.predict(c.DecayUS, c)
	if math.Abs(f.Weight-(1+math.Exp(-1))) > 1e-12 || f.ObservedReferences != 2 {
		t.Fatal("lost decayed history", f)
	}
	late := h.predict(1000*c.DecayUS, c)
	if math.IsNaN(late.NextProbability) || late.NextOffsetUS != 30_000_000 || late.NextProbability < 0 {
		t.Fatal("old history numerical instability", late)
	}
}

func TestBenefitRejectsRegressedClocksAndInvalidWindows(t *testing.T) {
	for _, c := range []HotPrefixBenefitConfig{{}, {HorizonUS: 1, DecayUS: -1}, {HorizonUS: 1 << 61, DecayUS: 1}} {
		if c.Validate() == nil {
			t.Fatal("invalid forecast config accepted")
		}
	}
	c := HotPrefixBenefitConfig{HorizonUS: 100, DecayUS: 100}
	for _, op := range []string{"completed", "predict"} {
		t.Run(op, func(t *testing.T) {
			var h hotPrefixReferenceHistory
			h.completed(20, c)
			defer func() {
				if recover() == nil {
					t.Fatal("used future history")
				}
			}()
			if op == "completed" {
				h.completed(10, c)
			} else {
				h.predict(10, c)
			}
		})
	}
}
