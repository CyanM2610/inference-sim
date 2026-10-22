package kvruntime

import (
	"math"
	"testing"
)

func TestServiceRepricingPreservesProgressAndPublication(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 1, map[string]int64{"hbm": 2})
	for _, id := range []string{"source", "target"} {
		if _, err := m.Reserve(id, "hbm", "device", 1); err != nil {
			t.Fatal(err)
		}
		if err := m.Allocate(id); err != nil {
			t.Fatal(err)
		}
	}
	if err := m.Fill("source", 7); err != nil {
		t.Fatal(err)
	}
	var finished, consumed int64
	task := e.MustAdd(Task{Name: "copy", Resource: "dma", ServiceClass: "copy", DurationNS: 100, Start: func() error {
		if err := m.Pin("source"); err != nil {
			return err
		}
		return m.BeginWrite("target")
	}, Finish: func() error {
		finished = e.NowNS
		if err := m.Copy("source", "target", 0, 0, 1); err != nil {
			return err
		}
		if err := m.Publish("target"); err != nil {
			return err
		}
		return m.Unpin("source")
	}})
	e.MustAdd(Task{Name: "consumer", Resource: "gpu", Parents: []int64{task}, DurationNS: 5, Start: func() error { return m.Pin("target") }, Finish: func() error { consumed = e.NowNS; return m.Unpin("target") }})
	_ = e.At(20, func() {
		if err := e.SetServiceRate("copy", 500000); err != nil {
			t.Fatal(err)
		}
	})
	_ = e.At(60, func() {
		if err := e.SetServiceRate("copy", 1000000); err != nil {
			t.Fatal(err)
		}
	})
	if err := e.RunUntil(110); err != nil {
		t.Fatal(err)
	}
	if e.NowNS != 60 || finished != 0 || m.Buffers["source"].Pins != 1 || m.Buffers["target"].Writers != 1 {
		t.Fatalf("cancelled deadline advanced time or released content: now=%d finish=%d", e.NowNS, finished)
	}
	if at, ok := e.NextNS(); !ok || at != 120 {
		t.Fatalf("wrong repriced deadline %d %v", at, ok)
	}
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	if finished != 120 || consumed != 125 || e.NowNS != 125 || e.Pending() != 0 {
		t.Fatalf("duplicate/stale completion: finish=%d consume=%d now=%d pending=%d", finished, consumed, e.NowNS, e.Pending())
	}
}

func TestServiceFractionSurvivesRepeatedRateChanges(t *testing.T) {
	e := NewEngine(nil)
	_ = e.SetServiceRate("work", 333333)
	var end int64
	e.MustAdd(Task{Name: "fraction", Resource: "gpu", ServiceClass: "work", DurationNS: 3, Finish: func() error { end = e.NowNS; return nil }})
	for i := int64(1); i <= 6; i++ {
		at := i
		_ = e.At(at, func() {
			rate := int64(333333)
			if at%2 == 1 {
				rate = 666667
			}
			if err := e.SetServiceRate("work", rate); err != nil {
				t.Fatal(err)
			}
		})
	}
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	// Three pairs each complete exactly one ns of service despite fractions.
	if end != 6 {
		t.Fatalf("fraction was lost across transitions: %d", end)
	}
}

func TestPausedServiceResumesAndDoesNotInventDeadline(t *testing.T) {
	e := NewEngine(nil)
	_ = e.SetServiceRate("work", 0)
	var end int64
	e.MustAdd(Task{Name: "paused", Resource: "gpu", ServiceClass: "work", DurationNS: 8, Finish: func() error { end = e.NowNS; return nil }})
	if _, ok := e.NextNS(); ok {
		t.Fatal("paused task has a false completion event")
	}
	_ = e.At(17, func() {
		if err := e.SetServiceRate("work", 500000); err != nil {
			t.Fatal(err)
		}
	})
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	if end != 33 {
		t.Fatalf("paused work progressed: %d", end)
	}
}

func TestServiceLargeWorkUsesOverflowSafeArithmetic(t *testing.T) {
	e := NewEngine(nil)
	work := int64(1) << 54
	_ = e.SetServiceRate("work", 500000)
	e.MustAdd(Task{Name: "large", Resource: "gpu", ServiceClass: "work", DurationNS: work})
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	if e.NowNS != 2*work {
		t.Fatal("scaled service arithmetic overflowed")
	}
	f := NewEngine(nil)
	_ = f.SetServiceRate("work", 1)
	f.MustAdd(Task{Name: "overflow", Resource: "gpu", ServiceClass: "work", DurationNS: math.MaxInt64})
	if f.Error() == nil {
		t.Fatal("unrepresentable completion was accepted")
	}
}
