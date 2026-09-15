package kvruntime

import (
	"strings"
	"testing"
)

func TestPagedInterferenceRepricesInflightWorkAtActualTransitions(t *testing.T) {
	e := NewEngine(nil)
	c, err := NewPagedInterference(e, PagedInterferenceRates{ComputePPM: map[string]int64{"forward": 500000}, CopyHostPPM: map[string]int64{"forward": 500000}, Evidence: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if err = c.Phase("forward", true); err != nil {
		t.Fatal(err)
	}
	var forwardEnd, hostEnd int64
	e.MustAdd(Task{Name: "forward", Resource: "gpu", DurationNS: 200, ServiceClass: PhaseServiceClass("forward"), Finish: func() error {
		forwardEnd = e.NowNS
		return c.Phase("forward", false)
	}})
	e.MustAdd(Task{Name: "copy_host", Resource: "copy_thread", DurationNS: 200, ServiceClass: CopyHostClass, Finish: func() error { hostEnd = e.NowNS; return nil }})
	for at, active := range map[int64]bool{40: true, 120: false} {
		_ = e.At(at, func() {
			if err := c.Copy(active); err != nil {
				t.Fatal(err)
			}
		})
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	// Forward loses 40 ns of work over the 80 ns copy window. The host
	// completes 120 ns of work by forward end, then its remaining 80 solo.
	if forwardEnd != 240 || hostEnd != 320 {
		t.Fatalf("transition work lost: forward=%d host=%d", forwardEnd, hostEnd)
	}
	if err = c.CheckIdle(); err != nil {
		t.Fatal(err)
	}
}

func TestPagedInterferenceRejectsUnmeasuredOverlap(t *testing.T) {
	for _, copyFirst := range []bool{false, true} {
		e := NewEngine(nil)
		c, err := NewPagedInterference(e, PagedInterferenceRates{ComputePPM: map[string]int64{"forward": 900000}, CopyHostPPM: map[string]int64{"forward": 900000}, Evidence: "test"})
		if err != nil {
			t.Fatal(err)
		}
		if copyFirst {
			if err = c.Copy(true); err != nil {
				t.Fatal(err)
			}
			err = c.Phase("observe_token", true)
		} else {
			if err = c.Phase("observe_token", true); err != nil {
				t.Fatal(err)
			}
			err = c.Copy(true)
		}
		if err == nil || !strings.Contains(err.Error(), "unmeasured") {
			t.Fatal("unmeasured output/copy overlap accepted", err)
		}
	}
}

func TestTransferHostRateDoesNotStretchDeviceWaitOrPublishEarly(t *testing.T) {
	r, _ := transferFixture(t, 8)
	e, m := r.Engine, r.Memory
	if err := e.SetServiceRate("host", 500000); err != nil {
		t.Fatal(err)
	}
	var transitions []int64
	v, err := r.Transfer(TransferSpec{ID: "host_rate", Source: "source", Destination: "target", CPU: "cpu", DMA: "dma", Spans: []Span{{Bytes: 128}}, HostServiceClass: "host",
		Cost: TransferCost{PlanNS: 10, SubmitNS: 10, DMANS: 100, ObserveNS: 10, HoldCPUUntilDMA: true},
		Activity: func(active bool) error {
			transitions = append(transitions, e.NowNS)
			if active {
				if m.Buffers["source"].Pins != 1 || m.Buffers["target"].Writers != 1 {
					t.Fatal("activity began without ownership")
				}
			} else if m.Buffers["source"].Pins != 0 || m.Buffers["target"].State != "ready" {
				t.Fatal("activity ended before publication")
			}
			return nil
		}})
	if err != nil {
		t.Fatal(err)
	}
	_ = e.At(60, func() {
		if err := e.SetServiceRate("host", 250000); err != nil {
			t.Fatal(err)
		}
	})
	if err = e.RunUntil(150); err != nil {
		t.Fatal(err)
	}
	if len(transitions) != 1 || m.Buffers["source"].Pins != 1 || m.Buffers["target"].Writers != 1 {
		t.Fatal("completed DMA bypassed host observation")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if v.FirstDMANS != 40 || v.LastDMANS != 140 || v.CPUFinishedNS != 140 || v.CompleteNS != 180 {
		t.Fatalf("host work and DMA wait conflated: %+v", v)
	}
	if len(transitions) != 2 || transitions[0] != 0 || transitions[1] != 180 {
		t.Fatal("wrong activity lifecycle", transitions)
	}
	if err = m.Check(); err != nil {
		t.Fatal(err)
	}
}

func TestPagedPhaseGapsExecuteAtTheirLocations(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 256, map[string]int64{"hbm": 917504, "workspace": 2 << 20})
	m.Reserve("slot", "hbm", "kv", 917504)
	m.Allocate("slot")
	r, _ := NewRuntime(e, m, 8)
	p := validPagedProfilePoint()
	p.Prefix, p.Query = 0, 16
	delete(p.Stages, "gather_prefix")
	delete(p.PhasePeakDelta, "gather_prefix")
	delete(p.PhaseEndDelta, "gather_prefix")
	p.PhaseGapsNS = map[string]int64{"forward": 11, "scatter_delta": 17, "hf_release": 23, "complete": 29}
	p.ResidualNS = 80
	weights := make([]float64, 30)
	weights[1] = 1
	origins := make([]uint64, 16)
	for i := range origins {
		origins[i] = uint64(i + 1)
	}
	starts, ends := map[string]int64{}, map[string]int64{}
	s := PagedStepSpec{ID: "gaps", Slots: []string{"slot"}, Query: 16, TokenOrigins: origins, Weights: weights, Point: p, WorkspacePool: "workspace", PhaseObserver: func(name string, active bool) error {
		if active {
			starts[name] = e.NowNS
		} else {
			ends[name] = e.NowNS
		}
		return nil
	}}
	// An unexecuted phase cost must be rejected before graph/memory mutation.
	s.Point.PhaseGapsNS["index_bind"] = 1
	s.Point.ResidualNS++
	if _, err := r.PagedStep(s); err == nil {
		t.Fatal("missing input phase silently dropped its cost")
	}
	if e.Pending() != 0 {
		t.Fatal("phase validation mutated the graph")
	}
	delete(s.Point.PhaseGapsNS, "index_bind")
	s.Point.ResidualNS--
	step, err := r.PagedStep(s)
	if err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if starts["forward"] != 11 || starts["scatter_delta"]-ends["forward"] != 17 || starts["hf_release"]-ends["scatter_delta"] != 23 || step.CompleteNS-ends["hf_release"] != 29 {
		t.Fatalf("gaps moved across phases: starts=%v ends=%v end=%d", starts, ends, step.CompleteNS)
	}
	for name, start := range starts {
		if step.StagesNS[name] != ends[name]-start {
			t.Fatal("gap included in phase service", name)
		}
	}
	if m.Used["workspace"] != 0 || step.CellsVerified != 16*224 {
		t.Fatal("gap scheduling changed memory lifecycle")
	}
}

func TestTransactionSynchronizeKeepsBatchSubmissionAndHoldsHost(t *testing.T) {
	r, _ := transferFixture(t, 8)
	e := r.Engine
	if err := e.SetServiceRate("host", 500000); err != nil {
		t.Fatal(err)
	}
	var waitID int64
	v, err := r.Transfer(TransferSpec{ID: "batch_sync", Source: "source", Destination: "target", CPU: "cpu", DMA: "dma", HostServiceClass: "host", Spans: []Span{{Bytes: 64}, {SourceOffset: 64, DestinationOffset: 64, Bytes: 64}}, Cost: TransferCost{SubmitNS: 5, DMANS: 100, ObserveNS: 5, HoldCPUUntilCompletion: true}})
	if err != nil {
		t.Fatal(err)
	}
	for id, task := range e.tasks {
		if task.Name == "host_transaction_synchronize" {
			waitID = id
		}
	}
	var otherHostStart int64
	_ = e.At(25, func() {
		e.MustAdd(Task{Name: "other_host_work", Resource: "cpu", DurationNS: 1, Start: func() error { otherHostStart = e.NowNS; return nil }})
	})
	if err = e.RunUntil(100); err != nil {
		t.Fatal(err)
	}
	if waitID == 0 || e.tasks[waitID].state != "running" || r.outstanding != 2 || otherHostStart != 0 {
		t.Fatal("batch issue serialized per DMA or host wait did not hold resource")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	// Two 10 ns submissions overlap two 100 ns DMA tasks. Host observation
	// can queue behind the independent 1 ns host task only after DMA end.
	if v.FirstDMANS != 10 || v.LastDMANS != 210 || v.CPUFinishedNS != 210 || otherHostStart != 210 || v.CompleteNS != 221 {
		t.Fatalf("incorrect final synchronization: %+v host=%d", v, otherHostStart)
	}
}
