package kvruntime

import "testing"

func TestPagedForwardPausePreservesUnpublishedKV(t *testing.T) {
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
	p.Stages["forward"] = BridgeStage{WallNS: 1000}
	weights := make([]float64, 30)
	weights[1] = 1 // Keep the first layer in flight through both rate changes.
	origins := make([]uint64, 16)
	for i := range origins {
		origins[i] = uint64(i + 1)
	}
	step, err := r.PagedStep(PagedStepSpec{ID: "step", Slots: []string{"slot"}, Query: 16,
		TokenOrigins: origins, Weights: weights, Point: p, WorkspacePool: "workspace",
		ServiceClasses: PagedServiceClasses{Forward: "forward"}})
	if err != nil {
		t.Fatal(err)
	}
	_ = e.At(250, func() {
		if err := e.SetServiceRate("forward", 0); err != nil {
			t.Fatal(err)
		}
	})
	_ = e.At(500, func() {
		if err := e.SetServiceRate("forward", 500000); err != nil {
			t.Fatal(err)
		}
	})
	if err := e.RunUntil(400); err != nil {
		t.Fatal(err)
	}
	if step.CompleteNS != 0 || m.Buffers["step/output/0"].Writers != 1 || m.Buffers["slot"].State == "ready" {
		t.Fatal("paused forward published or released unfinished KV")
	}
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	// 250 solo + 250 paused + 750/0.5 service; stale solo completion must not fire.
	if step.StagesNS["forward"] != 2000 || step.CompleteNS <= 2000 || step.CellsVerified != 16*224 {
		t.Fatalf("incorrect repriced paged completion: %+v", step)
	}
	if m.Used["workspace"] != 0 || m.Buffers["slot"].State != "ready" {
		t.Fatal("paged completion leaked workspace or content")
	}
	if err := m.Check(); err != nil {
		t.Fatal(err)
	}
}

func TestMappingKernelRepricingRetainsOwnership(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 1, map[string]int64{"hbm": 8})
	for _, id := range []string{"source", "target"} {
		m.Reserve(id, "hbm", "device", 4)
		m.Allocate(id)
		m.Fill(id, 7)
	}
	r, _ := NewRuntime(e, m, 8)
	if err := e.SetServiceRate("map_cpu", 500000); err != nil {
		t.Fatal(err)
	}
	_, completed, err := r.MapKernel("mapping", "cpu", "gpu", MappingKernel{Name: "map", Source: "source", Destination: "target",
		Spans: []Span{{Bytes: 4}}, SubmitNS: 10, ServiceNS: 100, SubmitServiceClass: "map_cpu", KernelServiceClass: "map_gpu"})
	if err != nil {
		t.Fatal(err)
	}
	_ = e.At(40, func() {
		if err := e.SetServiceRate("map_gpu", 0); err != nil {
			t.Fatal(err)
		}
	})
	_ = e.At(60, func() {
		if err := e.SetServiceRate("map_gpu", 500000); err != nil {
			t.Fatal(err)
		}
	})
	if err := e.RunUntil(50); err != nil {
		t.Fatal(err)
	}
	if m.Buffers["source"].Pins != 1 || m.Buffers["target"].Writers != 1 || e.tasks[completed].state != "running" {
		t.Fatal("paused mapping released ownership or completed early")
	}
	if err := m.Free("source"); err == nil {
		t.Fatal("paused mapping source was reclaimed")
	}
	if err := e.Run(); err != nil {
		t.Fatal(err)
	}
	if e.NowNS != 220 || m.Buffers["target"].State != "ready" || m.Buffers["target"].Cells[0] != m.Buffers["source"].Cells[0] {
		t.Fatal("mapping lost work, content or completion ordering")
	}
	if err := m.Check(); err != nil {
		t.Fatal(err)
	}
}
