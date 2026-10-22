package kvruntime

import (
	"strings"
	"testing"
)

func inputFixture(t *testing.T) (*Runtime, *RequestInputs) {
	t.Helper()
	e := NewEngine(nil)
	m, _ := NewMemory(e, 256, map[string]int64{"gpu": 1 << 20, "host": 640})
	r, _ := NewRuntime(e, m, 8)
	p := InputPipelineProfile{GPUIndexBytes: 512, HostIndexBytes: 320, GPUAllocationQuantum: 512, ReplacedGPUInputBytes: 1536, Evidence: "test", Index: []InputIndexPoint{{Count: 1, PrepareNS: 5, Cost: TransferCost{SubmitNS: 2, DMANS: 3, HoldCPUUntilDMA: true}}}, NextInput: TransferCost{SubmitNS: 2, DMANS: 3, HoldCPUUntilDMA: true}}
	i, err := NewRequestInputs(r, p, "gpu", "host", []string{"page"}, []InputRequest{{ID: "req", Prompt: []uint64{7}, Output: []uint64{12345, 54321}}})
	if err != nil {
		t.Fatal(err)
	}
	m.ReserveGranular("token", "gpu", "GPU producer", 512, 8)
	m.Allocate("token")
	m.Fill("token", 12345)
	return r, i
}

func TestInputPipelineWaitsForIndexAndTokenCopies(t *testing.T) {
	r, i := inputFixture(t)
	e, m := r.Engine, r.Memory
	for _, id := range []string{i.indexGPU, i.indexHost} {
		if m.Buffers[id].State != "allocated" {
			t.Fatal("empty index tensor advertised as initialized")
		}
		for _, cell := range m.Buffers[id].Cells {
			if cell.Valid {
				t.Fatal("invented index contents before binding")
			}
		}
	}
	for _, cell := range m.Buffers[i.requests["req"].buffer].Cells[3:] {
		if cell.Valid {
			t.Fatal("allocator padding advertised as known input tokens")
		}
	}
	e.MustAdd(Task{Name: "index_copy_backlog", Resource: "copy_h2d", DurationNS: 1000})
	b, _, err := i.Bind("first", "req", 0, 1, []string{"page"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(100); err != nil {
		t.Fatal(err)
	}
	if b.indicesPinned || m.Buffers[i.indexGPU].Writers != 1 {
		t.Fatal("indices became readable before actual H2D completion")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if err = b.AcquireQuery(); err != nil {
		t.Fatal(err)
	}
	if err = b.ReleaseQuery(); err != nil {
		t.Fatal(err)
	}
	started := e.NowNS
	e.MustAdd(Task{Name: "next_input_backlog", Resource: "copy_d2d", DurationNS: 1000})
	if _, err = b.ObserveOutput("token", 12345, nil); err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(started + 100); err != nil {
		t.Fatal(err)
	}
	if i.requests["req"].produced != 0 || m.Buffers["token"].Pins != 1 {
		t.Fatal("token declared produced or source released before D2D completion")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if i.requests["req"].produced != 1 || m.Buffers[i.requests["req"].buffer].Cells[1] != (Cell{12345, 0, true}) {
		t.Fatal("next input value was not physically copied")
	}
	if err = b.ReleaseIndices(); err != nil {
		t.Fatal(err)
	}
	decode, _, err := i.Bind("decode", "req", 1, 1, []string{"page"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if err = decode.AcquireQuery(); err != nil {
		t.Fatal(err)
	}
	decode.ReleaseQuery()
	decode.ReleaseIndices()
	if err = m.Check(); err != nil {
		t.Fatal(err)
	}
}

func TestInputPipelineRejectsUnproducedOrCorruptDecodeInput(t *testing.T) {
	for _, corrupt := range []bool{false, true} {
		r, i := inputFixture(t)
		if corrupt {
			i.requests["req"].produced = 1
		} // readiness alone cannot certify actual input bytes.
		b, _, err := i.Bind("decode", "req", 1, 1, []string{"page"}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err = r.Engine.Run(); err != nil {
			t.Fatal(err)
		}
		err = b.AcquireQuery()
		wanted := "before token production"
		if corrupt {
			wanted = "differs from independently"
		}
		if err == nil || !strings.Contains(err.Error(), wanted) {
			t.Fatalf("corrupt=%v, got %v", corrupt, err)
		}
	}
}

func TestPagedInputReleasesOutputAfterD2DBeforeScatter(t *testing.T) {
	r, i := inputFixture(t)
	e, m := r.Engine, r.Memory
	m.Capacity["hbm"] = 917504
	m.Reserve("page", "hbm", "kv", 917504)
	m.Allocate("page")
	m.ReserveGranular("host_output", "host", "scalar", 8, 8)
	m.Allocate("host_output")
	p := PagedForwardPoint{Prefix: 0, Query: 1, BeforeAllocatedBytes: 15 << 30, ReservedBytes: 16 << 30, Evidence: "test", Stages: map[string]BridgeStage{"forward": {WallNS: 30}, "observe_token": {WallNS: 2}, "scatter_delta": {WallNS: 57, SubmitNS: 1, ServiceNS: 1}, "hf_release": {WallNS: 1}}, PhasePeakDelta: map[string]int64{}, PhaseEndDelta: map[string]int64{}, Output: &OutputCopyPoint{GPUStorageBytes: 512, HostStorageBytes: 8, HostReservedBytes: 8, Cost: TransferCost{SubmitNS: 1, DMANS: 1, HoldCPUUntilDMA: true}, Evidence: "test"}}
	for name := range p.Stages {
		p.PhasePeakDelta[name] = 57344 + 512
		p.PhaseEndDelta[name] = 57344
	}
	p.PhaseEndDelta["forward"] = 57344 + 512
	p.PhaseEndDelta["hf_release"] = 0
	weights := make([]float64, 30)
	for n := range weights {
		weights[n] = 1.0 / 30
	}
	sawScatter := false
	e.Sink = func(record Record) {
		if record.Name == "task_start" && record.Operation == "full/scatter_delta" {
			sawScatter = true
			if m.Buffers["full/output_token"] != nil {
				t.Error("GPU argmax tensor survived into scatter")
			}
			if m.Buffers[i.requests["req"].buffer].Cells[1] != (Cell{12345, 0, true}) {
				t.Error("scatter started before next input D2D publication")
			}
		}
	}
	_, err := r.PagedStep(PagedStepSpec{Inputs: i, Request: "req", ID: "full", Slots: []string{"page"}, Query: 1, TokenOrigins: []uint64{1}, Weights: weights, Point: p, WorkspacePool: "gpu", HostOutputBuffer: "host_output", OutputCopyResource: "copy_d2h", OutputValue: 12345, OutputReferenceKind: "request_reference"})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if !sawScatter || i.requests["req"].produced != 1 {
		t.Fatal("input pipeline did not complete")
	}
}
