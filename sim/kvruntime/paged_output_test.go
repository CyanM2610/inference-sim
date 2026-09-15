package kvruntime

import "testing"

func TestPagedCompletionWaitsForActualOutputCopy(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 256, map[string]int64{"hbm": 917504, "hbm_workspace": 1 << 20, "host_output": 8})
	m.Reserve("slot", "hbm", "kv", 917504)
	m.Allocate("slot")
	m.ReserveGranular("host_output", "host_output", "pinned", 8, 8)
	m.Allocate("host_output")
	r, _ := NewRuntime(e, m, 8)
	p := PagedForwardPoint{Prefix: 0, Query: 16, BeforeAllocatedBytes: 15 << 30, ReservedBytes: 16 << 30, Evidence: "test control", Stages: map[string]BridgeStage{"forward": {WallNS: 30}, "observe_token": {WallNS: 2}, "scatter_delta": {WallNS: 57, SubmitNS: 1, ServiceNS: 1}, "hf_release": {WallNS: 1}}, PhasePeakDelta: map[string]int64{"forward": 1 << 20, "observe_token": 918272, "scatter_delta": 917760, "hf_release": 917760}, PhaseEndDelta: map[string]int64{"forward": 918272, "observe_token": 917760, "scatter_delta": 917760, "hf_release": 0}, Output: &OutputCopyPoint{Cost: TransferCost{SubmitNS: 1, DMANS: 1, HoldCPUUntilDMA: true}, GPUStorageBytes: 512, HostStorageBytes: 8, HostReservedBytes: 8, ReferenceToken: 7, Evidence: "test explicit output"}}
	weights := make([]float64, 30)
	for i := range weights {
		weights[i] = 1.0 / 30
	}
	origins := make([]uint64, 16)
	for i := range origins {
		origins[i] = uint64(i + 1)
	}
	e.MustAdd(Task{Name: "existing_d2h_backlog", Resource: "copy_d2h", DurationNS: 10000})
	var completed int64
	step, err := r.PagedStep(PagedStepSpec{Request: "request", ID: "step", Slots: []string{"slot"}, Query: 16, TokenOrigins: origins, Weights: weights, Point: p, WorkspacePool: "hbm_workspace", HostOutputBuffer: "host_output", OutputCopyResource: "copy_d2h", OutputValue: 7, OutputReferenceKind: "request_reference", Done: func(result *PagedStepResult) { completed = result.CompleteNS }})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(100); err != nil {
		t.Fatal(err)
	}
	if completed != 0 || step.OutputObservedNS != 0 || m.Buffers["step/output_token"].Pins != 1 || m.Buffers["host_output"].Writers != 1 {
		t.Fatal("request completed or output storage released while D2H was blocked")
	}
	if active := e.resources["cpu"].active; active == nil || active.Name != "host_stream_synchronize" {
		t.Fatal("caller CPU released while native scalar synchronization was blocked")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if step.OutputObservedNS != 10001 || completed <= step.OutputObservedNS || step.OutputBytes != 8 || step.CellsVerified != 16*224 {
		t.Fatalf("incorrect dependent completion: %+v", step)
	}
	if m.Buffers["step/output_token"] != nil || m.Used["hbm_workspace"] != 0 {
		t.Fatal("output/HF storage was not reclaimed")
	}
}
