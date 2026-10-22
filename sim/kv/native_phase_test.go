package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

// Artificial service values deliberately make output ready before the main
// stream tail. This is a causal-order test, not a fitted GPU timing model.
type nativePhaseTestCost struct{ nativeTestCost }

func (nativePhaseTestCost) ExecutionResources() []string { return []string{"cpu", "compute", "copy"} }
func (nativePhaseTestCost) CopyResource() string         { return "copy" }
func (nativePhaseTestCost) ExecutionPlan(shape []sim.NativeBatchShape) sim.NativeExecutionPlan {
	if len(shape) == 0 {
		return sim.NativeExecutionPlan{Stages: []sim.NativeStage{
			{Name: "post", Resource: "cpu", Stream: "host", DurationUS: 10},
			{Name: "producer", Stream: "main", After: []string{"post"}},
			{Name: "poll", Resource: "cpu", Stream: "host", DurationUS: 5, After: []string{"post"}},
		}, Computed: "post", ProducerReady: "producer", PostForward: "post", Poll: "poll", HostReturned: "poll", OutputReady: "poll"}
	}
	return sim.NativeExecutionPlan{Stages: []sim.NativeStage{
		{Name: "prepare", Resource: "cpu", Stream: "host", DurationUS: 10},
		{Name: "forward", Resource: "compute", Stream: "main", DurationUS: 100, After: []string{"prepare"}},
		{Name: "sample_submit", Resource: "cpu", Stream: "host", DurationUS: 5, After: []string{"prepare"}},
		{Name: "sample", Resource: "compute", Stream: "main", DurationUS: 10, After: []string{"forward", "sample_submit"}},
		{Name: "output_copy", Resource: "copy", Stream: "output", DurationUS: 5, After: []string{"sample"}},
		{Name: "main_tail", Resource: "compute", Stream: "main", DurationUS: 80, After: []string{"sample"}},
		{Name: "post", Resource: "cpu", Stream: "host", DurationUS: 10, After: []string{"sample_submit"}},
		{Name: "poll", Resource: "cpu", Stream: "host", DurationUS: 5, After: []string{"post"}},
	}, Computed: "forward", ProducerReady: "main_tail", PostForward: "post", Poll: "poll", HostReturned: "poll", OutputReady: "output_copy"}
}

func phaseEngine(t *testing.T, store *PeerCache) *sim.Simulator {
	t.Helper()
	e, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
		KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: 24, BlockSizeTokens: 2},
		BatchConfig:   sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: 16}}, store, nativeTestCost{})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.EnableNativeAsync(2, nativePhaseTestCost{}); err != nil {
		t.Fatal(err)
	}
	return e
}

func TestNativePhasesOutputOverlapsTailAndSubmittedStoreWaitsForProducer(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	e := phaseEngine(t, s)
	r := &sim.Request{ID: "branch", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6}, MaxOutputLen: 8}
	var events []*sim.NativeAsyncEvent
	e.EventObserver = func(event sim.Event, before bool) {
		if x, ok := event.(*sim.NativeAsyncEvent); ok && !before {
			events = append(events, x)
		}
	}
	e.InjectArrival(r)
	e.Run()
	times := map[string]int64{}
	for _, x := range events {
		if x.BatchID == 1 {
			times[x.Name+"/"+x.Phase] = x.At
		}
	}
	consumed := times["native_async_consumed/"]
	if consumed == 0 || consumed >= times["native_phase_end/main_tail"] || consumed < times["native_phase_end/output_copy"] {
		t.Fatal("output consumption incorrectly waited for main tail", times)
	}
	for _, x := range events {
		if x.BatchID == 2 && x.Name == "native_phase_start" && x.Phase == "forward" && x.At < times["native_phase_end/main_tail"] {
			t.Fatal("next forward bypassed previous main-stream tail")
		}
		if x.BatchID == 2 && x.Name == "native_async_scheduled" && x.At <= times["native_phase_end/poll"] {
			t.Fatal("next scheduler ignored synchronous worker host occupancy")
		}
	}
	queued, started := int64(-1), int64(-1)
	for _, x := range h.records {
		if x.Request == r.ID && x.Source == "hbm" {
			if x.Name == "transfer_queued" {
				queued = x.Time
			}
			if x.Name == "transfer_start" {
				started = x.Time
			}
		}
	}
	if queued != times["native_phase_end/post"] || queued >= times["native_phase_end/forward"] || started < times["native_phase_end/main_tail"] {
		t.Fatal("STORE lost host-submit / producer dependency", queued, started, times)
	}
	if e.Metrics.CompletedRequests != 1 || e.Metrics.TotalOutputTokens != 1 || s.fabric.Pending() != 0 || s.UsedBlocks() != 0 || s.nativeBatchPins != 0 {
		t.Fatal("phase execution did not finish/drain", s.PeerSnapshot())
	}
}

func TestNativePhasesLoadStartsAtPostForwardAndIsAdoptedAfterPoll(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	input := []sim.TokenID{1, 2, 3, 4}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	for _, b := range s.Blocks {
		if b.Hash != "" {
			s.invalidate(b)
		}
	}
	h.records = nil
	e := phaseEngine(t, s)
	r := &sim.Request{ID: "load", InputTokens: input, OutputTokens: []sim.TokenID{5}, MaxOutputLen: 1}
	e.InjectArrival(r)
	e.Run()
	prepared, started, physical, adopted := int64(-1), int64(-1), int64(-1), int64(-1)
	for _, x := range h.records {
		if x.Request == r.ID {
			switch x.Name {
			case "native_transfer_prepared":
				prepared = x.Time
			case "transfer_start":
				if x.Destination == "hbm" {
					started = x.Time
				}
			case "native_load_device_complete":
				physical = x.Time
			case "native_receive_adopted":
				adopted = x.Time
			}
		}
	}
	if prepared != 0 || started < 12 || physical < started || adopted <= physical || !r.TTFTSet {
		t.Fatal("LOAD bypassed post-forward or host poll", prepared, started, physical, adopted)
	}
	if s.fabric.Pending() != 0 || s.UsedBlocks() != 0 || len(s.nativeLoadPending) != 0 {
		t.Fatal("LOAD did not drain")
	}
}

func TestNativePhasesOutputAndKVCompeteForOneCopyResource(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	resource := s.fabric.resources["dram"]
	resource.BytesPerUS = 0.5
	s.fabric.resources["dram"] = resource
	e := phaseEngine(t, s)
	r := &sim.Request{ID: "copy-contention", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6, 7}, MaxOutputLen: 4}
	outputStart, sampleEnd := int64(-1), int64(-1)
	active := ""
	e.EventObserver = func(event sim.Event, before bool) {
		if x, ok := event.(*sim.NativeAsyncEvent); ok && !before {
			if x.BatchID == 2 && x.Name == "native_phase_end" && x.Phase == "sample" {
				sampleEnd = x.At
			}
			if x.BatchID == 2 && x.Name == "native_phase_start" && x.Phase == "output_copy" {
				outputStart = x.At
			}
			if x.Resource == "copy" {
				if x.Name == "native_phase_start" {
					if active != "" {
						t.Fatal("overlapping use of shared copy resource", active, x.Phase)
					}
					active = x.Phase
				}
				if x.Name == "native_phase_end" {
					if active != x.Phase {
						t.Fatal("copy resource completion mismatch")
					}
					active = ""
				}
			}
		}
	}
	e.InjectArrival(r)
	e.Run()
	storeStart, storeEnd := int64(-1), int64(-1)
	for _, x := range h.records {
		if x.Source == "hbm" {
			if x.Name == "transfer_start" && storeStart < 0 {
				storeStart = x.Time
			}
			if x.Name == "transfer_end" && storeEnd < 0 {
				storeEnd = x.Time
			}
		}
	}
	if !(storeStart < sampleEnd && sampleEnd < storeEnd && outputStart >= storeEnd) {
		t.Fatal("KV transfer did not delay output copy", storeStart, sampleEnd, storeEnd, outputStart)
	}
	if active != "" || s.fabric.Pending() != 0 || s.UsedBlocks() != 0 || e.Metrics.CompletedRequests != 1 {
		t.Fatal("shared execution failed to drain")
	}
}

func TestNativePhasesCancelBeforeHostSubmissionPreservesCapturedWork(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	e := phaseEngine(t, s)
	r := &sim.Request{ID: "cancel-phase", InputTokens: []sim.TokenID{1, 2, 3, 4, 5}, OutputTokens: []sim.TokenID{6}, MaxOutputLen: 1, Deadline: 10}
	protected := false
	e.EventObserver = func(event sim.Event, before bool) {
		if !before && event.Timestamp() == 10 && r.State == sim.StateTimedOut {
			protected = s.nativeBatchPins > 0 && s.UsedBlocks() > 0
		}
	}
	e.InjectArrival(r)
	e.Run()
	stores := 0
	for _, x := range h.records {
		if x.Name == "transfer_start" && x.Source == "hbm" {
			stores++
			if x.Time < 202 {
				t.Fatal("cancelled submitted STORE bypassed its device fence")
			}
		}
	}
	if !protected || stores != 1 || r.TTFTSet || s.UsedBlocks() != 0 || s.nativeBatchPins != 0 || s.fabric.Pending() != 0 {
		t.Fatal("cancelled native phases lost ownership", s.PeerSnapshot())
	}
}
