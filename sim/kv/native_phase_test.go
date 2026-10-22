package kv

import (
	"fmt"
	"github.com/inference-sim/inference-sim/sim"
	"strings"
	"testing"
)

func TestNativePhasePromotionRunsWithoutRequestsAndPublishesAfterReceipt(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	input := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8}
	if err := s.ConfigureNativeHostGeometry(1, 64<<20); err != nil {
		t.Fatal(err)
	}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	for _, b := range s.Blocks {
		if b.Hash != "" {
			s.invalidate(b)
		}
	}
	e, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
		KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: 24, BlockSizeTokens: 2},
		BatchConfig:   sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: 16}}, s, nativeTestCost{})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.EnableNativeAsync(2, nativePageAccessCost{}); err != nil {
		t.Fatal(err)
	}
	h.records = nil
	programs, consumed, pendingPolls, receiptPolls := 0, 0, 0, 0
	cuda, reply, published := int64(-1), int64(-1), int64(-1)
	e.EventObserver = func(event sim.Event, before bool) {
		x, ok := event.(*sim.NativeAsyncEvent)
		if !ok || before {
			return
		}
		if x.Name == "native_execution_plan" {
			programs++
			if len(x.Shape) != 0 {
				t.Fatal("promotion fabricated model computation")
			}
		}
		if x.QueueDepth > 2 {
			t.Fatal("control polling exceeded native output queue depth")
		}
		if x.Name == "native_phase_end" && x.Phase == "poll" && reply < 0 {
			pendingPolls++
			if !s.NativeControlPending() {
				t.Fatal("in-flight promotion lost native scheduler keepalive")
			}
		}
		if x.Name == "native_phase_end" && x.Phase == "poll" && reply >= 0 && len(s.GetCachedBlocks(input)) == 0 {
			receiptPolls++
			if !s.NativeControlPending() {
				t.Fatal("claimed receipt lost keepalive before FIFO adoption")
			}
		}
		if x.Name == "native_transfer_plan" {
			objects := x.TransferShape.Objects
			if len(objects) != 2 || objects[0].StagingBytes+objects[1].StagingBytes != 400 || objects[0].KernelBytes+objects[1].KernelBytes != 300 {
				t.Fatal("promotion changed native whole-object/partial-kernel accounting", objects)
			}
		}
		if x.Name == "native_phase_end" && strings.HasSuffix(x.Phase, "/copy") {
			cuda = x.At
			if len(s.GetCachedBlocks(input)) != 0 || s.UsedBlocks() != 3 {
				t.Fatal("CUDA completion published or released targets before modeled terminal")
			}
		}
		if x.Name == "native_phase_end" && strings.HasSuffix(x.Phase, "/reply") {
			reply = x.At
			if len(s.GetCachedBlocks(input)) != 0 || s.UsedBlocks() != 3 {
				t.Fatal("terminal reply bypassed worker poll/consumption")
			}
		}
		if x.Name == "native_async_consumed" {
			consumed++
			if len(s.GetCachedBlocks(input)) == 3 {
				published = x.At
			}
		}
	}
	s.SchedulePromotion(0, input, 3)
	e.Run()
	if cuda < 0 || reply <= cuda || published <= reply || pendingPolls < 2 || receiptPolls < 1 || programs != consumed {
		t.Fatal("promotion did not keep polling through physical/model completion and FIFO adoption", cuda, reply, published, programs, consumed, pendingPolls)
	}
	if e.Metrics.CompletedRequests != 0 || e.Metrics.TotalOutputTokens != 0 || s.nativeScheduledStep != 0 ||
		s.fabric.Pending() != 0 || s.UsedBlocks() != 0 || s.NativeControlPending() || s.PeerSnapshot()["pending_reads"] != 0 {
		t.Fatal("promotion fabricated request work or retained ownership", s.PeerSnapshot())
	}
}

// Native request LOOKUP/LOAD owns its own destination until control receipt
// adoption has actually published the Promotion. Do not silently coalesce the
// new request with a speculative, still-unpublished target or free its copy.
func TestNativePhaseForegroundUsesIndependentTargetsUntilPromotionAdoption(t *testing.T) {
	for _, arrival := range []string{"submitted", "cuda_complete", "reply_before_adoption"} {
		t.Run(arrival, func(t *testing.T) {
			h, s := hostChunkHarness(t, 2, 4)
			input := []sim.TokenID{1, 2, 3, 4, 5}
			if err := s.ConfigureNativeHostGeometry(1, 64<<20); err != nil {
				t.Fatal(err)
			}
			seedPeer(t, s, "seed", input)
			drainHost(t, h, s)
			for _, b := range s.Blocks {
				if b.Hash != "" {
					s.invalidate(b)
				}
			}
			e, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
				KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: 24, BlockSizeTokens: 2},
				BatchConfig:   sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: 16}}, s, nativeTestCost{})
			if err != nil {
				t.Fatal(err)
			}
			if err = e.EnableNativeAsync(2, nativePageAccessCost{}); err != nil {
				t.Fatal(err)
			}
			h.records = nil
			request := &sim.Request{ID: "overlap", InputTokens: input, OutputTokens: []sim.TokenID{9}, MaxOutputLen: 1}
			injected := false
			e.EventObserver = func(event sim.Event, before bool) {
				x, ok := event.(*sim.NativeAsyncEvent)
				if !ok || before || injected {
					return
				}
				trigger := arrival == "submitted" && x.Name == "native_transfer_plan" ||
					arrival == "cuda_complete" && x.Name == "native_phase_end" && strings.HasSuffix(x.Phase, "/copy") ||
					arrival == "reply_before_adoption" && x.Name == "native_phase_end" && strings.HasSuffix(x.Phase, "/reply")
				if trigger {
					if !s.NativeControlPending() || len(s.GetCachedBlocks(input)) != 0 {
						t.Fatal("foreground fixture did not arrive during unpublished control ownership")
					}
					request.ArrivalTime = x.At
					injected = true
					e.InjectArrival(request)
				}
			}
			s.SchedulePromotion(0, input, 2)
			e.Run()
			loads := map[string]PeerRecord{}
			for _, row := range h.records {
				if row.Name == "transfer_start" && row.Destination == "hbm" {
					if _, duplicate := loads[row.Request]; duplicate {
						t.Fatal("unexpected duplicate LOAD", row)
					}
					loads[row.Request] = row
				}
			}
			promotion, foreground := loads["@promotion"], loads[request.ID]
			if !injected || len(loads) != 2 || len(promotion.HBMBlocks) != 2 || len(foreground.HBMBlocks) != 2 {
				t.Fatal("unpublished promotion incorrectly replaced an independent demand LOAD", loads)
			}
			for _, a := range promotion.HBMBlocks {
				for _, b := range foreground.HBMBlocks {
					if a == b {
						t.Fatal("concurrent controls borrowed the same target", a)
					}
				}
			}
			if request.State != sim.StateCompleted || request.EmittedTokens() != 1 || s.NativeControlPending() ||
				s.fabric.Pending() != 0 || s.UsedBlocks() != 0 || s.nativeBatchPins != 0 || len(s.restoring) != 0 {
				t.Fatal("overlapping control/request did not complete and drain", s.PeerSnapshot())
			}
		})
	}
}

func TestNativePhasePromotionDuringWorkerSubmissionSurvivesForegroundCancellation(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	input := []sim.TokenID{1, 2, 3, 4, 5}
	if err := s.ConfigureNativeHostGeometry(1, 64<<20); err != nil {
		t.Fatal(err)
	}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	for _, b := range s.Blocks {
		if b.Hash != "" {
			s.invalidate(b)
		}
	}
	e, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
		KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: 24, BlockSizeTokens: 2},
		BatchConfig:   sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: 16}}, s, nativeTestCost{})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.EnableNativeAsync(2, nativePageAccessCost{}); err != nil {
		t.Fatal(err)
	}
	h.records = nil
	foreground := &sim.Request{ID: "cancel", InputTokens: []sim.TokenID{11, 12, 13, 14, 15}, OutputTokens: []sim.TokenID{16, 17}, MaxOutputLen: 2, Deadline: 50}
	// One new prompt token makes both promoted blocks an APC prefix; native
	// scheduling still computes that last token for logits.
	later := &sim.Request{ID: "reuse", ArrivalTime: 10000, InputTokens: input, OutputTokens: []sim.TokenID{9}, MaxOutputLen: 1}
	e.InjectArrival(foreground)
	e.InjectArrival(later)
	// First metadata was captured at zero, but its worker post has not run.
	// Late promotion must belong to a subsequent control/worker iteration.
	s.SchedulePromotion(3, input, 2)
	s.SchedulePromotion(4, input, 2)
	promotionTx, producer := int64(0), int64(0)
	duplicateChecked := false
	e.EventObserver = func(event sim.Event, before bool) {
		x, ok := event.(*sim.NativeAsyncEvent)
		if !ok || before {
			return
		}
		if x.Name == "native_transfer_plan" && x.TransferShape.Direction == "load" {
			if promotionTx != 0 {
				t.Fatal("duplicate promotion or missed later HBM reuse", x.At, x.TransferShape)
			}
			promotionTx, producer = x.Transaction, x.ProducerBatch
		}
		if promotionTx != 0 && x.Name == "native_phase_end" && x.Phase == fmt.Sprintf("kv_%d/reply", promotionTx) {
			duplicateChecked = true
			if s.Promote(input, 2) != 0 {
				t.Fatal("physical completion allowed duplicate LOAD before adoption")
			}
		}
	}
	e.Run()
	used := 0
	for _, r := range h.records {
		if r.Name == "promotion_consumed" && r.Request == later.ID {
			used++
		}
	}
	if !duplicateChecked || producer <= 1 || used != 2 || foreground.State != sim.StateTimedOut || later.State != sim.StateCompleted ||
		e.Metrics.CompletedRequests != 1 || e.Metrics.TotalOutputTokens != 1 {
		t.Fatal("late control/cancellation/reuse failed", producer, used, foreground.State, later.State)
	}
	if s.fabric.Pending() != 0 || s.UsedBlocks() != 0 || s.NativeControlPending() || s.nativeBatchPins != 0 || len(s.restoring) != 0 {
		t.Fatal("promotion or cancelled request retained ownership", s.PeerSnapshot())
	}
}

func TestNativePhasePromotionRequiresTransferModelBeforeReservation(t *testing.T) {
	h, s := hostChunkHarness(t, 2, 4)
	input := []sim.TokenID{1, 2, 3, 4}
	seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	for _, b := range s.Blocks {
		if b.Hash != "" {
			s.invalidate(b)
		}
	}
	phaseEngine(t, s) // Legacy aggregate copy costs do not define the extension.
	if s.Promote(input, 2) != 0 || s.UsedBlocks() != 0 || s.fabric.Pending() != 0 || s.NativeControlPending() {
		t.Fatal("unsupported promotion changed reservations before rejecting")
	}
}

// Artificial service values deliberately make output ready before the main
// stream tail. This is a causal-order test, not a fitted GPU timing model.
type nativePhaseTestCost struct{ nativeTestCost }

func (nativePhaseTestCost) ExecutionResources() []string { return []string{"cpu", "compute", "copy"} }
func (nativePhaseTestCost) CopyResource() string         { return "copy" }
func (nativePhaseTestCost) ExecutionPlan(shape []sim.NativeBatchShape) sim.NativeExecutionPlan {
	if len(shape) == 0 {
		return sim.NativeExecutionPlan{Stages: []sim.NativeStage{
			{Name: "kv_access", Stream: "host"},
			{Name: "post", Resource: "cpu", Stream: "host", DurationUS: 10},
			{Name: "producer", Stream: "main", After: []string{"post"}},
			{Name: "poll", Resource: "cpu", Stream: "host", DurationUS: 5, After: []string{"post"}},
		}, KVAccess: "kv_access", Computed: "post", ProducerReady: "producer", PostForward: "post", Poll: "poll", HostReturned: "poll", OutputReady: "poll"}
	}
	return sim.NativeExecutionPlan{Stages: []sim.NativeStage{
		{Name: "prepare", Resource: "cpu", Stream: "host", DurationUS: 10},
		{Name: "kv_access", Stream: "host", After: []string{"prepare"}},
		{Name: "forward", Resource: "compute", Stream: "main", DurationUS: 100, After: []string{"kv_access"}},
		{Name: "sample_submit", Resource: "cpu", Stream: "host", DurationUS: 5, After: []string{"kv_access"}},
		{Name: "sample", Resource: "compute", Stream: "main", DurationUS: 10, After: []string{"forward", "sample_submit"}},
		{Name: "output_copy", Resource: "copy", Stream: "output", DurationUS: 5, After: []string{"sample"}},
		{Name: "main_tail", Resource: "compute", Stream: "main", DurationUS: 80, After: []string{"sample"}},
		{Name: "post", Resource: "cpu", Stream: "host", DurationUS: 10, After: []string{"sample_submit"}},
		{Name: "poll", Resource: "cpu", Stream: "host", DurationUS: 5, After: []string{"post"}},
	}, KVAccess: "kv_access", Computed: "forward", ProducerReady: "main_tail", PostForward: "post", Poll: "poll", HostReturned: "poll", OutputReady: "output_copy"}
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
