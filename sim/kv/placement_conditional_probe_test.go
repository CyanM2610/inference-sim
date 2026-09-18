package kv

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// An explicitly synthetic, shape-dependent model. The predictor composes these
// boundaries arithmetically; the peer engine schedules and executes the events.
type conditionalProbeModel struct{}

func (conditionalProbeModel) PredictEngineStep(work []sim.BatchWork) EngineStepTiming {
	if len(work) == 0 {
		return EngineStepTiming{PreForwardUS: 3, PostForwardUS: 7, PollUS: 5, TailUS: 4}
	}
	prefix, query := work[0].PrefixTokens, work[0].NewTokens
	return EngineStepTiming{PreForwardUS: 5, PostForwardUS: 20 + query, GPUReadyUS: 800 + 3*query + prefix/4,
		OutputReadyUS: 40 + 2*query + prefix/8, PollUS: 5, TailUS: 4}
}
func (conditionalProbeModel) HostSubmitUS(_ string, blocks int64) int64 { return 11 + 2*blocks }
func (conditionalProbeModel) PredictLoadPlanning(loads, blocks int64) (int64, error) {
	return 5 + 3*loads + blocks, nil
}

type conditionalProbeResult struct {
	Environment string              `json:"environment"`
	Branch      string              `json:"branch"`
	GPUWaitUS   int64               `json:"gpu_wait_us"`
	Predicted   PlacementAccessCost `json:"predicted"`
	ActualUS    int64               `json:"actual_us"`
	Compute     int64               `json:"actual_compute_tokens"`
	Steps       int64               `json:"actual_compute_steps"`
	LoadBlocks  int64               `json:"actual_load_blocks"`
	LoadGroups  int64               `json:"actual_load_groups"`
	Events      []PeerRecord        `json:"events"`
}

func conditionalProbeStep(t *testing.T, h *peerHarness, s *PeerCache, now int64, work []sim.BatchWork) int64 {
	t.Helper()
	end := int64(-1)
	if ok, err := s.BeginBatch(now+7, work, func(at int64) { end = at }); !ok || err != nil {
		t.Fatalf("conditional execution could not start: %v %v", ok, err)
	}
	for iterations := 0; end < 0; iterations++ {
		if iterations > 10000 {
			t.Fatal("conditional event execution failed to converge")
		}
		phaseNext(t, h)
	}
	s.clock = end
	return end
}

func conditionalProbeRead(t *testing.T, h *peerHarness, s *PeerCache, now, chunk int64, id string, tokens []sim.TokenID) (int64, int64, int64) {
	t.Helper()
	r := &sim.Request{ID: id, InputTokens: tokens}
	var computed, steps int64
	for iteration := 0; r.ProgressIndex < int64(len(tokens)); iteration++ {
		if iteration > 10000 {
			t.Fatal("conditional reader failed to converge")
		}
		s.clock = now
		cached := s.GetCachedBlocks(tokens)
		start := max(r.ProgressIndex, int64(len(cached))*s.BlockSizeTokens)
		query := min(chunk, int64(len(tokens))-start)
		if query <= 0 {
			t.Fatal("conditional plus-one input unexpectedly had a full hit")
		}
		if !s.AllocateKVBlocks(r, start, start+query, cached) {
			if h.f.Pending() == 0 {
				t.Fatal("reader blocked without pending transfer")
			}
			now = conditionalProbeStep(t, h, s, now, nil)
			continue
		}
		now = conditionalProbeStep(t, h, s, now, []sim.BatchWork{{Request: r, PrefixTokens: start, NewTokens: query}})
		r.ProgressIndex = start + query
		s.MirrorToCPU([]*sim.Request{r})
		computed += query
		steps++
	}
	s.ReleaseKVBlocks(r)
	assertPeerConservation(t, s)
	for _, b := range s.Blocks {
		if b.RefCount != 0 {
			t.Fatal("completed conditional reader retained a physical reference")
		}
	}
	if h.f.Pending() != 0 {
		t.Fatal("completed reader left pending copies")
	}
	for _, state := range h.f.Snapshot() {
		if state["reserved"] != 0 || state["read_pins"] != 0 {
			t.Fatal("completed conditional reader retained host reservations/readers")
		}
	}
	return now, computed, steps
}

func conditionalProbe(t *testing.T, blocks, window int64, queued bool, branch string) conditionalProbeResult {
	t.Helper()
	h := &peerHarness{}
	var err error
	h.f, err = NewPeerFabric([]PeerResource{{ID: "link", BytesPerUS: 24000, LatencyUS: 3}},
		[]PeerPoolConfig{{ID: "dram", CapacityBlocks: blocks + 4}}, 917504, func(r PeerRecord) { h.records = append(h.records, r) })
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewPeerCache("probe", blocks+2, 16, h.f,
		[]PeerAccess{{Pool: "dram", ReadPath: []string{"link"}, WritePath: []string{"link"}}}, BuiltinPeerPolicy{Name: "lru_drop"})
	if err != nil {
		t.Fatal(err)
	}
	s.BindEvents(func(e sim.Event) { h.events = append(h.events, e) }, func(int64) {})
	if err := h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "dram", BackgroundStoreMode: "on_reclaim", GroupTransfers: true, ConcurrentRestores: true, RestoreWindow: int(window)}); err != nil {
		t.Fatal(err)
	}
	if err := s.ConfigureEnginePhases(conditionalProbeModel{}, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.EnableRestoreDecisions(); err != nil {
		t.Fatal(err)
	}
	if err := s.EnableStoreSourceReuse(); err != nil {
		t.Fatal(err)
	}
	zero := int64(0)
	if err := s.ConfigureHotPrefix(HotPrefixConfig{AgingIntervalRequests: 16, AdmissionThreshold: 1, ShadowTTLUS: &zero,
		Benefit: &HotPrefixBenefitConfig{HorizonUS: 60000000, DecayUS: 60000000}}); err != nil {
		t.Fatal(err)
	}
	chunk := min(int64(32), blocks*16)
	if err := s.ConfigureBenefitCosts(chunk, 7); err != nil {
		t.Fatal(err)
	}
	tokens := make([]sim.TokenID, blocks*16+1)
	for i := range tokens {
		tokens[i] = sim.TokenID(i + 100)
	}
	now, _, _ := conditionalProbeRead(t, h, s, 0, chunk, "warmup", tokens)
	now = max(now, h.f.phases.lastGPUReady) + 100
	s.clock = now
	ids := s.GetCachedBlocks(tokens)
	if int64(len(ids)) != blocks {
		t.Fatal("actual warmup did not publish the complete prefix")
	}
	if branch == "dram" {
		h.f.beginTransferGroup()
		for _, id := range ids {
			if !s.store(s.Blocks[id], "dram", "warmup-store") {
				t.Fatal("warmup STORE was rejected")
			}
		}
		h.f.endTransferGroup(now)
		for h.f.Pending() != 0 {
			now = conditionalProbeStep(t, h, s, now, nil)
		}
	}
	if branch != "keep" {
		for _, id := range ids {
			if s.Blocks[id].RefCount != 0 {
				t.Fatal("attempt to release a protected warmup block")
			}
			s.invalidate(s.Blocks[id])
		}
	}
	if queued {
		now = conditionalProbeStep(t, h, s, now, []sim.BatchWork{{Request: &sim.Request{ID: "known-backlog"}, NewTokens: 1}})
	}
	s.clock = now
	snapshot := s.benefitCostSnapshot("reader")
	path := s.hotPrefixKeys(tokens)
	prediction := snapshot.access(path, 0, 16)
	if queued != (snapshot.GPUWaitUS > 0) {
		t.Fatal("GPU debt condition was not reached")
	}
	start, firstRecord := now, len(h.records)
	end, computed, steps := conditionalProbeRead(t, h, s, now, chunk, "reader", tokens)
	result := conditionalProbeResult{Environment: fmt.Sprintf("blocks%d-window%d-queued%v", blocks, window, queued),
		Branch: branch, GPUWaitUS: snapshot.GPUWaitUS, Predicted: prediction, ActualUS: end - start, Compute: computed, Steps: steps,
		Events: append([]PeerRecord(nil), h.records[firstRecord:]...)}
	for _, e := range result.Events {
		if e.Name == "transfer_staged" && e.Destination == "hbm" {
			result.LoadGroups++
			result.LoadBlocks += e.Bytes / h.f.bytes
		}
	}
	return result
}

func TestPlacementConditionalAccessAgainstExecutedReader(t *testing.T) {
	var rows []conditionalProbeResult
	for _, blocks := range []int64{1, 4} {
		for _, window := range []int64{1, 16} {
			for _, queued := range []bool{false, true} {
				for _, branch := range []string{"keep", "dram", "drop"} {
					name := fmt.Sprintf("blocks%d/window%d/queued%v/%s", blocks, window, queued, branch)
					t.Run(name, func(t *testing.T) {
						r := conditionalProbe(t, blocks, window, queued, branch)
						rows = append(rows, r)
						if r.ActualUS != r.Predicted.TotalUS || r.Compute != r.Predicted.ComputedTokens || r.Steps != r.Predicted.ComputeSteps ||
							r.LoadBlocks != r.Predicted.LoadedBlocks || r.LoadGroups != r.Predicted.LoadGroups {
							t.Errorf("conditional access differs: actual=%dus compute=%d/%d load=%d/%d predicted=%+v", r.ActualUS, r.Compute, r.Steps, r.LoadBlocks, r.LoadGroups, r.Predicted)
						}
					})
				}
			}
		}
	}
	if output := os.Getenv("KV_CONDITIONAL_PROBE_OUTPUT"); output != "" {
		data, err := json.MarshalIndent(rows, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		file, err := os.OpenFile(output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write(data); err != nil {
			t.Fatal(err)
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
