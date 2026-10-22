package policylab

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func syntheticWorkerCosts(c *Config) {
	curve := func(value float64) []float64 { return []float64{value, 0, 0, 0, 40, 10} }
	x := EnginePhaseCurve{PreForward: curve(2), PostForward: curve(10), GPUReady: curve(100), OutputReady: curve(100), Poll: []float64{1, 0, 0, 0, 0, 0}, Tail: []float64{2, 0, 0, 0, 0, 0}}
	c.EnginePhases.WorkerMetadata = &WorkerMetadataCostConfig{Compute: x, SingleToken: x, MaxNewRequests: 2, Provenance: "synthetic worker fixture", Coverage: "functional only"}
}

func TestWorkerMetadataProfileRequiresMeasuredStateAndBranches(t *testing.T) {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	syntheticWorkerCosts(&c)
	if err := c.EnginePhases.Validate(); err != nil {
		t.Fatal(err)
	}
	work := []sim.BatchWork{{PrefixTokens: 32, NewTokens: 16}, {PrefixTokens: 0, NewTokens: 16}}
	fresh, err := c.EnginePhases.PredictEngineStepWithMetadata(work, []bool{false, true})
	if err != nil {
		t.Fatal(err)
	}
	resident, err := c.EnginePhases.PredictEngineStepWithMetadata(work, []bool{false, false})
	if err != nil || fresh.OutputReadyUS-resident.OutputReadyUS != 50 {
		t.Fatal(fresh, resident, err)
	}
	if _, err = c.EnginePhases.PredictEngineStepWithMetadata(work, nil); err == nil {
		t.Fatal("missing worker state accepted")
	}
	if _, err = c.EnginePhases.PredictEngineStepWithMetadata([]sim.BatchWork{{NewTokens: 1}}, []bool{true}); err == nil {
		t.Fatal("unmeasured single-token new worker accepted")
	}
	if _, err = c.EnginePhases.PredictEngineStepWithMetadata([]sim.BatchWork{{NewTokens: 1}}, []bool{false}); err != nil {
		t.Fatal(err)
	}
}

func TestWorkerMetadataPhaseActualRun(t *testing.T) {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	syntheticWorkerCosts(&c)
	c.Requests = []RequestConfig{{ID: "r", Input: make([]sim.TokenID, 32), Output: []sim.TokenID{1, 2}}}
	c.PrefillChunk = 16
	x := c.EnginePhases.WorkerMetadata.SingleToken
	c.EnginePhases.WorkerMetadata.SingleTokenWithNew = &x
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.FinishedUS) != 1 {
		t.Fatal("request did not complete")
	}
	adds, continues := 0, 0
	for _, e := range r.Events {
		if e.Name == "engine_worker_metadata" {
			adds += int(e.Counters["new_requests"])
			continues += int(e.Counters["resident_requests"])
		}
	}
	if adds != 1 || continues < 1 {
		t.Fatal("worker metadata did not follow actual execution", adds, continues)
	}
	if r.HBM["instance_0"]["worker_resident_requests"] != 0 {
		t.Fatal("worker state did not drain")
	}
}

func TestWorkerMetadataEstimatorChargesFirstStepOnlyEvenAfterRestore(t *testing.T) {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	syntheticWorkerCosts(&c)
	m := &restoreEstimator{phase: *c.EnginePhases}
	r := sim.DecisionRequest{ID: "r", InputTokens: 48}
	v := sim.DecisionView{MaxBatchTokens: 32, PrefillChunk: 16, KV: sim.DecisionKVState{WorkerStateKnown: true, Requests: []sim.DecisionKVRequest{{ID: "r"}}}}
	fresh, err := m.computeUS(v, r, 16, nil)
	if err != nil {
		t.Fatal(err)
	}
	v.KV.Requests[0].WorkerResident = true
	resident, err := m.computeUS(v, r, 16, nil)
	if err != nil || fresh-resident != 50 {
		t.Fatal(fresh, resident, err)
	}
	v.KV.WorkerStateKnown = false
	if _, err := m.computeUS(v, r, 16, nil); err == nil {
		t.Fatal("unknown worker state predicted")
	}
}
