package policylab

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func requireProfileWarning(t *testing.T, warnings []ProfileWarning, component, parameter string) ProfileWarning {
	t.Helper()
	for _, w := range warnings {
		if w.Component == component && w.Parameter == parameter && w.Observations > 0 {
			return w
		}
	}
	t.Fatalf("missing warning %s.%s: %+v", component, parameter, warnings)
	return ProfileWarning{}
}

func TestEightNewWorkersCompleteWithoutClippingMeasuredLimit(t *testing.T) {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	syntheticWorkerCosts(&c)
	c.MaxSequences, c.MaxBatchTokens, c.PrefillChunk = 8, 16, 2
	c.Requests = nil
	for i := 0; i < 8; i++ {
		c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), Input: []sim.TokenID{sim.TokenID(i), 99}, Output: []sim.TokenID{1}})
	}
	before, _ := json.Marshal(c)
	r, err := Run(c)
	if err != nil || len(r.FinishedUS) != 8 {
		t.Fatal("eight workers did not complete", err)
	}
	w := requireProfileWarning(t, r.ProfileWarnings, "worker_metadata", "new_requests")
	if w.ProfileMax != 2 || w.ObservedMax != 8 || r.ProfileCostCoverage == "" {
		t.Fatal("wrong profile warning", w)
	}
	actual := int64(0)
	for _, e := range r.Events {
		if e.Name == "engine_worker_metadata" {
			actual = max(actual, e.Counters["new_requests"])
		}
	}
	if actual != 8 {
		t.Fatal("requests were clipped to profile", actual)
	}
	after, _ := json.Marshal(c)
	if string(before) != string(after) {
		t.Fatal("runtime rewrote the caller's measured envelope")
	}
	// Altering only coverage metadata must not change the execution or cost.
	p := *c.EnginePhases.WorkerMetadata
	p.MaxNewRequests = 8
	c.EnginePhases.WorkerMetadata = &p
	inRange, err := Run(c)
	if err != nil || len(inRange.ProfileWarnings) != 0 || !reflect.DeepEqual(r.Events, inRange.Events) {
		t.Fatal("measured envelope affected execution", err)
	}
}

func TestFortyNineBlockRestoreCompletesWithOriginalProfile(t *testing.T) {
	c := labConfig(1, "dram")
	c.BlockTokens = 16
	c.Instances[0].HBMBlocks = 64
	c.Pools[0].CapacityBlocks = 128
	phaseTestConfig(&c)
	c.Mechanisms.RestoreWindow = 64
	c.EnginePhases.LoadPlanning = &EngineLoadPlanningCost{Coefficients: []float64{1, 2, 3}, MaxLoads: 3, MaxBlocks: 48, Provenance: "test load formula", Coverage: "synthetic"}
	request := func(id string, at int64, n, offset int) RequestConfig {
		x := RequestConfig{ID: id, At: at, Input: make([]sim.TokenID, n), Output: []sim.TokenID{1}}
		for i := range x.Input {
			x.Input[i] = sim.TokenID(i + offset)
		}
		return x
	}
	c.Requests = []RequestConfig{request("warm", 0, 785, 1000), request("evict", 100000, 1009, 2000), request("restore", 200000, 785, 1000)}
	r, err := Run(c)
	if err != nil || len(r.FinishedUS) != 3 {
		t.Fatal("large restore failed", err)
	}
	w := requireProfileWarning(t, r.ProfileWarnings, "load_planning", "blocks")
	if w.ProfileMax != 48 || w.ObservedMax != 49 || c.EnginePhases.LoadPlanning.MaxBlocks != 48 {
		t.Fatal("original profile boundary lost", w)
	}
	var charged bool
	for _, e := range r.Events {
		if e.Name == "engine_load_planning" && e.Counters["blocks"] == 49 {
			charged = e.Duration == 150
		}
	}
	if !charged || r.Counts["transfer_end"] != r.Counts["transfer_adopted"] || r.HBM["instance_0"]["total_refs"] != 0 || r.Pools["dram"]["reserved"] != 0 {
		t.Fatal("extrapolated work was uncharged or did not drain")
	}
	p := *c.EnginePhases.LoadPlanning
	p.MaxBlocks = 64
	c.EnginePhases.LoadPlanning = &p
	inRange, err := Run(c)
	if err != nil || len(inRange.ProfileWarnings) != 0 || !reflect.DeepEqual(r.Events, inRange.Events) {
		t.Fatal("coverage-only edit changed load execution", err)
	}
}

func TestServiceGeometryExtrapolatesUsingActualWorkAndKeepsBounds(t *testing.T) {
	c := budgetLabConfig()
	c.DecisionPolicy.RestoreServiceCost = serviceProfile(c)
	c.DecisionPolicy.RestoreServiceCost.RatesUS["load_windows"] = 3
	c.DecisionPolicy.RestoreServiceCost.Shape.RestoreWindowBlocks++
	c.profileWarnings = &profileWarnings{}
	m, err := newRestoreServiceCost(c)
	if err != nil || m.shape != serviceShape(c) {
		t.Fatal("runtime work shape was replaced by measured geometry", err)
	}
	requireProfileWarning(t, c.profileWarnings.snapshot(), "restore_service", "restore_window_blocks")
	for _, fixture := range []Config{queueCostFixture(1), decodeFillCostFixture()} {
		fixture.profileWarnings = &profileWarnings{}
		fixture.MaxSequences++
		if err := fixture.Validate(); err != nil {
			t.Fatal("profile geometry became scheduler capacity", err)
		}
		if len(fixture.profileWarnings.snapshot()) == 0 {
			t.Fatal("geometry extrapolation was silent")
		}
	}
}
