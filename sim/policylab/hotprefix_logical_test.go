package policylab

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestLogicalSegmentsNativeExecutionFencesAndDiagnostics(t *testing.T) {
	c := nativeLabConfig()
	c.HotPrefix.PromotionBlocks = 0
	c.HotPrefix.HBMEvictionUnit = "logical_segment"
	c.HotPrefix.AdmissionThreshold = 1
	c.Instances[0].HBMBlocks = 6
	c.Pools[0].CapacityBlocks = 32
	c.MaxSequences = 2
	c.MaxBatchTokens = 64
	c.PrefillChunk = 32
	c.StoreSourceReuse = true
	c.TraceBatchShapes = true
	c.Requests = nil
	for i := 0; i < 12; i++ {
		input := make([]sim.TokenID, 65)
		for j := range input {
			input[j] = sim.TokenID((i%4)*1000 + j)
		}
		c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i) * 100000, Input: input, Output: []sim.TokenID{9, 8}})
	}
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.HotPrefix.Diagnostics = &kv.HotPrefixDiagnosticsConfig{TraceCandidates: true, MeasureCPU: true}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.BatchShapes, b.BatchShapes) || !reflect.DeepEqual(a.FirstTokenUS, b.FirstTokenUS) || !reflect.DeepEqual(a.CacheMetrics, b.CacheMetrics) {
		t.Fatal("logical diagnostics changed execution")
	}
	if len(a.FinishedUS) != 12 || a.HBM["instance_0"]["total_refs"] != 0 || a.Pools["dram"]["reserved"] != 0 {
		t.Fatal("incomplete or ownership leaked")
	}
	groups, overshoot, edges := 0, int64(0), 0
	ends := map[int64]int64{}
	waiting := map[int64]bool{}
	credit := map[string]bool{}
	for _, e := range a.Events {
		switch e.Name {
		case "hotprefix_group_reclaim":
			if len(e.HBMBlocks) > 1 {
				groups++
			}
			overshoot += e.Counters["overshoot_blocks"]
		case "transfer_end":
			ends[e.Transaction] = e.Time
		case "hbm_reuse_dependency", "preemption_store_dependency":
			waiting[e.Transaction] = true
		case "engine_forward_ready":
			for tx := range waiting {
				at, ok := ends[tx]
				if !ok || at > e.Time {
					t.Fatal("group overwrite before STORE physical completion")
				}
				edges++
			}
			waiting = map[int64]bool{}
		case "hotprefix_reuse":
			key := e.Request + "/" + e.Hash
			if credit[key] {
				t.Fatal("group heat double credit")
			}
			credit[key] = true
		}
	}
	if groups == 0 || overshoot == 0 || edges == 0 || len(waiting) != 0 {
		t.Fatal("fixture lacks real whole groups, overshoot, or resolved STORE fences", groups, overshoot, edges)
	}
	variable := false
	for _, r := range b.HotPrefixDiagnostics["instance_0"].Records {
		if r.CandidateUnit != "logical_segment" {
			t.Fatal("ambiguous diagnostic unit")
		}
		for _, x := range r.Candidates {
			if x.LengthTokens > c.BlockTokens {
				variable = true
			}
			if int64(len(x.MemberBlocks))*c.BlockTokens != x.LengthTokens {
				t.Fatal("length not actual group size")
			}
		}
	}
	if !variable {
		t.Fatal("no variable length candidates")
	}
}
