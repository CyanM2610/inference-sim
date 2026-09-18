package policylab

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func benefitLabConfig() Config {
	c := nativeLabConfig()
	c.HotPrefix.PromotionBlocks = 0
	c.HotPrefix.HBMEvictionUnit = "logical_segment"
	c.Instances[0].HBMBlocks = 6
	c.Pools[0].CapacityBlocks = 12
	c.MaxSequences = 2
	c.MaxBatchTokens = 64
	c.PrefillChunk = 32
	c.StoreSourceReuse = true
	c.TraceBatchShapes = true
	c.Requests = nil
	for i := 0; i < 16; i++ {
		input := make([]sim.TokenID, 65)
		for j := range input {
			input[j] = sim.TokenID((i%3)*1000 + j)
		}
		c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i) * 100000, Input: input, Output: []sim.TokenID{9, 8}})
	}
	return c
}

func TestBenefitHistoryIsObserverOnlyForExistingScoreAndDeduplicated(t *testing.T) {
	c := benefitLabConfig()
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.HotPrefix.Benefit = &kv.HotPrefixBenefitConfig{HorizonUS: 60_000_000, DecayUS: 60_000_000}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	var filtered []kv.PeerRecord
	seen := map[string]bool{}
	for _, e := range b.Events {
		if e.Name == "hotprefix_reference_completed" {
			key := e.Request + "/" + e.Hash
			if seen[key] {
				t.Fatal("repeated mirror/use doubled forecast observation")
			}
			seen[key] = true
		} else {
			filtered = append(filtered, e)
		}
	}
	if len(seen) == 0 {
		t.Fatal("no observed completed references")
	}
	if !reflect.DeepEqual(a.Events, filtered) || !reflect.DeepEqual(a.BatchShapes, b.BatchShapes) || !reflect.DeepEqual(a.FirstTokenUS, b.FirstTokenUS) || !reflect.DeepEqual(a.CacheMetrics, b.CacheMetrics) || !reflect.DeepEqual(a.ProfileWarnings, b.ProfileWarnings) {
		t.Fatal("forecast or hypothetical coverage calls changed baseline execution")
	}
}

func TestBenefitNativePolicyCompletesWithPastOnlyPredictionsAndDiagnosticTransparency(t *testing.T) {
	c := benefitLabConfig()
	c.HotPrefix.Benefit = &kv.HotPrefixBenefitConfig{HorizonUS: 1_000_000, DecayUS: 1_000_000}
	c.HotPrefix.HBMScore = "benefit_next"
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.HotPrefix.Diagnostics = &kv.HotPrefixDiagnosticsConfig{TraceCandidates: true, MeasureCPU: true}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.FinishedUS) != len(c.Requests) || a.HBM["instance_0"]["total_refs"] != 0 || a.Pools["dram"]["reserved"] != 0 {
		t.Fatal("incomplete or leaked benefit run")
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.FinishedUS, b.FinishedUS) {
		t.Fatal("benefit diagnostics changed policy behavior")
	}
	count := 0
	for _, r := range b.HotPrefixDiagnostics["instance_0"].Records {
		for _, candidate := range r.Candidates {
			e := candidate.Benefit
			if e == nil {
				t.Fatal("missing physical alternative estimate")
			}
			count++
			if e.Forecast.LastReferenceUS > r.TimeUS || e.Forecast.ObservedThroughUS != r.TimeUS || e.Forecast.NextProbability < 0 || e.Forecast.NextProbability > 1 {
				t.Fatal("future history or invalid probability", e.Forecast, r.TimeUS)
			}
			if e.FreeBytes != int64(len(candidate.MemberBlocks))*b.BlockBytes || candidate.Score != e.ValueUSPerByte {
				t.Fatal("wrong actual density score")
			}
		}
	}
	if count == 0 {
		t.Fatal("fixture lacked real benefit choices")
	}
}
