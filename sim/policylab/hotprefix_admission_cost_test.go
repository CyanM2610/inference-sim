package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kv"
)

func admissionLabConfig() Config {
	c := benefitLabConfig()
	c.HotPrefix.HBMEvictionUnit = "block"
	c.HotPrefix.HBMScore = "clock"
	c.HotPrefix.Benefit = &kv.HotPrefixBenefitConfig{HorizonUS: 60_000_000, DecayUS: 60_000_000}
	return c
}

func TestCostAdmissionObservationPreservesThresholdNativeExecution(t *testing.T) {
	c := admissionLabConfig()
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.HotPrefix.AdmissionCost = &kv.HotPrefixAdmissionConfig{Rule: "threshold"}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	var filtered []kv.PeerRecord
	count := 0
	for _, e := range b.Events {
		if e.Name == "hotprefix_admission_cost" {
			count++
		} else {
			filtered = append(filtered, e)
		}
	}
	if count == 0 {
		t.Fatal("fixture did not reach admission")
	}
	if !reflect.DeepEqual(a.Events, filtered) || !reflect.DeepEqual(a.FirstTokenUS, b.FirstTokenUS) || !reflect.DeepEqual(a.FinishedUS, b.FinishedUS) || !reflect.DeepEqual(a.CacheMetrics, b.CacheMetrics) || !reflect.DeepEqual(a.BatchShapes, b.BatchShapes) || !reflect.DeepEqual(a.ProfileWarnings, b.ProfileWarnings) {
		t.Fatal("admission observation altered threshold baseline")
	}
}

func TestCostAdmissionNativeCompletesAndExecutesDeclaredGate(t *testing.T) {
	c := admissionLabConfig()
	c.HotPrefix.AdmissionCost = &kv.HotPrefixAdmissionConfig{Rule: "cost_next"}
	c.Pools[0].CapacityBlocks = 3
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.FinishedUS) != len(c.Requests) || a.HBM["instance_0"]["total_refs"] != 0 || a.Pools["dram"]["reserved"] != 0 {
		t.Fatal("cost admission left unfinished requests or reservations")
	}
	var pending *kv.PeerRecord
	count := 0
	for i := range a.Events {
		e := a.Events[i]
		if e.Name == "hotprefix_admission_cost" {
			if pending != nil {
				t.Fatal("unresolved admission prediction")
			}
			pending = &a.Events[i]
			count++
			f := e.AdmissionCost.Forecast
			if f.ObservedThroughUS != e.Time || f.LastReferenceUS > e.Time {
				t.Fatal("future forecast history")
			}
		}
		if e.Name == "l2_admission_decision" {
			if pending == nil || pending.Hash != e.Hash || pending.Time != e.Time {
				t.Fatal("decision lacked matching estimate")
			}
			if (e.Counters["accepted"] == 1) != pending.AdmissionCost.CostAccept {
				t.Fatal("executed a different cost gate")
			}
			pending = nil
		}
	}
	if pending != nil || count == 0 {
		t.Fatal("missing complete cost admission evidence")
	}
	c.HotPrefix.Diagnostics = &kv.HotPrefixDiagnosticsConfig{TraceCandidates: true, MeasureCPU: true}
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.FinishedUS, b.FinishedUS) {
		t.Fatal("diagnostics changed cost policy execution")
	}
}

func TestCostAdmissionRejectsUnsupportedGroupPrediction(t *testing.T) {
	c := admissionLabConfig()
	c.HotPrefix.AdmissionCost = &kv.HotPrefixAdmissionConfig{Rule: "cost_next"}
	c.HotPrefix.HBMEvictionUnit = "logical_segment"
	if _, err := Run(c); err == nil {
		t.Fatal("silently used single-block cost for whole-group action")
	}
	c.HotPrefix.HBMEvictionUnit = "block"
	c.HotPrefix.Benefit = nil
	if _, err := Run(c); err == nil {
		t.Fatal("cost admission accepted missing forecast")
	}
}
