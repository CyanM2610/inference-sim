package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestTailNativeRuntimeRespectsDeficitAndDiagnosticSelectionIdentity(t *testing.T) {
	c := benefitLabConfig()
	c.HotPrefix.Benefit = &kv.HotPrefixBenefitConfig{HorizonUS: 60_000_000, DecayUS: 60_000_000}
	c.HotPrefix.HBMScore = "benefit_next"
	c.HotPrefix.ReclaimMode = "deficit_tail"
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
		t.Fatal("tail run failed to drain")
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.FinishedUS, b.FinishedUS) {
		t.Fatal("tail diagnostics changed execution")
	}
	groups, truncated := 0, 0
	for _, e := range b.Events {
		if e.Name != "hotprefix_group_reclaim" {
			continue
		}
		groups++
		if e.Counters["selected_blocks"] > e.Counters["deficit_blocks"] || e.Counters["overshoot_blocks"] != 0 || e.Reason != "deficit_capped_contiguous_tail" {
			t.Fatal("tail runtime exceeded deficit", e)
		}
	}
	for _, r := range b.HotPrefixDiagnostics["instance_0"].Records {
		selected := 0
		for _, x := range r.Candidates {
			if int64(len(x.MemberBlocks)) > r.DeficitBlocks || !reflect.DeepEqual(x.MemberBlocks, x.ResidentMemberBlocks[:len(x.MemberBlocks)]) {
				t.Fatal("illegal tail candidate")
			}
			if reflect.DeepEqual(x.MemberBlocks, r.SelectedMemberBlocks) {
				selected++
				if len(x.MemberBlocks) < len(x.ResidentMemberBlocks) {
					truncated++
				}
			}
		}
		if !r.Declined && selected != 1 {
			t.Fatal("physical representative confused multiple lengths")
		}
	}
	if groups == 0 || truncated == 0 {
		t.Fatal("fixture missed real tail truncation", groups, truncated)
	}
}

func TestWholeReclaimModePreservesExistingBenefitExecution(t *testing.T) {
	c := benefitLabConfig()
	c.HotPrefix.Benefit = &kv.HotPrefixBenefitConfig{HorizonUS: 60_000_000, DecayUS: 60_000_000}
	c.HotPrefix.HBMScore = "benefit_next"
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	c.HotPrefix.ReclaimMode = "whole"
	b, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.FinishedUS, b.FinishedUS) || !reflect.DeepEqual(a.CacheMetrics, b.CacheMetrics) {
		t.Fatal("whole mode changed existing B execution")
	}
}
