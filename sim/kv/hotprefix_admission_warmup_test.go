package kv

import (
	"reflect"
	"testing"
)

func TestAdmissionWarmupShadowEligibilityDoesNotLeakTargetThreshold(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		_, s := shadowFixture(t, nil)
		p := s.hotprefix.policy
		p.config.AdmissionCost = &HotPrefixAdmissionConfig{Rule: "cost_next", WarmupUntilFullReady: &HotPrefixAdmissionWarmupConfig{AdmissionThreshold: 1}}
		p.config.Benefit = &HotPrefixBenefitConfig{HorizonUS: 1000, DecayUS: 1000}
		if explicit {
			threshold := int64(2)
			p.config.ShadowThreshold = &threshold
		}
		p.remember([]string{"dropped"})
		p.published("dropped")
		s.shadowHotPrefixDrop("dropped", "writer")
		if (len(s.hotprefix.shadows) == 1) != explicit {
			t.Fatal("warmup Shadow eligibility used the future admission threshold")
		}
		p.admissionWarmupFinished = true
		p.remember([]string{"later"})
		p.published("later")
		s.shadowHotPrefixDrop("later", "writer")
		if _, exists := s.hotprefix.shadows["later"]; !exists {
			t.Fatal("measurement Shadow eligibility was lost")
		}
	}
}

func TestAdmissionWarmupUsesCommonThresholdUntilFullReadyAndNeverReverts(t *testing.T) {
	p, c := admissionFixture(t)
	p.config.AdmissionCost.WarmupUntilFullReady = &HotPrefixAdmissionWarmupConfig{AdmissionThreshold: 1}
	// The target cost gate rejects this zero-history incoming block. Warmup
	// must still execute the same real threshold=1 rule used by comparators.
	p.nodes[c.Hash].references = hotPrefixReferenceHistory{}
	policy := hotPrefixPoolPolicy{state: p}
	d := policy.Admit(c)
	if !d.Accept || d.AdmissionCost.CostAccept || d.AdmissionCost.Rule != "threshold" || d.AdmissionCost.Phase != "warmup" || *d.AdmissionCost.ActiveThreshold != 1 {
		t.Fatal("target rule leaked into common warmup", d)
	}
	p.remember([]string{"victim"})
	p.published("victim")
	c.CapacityBlocks, c.UsedBlocks, c.Costs.HostCapacity = 1, 1, 1
	c.Costs.Host["victim"] = PlacementHostCopy{}
	if policy.Admit(c).AdmissionCost.Phase != "warmup" {
		t.Fatal("reserved but unpublished copy ended warmup")
	}
	c.Costs.Host["victim"] = PlacementHostCopy{Ready: true, Pinned: true}
	if policy.Admit(c).AdmissionCost.Phase != "warmup" {
		t.Fatal("pool without an eligible victim ended warmup")
	}
	c.Costs.Host["victim"] = PlacementHostCopy{Ready: true}
	c.Candidates = []PoolEvictionCandidate{{Hash: "victim"}}
	before := c.Costs.clone()
	d = policy.Admit(c)
	if d.Accept || d.AdmissionCost.Rule != "cost_next" || d.AdmissionCost.Phase != "measurement" || len(d.AdmissionCost.WarmupStateSHA256) != 64 || d.AdmissionCost.WarmupReadyBlocks != 1 {
		t.Fatal("did not switch to target at legal full-pool boundary", d)
	}
	if !reflect.DeepEqual(before, c.Costs) {
		t.Fatal("warmup changed current cache state")
	}
	delete(c.Costs.Host, "victim")
	c.UsedBlocks, c.Candidates = 0, nil
	d = policy.Admit(c)
	if d.Accept || d.AdmissionCost.Rule != "cost_next" || d.AdmissionCost.WarmupStateSHA256 != "" || d.AdmissionCost.Phase != "measurement" {
		t.Fatal("target rule reverted or repeated its boundary after capacity became free", d)
	}
}

func TestAdmissionWarmupBoundaryDigestIgnoresTargetButCoversHeatAndCopies(t *testing.T) {
	var expected string
	for _, target := range []struct {
		rule      string
		threshold int64
	}{{"threshold", 1}, {"threshold", 2}, {"cost_next", 2}} {
		p, c := admissionFixture(t)
		p.config.AdmissionCost.Rule, p.config.AdmissionThreshold = target.rule, target.threshold
		p.config.AdmissionCost.WarmupUntilFullReady = &HotPrefixAdmissionWarmupConfig{AdmissionThreshold: 1}
		p.remember([]string{"victim"})
		p.published("victim")
		c.CapacityBlocks, c.UsedBlocks, c.Costs.HostCapacity = 1, 1, 1
		c.Costs.Host["victim"] = PlacementHostCopy{Ready: true}
		c.Candidates = []PoolEvictionCandidate{{Hash: "victim"}}
		d := (hotPrefixPoolPolicy{state: p}).Admit(c)
		if expected == "" {
			expected = d.AdmissionCost.WarmupStateSHA256
		}
		if d.AdmissionCost.WarmupStateSHA256 != expected || d.AdmissionCost.Rule != target.rule {
			t.Fatal("different target altered shared placement state")
		}
		p.nodes["victim"].references.completed(1000, *p.config.Benefit)
		if p.admissionStateDigest(c) == expected {
			t.Fatal("digest ignored forecast history")
		}
		p.nodes["victim"].references = hotPrefixReferenceHistory{}
		c.Costs.HBM["victim"] = 1
		if p.admissionStateDigest(c) == expected {
			t.Fatal("digest ignored physical copies")
		}
	}
}
