package kv

import (
	"reflect"
	"testing"
)

func admissionFixture(t *testing.T) (*HotPrefixPolicy, PoolAdmissionContext) {
	t.Helper()
	p, err := NewHotPrefixPolicy(HotPrefixConfig{AgingIntervalRequests: 16, AdmissionThreshold: 2, HBMEvictionUnit: "block", HBMScore: "clock",
		Benefit: &HotPrefixBenefitConfig{HorizonUS: 1000, DecayUS: 1000}, AdmissionCost: &HotPrefixAdmissionConfig{Rule: "cost_next"}}, 16)
	if err != nil {
		t.Fatal(err)
	}
	p.remember([]string{"incoming"})
	p.published("incoming")
	for i := 0; i < 10; i++ {
		p.nodes["incoming"].references.completed(1000, *p.config.Benefit)
	}
	s := &PlacementCostSnapshot{NowUS: 1000, HBM: map[string]int{"incoming": 1}, Host: map[string]PlacementHostCopy{},
		HostCapacity: 10, RestoreWindow: 100, model: costFixture(t)}
	return p, PoolAdmissionContext{Hash: "incoming", CapacityBlocks: 10, Costs: s}
}

func TestCostAdmissionWeighsProbabilityStoreAndActualNativeLoad(t *testing.T) {
	p, c := admissionFixture(t)
	policy := hotPrefixPoolPolicy{state: p}
	d := policy.Admit(c)
	if !d.Accept || d.Victim != "" || d.AdmissionCost.OccupancyUS != 0 || d.AdmissionCost.Store.LoadedBlocks != 1 || d.AdmissionCost.Drop.ComputedTokens != 17 {
		t.Fatal("did not admit profitable free slot", d)
	}
	before := c.Costs.clone()
	c.Costs.model.readPerByte = 1000
	d = policy.Admit(c)
	if d.Accept || d.AdmissionCost.Store.LoadedBlocks != 1 || d.AdmissionCost.ExpectedGainUS != 0 {
		t.Fatal("pretended native could choose cheaper recompute", d)
	}
	if !reflect.DeepEqual(before.HBM, c.Costs.HBM) || !reflect.DeepEqual(before.Host, c.Costs.Host) {
		t.Fatal("prediction mutated cache copies")
	}
	c.Costs.model = costFixture(t)
	p.nodes["incoming"].references = hotPrefixReferenceHistory{}
	if policy.Admit(c).Accept {
		t.Fatal("zero observed reuse justified a store")
	}
}

func TestCostAdmissionThresholdObservationDoesNotChangeBaselineGate(t *testing.T) {
	p, c := admissionFixture(t)
	p.config.AdmissionCost.Rule = "threshold"
	d := (hotPrefixPoolPolicy{state: p}).Admit(c)
	if d.Accept || d.Reason != "frequency_below_threshold" || !d.AdmissionCost.CostAccept {
		t.Fatal(d)
	}
	p.config.AdmissionThreshold = 1
	if !(hotPrefixPoolPolicy{state: p}).Admit(c).Accept {
		t.Fatal("threshold=1 gate changed")
	}
	n := c.Costs.BufferedStoreBlocks
	c.Costs.BufferedStoreBlocks = 4
	e := p.assessAdmission(c)
	if n != 0 || e.CurrentStoreUS != c.Costs.model.storeService(5)-c.Costs.model.storeService(4) || e.CurrentStoreUS >= c.Costs.model.storeService(1) {
		t.Fatal("charged vector setup repeatedly", e)
	}
}

func TestCostAdmissionMissingAncestorDoesNotValueUnusableSuffix(t *testing.T) {
	p, c := admissionFixture(t)
	p.remember([]string{"root", "tail"})
	p.published("root")
	p.published("tail")
	p.nodes["tail"].references = p.nodes["incoming"].references
	c.Hash = "tail"
	c.Costs.HBM = map[string]int{"tail": 1}
	d := (hotPrefixPoolPolicy{state: p}).Admit(c)
	if d.Accept || d.AdmissionCost.ExpectedGainUS != 0 || d.AdmissionCost.Store.LoadedBlocks != 0 {
		t.Fatal(d)
	}
	c.Costs.Host["root"] = PlacementHostCopy{Ready: true}
	c.UsedBlocks = 1
	if !(hotPrefixPoolPolicy{state: p}).Admit(c).Accept {
		t.Fatal("recoverable ancestor failed to unlock suffix")
	}
}

func TestCostAdmissionFullPoolChargesReplacementAndRespectsHBMCopy(t *testing.T) {
	p, c := admissionFixture(t)
	p.remember([]string{"victim"})
	p.published("victim")
	c.CapacityBlocks = 1
	c.UsedBlocks = 1
	c.Costs.HostCapacity = 1
	c.Costs.Host["victim"] = PlacementHostCopy{Ready: true}
	c.Candidates = []PoolEvictionCandidate{{Hash: "victim"}}
	for i := 0; i < 20; i++ {
		p.nodes["victim"].references.completed(1000, *p.config.Benefit)
	}
	d := (hotPrefixPoolPolicy{state: p}).Admit(c)
	if d.Accept || d.AdmissionCost.OccupancyUS <= 0 || d.Victim != "" {
		t.Fatal("ignored full-pool opportunity", d)
	}
	p.nodes["victim"].references = hotPrefixReferenceHistory{}
	d = (hotPrefixPoolPolicy{state: p}).Admit(c)
	if !d.Accept || d.Victim != "victim" || d.AdmissionCost.OccupancyUS != 0 {
		t.Fatal("wrong zero-history replacement", d)
	}
	p.nodes["victim"].references = p.nodes["incoming"].references
	c.Costs.HBM["victim"] = 1
	d = (hotPrefixPoolPolicy{state: p}).Admit(c)
	if !d.Accept || d.AdmissionCost.OccupancyUS != 0 {
		t.Fatal("charged loss despite READY HBM copy", d)
	}
	c.Candidates = nil
	d = (hotPrefixPoolPolicy{state: p}).Admit(c)
	if d.Accept || d.Reason != "no_evictable_copy" {
		t.Fatal("fabricated space", d)
	}
}

func TestCostAdmissionReplacementCannotBreakIncomingAncestorAndClaimGain(t *testing.T) {
	p, c := admissionFixture(t)
	p.remember([]string{"root", "tail"})
	p.published("root")
	p.published("tail")
	p.nodes["tail"].references = p.nodes["incoming"].references
	c.Hash = "tail"
	c.Costs.HBM = map[string]int{"tail": 1}
	c.Costs.Host = map[string]PlacementHostCopy{"root": {Ready: true}}
	c.UsedBlocks = 1
	c.CapacityBlocks = 1
	c.Costs.HostCapacity = 1
	c.Candidates = []PoolEvictionCandidate{{Hash: "root"}}
	d := (hotPrefixPoolPolicy{state: p}).Admit(c)
	if d.Accept || d.AdmissionCost.ExpectedGainUS != 0 {
		t.Fatal("ignored displaced ancestor", d)
	}
}

type constantSubmitPhaseFixture struct{ benefitPhaseFixture }

func (constantSubmitPhaseFixture) HostSubmitUS(string, int64) int64 { return 11 }

func TestCostAdmissionPendingCopyWaitAndStrictEquality(t *testing.T) {
	p, c := admissionFixture(t)
	c.Costs.WriteQueueUS = 10000
	d := (hotPrefixPoolPolicy{state: p}).Admit(c)
	if d.Accept || d.AdmissionCost.Store.PendingStoreWaitUS <= 0 {
		t.Fatal("unpublished STORE treated as READY", d)
	}
	c.Costs.WriteQueueUS = 0
	// Zero gain equals zero store fee: strict > must still reject.
	p.nodes["incoming"].references = hotPrefixReferenceHistory{}
	c.Costs.BufferedStoreBlocks = 1
	c.Costs.model.writePerByte = 0
	c.Costs.model.writeLatency = 0
	c.Costs.model.model = constantSubmitPhaseFixture{}
	d = (hotPrefixPoolPolicy{state: p}).Admit(c)
	if d.Accept || d.AdmissionCost.CurrentStoreUS != 0 || d.AdmissionCost.ExpectedGainUS != 0 {
		t.Fatal("zero gain equal to zero marginal fee was admitted", d)
	}
}

type explicitAdmissionPolicy struct {
	victim string
	accept bool
}

func (explicitAdmissionPolicy) Observe(PoolPolicyEvent) {}
func (explicitAdmissionPolicy) Victim([]PoolEvictionCandidate) string {
	panic("explicit victim must not be selected twice")
}
func (p explicitAdmissionPolicy) Admit(c PoolAdmissionContext) PoolAdmissionDecision {
	if len(c.Candidates) > 0 {
		c.Candidates[0].Hash = "mutated"
	}
	return PoolAdmissionDecision{Accept: p.accept, Victim: p.victim}
}

func TestAdmissionExplicitVictimValidatedBeforeMutation(t *testing.T) {
	for _, mode := range []string{"valid", "pinned", "unready", "missing", "free_capacity", "declined"} {
		t.Run(mode, func(t *testing.T) {
			f, _ := NewPeerFabric(nil, []PeerPoolConfig{{ID: "cpu", CapacityBlocks: 2}}, 100, nil)
			a, _ := f.reserve("cpu", "a", nil, 0)
			a.ready = true
			b, _ := f.reserve("cpu", "b", nil, 0)
			b.ready = true
			policy := explicitAdmissionPolicy{victim: "b", accept: true}
			switch mode {
			case "pinned":
				b.readers = 1
			case "unready":
				b.ready = false
			case "missing":
				policy.victim = "missing"
			case "free_capacity":
				f.pools["cpu"].config.CapacityBlocks = 3
			case "declined":
				policy.accept = false
			}
			f.pools["cpu"].policy = policy
			if mode != "valid" {
				defer func() {
					if recover() == nil {
						t.Error("invalid victim accepted")
					}
					if len(f.pools["cpu"].entries) != 2 || f.pools["cpu"].entries["a"] != a || f.pools["cpu"].entries["b"] != b {
						t.Error("invalid decision mutated pool")
					}
				}()
			}
			f.reserve("cpu", "new", nil, 1)
			if mode == "valid" && (f.pools["cpu"].entries["b"] != nil || f.pools["cpu"].entries["a"] != a || f.pools["cpu"].entries["new"] == nil) {
				t.Fatal("wrong explicit victim")
			}
		})
	}
}
