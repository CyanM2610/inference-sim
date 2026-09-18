package kv

import "testing"

func benefitPolicy(t *testing.T) *HotPrefixPolicy {
	p := logicalPolicy(t, "paper")
	p.config.HBMScore = "benefit_next"
	p.config.Benefit = &HotPrefixBenefitConfig{HorizonUS: 1000, DecayUS: 1000}
	return p
}

func benefitState(t *testing.T, c PeerReclaimContext) *PlacementCostSnapshot {
	s := &PlacementCostSnapshot{NowUS: 1000, HBM: map[string]int{}, Host: map[string]PlacementHostCopy{}, HostCapacity: 10,
		RestoreWindow: 100, model: costFixture(t)}
	for _, hash := range c.ResidentHashes {
		s.HBM[hash]++
	}
	return s
}

func TestBenefitDensityUsesForecastAndLegalPhysicalBytes(t *testing.T) {
	p := benefitPolicy(t)
	c := logicalContext(p, []string{"a", "ab"}, []string{"x", "xy"})
	c.BenefitCosts = benefitState(t, c)
	p.nodes["ab"].references.completed(1000, *p.config.Benefit)
	for i := 0; i < 3; i++ {
		p.nodes["xy"].references.completed(1000, *p.config.Benefit)
	}
	d := p.Choose(c)
	if len(d.Reclaim) != 2 || d.BlockID != 1 {
		t.Fatal("did not prefer colder forecast", d)
	}
	e := p.lastSegments[0].benefit
	if e.FreeBytes != 200 || e.Keep.ComputedTokens != 1 || e.Evict.ComputedTokens != 33 || e.CurrentStoreUS != 0 || e.NextMarginalUS <= 0 {
		t.Fatal("wrong physical or conditional cost accounting", e)
	}
	if c.BenefitCosts.HBM["a"] != 1 || len(c.BenefitCosts.Host) != 0 {
		t.Fatal("benefit evaluation mutated runtime snapshot")
	}
}

func TestBenefitDoesNotValueIsolatedSuffixUntilAncestorRecoverable(t *testing.T) {
	p := benefitPolicy(t)
	c := logicalContext(p, []string{"a", "ab", "abc"})
	c.Candidates = c.Candidates[1:]
	c.ResidentHashes = c.ResidentHashes[1:]
	c.BenefitCosts = benefitState(t, c)
	p.nodes["abc"].references.completed(1000, *p.config.Benefit)
	p.Choose(c)
	e := p.lastSegments[0].benefit
	if e.NextMarginalUS != 0 || e.ValueUSPerByte != 0 {
		t.Fatal("valued a suffix behind a real missing ancestor", e)
	}
	c.BenefitCosts.Host["a"] = PlacementHostCopy{Ready: true}
	p.Choose(c)
	e = p.lastSegments[0].benefit
	if e.Keep.LoadedBlocks != 1 || e.NextMarginalUS <= 0 {
		t.Fatal("failed to restore ancestor dependency", e)
	}
}

func TestBenefitHonorsNativeLoadEvenWhenRecomputeWouldBeFaster(t *testing.T) {
	p := benefitPolicy(t)
	c := logicalContext(p, []string{"a", "ab"})
	p.nodes["a"].inputEndpoint = true
	c.BenefitCosts = benefitState(t, c)
	c.BenefitCosts.Host["ab"] = PlacementHostCopy{Ready: true}
	c.BenefitCosts.model.readPerByte = 1000
	p.nodes["ab"].references.completed(1000, *p.config.Benefit)
	p.Choose(c)
	e := p.lastSegments[0].benefit
	if e.Evict.LoadedBlocks != 1 || e.Evict.ComputedTokens != 1 || e.NewStoreBlocks != 0 || e.Evict.TotalUS <= c.BenefitCosts.model.compute(16, 17).TotalUS {
		t.Fatal("used an unimplemented restore/recompute minimum", e)
	}
}

func TestBenefitSimulatesPartialAdmissionAndExcludesReservedVictims(t *testing.T) {
	p := benefitPolicy(t)
	c := logicalContext(p, []string{"a", "ab"})
	p.remember([]string{"cold"})
	p.published("cold")
	for _, n := range p.nodes {
		if n.depth > 0 && n != p.nodes["cold"] {
			n.frequency = 2
		}
	}
	c.Targets = []PeerTarget{{Pool: "dram", Available: true}}
	c.BenefitCosts = benefitState(t, c)
	c.BenefitCosts.HostCapacity = 1
	c.BenefitCosts.Host["cold"] = PlacementHostCopy{Ready: true}
	p.nodes["ab"].references.completed(1000, *p.config.Benefit)
	p.Choose(c)
	e := p.lastSegments[0].benefit
	if e.NewStoreBlocks != 1 || len(e.HostVictims) != 1 || len(e.StoreRejected) != 1 || e.Evict.LoadedBlocks != 0 || e.Evict.ComputedTokens != 33 {
		t.Fatal("wrong partial host prefix / reserved-copy eligibility", e)
	}
	if e.CurrentStoreUS != c.BenefitCosts.model.storeService(1) {
		t.Fatal("missing one vector store fee")
	}
	c.BenefitCosts.BufferedStoreBlocks = 1
	p.Choose(c)
	e = p.lastSegments[0].benefit
	if e.CurrentStoreUS != c.BenefitCosts.model.storeService(2)-c.BenefitCosts.model.storeService(1) {
		t.Fatal("repeated already buffered vector setup")
	}
}

func TestBenefitRedundantPhysicalCopyHasZeroMarginalLoss(t *testing.T) {
	p := benefitPolicy(t)
	c := logicalContext(p, []string{"a"})
	c.Candidates = append(c.Candidates, PeerCandidate{ID: 1, Hash: "a"})
	c.ResidentHashes = append(c.ResidentHashes, "a")
	c.BenefitCosts = benefitState(t, c)
	p.nodes["a"].references.completed(1000, *p.config.Benefit)
	p.Choose(c)
	for _, g := range p.lastSegments {
		if g.benefit.FreeBytes != 100 || g.benefit.ValueUSPerByte != 0 {
			t.Fatal("duplicate inflated loss or free bytes", g.benefit)
		}
	}
}

func TestBenefitPendingStoreCannotBeUsedBeforeProjectedReadiness(t *testing.T) {
	p := benefitPolicy(t)
	c := logicalContext(p, []string{"a"})
	s := benefitState(t, c)
	s.HBM["a"] = 0
	s.Host["a"] = PlacementHostCopy{Pinned: true, ReadyInUS: 1000}
	r := s.access([]string{"a"}, 100, 16)
	if r.PendingStoreWaitUS != 900 || r.LoadedBlocks != 1 || r.ComputedTokens != 1 {
		t.Fatal("pending STORE treated as READY", r)
	}
	clone := s.clone()
	clone.Host["a"] = PlacementHostCopy{Ready: true}
	clone.HBM["a"] = 1
	if s.Host["a"].Ready || s.HBM["a"] != 0 {
		t.Fatal("policy mutation escaped snapshot")
	}
}
