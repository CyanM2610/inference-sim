package kv

// PeerCandidate is a physical idle HBM block. A policy sees historical frequency,
// current copies, and isolated transfer estimates, never future requests.
type PeerCandidate struct {
	ID          int64
	Hash        string
	Frequency   int64
	ReadyCopies []string
}
type PeerTarget struct {
	Pool      string
	StoreUS   int64
	Available bool
	QueueUS   int64
}
type PeerReclaimContext struct {
	BenefitCosts   *PlacementCostSnapshot
	Candidates     []PeerCandidate
	Targets        []PeerTarget
	ResidentHashes []string // includes protected resident descendants
}
type PeerReclaimAction struct {
	BlockID int64
	Pool    string
}
type PeerDecision struct {
	BlockID int64
	Pool    string
	Decline bool                // no eligible victim; runtime waits instead of fabricating space
	Reclaim []PeerReclaimAction // when nonempty, apply the whole validated group
} // empty Pool means drop
type PeerPolicy interface {
	// Choose receives a detached candidate/target snapshot. The runtime retains
	// its own eligibility list; modifying this input cannot authorize a victim.
	Choose(PeerReclaimContext) PeerDecision
}

func validatedReclaimActions(d PeerDecision, c PeerReclaimContext) []PeerReclaimAction {
	if len(d.Reclaim) == 0 {
		return []PeerReclaimAction{{BlockID: d.BlockID, Pool: d.Pool}}
	}
	if d.BlockID != d.Reclaim[0].BlockID || d.Pool != d.Reclaim[0].Pool {
		panic("peer group representative disagrees with first action")
	}
	legal := map[int64]bool{}
	for _, v := range c.Candidates {
		legal[v.ID] = true
	}
	targets := map[string]bool{"": true}
	for _, t := range c.Targets {
		targets[t.Pool] = t.Available
	}
	seen := map[int64]bool{}
	for _, a := range d.Reclaim {
		if !legal[a.BlockID] || seen[a.BlockID] || !targets[a.Pool] {
			panic("peer policy selected an invalid reclaim group")
		}
		seen[a.BlockID] = true
	}
	return append([]PeerReclaimAction(nil), d.Reclaim...)
}

func cloneReclaimContext(c PeerReclaimContext) PeerReclaimContext {
	copy := PeerReclaimContext{
		BenefitCosts:   c.BenefitCosts.clone(),
		Candidates:     append([]PeerCandidate(nil), c.Candidates...),
		Targets:        append([]PeerTarget(nil), c.Targets...),
		ResidentHashes: append([]string(nil), c.ResidentHashes...),
	}
	for i := range copy.Candidates {
		copy.Candidates[i].ReadyCopies = append([]string(nil), c.Candidates[i].ReadyCopies...)
	}
	return copy
}

// BuiltinPeerPolicy is deliberately a small, inspectable baseline. Target order
// is supplied by topology access order; it is not a hidden DRAM->CXL hierarchy.
type BuiltinPeerPolicy struct {
	Name         string
	MinFrequency int64
	MaxStoreUS   int64
}

func (p BuiltinPeerPolicy) Choose(c PeerReclaimContext) PeerDecision {
	v := c.Candidates[0] // candidates are ordered by native HBM LRU
	if p.Name == "ready_first" || p.Name == "cost_aware" {
		for _, b := range c.Candidates {
			if len(b.ReadyCopies) > 0 {
				return PeerDecision{BlockID: b.ID}
			}
		}
	}
	d := PeerDecision{BlockID: v.ID}
	if p.Name == "lru_drop" || len(v.ReadyCopies) > 0 || v.Frequency < p.MinFrequency {
		return d
	}
	for _, t := range c.Targets {
		if t.Available && (p.MaxStoreUS == 0 || t.StoreUS+t.QueueUS <= p.MaxStoreUS) {
			d.Pool = t.Pool
			return d
		}
	}
	return d
}
