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
	Candidates []PeerCandidate
	Targets    []PeerTarget
}
type PeerDecision struct {
	BlockID int64
	Pool    string
} // empty Pool means drop
type PeerPolicy interface {
	// Choose receives a detached candidate/target snapshot. The runtime retains
	// its own eligibility list; modifying this input cannot authorize a victim.
	Choose(PeerReclaimContext) PeerDecision
}

func cloneReclaimContext(c PeerReclaimContext) PeerReclaimContext {
	copy := PeerReclaimContext{
		Candidates: append([]PeerCandidate(nil), c.Candidates...),
		Targets:    append([]PeerTarget(nil), c.Targets...),
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
