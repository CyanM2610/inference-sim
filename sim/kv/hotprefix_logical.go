package kv

// Logical segments are block-aligned, heat-coherent resident radix fragments.
// Persistent heat and shadow identity stay with each physical prefix hash. A
// segment does not sum its members' access counts or use cumulative depth as L.
type hotPrefixLogicalSegment struct {
	residentMembers []int64 // C diagnostic provenance; empty for unchanged whole mode
	benefit         *HotPrefixBenefitEstimate
	members         []PeerReclaimAction // tail to head; runtime preserves this order
	hashes          []string
	lru             int
	freq            int64
	clock           int64
	depth           int64
	score           float64
}

func (p *HotPrefixPolicy) segmentScore(freq, clock, length int64) float64 {
	switch p.config.HBMScore {
	case "frequency":
		return float64(freq)
	case "clock":
		return float64(clock)
	case "lru":
		return 0
	case "paper_fixed":
		length = p.config.LengthReferenceTokens
	}
	return float64(freq) + float64(clock)/float64(length)
}

func (p *HotPrefixPolicy) logicalSegments(c PeerReclaimContext) []hotPrefixLogicalSegment {
	parents := p.nonLeaves(c.ResidentHashes)
	copies := map[string]int{}
	for _, hash := range c.ResidentHashes {
		copies[hash]++
	}
	byHash := map[string]int{}
	for i, v := range c.Candidates {
		if copies[v.Hash] == 1 {
			byHash[v.Hash] = i
		}
	}
	var segments []hotPrefixLogicalSegment
	for i, leaf := range c.Candidates {
		if !p.eligible(leaf, parents, copies) {
			continue
		}
		n := p.nodes[leaf.Hash]
		g := hotPrefixLogicalSegment{lru: i, freq: n.frequency, clock: n.clock, depth: n.depth}
		index := i
		for {
			v := c.Candidates[index]
			a := PeerReclaimAction{BlockID: v.ID}
			if len(v.ReadyCopies) == 0 && copies[v.Hash] == 1 {
				for _, target := range c.Targets {
					if target.Available {
						a.Pool = target.Pool
						break
					}
				}
			}
			g.members = append(g.members, a)
			g.hashes = append(g.hashes, v.Hash)
			g.lru = min(g.lru, index)
			if copies[v.Hash] != 1 {
				break // duplicate copy is a separate, one-block physical action
			}
			parent := p.nodes[v.Hash].parent
			pn := p.nodes[parent]
			pi, idle := byHash[parent]
			if !idle || pn == nil || pn.children != 1 || pn.inputEndpoint ||
				pn.frequency != g.freq || pn.clock != g.clock {
				break
			}
			index = pi
		}
		g.score = p.segmentScore(g.freq, g.clock, int64(len(g.members))*p.blockTokens)
		segments = append(segments, g)
	}
	return segments
}

func (p *HotPrefixPolicy) chooseLogicalSegment(c PeerReclaimContext) PeerDecision {
	segments := p.logicalSegments(c)
	if p.config.ReclaimMode == "deficit_tail" {
		segments = tailOptions(segments, c.DeficitBlocks)
	}
	if p.config.Benefit != nil {
		if c.BenefitCosts == nil {
			panic("logical benefit choice lacks a current cost snapshot")
		}
		for i := range segments {
			segments[i].benefit = p.assessBenefit(segments[i], c.BenefitCosts)
			if p.config.HBMScore == "benefit_next" {
				segments[i].score = segments[i].benefit.ValueUSPerByte
			}
		}
	}
	p.lastSegments = segments
	best := -1
	for i, g := range segments {
		if best < 0 || g.score < segments[best].score || g.score == segments[best].score &&
			(g.lru < segments[best].lru || p.config.ReclaimMode == "deficit_tail" && g.lru == segments[best].lru && len(g.members) > len(segments[best].members)) {
			best = i
		}
	}
	if best < 0 {
		return PeerDecision{Decline: true}
	}
	g := segments[best]
	return PeerDecision{BlockID: g.members[0].BlockID, Pool: g.members[0].Pool, Reclaim: g.members}
}

func (p *HotPrefixPolicy) recordLogicalChoice(c PeerReclaimContext, r *HotPrefixChoice) {
	r.CandidateUnit = "logical_segment"
	if p.config.ReclaimMode == "deficit_tail" {
		r.CandidateUnit = "logical_tail"
	}
	r.PhysicalIdleBlocks = int64(len(c.Candidates))
	for _, g := range p.lastSegments {
		x := HotPrefixChoiceCandidate{BlockID: g.members[0].BlockID, Hash: g.hashes[0], LRUIndex: g.lru,
			Eligible: true, Frequency: g.freq, Clock: g.clock, LengthTokens: int64(len(g.members)) * p.blockTokens,
			Depth: g.depth, Score: g.score, MemberHashes: append([]string(nil), g.hashes...), Benefit: g.benefit}
		if len(g.residentMembers) > 0 {
			x.ResidentMemberBlocks = append([]int64(nil), g.residentMembers...)
			x.ResidentLengthTokens = int64(len(g.residentMembers)) * p.blockTokens
		}
		for _, a := range g.members {
			x.MemberBlocks = append(x.MemberBlocks, a.BlockID)
		}
		r.Candidates = append(r.Candidates, x)
	}
}
