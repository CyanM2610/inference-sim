package kv

import (
	"encoding/json"

	kvhash "github.com/inference-sim/inference-sim/sim/internal/hash"
)

// HotPrefixAdmissionWarmupConfig runs a common threshold until a current
// admission sees a full, published pool with at least one legal replacement.
// It changes policy state once; it never seeds or rewrites cache/history state.
type HotPrefixAdmissionWarmupConfig struct {
	AdmissionThreshold int64 `json:"admission_threshold"`
}

func (p *HotPrefixPolicy) admissionRule(c PoolAdmissionContext, e *HotPrefixAdmissionEstimate) (string, int64) {
	config := p.config.AdmissionCost
	rule, threshold := config.Rule, p.config.AdmissionThreshold
	if config.WarmupUntilFullReady == nil {
		return rule, threshold
	}
	e.Phase = "measurement"
	if !p.admissionWarmupFinished {
		fullReady := c.UsedBlocks == c.CapacityBlocks && len(c.Candidates) > 0
		for _, copy := range c.Costs.Host {
			fullReady = fullReady && copy.Ready
		}
		if fullReady {
			e.WarmupStateSHA256 = p.admissionStateDigest(c)
			e.WarmupReadyBlocks = c.UsedBlocks
			p.admissionWarmupFinished = true
		} else {
			e.Phase = "warmup"
			rule, threshold = "threshold", config.WarmupUntilFullReady.AdmissionThreshold
		}
	}
	e.Rule = rule
	if rule == "threshold" {
		e.ActiveThreshold = &threshold
	}
	return rule, threshold
}

// This is a deterministic digest of placement-visible state, not a restorable
// engine checkpoint. The experiment also compares the entire emitted event
// prefix and common cost configuration before trusting a shared warmup.
func (p *HotPrefixPolicy) admissionStateDigest(c PoolAdmissionContext) string {
	type node struct {
		Parent                      string
		Frequency, Clock, Depth     int64
		Children                    int
		InputEndpoint               bool
		ReferenceWeight             float64
		ReferenceLastUS, References int64
	}
	nodes := make(map[string]node, len(p.nodes))
	for hash, n := range p.nodes {
		nodes[hash] = node{n.parent, n.frequency, n.clock, n.depth, n.children, n.inputEndpoint,
			n.references.weight, n.references.lastUS, n.references.observations}
	}
	state := struct {
		Incoming      string
		RequestSerial int64
		BlockTokens   int64
		Costs         *PlacementCostSnapshot
		Candidates    []PoolEvictionCandidate
		Nodes         map[string]node
	}{c.Hash, p.requests, p.blockTokens, c.Costs, c.Candidates, nodes}
	encoded, err := json.Marshal(state)
	if err != nil {
		panic("invalid placement state at admission warmup boundary: " + err.Error())
	}
	return kvhash.SHA256Hex(encoded)
}
