package kv

import "fmt"

// HotPrefixMemoryRoots retains bookkeeping independently of the physical cache
// and event engine. SharedModel must stay alive when measuring the incremental
// heap reachable from Metadata. Counts describe this terminal state, not peaks.
// These opaque roots are for heap measurement only, never policy decisions.
type HotPrefixMemoryRoots struct {
	Metadata    any
	SharedModel any
	Counts      map[string]int64
}

func (s *PeerCache) CaptureHotPrefixMemoryRoots() (HotPrefixMemoryRoots, error) {
	h := s.hotprefix
	if h == nil {
		return HotPrefixMemoryRoots{}, fmt.Errorf("HotPrefix memory capture requires HotPrefix")
	}
	if h.policy.diagnostics != nil && len(h.policy.diagnostics.Records) != 0 {
		return HotPrefixMemoryRoots{}, fmt.Errorf("HotPrefix memory capture excludes retained candidate traces")
	}
	c := map[string]int64{
		"history_nodes": int64(len(h.policy.nodes)), "request_records": int64(len(h.requests)),
		"shadows": int64(len(h.shadows)), "expiry_length": int64(len(h.expiry)), "expiry_capacity": int64(cap(h.expiry)),
		"prefills": int64(len(h.prefills)), "parent_holds": int64(len(h.parentHolds)),
		"last_options": int64(len(h.policy.lastSegments)), "last_options_capacity": int64(cap(h.policy.lastSegments)),
	}
	for _, r := range h.requests {
		c["credited_pairs"] += int64(len(r.credited))
		c["shadow_matches"] += int64(len(r.shadowMatches))
		c["pending_length"] += int64(len(r.pending))
		c["pending_capacity"] += int64(cap(r.pending))
	}
	for _, g := range h.policy.lastSegments {
		c["last_option_members"] += int64(len(g.members))
		c["last_option_members_capacity"] += int64(cap(g.members))
		c["last_option_hashes_capacity"] += int64(cap(g.hashes))
		c["last_option_resident_members_capacity"] += int64(cap(g.residentMembers))
	}
	r := HotPrefixMemoryRoots{Metadata: h, Counts: c}
	if s.fabric.phases != nil {
		r.SharedModel = s.fabric.phases.model
	}
	return r, nil
}
