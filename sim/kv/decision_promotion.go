package kv

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// EnablePromotionDecisions uses the same staged whole-block copy backend as
// demand restores. It does not bypass host submission, DMA, poll or adoption.
func (s *PeerCache) EnablePromotionDecisions() error {
	return s.EnablePromotionDecisionsWithRetention("cache")
}

func (s *PeerCache) EnablePromotionDecisionsWithRetention(retention string) error {
	if retention != "cache" && retention != "request" {
		return fmt.Errorf("promotion retention must be cache or request")
	}
	if retention == "request" && !s.restoreControl {
		return fmt.Errorf("request-retained promotion requires restore control")
	}
	p := s.fabric.phases
	if p == nil || p.active || s.fabric.Pending() != 0 || s.promotionControl || len(s.access) == 0 {
		return fmt.Errorf("promotion control requires unused single-instance offload engine phases")
	}
	s.promotionControl = true
	s.promotionRetention = retention
	return nil
}

func (s *PeerCache) PromotionDecisionsEnabled() bool { return s.promotionControl }

func (s *PeerCache) PromotionRetention() string { return s.promotionRetention }

func (s *PeerCache) PromoteRequestPrefix(q sim.DecisionKVQuery, endpoint int64) sim.DecisionPromotionOutcome {
	if !s.promotionControl || q.ID == "" || endpoint <= 0 || endpoint > int64(max(0, len(q.Input)-1))/s.BlockSizeTokens {
		panic("invalid trusted prefix promotion")
	}
	s.observeCacheLookup(q.ID, q.Input)
	out := sim.DecisionPromotionOutcome{Request: q.ID, MaxPrefixBlocks: endpoint, Reason: "ready_or_pending"}
	if len(s.RequestMap[q.ID]) > 0 || (len(s.holds[q.ID]) > 0 && s.readPending[q.ID] == 0) {
		out.BlockedBlocks, out.Reason = endpoint, "request_has_owned_kv"
		return out
	}
	// Retain all currently resident prefix blocks while reserving the missing
	// suffix. Source read pins and target reservations are owned by fetch.
	protect := map[int64]bool{}
	keys := s.hashes(q.Input)[:endpoint]
	localPrefix := s.GetCachedBlocks(q.Input)
	if int64(len(localPrefix)) > endpoint {
		localPrefix = localPrefix[:endpoint]
	}
	if s.promotionRetention == "request" {
		return s.promoteRetainedPrefix(q, keys, localPrefix, out)
	}
	for _, key := range keys {
		if id, ok := s.lookup[key]; ok {
			protect[id] = true
		}
	}
	s.fabric.beginTransferGroup()
	defer s.fabric.endTransferGroup(s.clock)
	for i, key := range keys {
		if _, ok := s.lookup[key]; ok {
			out.ReadyBlocks++
			continue
		}
		if s.restoring[key] {
			out.PendingBlocks++
			continue
		}
		entry, access := s.find(key)
		if entry == nil {
			out.BlockedBlocks, out.Reason = endpoint-int64(i), "source_not_ready"
			break
		}
		if limit := s.fabric.mechanisms.MaxPromotionQueueUS; limit > 0 && s.fabric.EstimateQueueUS(s.clock, access.ReadPath) > limit {
			out.BlockedBlocks, out.Reason = endpoint-int64(i), "queue_budget"
			break
		}
		// Reclaim can write a victim into this same pool. Protect the read
		// source before it can evict/reuse the source slot; fetch adds its own
		// lasting pin only after target reservation succeeds.
		entry.readers++
		started := s.fetch(key, q.ID, entry, access, protect, true)
		entry.readers--
		if !started {
			out.BlockedBlocks, out.Reason = endpoint-int64(i), "hbm_capacity"
			break
		}
		out.StartedBlocks++
	}
	if out.StartedBlocks > 0 && out.BlockedBlocks == 0 {
		out.Reason = "submitted"
	}
	return out
}

// Request retention follows native allocation: discover a contiguous readable
// range, check that complete range against reservations, then acquire it. A
// missing source can shorten the range; capacity cannot strand half a load.
func (s *PeerCache) promoteRetainedPrefix(q sim.DecisionKVQuery, keys []string, local []int64, out sim.DecisionPromotionOutcome) sim.DecisionPromotionOutcome {
	if s.readPending[q.ID] == 0 {
		s.restoreLimits[q.ID] = out.MaxPrefixBlocks
	}
	out.ReadyBlocks = int64(len(local))
	missing := out.MaxPrefixBlocks - out.ReadyBlocks
	for _, key := range keys[len(local):] {
		if s.restoring[key] {
			out.PendingBlocks++
		}
	}
	if out.PendingBlocks > 0 {
		out.BlockedBlocks = missing - out.PendingBlocks
		if out.BlockedBlocks > 0 {
			out.Reason = "shared_prefix_pending"
		}
		s.waiting[q.ID] = true
		return out
	}
	if missing == 0 {
		return out
	}
	type page struct {
		key    string
		entry  *peerEntry
		access *PeerAccess
	}
	var pages []page
	for _, key := range keys[len(local):] {
		entry, access := s.find(key)
		if entry == nil {
			out.Reason = "source_not_ready"
			break
		}
		if limit := s.fabric.mechanisms.MaxPromotionQueueUS; limit > 0 && s.fabric.EstimateQueueUS(s.clock, access.ReadPath) > limit {
			out.Reason = "queue_budget"
			break
		}
		pages = append(pages, page{key, entry, access})
	}
	out.BlockedBlocks = missing
	if len(pages) == 0 {
		return out
	}
	needed := int64(len(pages))
	protect := map[int64]bool{}
	for _, id := range local {
		protect[id] = true
		if !s.Blocks[id].InUse {
			needed++
		}
	}
	request := &sim.Request{ID: q.ID, InputTokens: q.Input}
	if !s.fullPrefillFits(request, local) || needed > s.FreeBlockCnt-s.otherPrefillReservations(q.ID) {
		out.Reason = "hbm_capacity"
		return out
	}
	// Victim stores can target the same CPU pool. Protect every planned source
	// before reclaim, including later blocks in this grouped copy.
	for _, p := range pages {
		p.entry.readers++
	}
	defer func() {
		for _, p := range pages {
			p.entry.readers--
		}
	}()
	s.fabric.beginTransferGroup()
	defer s.fabric.endTransferGroup(s.clock)
	if !s.makeSpace(int64(len(pages)), protect, q.ID) {
		out.Reason = "hbm_capacity"
		return out
	}
	s.holdCachedPrefix(q.ID, local)
	s.prefillTargets[q.ID] = (int64(len(q.Input)) + s.BlockSizeTokens - 1) / s.BlockSizeTokens
	for _, p := range pages {
		if !s.fetchWithRetention(p.key, q.ID, p.entry, p.access, protect, true, true) {
			panic("prechecked promotion reservation failed")
		}
		out.StartedBlocks++
	}
	out.BlockedBlocks -= out.StartedBlocks
	if out.BlockedBlocks == 0 {
		out.Reason = "submitted"
	}
	return out
}
