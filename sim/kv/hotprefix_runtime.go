package kv

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

type hotPrefixRuntime struct {
	policy        *HotPrefixPolicy
	pool          string
	prefills      map[string]bool
	parentHolds   map[string]int64
	wantPromotion bool
}

// ConfigureHotPrefix installs the coupled HBM/host rules before execution.
// The caller chooses request policy independently. Existing reserve, copy and
// adoption machinery remains the authority for capacity and ownership.
func (s *PeerCache) ConfigureHotPrefix(c HotPrefixConfig) error {
	if s.hotprefix != nil || len(s.fabric.stores) != 1 || len(s.access) != 1 || len(s.fabric.pools) != 1 ||
		len(s.ready) != 0 || len(s.RequestMap) != 0 || s.fabric.Pending() != 0 || s.fabric.native != nil || s.fabric.external != nil {
		return fmt.Errorf("HotPrefix requires one unused abstract HBM instance and one host pool")
	}
	if s.fabric.mechanisms.BackgroundStoreMode != "on_reclaim" || s.fabric.mechanisms.Online != nil {
		return fmt.Errorf("HotPrefix requires on_reclaim storage without online promotion")
	}
	policy, err := NewHotPrefixPolicy(c, s.BlockSizeTokens)
	if err != nil {
		return err
	}
	pool := s.access[0].Pool
	if err := s.fabric.SetPoolEvictionPolicy(pool, hotPrefixPoolPolicy{state: policy}); err != nil {
		return err
	}
	s.hotprefix = &hotPrefixRuntime{policy: policy, pool: pool, prefills: map[string]bool{}, parentHolds: map[string]int64{}}
	s.policy = policy
	return nil
}

func (s *PeerCache) hotPrefixKeys(tokens []sim.TokenID) []string {
	var keys []string
	previous := ""
	for i := int64(0); i+s.BlockSizeTokens <= int64(len(tokens)); i += s.BlockSizeTokens {
		previous = s.hashBlock(previous, tokens[i:i+s.BlockSizeTokens])
		keys = append(keys, previous)
	}
	return keys
}

func (s *PeerCache) observeHotPrefix(req *sim.Request) {
	keys := s.hotPrefixKeys(req.FullInputTokens())
	matched := 0
	for _, hash := range keys {
		_, local := s.lookup[hash]
		e := s.fabric.pools[s.hotprefix.pool].entries[hash]
		if !local && (e == nil || !e.ready) {
			break
		}
		matched++
	}
	s.hotprefix.policy.observe(keys, matched)
	s.fabric.emit(PeerRecord{Time: s.clock, Name: "hotprefix_observe", Instance: s.id, Request: req.ID,
		Hashes: keys, Counters: map[string]int64{"matched_blocks": int64(matched), "request_serial": s.hotprefix.policy.requests}})
}

func (s *PeerCache) hotPrefixResidents() []string {
	var hashes []string
	for _, hash := range s.ready {
		hashes = append(hashes, hash)
	}
	sort.Strings(hashes)
	return hashes
}

func (s *PeerCache) hotPrefixRecord(name, request, hash, destination string) {
	n := s.hotprefix.policy.nodes[hash]
	if n == nil {
		panic("HotPrefix decision refers to unknown history")
	}
	s.fabric.emit(PeerRecord{Time: s.clock, Name: name, Instance: s.id, Request: request, Hash: hash,
		Destination: destination, Counters: map[string]int64{"frequency": n.frequency, "clock": n.clock,
			"depth": n.depth, "length_tokens": s.BlockSizeTokens, "hotness": n.frequency * n.clock}})
}

// afterHotPrefixPlacement charges declared planning service before taking the
// decision snapshot. With placement disabled the old phase callback is exact.
func (s *PeerCache) afterHotPrefixPlacement(now int64, work []sim.BatchWork, done func(int64)) {
	h := s.hotprefix
	if h == nil || !h.wantPromotion || s.readPending["@promotion"] > 0 {
		done(now)
		return
	}
	decode := false
	for _, w := range work {
		decode = decode || w.PrefixTokens >= w.Request.PrefillEnd() && w.NewTokens > 0
	}
	if !decode {
		done(now)
		return
	}
	finish := func(at int64) {
		s.clock = at
		s.executeHotPrefixPromotion()
		done(at)
	}
	cost := h.policy.config.PlannerUS
	if cost == 0 {
		finish(now)
		return
	}
	if now+cost < now {
		panic("HotPrefix planner clock overflow")
	}
	s.fabric.emit(PeerRecord{Time: now, Name: "hotprefix_planner_service", Instance: s.id, Duration: cost,
		Reason: "declared_not_native_calibrated"})
	s.fabric.phases.schedule(now+cost, finish)
}

func (s *PeerCache) executeHotPrefixPromotion() {
	h := s.hotprefix
	var host []string
	for key, e := range s.fabric.pools[h.pool].entries {
		if e.ready && !s.restoring[key] {
			host = append(host, key)
		}
	}
	var idle []PeerCandidate
	var empty int64
	for b := s.FreeHead; b != nil; b = b.NextFree {
		if s.ready[b.ID] != b.Hash || b.Hash == "" {
			empty++
		} else {
			idle = append(idle, PeerCandidate{ID: b.ID, Hash: b.Hash})
		}
	}
	// Foreground prefill reservations retain headroom even though they are not
	// physical refs yet. Current compute and pending copies are not on FreeHead.
	budget := min(h.policy.config.PromotionBlocks, max(0, s.FreeBlockCnt-s.otherPrefillReservations("")))
	plan := h.policy.planPromotions(s.hotPrefixResidents(), idle, host, empty, budget)
	h.wantPromotion = len(plan) > 0 // continue a budget-limited plan only on a later decode step
	s.fabric.emit(PeerRecord{Time: s.clock, Name: "hotprefix_promotion_plan", Instance: s.id,
		Counters: map[string]int64{"host_candidates": int64(len(host)), "submitted_blocks": int64(len(plan)), "block_budget": budget}})
	if len(plan) == 0 {
		return
	}
	protect := map[int64]bool{}
	victims := map[int64]bool{}
	for _, action := range plan {
		if action.victim >= 0 {
			victims[action.victim] = true
		}
	}
	for _, action := range plan {
		parent := h.policy.parent(action.hash)
		for id, hash := range s.ready {
			if hash == parent && !victims[id] {
				protect[id] = true
			}
		}
	}
	s.fabric.beginTransferGroup()
	defer s.fabric.endTransferGroup(s.clock)
	for _, action := range plan {
		e, access := s.find(action.hash)
		if e == nil || s.restoring[action.hash] {
			panic("HotPrefix source changed during atomic plan application")
		}
		if action.victim >= 0 {
			b := s.Blocks[action.victim]
			if b.RefCount != 0 || protect[b.ID] || s.ready[b.ID] != b.Hash {
				panic("HotPrefix selected protected promotion victim")
			}
			s.hotPrefixRecord("hotprefix_promotion_drop", "@promotion", b.Hash, "")
			s.emit("hbm_drop", "@promotion", b.Hash, "hbm", "", "hotprefix_promotion")
			s.invalidate(b) // Algorithm 2 discards cold GPU victims; no STORE here.
		}
		s.hotPrefixRecord("hotprefix_promote", "@promotion", action.hash, "hbm")
		parent := h.policy.parent(action.hash)
		if parent != "" {
			id, ok := s.lookup[parent]
			if !ok {
				panic("HotPrefix promotion lost its resident parent")
			}
			s.pin(s.Blocks[id])
			h.parentHolds[action.hash] = id
			s.emit("hotprefix_parent_pin", "@promotion", parent, "hbm", "", "pending_child")
		}
		if !s.fetch(action.hash, "@promotion", e, access, protect, true) {
			panic("HotPrefix reserved plan could not allocate its destination")
		}
	}
}

func (s *PeerCache) releaseHotPrefixParent(hash string) {
	if s.hotprefix == nil {
		return
	}
	if id, ok := s.hotprefix.parentHolds[hash]; ok {
		b := s.Blocks[id]
		delete(s.hotprefix.parentHolds, hash)
		s.unpin(b)
		s.emit("hotprefix_parent_release", "@promotion", b.Hash, "hbm", "", "child_adopted")
	}
}
