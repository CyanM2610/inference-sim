package kv

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// EnableRestoreDecisions opts into full CPU-prefix hits and explicit loading
// limits. Legacy APC and unsupported execution backends retain their semantics.
func (s *PeerCache) EnableRestoreDecisions() error {
	p := s.fabric.phases
	if p == nil || p.active || s.fabric.Pending() != 0 || !s.fabric.mechanisms.ConcurrentRestores || !s.fabric.mechanisms.GroupTransfers {
		return fmt.Errorf("restore control requires idle engine phases with grouped concurrent restores")
	}
	if s.restoreControl {
		return fmt.Errorf("restore control already enabled")
	}
	s.restoreControl = true
	s.restoreLimits = map[string]int64{}
	s.restoredPrefixBlocks = map[string]int64{}
	return nil
}

func (s *PeerCache) RestoreDecisionsEnabled() bool { return s.restoreControl }

func (s *PeerCache) PreservesOutputHistory() bool { return s.restoreControl }

// New loads on this backend reserve/submit first and defer compute until a
// later scheduler adoption. A failed lookup may instead fall back to compute.
func (s *PeerCache) RestoreDecisionsDeferCompute() bool {
	return s.restoreControl && s.fabric.phases != nil
}

func (s *PeerCache) ValidateRestoreDecisions(updates []sim.DecisionRestore) error {
	if len(updates) > 0 && !s.restoreControl {
		return fmt.Errorf("restore control is disabled")
	}
	seen := map[string]bool{}
	for _, u := range updates {
		if u.Request == "" || u.MaxPrefixBlocks < 0 || seen[u.Request] || len(s.RequestMap[u.Request]) > 0 {
			return fmt.Errorf("invalid restore update for %q", u.Request)
		}
		seen[u.Request] = true
		old, set := s.restoreLimits[u.Request]
		if s.readPending[u.Request] > 0 && (!set || old != u.MaxPrefixBlocks) {
			return fmt.Errorf("cannot change restore limit while request %q has pending loads", u.Request)
		}
	}
	return nil
}

func (s *PeerCache) ApplyRestoreDecisions(updates []sim.DecisionRestore) {
	for _, u := range updates {
		s.restoreLimits[u.Request] = u.MaxPrefixBlocks
	}
}

// restoreHashes uses complete input blocks, including the final prompt block.
// The applied endpoint only bounds loading; it does not invalidate local hits.
func (s *PeerCache) restoreHashes(req *sim.Request) []string {
	if !s.restoreControl {
		return s.hashes(req.FullInputTokens())
	}
	keys := s.fullPrefixHashes(req.PrefillTokens())
	if limit, set := s.restoreLimits[req.ID]; set && limit < int64(len(keys)) {
		keys = keys[:limit]
	}
	return keys
}

// The native offload lookup distinguishes an absent prefix from HIT_PENDING:
// an unpublished STORE in the contiguous lookup range defers the entire lookup,
// even if an earlier external block is ready. Do this before capacity checks so
// the lookup cannot reserve HBM or trigger a victim. Loads and already-running
// requests retain their own admission/ownership paths.
func (s *PeerCache) observeRestoreLookup(req *sim.Request, cached []int64) (known bool, readyBlocks int64, deferred bool) {
	if !s.restoreControl || len(s.RequestMap[req.ID]) > 0 || s.readPending[req.ID] > 0 {
		return false, 0, false
	}
	keys := s.restoreHashes(req)
	if len(cached) >= len(keys) {
		return true, 0, false
	}
	for _, key := range keys[len(cached):] {
		if ready, _ := s.find(key); ready != nil {
			readyBlocks++
			continue
		}
		for _, access := range s.access {
			if entry := s.fabric.pools[access.Pool].entries[key]; entry != nil && !entry.ready {
				s.waiting[req.ID] = true
				s.noteAllocationWait(req.ID, "restore_lookup_pending")
				s.emit("allocation_wait", req.ID, key, access.Pool, "hbm", "restore_lookup_pending")
				return true, 0, true
			}
		}
		// A true miss terminates the contiguous prefix; later pending stores
		// cannot make any of that suffix loadable for this request.
		break
	}
	return true, readyBlocks, false
}

func (s *PeerCache) fullPrefixHashes(tokens []sim.TokenID) []string {
	var keys []string
	previous := ""
	for i := int64(0); i+s.BlockSizeTokens <= int64(len(tokens)); i += s.BlockSizeTokens {
		previous = s.hashBlock(previous, tokens[i:i+s.BlockSizeTokens])
		keys = append(keys, previous)
	}
	return keys
}

// localPrefix consumes ready lookup entries only. A completed load receipt is
// needed to reuse a full prompt minus one token; plain APC still leaves the
// final block out. The receipt never makes a reserved/unadopted block visible.
func (s *PeerCache) localPrefix(id string, inputTokens int64, keys []string) sim.CachedPrefix {
	end := max(int64(0), (inputTokens-1)/s.BlockSizeTokens)
	if s.restoreControl && s.readPending[id] == 0 {
		end = max(end, s.restoredPrefixBlocks[id])
	}
	end = min(end, int64(len(keys)))
	var blocks []int64
	for _, key := range keys[:end] {
		bid, ok := s.lookup[key]
		if !ok {
			break
		}
		blocks = append(blocks, bid)
	}
	return sim.CachedPrefix{Blocks: blocks, Tokens: min(int64(len(blocks))*s.BlockSizeTokens, max(0, inputTokens-1))}
}

func (s *PeerCache) GetRequestCachedPrefix(req *sim.Request) sim.CachedPrefix {
	if s.batchPrefixes != nil && s.batchPrefixes.planning {
		return s.batchCachedPrefix(req)
	}
	if !s.restoreControl {
		blocks := s.GetCachedBlocks(req.FullInputTokens())
		return sim.CachedPrefix{Blocks: blocks, Tokens: int64(len(blocks)) * s.BlockSizeTokens}
	}
	return s.localPrefix(req.ID, req.PrefillEnd(), s.fullPrefixHashes(req.PrefillTokens()))
}
