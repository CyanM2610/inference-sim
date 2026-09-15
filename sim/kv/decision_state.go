package kv

import (
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// DecisionState is a pure snapshot. It neither touches recency nor accounts a
// policy hash as execution work. Device completion is deliberately not exposed:
// a copy stays pending until the scheduler has adopted it.
func (s *PeerCache) DecisionState(queries []sim.DecisionKVQuery) sim.DecisionKVState {
	return s.decisionState(queries, false)
}

func (s *PeerCache) DecisionStateWithSharing(queries []sim.DecisionKVQuery) sim.DecisionKVState {
	return s.decisionState(queries, s.fabric.phases != nil)
}

func (s *PeerCache) decisionState(queries []sim.DecisionKVQuery, sharing bool) sim.DecisionKVState {
	v := sim.DecisionKVState{PrefixStateKnown: true, TransfersKnown: s.fabric.phases != nil}
	if p := s.fabric.phases; p != nil {
		v.WorkerStateKnown = p.workerResidents != nil
	}
	v.PrefixSharingKnown = sharing
	for _, q := range queries {
		var keys []string
		previous := ""
		end := int64(len(q.Input)) - 1
		if s.restoreControl {
			end++
		}
		for i := int64(0); i+s.BlockSizeTokens <= end; i += s.BlockSizeTokens {
			previous = hash.HashBlock(previous, q.Input[i:i+s.BlockSizeTokens])
			keys = append(keys, previous)
		}
		prefix := s.localPrefix(q.ID, int64(len(q.Input)), keys)
		local := len(prefix.Blocks)
		recoverable := local
		for _, key := range keys[local:] {
			found := false
			for _, a := range s.access {
				if e := s.fabric.pools[a.Pool].entries[key]; e != nil && e.ready {
					found = true
					break
				}
			}
			if !found {
				break
			}
			recoverable++
		}
		limit, set := s.restoreLimits[q.ID]
		v.Requests = append(v.Requests, sim.DecisionKVRequest{ID: q.ID, LocalPrefixTokens: prefix.Tokens,
			LocalPrefixBlocks: int64(local), RecoverablePrefixBlocks: int64(recoverable), RestoreLimitSet: set, RestoreLimitBlocks: limit,
			RecoverablePrefixTokens: min(int64(recoverable)*s.BlockSizeTokens, max(0, int64(len(q.Input))-1)), TransferPending: s.readPending[q.ID] > 0, Deferred: s.IsDeferred(q.ID)})
		if v.WorkerStateKnown {
			v.Requests[len(v.Requests)-1].WorkerResident = s.fabric.phases.workerResidents[q.ID] != nil
		}
		if q.SpillComputedTokens > 0 {
			v.Requests[len(v.Requests)-1].SpillTargets = s.spillTargets(q.ID, q.SpillComputedTokens)
		}
		if q.CapacityReservation {
			if !s.SupportsCapacityReservationView() {
				panic("capacity reservation view requires an enabled backend")
			}
			_, needed := s.decodeCapacityRequirement(q.ID, q.InputTokens, q.ClientOutputLimit, prefix.Blocks)
			v.Requests[len(v.Requests)-1].CapacityReservation = &sim.DecisionCapacityReservation{
				NeededBlocks: needed, FreeBlocks: s.FreeBlockCnt, Fits: needed <= s.FreeBlockCnt}
		}
		if sharing {
			row := &v.Requests[len(v.Requests)-1]
			for _, key := range keys {
				row.PrefixBlocks = append(row.PrefixBlocks, sim.DecisionPrefixBlock{ID: key, LoadPending: s.restoring[key]})
			}
		}
	}
	for _, a := range s.access {
		p := s.fabric.pools[a.Pool]
		r := sim.DecisionPool{ID: a.Pool, CapacityBlocks: p.config.CapacityBlocks, FreeBlocks: p.config.CapacityBlocks - int64(len(p.entries))}
		for _, e := range p.entries {
			if e.ready {
				r.ReadyBlocks++
				if e.readers > 0 {
					r.ProtectedBlocks++
				}
			} else {
				r.ReservedBlocks++
			}
		}
		v.Pools = append(v.Pools, r)
	}
	if p := s.fabric.phases; p != nil {
		ids := make([]int64, 0, len(p.jobs))
		for id := range p.jobs {
			ids = append(ids, id)
		}
		sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
		for _, id := range ids {
			r := p.jobs[id].job.record
			v.PendingTransfers = append(v.PendingTransfers, sim.DecisionTransfer{ID: id, Request: r.Request, Source: r.Source, Destination: r.Destination, Bytes: r.Bytes})
		}
	}
	return v
}
