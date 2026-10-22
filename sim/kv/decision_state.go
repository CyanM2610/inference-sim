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
	v := sim.DecisionKVState{PrefixStateKnown: true, TransfersKnown: s.fabric.phases != nil}
	for _, q := range queries {
		var keys []string
		previous := ""
		for i := int64(0); i+s.BlockSizeTokens < int64(len(q.Input)); i += s.BlockSizeTokens {
			previous = hash.HashBlock(previous, q.Input[i:i+s.BlockSizeTokens])
			keys = append(keys, previous)
		}
		local := 0
		for _, key := range keys {
			if _, ok := s.lookup[key]; !ok {
				break
			}
			local++
		}
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
		v.Requests = append(v.Requests, sim.DecisionKVRequest{ID: q.ID, LocalPrefixTokens: int64(local) * s.BlockSizeTokens,
			RecoverablePrefixTokens: int64(recoverable) * s.BlockSizeTokens, TransferPending: s.readPending[q.ID] > 0, Deferred: s.IsDeferred(q.ID)})
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
