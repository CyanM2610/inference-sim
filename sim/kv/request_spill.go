package kv

import "github.com/inference-sim/inference-sim/sim"

func (s *PeerCache) SupportsRequestSpill() bool {
	m := s.fabric.mechanisms
	return s.fabric.native == nil && s.fabric.external == nil && !m.ReserveAtDispatch && m.Online == nil && len(s.access) > 0
}

// SpillPrefill never releases the request. All prefix destination reservations
// are obtained before the first new DMA; existing copies are temporarily pinned
// against eviction during this transaction. Ordinary pool eviction resumes on
// return. The caller must attach preemption fences before releasing source refs.
func (s *PeerCache) SpillPrefill(req *sim.Request, pool string) sim.RequestSpillOutcome {
	if !s.SupportsRequestSpill() || req == nil || req.State != sim.StateRunning || req.TTFTSet || req.EmittedTokens() != 0 || req.ProgressIndex <= 0 || req.ProgressIndex >= req.InputLen() {
		panic("spill requires a running, completed-chunk prefill")
	}
	var path []string
	for _, access := range s.access {
		if access.Pool == pool {
			path = access.WritePath
			break
		}
	}
	if path == nil {
		panic("spill selected an inaccessible pool")
	}
	blocks := s.spillBlocks(req.ID, req.ProgressIndex)
	out := sim.RequestSpillOutcome{Request: req.ID, Pool: pool, Status: "spill_ready", TailTokens: req.ProgressIndex % s.BlockSizeTokens}
	p := s.fabric.pools[pool]
	var protected []*peerEntry
	var missing []*KVBlock
	for _, b := range blocks {
		if e := p.entries[b.Hash]; e != nil {
			e.readers++
			protected = append(protected, e)
			if e.ready {
				out.ReadyBlocks++
			} else {
				out.PendingBlocks++
			}
		} else {
			missing = append(missing, b)
		}
	}
	releaseTargets := func() {
		for _, e := range protected {
			e.readers--
		}
		protected = nil
	}
	defer releaseTargets()
	out = s.commitSpill(req, pool, path, out, missing)
	releaseTargets()
	// Native STORE construction touches the request's existing prefix copies
	// after its full reservation succeeds. Temporary transaction protection must
	// not hide those idle copies from the ordinary pool access policy.
	if out.NewBlocks > 0 {
		s.touchRequestPools(req.ID, req.FullInputTokens())
	}
	return out
}

// Shared by execution and observation: never include allocated-but-uncomputed
// blocks, and never equate a hash assigned at allocation with published KV.
func (s *PeerCache) spillBlocks(id string, computed int64) []*KVBlock {
	n := computed / s.BlockSizeTokens
	ids := s.RequestMap[id]
	if computed < 0 {
		panic("negative spill prefix")
	}
	if int64(len(ids)) < n {
		panic("spill prefix exceeds request allocation")
	}
	blocks := make([]*KVBlock, 0, n)
	for _, id := range ids[:n] {
		b := s.Blocks[id]
		if b.RefCount <= 0 || b.Hash == "" || s.ready[id] != b.Hash || int64(len(b.Tokens)) != s.BlockSizeTokens {
			panic("spill encountered unpublished or incomplete KV")
		}
		blocks = append(blocks, b)
	}
	return blocks
}

func (s *PeerCache) spillTargets(id string, computed int64) []sim.DecisionSpillTarget {
	blocks := s.spillBlocks(id, computed)
	keys := map[string]bool{}
	for _, b := range blocks {
		keys[b.Hash] = true
	}
	var targets []sim.DecisionSpillTarget
	for _, access := range s.access {
		p := s.fabric.pools[access.Pool]
		target := sim.DecisionSpillTarget{Pool: access.Pool, CompleteBlocks: int64(len(blocks)), TailTokens: computed % s.BlockSizeTokens,
			AvailableBlocks: p.config.CapacityBlocks - int64(len(p.entries))}
		for _, b := range blocks {
			e := p.entries[b.Hash]
			if e == nil {
				target.MissingBlocks++
			} else if e.ready {
				target.ReadyBlocks++
			} else {
				target.PendingBlocks++
			}
		}
		for key, e := range p.entries {
			if !keys[key] && e.ready && e.readers == 0 {
				target.AvailableBlocks++
			}
		}
		targets = append(targets, target)
	}
	return targets
}

func (s *PeerCache) commitSpill(req *sim.Request, pool string, path []string, out sim.RequestSpillOutcome, missing []*KVBlock) sim.RequestSpillOutcome {
	p := s.fabric.pools[pool]
	// Capacity may change as a pool policy declines eviction. Keep every new
	// reservation unsubmitted until the whole request prefix is accepted.
	var reserved []*peerEntry
	for _, b := range missing {
		e, fresh := s.fabric.reserve(pool, b.Hash, b.Tokens, s.clock)
		if e == nil {
			for _, entry := range reserved {
				delete(p.entries, entry.hash)
				s.fabric.emit(PeerRecord{Time: s.clock, Name: "l2_reservation_cancel", Instance: s.id, Request: req.ID,
					Destination: pool, Hash: entry.hash, Bytes: s.fabric.bytes, Reason: "request_spill_capacity"})
			}
			out.Status = "recompute_capacity"
			s.emitSpillOutcome(out)
			return out
		}
		if !fresh {
			panic("spill destination changed during synchronous reservation")
		}
		reserved = append(reserved, e)
	}
	if s.fabric.mechanisms.GroupTransfers {
		s.fabric.beginTransferGroup()
	}
	for i, b := range missing {
		s.storeReservedCopy(b, pool, req.ID, true, path, reserved[i], true)
		out.NewBlocks++
	}
	if s.fabric.mechanisms.GroupTransfers {
		s.fabric.endTransferGroup(s.clock)
	}
	if out.PendingBlocks+out.NewBlocks > 0 {
		out.Status = "spill_pending"
	}
	s.emitSpillOutcome(out)
	return out
}

func (s *PeerCache) emitSpillOutcome(out sim.RequestSpillOutcome) {
	s.fabric.emit(PeerRecord{Time: s.clock, Name: "request_spill", Instance: s.id, Request: out.Request,
		Source: "hbm", Destination: out.Pool, Reason: out.Status, Counters: map[string]int64{
			"ready_blocks": out.ReadyBlocks, "pending_blocks": out.PendingBlocks,
			"new_blocks": out.NewBlocks, "tail_tokens": out.TailTokens}})
}
