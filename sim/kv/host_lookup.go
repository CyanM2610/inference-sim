package kv

import "github.com/inference-sim/inference-sim/sim"

// Lookup results and ownership outlive scheduler-side result bookkeeping.
// A request owns only its unconsumed logical references; submitted LOADs own
// separate physical references and survive request cancellation/END.
type hostLookup struct {
	prefix    []*hostObject
	remaining map[string]*hostObject
	admitted  bool
}

func (s *PeerCache) ensureHostLookup(request string, input []sim.TokenID) *hostLookup {
	h := s.fabric.hostChunks
	if h == nil {
		return nil
	}
	if existing := h.lookups[request]; existing != nil {
		return existing
	}
	if h.ended[request] {
		return nil
	} // a late control message cannot resurrect END
	lookup := &hostLookup{remaining: map[string]*hostObject{}}
	// Native LOOKUP hashes from token zero. An earlier host miss terminates
	// the host prefix even when the GPU happens to contain those earlier tokens.
	for _, key := range s.hostInput(request, input).objects {
		object := h.objects[key]
		if object == nil {
			break
		}
		ready := true
		for _, entry := range object.entries {
			ready = ready && entry.ready
		}
		if !ready {
			break
		} // write-locked is a miss, not HIT_PENDING
		object.lookupReaders++
		for _, entry := range object.entries {
			entry.last = s.clock
		}
		lookup.prefix = append(lookup.prefix, object)
		lookup.remaining[key] = object
		s.hostOwnershipEvent("host_lookup_reserved", request, object, "lookup")
	}
	h.lookups[request] = lookup // retain an empty/miss result across retries too
	return lookup
}

func (s *PeerCache) hostOwnershipEvent(name, request string, object *hostObject, reason string) {
	s.fabric.emit(PeerRecord{Time: s.clock, Name: name, Instance: s.id, Request: request,
		Source: object.pool, HostObjects: []string{object.key}, HostChunkBlocks: s.fabric.hostChunks.blocks,
		Reason: reason, Counters: map[string]int64{"lookup_references": object.lookupReaders, "transfer_reservations": object.transferRefs}})
}

func (s *PeerCache) releaseHostLookupObject(request string, lookup *hostLookup, key, reason string) {
	object := lookup.remaining[key]
	if object == nil {
		return
	} // duplicate FREE/END is scoped and idempotent
	if object.lookupReaders <= 0 || s.fabric.hostChunks.objects[key] != object {
		panic("host lookup lost its owned source")
	}
	object.lookupReaders--
	delete(lookup.remaining, key)
	s.hostOwnershipEvent("host_lookup_released", request, object, reason)
}

func (s *PeerCache) admitHostLookup(request string) {
	if s.fabric.hostChunks == nil {
		return
	}
	lookup := s.fabric.hostChunks.lookups[request]
	if lookup == nil || lookup.admitted {
		return
	}
	lookup.admitted = true
	// All external work has completed, or native admission needed no LOAD.
	// Retire unused logical refs only; the session remains until request END.
	for _, object := range lookup.prefix {
		s.releaseHostLookupObject(request, lookup, object.key, "no_remaining_retrieve")
	}
}

func (s *PeerCache) endHostLookup(request string) {
	h := s.fabric.hostChunks
	if h == nil || h.ended[request] {
		return
	}
	h.ended[request] = true
	lookup := h.lookups[request]
	if lookup != nil {
		for _, object := range lookup.prefix {
			s.releaseHostLookupObject(request, lookup, object.key, "end_session")
		}
		delete(h.lookups, request)
	}
	// Native END unifies touches for retrieved and newly stored prompt keys.
	// This is recency bookkeeping, never a release of submitted physical pins.
	for _, key := range h.inputs[request].objects {
		if object := h.objects[key]; object != nil {
			for _, entry := range object.entries {
				if entry.ready {
					entry.last = s.clock
				}
			}
		}
	}
	delete(h.inputs, request)
}

// findForRestore reads this request's already completed lookup result. It does
// not opportunistically refresh a miss while waiting for HBM capacity.
func (s *PeerCache) findForRestore(request, key string, input []sim.TokenID) (*peerEntry, *PeerAccess) {
	if s.fabric.hostChunks == nil {
		return s.findForInput(request, key, input)
	}
	lookup := s.fabric.hostChunks.lookups[request]
	if lookup == nil || lookup.admitted {
		return nil, nil
	}
	entry, access := s.hostEntryForInput(request, key, input)
	if entry == nil || lookup.remaining[entry.object.key] != entry.object {
		return nil, nil
	}
	return entry, access
}

// Handoff is atomic with respect to reclamation: all physical source pins have
// been acquired by fetch before this releases one logical reference per object.
// The returned completion releases physical ownership only after the whole
// operation (including every pool) has drained.
func (s *PeerCache) handoffHostLookup(jobs []*peerJob) func(int64) {
	request := jobs[0].record.Request
	h := s.fabric.hostChunks
	lookup := h.lookups[request]
	seen := map[*hostObject]bool{}
	var objects []*hostObject
	for _, job := range jobs {
		object := job.hostObject
		if object == nil || h.objects[object.key] != object {
			panic("LOAD lost its immutable host source")
		}
		if !seen[object] {
			objects = append(objects, object)
			seen[object] = true
		}
	}
	// Explicit background promotion has its own immediate read reservation;
	// it must not borrow or retire a foreground request's lookup lease.
	leased := jobs[0].record.Reason == "restore" || jobs[0].nativeControl != nil
	if leased {
		for _, object := range objects {
			if lookup == nil || lookup.remaining[object.key] != object {
				panic("LOAD does not own its lookup reservation")
			}
		}
	}
	for _, object := range objects {
		object.transferRefs++
		if leased {
			s.releaseHostLookupObject(request, lookup, object.key, "transfer_handoff")
		}
		s.hostOwnershipEvent("host_transfer_reserved", request, object, "load")
	}
	if leased {
		// Complete chunks preceding the first LOAD are covered by HBM.
		for _, object := range lookup.prefix {
			if seen[object] {
				break
			}
			s.releaseHostLookupObject(request, lookup, object.key, "hbm_covered_prefix")
		}
	}
	return func(at int64) {
		s.clock = at
		for _, object := range objects {
			if object.transferRefs <= 0 || h.objects[object.key] != object {
				panic("unmatched host physical reference release")
			}
			object.transferRefs--
			s.hostOwnershipEvent("host_transfer_released", request, object, "load_complete")
		}
	}
}
