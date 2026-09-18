package kv

import "github.com/inference-sim/inference-sim/sim"

func (s *PeerCache) ReadyDeferredRequests() []string {
	if !s.fabric.mechanisms.ConcurrentRestores {
		return nil
	}
	ready := map[string]bool{}
	for id, held := range s.holds {
		if len(held) > 0 && s.readPending[id] == 0 {
			ready[id] = true
		}
	}
	return sortedPeerIDs(ready)
}

// fullPrefillFits is an admission check, not an eager allocation of the whole
// prompt. Free cached hits consume capacity when pinned, whereas shared hits
// already in use do not. Actual future output length is never consulted.
func (s *PeerCache) fullPrefillFits(req *sim.Request, cached []int64) bool {
	full := (req.PrefillEnd() + s.BlockSizeTokens - 1) / s.BlockSizeTokens
	needed := full - int64(len(cached))
	for _, id := range cached {
		if !s.Blocks[id].InUse {
			needed++
		}
	}
	if needed > s.FreeBlockCnt {
		protect := map[int64]bool{}
		for _, id := range cached {
			protect[id] = true
		}
		s.noteAllocationPressure(req.ID, "full_prefill_capacity", needed, s.FreeBlockCnt, protect)
		s.emit("allocation_wait", req.ID, "", "", "", "full_prefill_capacity")
		return false
	}
	return true
}

func (s *PeerCache) otherPrefillReservations(req string) int64 {
	var reserved int64
	for id, target := range s.prefillTargets {
		if id != req {
			allocated := int64(len(s.RequestMap[id]) + len(s.holds[id]) + s.readPending[id])
			reserved += max(0, target-allocated)
		}
	}
	return reserved
}

// concurrentRestore returns true when compute admission must stop or defer.
// Inspect the entire range before mutating storage, so capacity rejection does
// not strand a partial restore. The existing allocator/transfer callbacks own
// physical blocks, pin counts and publication.
func (s *PeerCache) concurrentRestore(req *sim.Request, cached []int64, hashes []string) bool {
	if s.readPending[req.ID] > 0 {
		s.noteAllocationWait(req.ID, "restore_in_flight")
		return true
	}
	if len(cached) >= len(hashes) {
		return false
	}
	if len(s.access) > 0 && !s.directoryReady(req.ID) {
		s.noteAllocationWait(req.ID, "directory")
		return true
	}
	type page struct {
		hash   string
		entry  *peerEntry
		access *PeerAccess
	}
	var pages []page
	for _, h := range hashes[len(cached):] {
		if !s.nativeRestoreAdmission && len(pages) == max(1, s.fabric.mechanisms.RestoreWindow) {
			break
		}
		e, a := s.find(h)
		if e == nil {
			break
		}
		if s.restoring[h] {
			// Join the existing physical load by retrying its published prefix.
			// Do not duplicate its target allocation or increment its read count.
			s.waiting[req.ID] = true
			s.noteAllocationWait(req.ID, "shared_restore_in_flight")
			return true
		}
		pages = append(pages, page{h, e, a})
	}
	if len(pages) == 0 {
		return false
	}
	needed := int64(len(pages))
	protect := map[int64]bool{}
	for _, id := range cached {
		protect[id] = true
		if !s.Blocks[id].InUse {
			needed++
		}
	}
	reserved := s.otherPrefillReservations(req.ID)
	if s.nativeRestoreAdmission {
		s.fabric.emit(PeerRecord{Time: s.clock, Name: "restore_admission_check", Instance: s.id, Request: req.ID,
			Counters: map[string]int64{"lookup_blocks": int64(len(pages)), "needed_blocks": needed,
				"free_blocks": s.FreeBlockCnt, "other_reserved_blocks": reserved}})
	}
	if needed > s.FreeBlockCnt-reserved {
		s.noteAllocationWait(req.ID, "prefill_reservations")
		if s.nativeRestoreAdmission {
			s.allocationFailure.NeededBlocks = needed
			s.allocationFailure.ReservedBlocks = reserved
		}
		s.emit("allocation_wait", req.ID, "", "", "", "prefill_reservations")
		return true
	}
	// Reclamation can store victims into the same pool. Protect the planned
	// read sources while that may evict CPU entries; fetch takes its own pins.
	for _, p := range pages {
		p.entry.readers++
	}
	defer func() {
		for _, p := range pages {
			p.entry.readers--
		}
	}()
	if !s.makeSpace(int64(len(pages)), protect, req.ID) {
		return true
	}
	s.holdCachedPrefix(req.ID, cached)
	if s.restoreControl {
		s.restoredPrefixBlocks[req.ID] = max(s.restoredPrefixBlocks[req.ID], int64(len(cached)+len(pages)))
	}
	s.prefillTargets[req.ID] = (req.PrefillEnd() + s.BlockSizeTokens - 1) / s.BlockSizeTokens
	for _, p := range pages {
		if !s.fetch(p.hash, req.ID, p.entry, p.access, protect, false) {
			panic("prechecked restore reservation failed")
		}
		s.emit("allocation_wait", req.ID, p.hash, p.access.Pool, "hbm", "restore")
	}
	s.noteAllocationWait(req.ID, "restore_submitted")
	return true
}
