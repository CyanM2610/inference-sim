package kv

import "github.com/inference-sim/inference-sim/sim"

func (s *KVCacheState) LastAllocationFailure() sim.AllocationFailure { return s.allocationFailure }

func (s *KVCacheState) capacityFailure(req string, need int64) bool {
	s.allocationFailure = sim.AllocationFailure{Request: req, Kind: "capacity", Reason: "hbm_blocks", NeededBlocks: need, FreeBlocks: s.FreeBlockCnt}
	return false
}

func (s *PeerCache) LastAllocationFailure() sim.AllocationFailure { return s.allocationFailure }

// Only physical sources with exclusively STORE references can become free
// when the pending copies are adopted. Multiple pins promise one block.
func (s *PeerCache) reclaimingBlocks(protect map[int64]bool) int64 {
	var n int64
	for id, pins := range s.storePins {
		if pins > 0 && s.Blocks[id].RefCount == pins && !protect[id] {
			n++
		}
	}
	return n
}

func (s *PeerCache) noteAllocationPressure(req, reason string, need, free int64, protect map[int64]bool) {
	pending := s.reclaimingBlocks(protect)
	kind := "capacity"
	if pending > 0 && free+pending >= need {
		kind, reason = "wait", "reclaim_writeback"
	}
	s.allocationFailure = sim.AllocationFailure{Request: req, Kind: kind, Reason: reason, NeededBlocks: need, FreeBlocks: free, ReclaimingBlocks: pending}
}

func (s *PeerCache) noteAllocationWait(req, reason string) {
	s.allocationFailure = sim.AllocationFailure{Request: req, Kind: "wait", Reason: reason, FreeBlocks: s.FreeBlockCnt}
}
