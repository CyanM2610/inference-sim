package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// Logical capacity promises use the client output limit, never actual future
// output. They do not allocate physical blocks or publish unfinished KV.
func (s *PeerCache) EnableDecodeCapacityReservation() error {
	if s.decodeTargets != nil || len(s.RequestMap) != 0 || s.fabric.Pending() != 0 {
		return fmt.Errorf("decode capacity reservation requires an idle store")
	}
	s.decodeTargets = map[string]int64{}
	return nil
}

func (s *PeerCache) DecodeCapacityReservationEnabled() bool { return s.decodeTargets != nil }

func (s *PeerCache) decodeCapacityReserved(except string) int64 {
	var total int64
	for id, target := range s.decodeTargets {
		if id != except {
			allocated := int64(len(s.RequestMap[id]) + len(s.holds[id]) + s.readPending[id])
			total += max(0, target-allocated)
		}
	}
	return total
}

func (s *PeerCache) SupportsCapacityReservationView() bool { return s.decodeTargets != nil }

// Shared pure requirement calculation for observation and actual reservation.
func (s *PeerCache) decodeCapacityRequirement(id string, inputTokens, limit int64, cached []int64) (int64, int64) {
	if limit <= 0 || inputTokens > math.MaxInt64-limit+1 {
		panic("decode capacity reservation requires a finite client output budget")
	}
	tokens := inputTokens + limit - 1
	target := (tokens-1)/s.BlockSizeTokens + 1
	if target > s.TotalBlocks {
		panic("declared request capacity exceeds HBM")
	}
	if previous, ok := s.decodeTargets[id]; ok && previous != target {
		panic("declared request capacity changed")
	}
	owned := int64(len(s.RequestMap[id]))
	if owned == 0 {
		owned = int64(len(cached) + s.readPending[id])
		for _, id := range cached {
			if !s.Blocks[id].InUse {
				owned--
			}
		}
	}
	needed := max(0, target-owned) + s.decodeCapacityReserved(id)
	return target, needed
}

func (s *PeerCache) reserveDecodeCapacity(req *sim.Request, end int64, cached []int64) bool {
	if s.decodeTargets == nil {
		return true
	}
	limit := int64(req.MaxOutputLen)
	target, needed := s.decodeCapacityRequirement(req.ID, req.InputLen(), limit, cached)
	if end > req.InputLen()+limit-1 {
		panic("allocation exceeds declared client output budget")
	}
	if needed > s.FreeBlockCnt {
		protect := map[int64]bool{}
		for _, id := range cached {
			protect[id] = true
		}
		s.noteAllocationPressure(req.ID, "decode_capacity_reservation", needed, s.FreeBlockCnt, protect)
		if s.readPending[req.ID] > 0 {
			s.noteAllocationWait(req.ID, "restore_in_flight")
		}
		return false
	}
	if _, exists := s.decodeTargets[req.ID]; !exists {
		s.decodeTargets[req.ID] = target
		s.fabric.emit(PeerRecord{Time: s.clock, Name: "decode_capacity_reserve", Instance: s.id, Request: req.ID,
			Counters: map[string]int64{"target_blocks": target, "client_output_limit": limit}})
	}
	return true
}

func (s *PeerCache) releaseDecodeCapacity(id string) {
	if target, exists := s.decodeTargets[id]; exists {
		delete(s.decodeTargets, id)
		s.fabric.emit(PeerRecord{Time: s.clock, Name: "decode_capacity_release", Instance: s.id, Request: id, Counters: map[string]int64{"target_blocks": target}})
	}
}
