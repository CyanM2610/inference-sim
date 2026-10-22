package policylab

import (
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

// requestScheduler only sees arrived requests in the waiting queue. EDF uses
// their supplied TTFT targets; it never reads actual future output lengths.
// Existing running requests and the core batch/preemption policy are unchanged.
type requestScheduler struct {
	name      string
	instance  string
	deadlines map[string]int64
	sink      func(kv.PeerRecord)
}

func (s *requestScheduler) OrderQueue(requests []*sim.Request, clock int64) {
	switch s.name {
	case "sjf":
		(&sim.SJFScheduler{}).OrderQueue(requests, clock)
	case "edf":
		deadline := func(r *sim.Request) int64 {
			if d, ok := s.deadlines[r.ID]; ok {
				return d
			}
			return math.MaxInt64
		}
		sort.SliceStable(requests, func(i, j int) bool {
			a, b := requests[i], requests[j]
			if deadline(a) != deadline(b) {
				return deadline(a) < deadline(b)
			}
			if a.ArrivalTime != b.ArrivalTime {
				return a.ArrivalTime < b.ArrivalTime
			}
			return a.ID < b.ID
		})
	}
	if s.sink != nil && len(requests) > 1 {
		ids := make([]string, len(requests))
		for i, r := range requests {
			ids[i] = r.ID
		}
		s.sink(kv.PeerRecord{Time: clock, Name: "request_queue_order", Instance: s.instance, Reason: s.name, Requests: ids})
	}
}
