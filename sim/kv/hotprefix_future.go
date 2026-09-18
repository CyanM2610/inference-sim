package kv

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// FuturePlacementRequest is privileged experiment input, never an online
// scheduler view. Outputs and policy-dependent future completion times are absent.
type FuturePlacementRequest struct {
	ID           string
	AtUS         int64
	AfterRequest string
	Input        []sim.TokenID
}

type futurePlacementDemand struct {
	request  string
	at       int64
	consumed bool
}

type hotPrefixFuture struct {
	byHash    map[string][]*futurePlacementDemand
	byRequest map[string]map[string]*futurePlacementDemand
	now       func() int64
}

func isFutureScore(mode string) bool {
	return mode == "oracle_next_arrival" || mode == "oracle_remaining"
}

// ConfigureHotPrefixFutureDemand cannot be installed on an online policy. It
// hashes inputs without calling remember: future structure must not split the
// online policy's legal logical segments or create hotness observations.
func (s *PeerCache) ConfigureHotPrefixFutureDemand(requests []FuturePlacementRequest) error {
	if s.hotprefix == nil || !isFutureScore(s.hotprefix.policy.config.HBMScore) ||
		s.hotprefix.policy.future != nil || len(s.hotprefix.requests) != 0 || len(requests) == 0 {
		return fmt.Errorf("future placement requires an unused explicitly named oracle")
	}
	f := &hotPrefixFuture{byHash: map[string][]*futurePlacementDemand{}, byRequest: map[string]map[string]*futurePlacementDemand{}, now: func() int64 { return s.clock }}
	for _, r := range requests {
		if r.ID == "" || r.AtUS < 0 || f.byRequest[r.ID] != nil {
			return fmt.Errorf("invalid or duplicate future request")
		}
		if r.AfterRequest != "" && s.hotprefix.policy.config.HBMScore == "oracle_next_arrival" {
			return fmt.Errorf("next-arrival oracle cannot use endogenous conversation arrivals")
		}
		f.byRequest[r.ID] = map[string]*futurePlacementDemand{}
		for _, h := range s.hotPrefixKeys(r.Input) {
			if f.byRequest[r.ID][h] != nil {
				continue
			}
			d := &futurePlacementDemand{request: r.ID, at: r.AtUS}
			f.byRequest[r.ID][h] = d
			f.byHash[h] = append(f.byHash[h], d)
		}
	}
	s.hotprefix.policy.future = f
	return nil
}

func (s *PeerCache) consumeHotPrefixFuture(hash, request string) {
	if s.hotprefix == nil || s.hotprefix.policy.future == nil {
		return
	}
	f := s.hotprefix.policy.future
	if d := f.byRequest[request][hash]; d != nil && !d.consumed {
		d.consumed = true
		if s.fabric != nil {
			s.fabric.emit(PeerRecord{Time: s.clock, Name: "offline_future_reference_consumed", Instance: s.id, Request: request, Hash: hash, Reason: "completed_input_use"})
		}
	}
}

// Smaller scores are evicted first. Negative delay means farther future use
// loses first; no remaining use receives a finite sentinel for JSON diagnostics.
// Counts are exact remaining input references, not exact attainable reuse.
func (p *HotPrefixPolicy) futureScore(hashes []string) float64 {
	f := p.future
	if f == nil {
		panic("oracle lacks explicit future demand installation")
	}
	if p.config.HBMScore == "oracle_remaining" {
		var count int64
		for _, h := range hashes {
			for _, d := range f.byHash[h] {
				if !d.consumed {
					count++
				}
			}
		}
		// Equal-size physical blocks: token demand / actual bytes is this mean
		// times a common positive constant, irrelevant to candidate ordering.
		return float64(count) / float64(len(hashes))
	}
	next := int64(math.MaxInt64)
	for _, h := range hashes {
		for _, d := range f.byHash[h] {
			if !d.consumed {
				next = min(next, d.at)
			}
		}
	}
	if next == math.MaxInt64 {
		return -math.MaxFloat64
	}
	return -float64(max(int64(0), next-f.now()))
}
