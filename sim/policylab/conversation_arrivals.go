package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// Conversation dependencies are client arrival constraints, not policy-visible
// queued work. Each completed request can release at most one next turn.
func (c Config) validateConversationArrivals() error {
	byID := map[string]RequestConfig{}
	children := map[string]string{}
	for _, r := range c.Requests {
		byID[r.ID] = r
	}
	for _, r := range c.Requests {
		if r.ThinkTimeUS < 0 || (r.AfterRequest == "" && r.ThinkTimeUS != 0) {
			return fmt.Errorf("invalid think time for %s", r.ID)
		}
		if r.AfterRequest == "" {
			continue
		}
		if len(c.Instances) != 1 || c.Native != nil {
			return fmt.Errorf("conversation arrivals require the single-instance simulated execution path")
		}
		if _, ok := byID[r.AfterRequest]; !ok || r.AfterRequest == r.ID || r.At != 0 {
			return fmt.Errorf("invalid predecessor or absolute arrival for %s", r.ID)
		}
		if child, exists := children[r.AfterRequest]; exists {
			return fmt.Errorf("conversation branches at %s: %s and %s", r.AfterRequest, child, r.ID)
		}
		children[r.AfterRequest] = r.ID
	}
	// Each node has at most one predecessor; walking with colors is linear and
	// avoids stack growth for conversations with many turns.
	color := map[string]int{}
	for id := range byID {
		var path []string
		for current := id; current != "" && color[current] != 2; current = byID[current].AfterRequest {
			if color[current] == 1 {
				return fmt.Errorf("cyclic conversation dependency at %s", current)
			}
			color[current] = 1
			path = append(path, current)
		}
		for _, current := range path {
			color[current] = 2
		}
	}
	return nil
}

type conversationArrivals struct {
	config    map[string]RequestConfig
	requests  map[string]*sim.Request
	child     map[string]string
	released  map[string]bool
	deadlines map[string]int64
	err       error
}

func newConversationArrivals(c Config, requests []*sim.Request, config map[string]RequestConfig, deadlines map[string]int64) *conversationArrivals {
	s := &conversationArrivals{config: config, requests: map[string]*sim.Request{}, child: map[string]string{}, released: map[string]bool{}, deadlines: deadlines}
	for _, r := range requests {
		s.requests[r.ID] = r
	}
	for _, r := range c.Requests {
		if r.AfterRequest != "" {
			s.child[r.AfterRequest] = r.ID
		}
	}
	return s
}

func (s *conversationArrivals) complete(previous *sim.Request, at int64) []*sim.Request {
	id, exists := s.child[previous.ID]
	if !exists || s.err != nil {
		return nil
	}
	if previous.State != sim.StateCompleted || s.released[id] {
		s.err = fmt.Errorf("conversation predecessor %s did not complete once", previous.ID)
		return nil
	}
	r := s.config[id]
	if at < 0 || r.ThinkTimeUS > math.MaxInt64-at || r.TTFTSLOUS > math.MaxInt64-at-r.ThinkTimeUS {
		s.err = fmt.Errorf("conversation arrival/deadline overflows for %s", id)
		return nil
	}
	r.At = at + r.ThinkTimeUS
	s.config[id] = r // request timing consumers use the actual client arrival
	next := s.requests[id]
	next.ArrivalTime = r.At
	if r.TTFTSLOUS > 0 {
		s.deadlines[id] = r.At + r.TTFTSLOUS
	}
	s.released[id] = true
	return []*sim.Request{next}
}
