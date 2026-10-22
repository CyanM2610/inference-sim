package policylab

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

func (r RequestConfig) hasArrivalDependencies() bool {
	return r.AfterRequest != "" || len(r.AfterRequests) > 0
}

func (r RequestConfig) arrivalPredecessors() []string {
	if r.AfterRequest != "" {
		return []string{r.AfterRequest}
	}
	return r.AfterRequests
}

// Arrival dependencies belong to the client, not to policy-visible queued work.
// after_request preserves the linear conversation contract; after_requests is an
// explicit fork/join whose think time starts after all predecessors complete.
func (c Config) validateConversationArrivals() error {
	byID := map[string]RequestConfig{}
	linearChildren := map[string]string{}
	children := map[string][]string{}
	remaining := map[string]int{}
	var ready []string
	for _, r := range c.Requests {
		byID[r.ID] = r
	}
	for _, r := range c.Requests {
		if r.AfterRequest != "" && len(r.AfterRequests) > 0 {
			return fmt.Errorf("choose after_request or after_requests for %s", r.ID)
		}
		if r.ThinkTimeUS < 0 || (!r.hasArrivalDependencies() && r.ThinkTimeUS != 0) {
			return fmt.Errorf("invalid think time for %s", r.ID)
		}
		predecessors := r.arrivalPredecessors()
		remaining[r.ID] = len(predecessors)
		if len(predecessors) == 0 {
			ready = append(ready, r.ID)
			continue
		}
		if len(c.Instances) != 1 || c.Native != nil {
			return fmt.Errorf("conversation arrivals require the single-instance simulated execution path")
		}
		seen := map[string]bool{}
		for _, parent := range predecessors {
			if _, ok := byID[parent]; !ok || parent == r.ID || seen[parent] || r.At != 0 {
				return fmt.Errorf("invalid predecessor or absolute arrival for %s", r.ID)
			}
			seen[parent] = true
			children[parent] = append(children[parent], r.ID)
		}
		if r.AfterRequest != "" {
			if child, exists := linearChildren[r.AfterRequest]; exists {
				return fmt.Errorf("conversation branches at %s: %s and %s; use after_requests for a fork", r.AfterRequest, child, r.ID)
			}
			linearChildren[r.AfterRequest] = r.ID
		}
	}
	// Kahn's traversal is linear and avoids recursion on long conversations.
	for i := 0; i < len(ready); i++ {
		for _, child := range children[ready[i]] {
			remaining[child]--
			if remaining[child] == 0 {
				ready = append(ready, child)
			}
		}
	}
	if len(ready) != len(c.Requests) {
		return fmt.Errorf("cyclic conversation dependency")
	}
	return nil
}

type conversationArrivals struct {
	config     map[string]RequestConfig
	requests   map[string]*sim.Request
	children   map[string][]string
	remaining  map[string]int
	lastFinish map[string]int64
	completed  map[string]bool
	deadlines  map[string]int64
	err        error
}

func newConversationArrivals(c Config, requests []*sim.Request, config map[string]RequestConfig, deadlines map[string]int64) *conversationArrivals {
	s := &conversationArrivals{
		config: config, requests: map[string]*sim.Request{}, children: map[string][]string{},
		remaining: map[string]int{}, lastFinish: map[string]int64{}, completed: map[string]bool{}, deadlines: deadlines,
	}
	for _, r := range requests {
		s.requests[r.ID] = r
	}
	for _, r := range c.Requests {
		predecessors := r.arrivalPredecessors()
		s.remaining[r.ID] = len(predecessors)
		for _, parent := range predecessors {
			s.children[parent] = append(s.children[parent], r.ID)
		}
	}
	return s
}

func (s *conversationArrivals) complete(previous *sim.Request, at int64) []*sim.Request {
	children := s.children[previous.ID]
	if len(children) == 0 || s.err != nil {
		return nil
	}
	if previous.State != sim.StateCompleted || s.completed[previous.ID] || at < 0 {
		s.err = fmt.Errorf("conversation predecessor %s did not complete once at a valid time", previous.ID)
		return nil
	}
	s.completed[previous.ID] = true
	var released []*sim.Request
	for _, id := range children {
		s.remaining[id]--
		if at > s.lastFinish[id] {
			s.lastFinish[id] = at
		}
		if s.remaining[id] != 0 {
			continue
		}
		r := s.config[id]
		finish := s.lastFinish[id]
		if r.ThinkTimeUS > math.MaxInt64-finish || r.TTFTSLOUS > math.MaxInt64-finish-r.ThinkTimeUS {
			s.err = fmt.Errorf("conversation arrival/deadline overflows for %s", id)
			return nil
		}
		r.At = finish + r.ThinkTimeUS
		s.config[id] = r // timing consumers use the actual client arrival
		next := s.requests[id]
		next.ArrivalTime = r.At
		if r.TTFTSLOUS > 0 {
			s.deadlines[id] = r.At + r.TTFTSLOUS
		}
		released = append(released, next)
	}
	return released
}
