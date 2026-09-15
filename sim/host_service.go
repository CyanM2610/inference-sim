package sim

import "fmt"

// HostServiceWork contains work known at a host boundary. Completion requests
// actually participated in the returned batch; Finishing identifies results
// that reach their stop condition now. No output length or future arrival is
// exposed to the model or request policy.
type HostServiceWork struct {
	Stage     string   `json:"stage"`
	Requests  []string `json:"requests"`
	Finishing []string `json:"finishing,omitempty"`
}

type HostServiceCostModel interface {
	EstimateHostService(HostServiceWork) (DecisionCostEstimate, error)
}

type HostServiceRecord struct {
	Instance string          `json:"instance,omitempty"`
	Version  uint64          `json:"version"`
	Work     HostServiceWork `json:"work"`
	Service  DecisionService `json:"service"`
}

// HostCompletionBarrier runs before any batch/transfer publication. The
// backend retains its poll snapshot and all leases until publish is called.
// The barrier must call publish exactly once, at or after now.
type HostCompletionBarrier func(now int64, work []BatchWork, publish func(int64))
type HostCompletionStore interface {
	BindHostCompletionBarrier(HostCompletionBarrier) error
}

type registrationWork struct {
	request *Request
	cost    DecisionCostEstimate
}
type hostServiceRuntime struct {
	model                    HostServiceCostModel
	completion, registration bool
	observe                  func(HostServiceRecord)
	pending                  *hostServiceEvent
	registrations            []registrationWork
}

func (s *Simulator) SetHostServiceCosts(model HostServiceCostModel, completion, registration bool, observe func(HostServiceRecord)) error {
	if model == nil || (!completion && !registration) || s.hostServices != nil || s.stepCount != 0 || !s.batchCompletionEvents {
		return fmt.Errorf("host services require a model and explicit stages before event-driven execution")
	}
	if completion {
		store, ok := s.KVCache.(HostCompletionStore)
		if !ok {
			return fmt.Errorf("completion service requires a pre-publication backend barrier")
		}
		if err := store.BindHostCompletionBarrier(s.completeHostService); err != nil {
			return err
		}
	}
	s.hostServices = &hostServiceRuntime{model: model, completion: completion, registration: registration, observe: observe}
	return nil
}

func cloneHostWork(w HostServiceWork) HostServiceWork {
	w.Requests = append([]string(nil), w.Requests...)
	w.Finishing = append([]string(nil), w.Finishing...)
	return w
}

func (s *Simulator) priceHostWork(w HostServiceWork, now int64) DecisionCostEstimate {
	cost, err := s.hostServices.model.EstimateHostService(cloneHostWork(w))
	if err == nil {
		err = cost.validate(now)
	}
	if err != nil {
		panic(err)
	}
	return cost
}

func (s *Simulator) serveHostWork(now int64, w HostServiceWork, cost DecisionCostEstimate, done func(int64)) {
	h := s.hostServices
	if h.pending != nil && cost.ExtraUS > 0 {
		panic("overlapping host service")
	}
	if err := cost.validate(now); err != nil {
		panic(err)
	}
	version := uint64(0)
	if s.decision != nil {
		version = s.decision.record.View.Version
	}
	if h.observe != nil {
		h.observe(HostServiceRecord{Version: version, Work: cloneHostWork(w), Service: DecisionService{StartUS: now, EndUS: now + cost.ExtraUS, ExtraUS: cost.ExtraUS, Provenance: cost.Provenance}})
	}
	if cost.ExtraUS == 0 {
		done(now)
		return
	}
	e := &hostServiceEvent{at: now + cost.ExtraUS, done: done, waiting: map[*Request]bool{}}
	h.pending = e
	for _, r := range s.WaitQ.Items() {
		e.waiting[r] = true
	}
	// Preserve an already submitted batch marker; an empty control round also
	// needs an owner so arrivals cannot start a second host operation.
	if s.stepEvent == nil {
		s.stepEvent = e
	}
	s.Schedule(e)
}

type hostServiceEvent struct {
	at      int64
	done    func(int64)
	waiting map[*Request]bool
	revisit bool
}

func (e *hostServiceEvent) Timestamp() int64 { return e.at }
func (e *hostServiceEvent) Priority() int    { return PriorityStep }
func (e *hostServiceEvent) Execute(s *Simulator) {
	if s.hostServices == nil || s.hostServices.pending != e {
		panic("unexpected host completion")
	}
	late := e.revisit
	for _, r := range s.WaitQ.Items() {
		late = late || !e.waiting[r]
	}
	s.hostServices.pending = nil
	s.KVCache.SetClock(e.at)
	e.done(e.at)
	if late {
		s.ScheduleStepIfIdle(e.at)
	}
}

// The result is known only after the engine has returned output. Stop-condition
// inspection stays in the trusted runtime, never in a scheduling view. Work
// accepted at this boundary remains paid if cancellation arrives during service;
// normal timeout and completion paths still determine whether it is published.
func (s *Simulator) completeHostService(now int64, batch []BatchWork, publish func(int64)) {
	if s.hostServices == nil || !s.hostServices.completion {
		publish(now)
		return
	}
	w := HostServiceWork{Stage: "completion"}
	for _, b := range batch {
		if b.NewTokens <= 0 {
			continue
		}
		r := b.Request
		w.Requests = append(w.Requests, r.ID)
		if r.State == StateTimedOut || r.State == StateCompleted {
			continue
		}
		next := b.PrefixTokens + b.NewTokens
		if next >= r.completionProgressIndex() || s.maxModelLen > 0 && next >= s.maxModelLen-1 {
			w.Finishing = append(w.Finishing, r.ID)
		}
	}
	s.serveHostWork(now, w, s.priceHostWork(w, now), publish)
}

// Registration is queued after ordinary servability/deadline checks. Its
// service runs between engine iterations, not concurrently with a synchronous
// native engine call. The scheduled client arrival is never rewritten.
func (s *Simulator) registerOrEnqueue(r *Request) {
	if s.hostServices == nil || !s.hostServices.registration {
		if s.postStep != nil {
			// A synchronous driver cannot inject input during engine execution
			// or post-output processing, even when registration costs zero.
			s.postStep.inputs = append(s.postStep.inputs, r)
			s.ScheduleStepIfIdle(s.Clock)
			return
		}
		s.publishEnqueuedRequest(r)
		return
	}
	w := HostServiceWork{Stage: "registration", Requests: []string{r.ID}}
	cost := s.priceHostWork(w, s.Clock)
	if cost.ExtraUS == 0 && s.postStep == nil {
		s.serveHostWork(s.Clock, w, cost, func(int64) { s.publishEnqueuedRequest(r) })
		return
	}
	s.hostServices.registrations = append(s.hostServices.registrations, registrationWork{r, cost})
	s.ScheduleStepIfIdle(s.Clock)
}

func (s *Simulator) publishEnqueuedRequest(r *Request) {
	if r.State == StateTimedOut {
		return
	}
	s.WaitQ.Enqueue(r)
	s.notifyDecisionEvent(DecisionEvent{Kind: "request_arrived", OccurredUS: s.Clock, Request: r.ID})
}

func (s *Simulator) pendingRegistrations() bool {
	return s.postStep != nil && len(s.postStep.inputs) > 0 || s.hostServices != nil && len(s.hostServices.registrations) > 0
}

func (s *Simulator) beginRegistrationService(now int64) bool {
	if s.postStep != nil {
		inputs := s.postStep.inputs
		s.postStep.inputs = nil
		for _, r := range inputs {
			s.publishEnqueuedRequest(r)
		}
	}
	for s.hostServices != nil && len(s.hostServices.registrations) > 0 {
		h := s.hostServices
		item := h.registrations[0]
		h.registrations = h.registrations[1:]
		if item.request.State == StateTimedOut {
			continue
		}
		w := HostServiceWork{Stage: "registration", Requests: []string{item.request.ID}}
		s.serveHostWork(now, w, item.cost, func(at int64) {
			s.publishEnqueuedRequest(item.request)
			if s.RunningBatch != nil && len(s.RunningBatch.Requests) == 0 {
				s.RunningBatch = nil
			}
			s.Step(at)
		})
		return true
	}
	return false
}
