package sim

import "fmt"

// PostStepWork is observed after output publication. Only already registered,
// live requests are visible; pending external input is a private driver fact.
type PostStepWork struct {
	Stage    string   `json:"stage"`
	Requests []string `json:"requests"`
	Held     []string `json:"held,omitempty"`
}

type PostStepCostModel interface {
	EstimatePostStep(PostStepWork) (DecisionCostEstimate, error)
}

type PostStepRecord struct {
	Instance  string          `json:"instance,omitempty"`
	Iteration uint64          `json:"iteration,omitempty"`
	Version   uint64          `json:"version"`
	Work      PostStepWork    `json:"work"`
	Service   DecisionService `json:"service"`
}

type postStepRuntime struct {
	model                                     PostStepCostModel
	observe                                   func(PostStepRecord)
	remainingInputs                           int
	arrived                                   map[string]bool
	inputs                                    []*Request
	loopCosts, loopOpen, waitingInput, exited bool
	iteration                                 uint64
	active, revisit                           bool
	pending                                   *postStepServiceEvent
}

// SetPostStepCostModel models the fixed-input synchronous driver. Input count
// determines idle/loop termination only; neither future arrival times nor
// future request attributes are passed to the model. An optional loop-cost
// capability also prices work before input registration and the final test.
func (s *Simulator) SetPostStepCostModel(model PostStepCostModel, inputRequests int, observe func(PostStepRecord)) error {
	if model == nil || inputRequests <= 0 || s.postStep != nil || s.decision == nil || s.stepCount != 0 || !s.batchCompletionEvents {
		return fmt.Errorf("post-step services require a fixed input count, decision policy and batch completion events before execution")
	}
	s.postStep = &postStepRuntime{model: model, observe: observe, remainingInputs: inputRequests, arrived: map[string]bool{}}
	if loop, ok := model.(interface{ DriverLoopCostsEnabled() bool }); ok {
		s.postStep.loopCosts = loop.DriverLoopCostsEnabled()
	}
	if s.postStep.loopCosts && s.stepEvent == nil {
		// The native runner begins at its epoch even if the first input is
		// later. Its first empty loop must not disappear from the cost model.
		s.stepEvent = &StepEvent{time: s.Clock}
		s.Schedule(s.stepEvent)
	}
	return nil
}

func (s *Simulator) notePostStepInput(r *Request) {
	p := s.postStep
	if p == nil || p.arrived[r.ID] {
		return
	}
	if p.remainingInputs <= 0 {
		panic("post-step driver received more inputs than declared")
	}
	p.arrived[r.ID] = true
	p.remainingInputs--
	p.waitingInput = false
	p.revisit = p.revisit || p.active
}

func (s *Simulator) postStepWork(stage string, at int64) PostStepWork {
	w := PostStepWork{Stage: stage}
	seen := map[string]bool{}
	groups := [][]*Request{s.WaitQ.Items()}
	if s.RunningBatch != nil {
		groups = append(groups, s.RunningBatch.Requests)
	}
	for _, group := range groups {
		for _, r := range group {
			if seen[r.ID] || r.State == StateCompleted || r.State == StateTimedOut {
				continue
			}
			seen[r.ID] = true
			w.Requests = append(w.Requests, r.ID)
			if s.decisionWaitsUntil(r.ID) > at {
				w.Held = append(w.Held, r.ID)
			}
		}
	}
	return w
}

func clonePostStepWork(w PostStepWork) PostStepWork {
	w.Requests = append([]string(nil), w.Requests...)
	w.Held = append([]string(nil), w.Held...)
	return w
}

// finishPostStep pays output bookkeeping before the optional idle check. The
// existing deadline event advances idle time; sleep is never charged as CPU.
// Zero service executes inline and retains the prior event ordering.
func (s *Simulator) finishPostStep(now int64, done func(int64)) {
	p := s.postStep
	if p == nil {
		done(now)
		return
	}
	if p.active {
		panic("overlapping post-step iterations")
	}
	p.active, p.revisit = true, false
	finish := func(at int64) { s.endDriverLoop(at, true, done) }
	s.servePostStep(now, s.postStepWork("after_output", now), func(at int64) {
		w := s.postStepWork("idle_check", at)
		if len(w.Requests) == 0 && p.remainingInputs == 0 && !s.pendingRegistrations() && !pendingEngineWork(s.KVCache) && !s.pendingDecisionControl() {
			finish(at)
			return
		}
		s.servePostStep(at, w, finish)
	})
}

func (s *Simulator) servePostStep(now int64, w PostStepWork, done func(int64)) {
	p := s.postStep
	cost, err := p.model.EstimatePostStep(clonePostStepWork(w))
	if err == nil {
		err = cost.validate(now)
	}
	if err != nil {
		panic(err)
	}
	if p.observe != nil {
		p.observe(PostStepRecord{Iteration: p.iteration, Version: s.decision.record.View.Version, Work: clonePostStepWork(w), Service: DecisionService{StartUS: now, EndUS: now + cost.ExtraUS, ExtraUS: cost.ExtraUS, Provenance: cost.Provenance}})
	}
	if cost.ExtraUS == 0 {
		done(now)
		return
	}
	e := &postStepServiceEvent{at: now + cost.ExtraUS, done: done}
	p.pending = e
	// Own the next formation even if this was a no-request control iteration.
	s.stepEvent = e
	s.Schedule(e)
}

type postStepServiceEvent struct {
	at   int64
	done func(int64)
}

func (e *postStepServiceEvent) Timestamp() int64 { return e.at }
func (e *postStepServiceEvent) Priority() int    { return PriorityStep }
func (e *postStepServiceEvent) Execute(s *Simulator) {
	if s.postStep == nil || s.postStep.pending != e {
		panic("unexpected post-step service completion")
	}
	s.postStep.pending = nil
	s.KVCache.SetClock(e.at)
	e.done(e.at)
}

func (s *Simulator) driverHasCurrentWork() bool {
	return len(s.postStepWork("", s.Clock).Requests) > 0 || s.pendingRegistrations() || pendingEngineWork(s.KVCache) || s.pendingDecisionControl()
}

// One outer-loop prefix precedes all registration calls and policy retries in
// this iteration. Registration continuations re-enter Step with loopOpen set.
func (s *Simulator) beginDriverLoop(now int64) bool {
	p := s.postStep
	if p == nil || !p.loopCosts || p.loopOpen || p.exited {
		return false
	}
	if !s.driverHasCurrentWork() && (p.remainingInputs == 0 || p.waitingInput) {
		return false
	}
	p.loopOpen, p.active, p.waitingInput, p.revisit = true, true, false, false
	p.iteration++
	s.servePostStep(now, s.postStepWork("before_step", now), func(at int64) {
		p.active, p.revisit = false, false
		s.Step(at)
	})
	return true
}

func (s *Simulator) endDriverLoop(now int64, afterOutput bool, done func(int64)) {
	p := s.postStep
	p.loopOpen = false
	late := p.revisit
	finish := func(at int64) {
		p.active, p.revisit = false, false
		done(at)
		if late {
			s.ScheduleStepIfIdle(at)
		}
		// With only future input, the real driver enters an empty outer loop,
		// tests its gate, then sleeps. That loop has no engine/output work.
		if afterOutput && p.loopCosts && p.remainingInputs > 0 && !s.driverHasCurrentWork() && !p.waitingInput {
			s.Step(at)
		}
	}
	if p.loopCosts && !p.exited && p.remainingInputs == 0 && !s.driverHasCurrentWork() {
		p.exited, p.active = true, true
		s.servePostStep(now, s.postStepWork("loop_exit", now), finish)
		return
	}
	finish(now)
}

// The no-engine gate does not create post-output work. Redundant wakes while
// awaiting future input cannot repeatedly bill the same empty driver loop.
func (s *Simulator) finishDriverWithoutOutput(now int64) {
	s.stepEvent = nil
	p := s.postStep
	if p == nil || !p.loopCosts || !p.loopOpen {
		return
	}
	p.waitingInput = p.remainingInputs > 0 && !s.driverHasCurrentWork()
	s.endDriverLoop(now, false, func(int64) { s.stepEvent = nil })
}
