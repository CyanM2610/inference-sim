package kvruntime

import "fmt"

type DriverCompletion struct {
	ID       string `json:"id"`
	Kind     string `json:"kind"`
	QueuedNS int64  `json:"queued_ns"`
}
type DriverObservation struct {
	TimeNS    int64              `json:"time_ns"`
	Completed []DriverCompletion `json:"completed"`
}
type DriverDecision struct {
	Control     *ControlDecision
	NextEventNS int64 // -1 means no scheduled policy event
	Done        bool
}
type DriverCycle struct {
	Observation DriverObservation   `json:"observation"`
	Round       *ControlRoundResult `json:"round"`
}
type ControlDriverResult struct {
	Cycles     []*DriverCycle `json:"cycles"`
	CompleteNS int64          `json:"complete_ns"`
}
type ControlDriverSpec struct {
	Profile   *ControlProfile
	Prepare   func(DriverObservation) (ControlFeatures, error)
	Decide    func(DriverObservation) (*DriverDecision, error)
	MaxRounds int
	// HoldNS is an explicit diagnostic delay, not a fitted service coefficient.
	HoldNS func(round, commands int) int64
}

type controlDriver struct {
	r                            *Runtime
	s                            ControlDriverSpec
	out                          *ControlDriverResult
	inbox, received              []DriverCompletion
	pending                      map[string]string
	queued                       map[string]bool
	decision                     *DriverDecision
	operation                    string
	waitGeneration               int64
	waiting, signalled, finished bool
	signal                       *DriverCompletion
}

// ControlDriver executes the native receive/drain/snapshot/send/wait protocol.
// Its one driver thread cannot enter policy twice at once. Completions commit
// independently while it is busy; only a later observed snapshot acknowledges
// them. Timers wake the driver, never advance policy directly.
func (r *Runtime) ControlDriver(s ControlDriverSpec) (*ControlDriverResult, error) {
	if s.Profile == nil || s.Prepare == nil || s.Decide == nil || s.MaxRounds <= 0 {
		return nil, fmt.Errorf("invalid control driver")
	}
	d := &controlDriver{r: r, s: s, out: &ControlDriverResult{}, pending: map[string]string{}, queued: map[string]bool{}, operation: "driver/startup"}
	d.stage("driver_initial", nil, func() { d.beginDrain() })
	return d.out, r.Engine.Error()
}

func (d *controlDriver) stage(name string, features ControlFeatures, after func()) {
	e := d.r.Engine
	e.MustAdd(Task{Name: name, Operation: d.operation, Resource: "protocol_driver", Duration: func() int64 {
		ns, err := d.s.Profile.Predict(name, features)
		if err != nil {
			e.Fail(err)
			return 0
		}
		return ns
	}, After: after})
}
func (d *controlDriver) beginDrain() {
	if len(d.out.Cycles) >= d.s.MaxRounds {
		d.r.Engine.Fail(fmt.Errorf("control driver exceeded round limit"))
		return
	}
	d.operation = fmt.Sprintf("driver/round/%d", len(d.out.Cycles))
	d.stage("driver_drain_begin", nil, d.drainNext)
}
func (d *controlDriver) take() DriverCompletion {
	x := d.inbox[0]
	d.inbox = d.inbox[1:]
	return x
}
func (d *controlDriver) drainNext() {
	if len(d.inbox) > 0 {
		x := d.take()
		d.stage("driver_drain_get", nil, func() {
			d.received = append(d.received, x)
			d.stage("driver_drain_advance", nil, d.drainNext)
		})
		return
	}
	// Empty is decided at the queue access. A completion committed during the
	// empty-get/clock tail belongs to a subsequent round, just as in Python.
	d.stage("driver_drain_empty", nil, func() { d.stage("driver_clock", nil, d.snapshot) })
}
func (d *controlDriver) snapshot() {
	e := d.r.Engine
	observation := DriverObservation{TimeNS: e.NowNS, Completed: append([]DriverCompletion(nil), d.received...)}
	d.received = nil
	for _, c := range observation.Completed {
		if d.pending[c.Kind] != c.ID || !d.queued[c.ID] {
			e.Fail(fmt.Errorf("unknown or repeated driver acknowledgement"))
			return
		}
		delete(d.pending, c.Kind)
		delete(d.queued, c.ID)
	}
	features, err := d.s.Prepare(observation)
	if err != nil {
		e.Fail(err)
		return
	}
	cycle := &DriverCycle{Observation: observation}
	d.out.Cycles = append(d.out.Cycles, cycle)
	e.Emit(Record{Name: "driver_observation", Operation: d.operation, Bytes: int64(len(observation.Completed))})
	d.stage("driver_ack_pack", ControlFeatures{"acks": float64(len(observation.Completed))}, func() {
		round, err := d.r.ControlRound(ControlRoundSpec{ID: d.operation, Profile: d.s.Profile, InputFeatures: features, DriverProtocol: true, Decide: func() (*ControlDecision, error) {
			decision, err := d.s.Decide(observation)
			if err != nil {
				return nil, err
			}
			if decision == nil || decision.Control == nil || decision.NextEventNS < -1 {
				return nil, fmt.Errorf("invalid driver decision")
			}
			d.decision = decision
			for i := range decision.Control.Commands {
				command := &decision.Control.Commands[i]
				if d.pending[command.Kind] != "" || command.Ready != nil || command.Queued != nil {
					return nil, fmt.Errorf("worker still pending or externally owned notification")
				}
				d.pending[command.Kind] = command.ID
				id, kind := command.ID, command.Kind
				command.Queued = func(at int64) { d.enqueue(DriverCompletion{ID: id, Kind: kind, QueuedNS: at}) }
			}
			if decision.Done && (len(d.pending) > 0 || len(d.inbox) > 0) {
				return nil, fmt.Errorf("controller done with outstanding workers")
			}
			return decision.Control, nil
		}, Released: func(int64) { d.released() }})
		cycle.Round = round
		if err != nil {
			e.Fail(err)
		}
	})
}
func (d *controlDriver) enqueue(c DriverCompletion) {
	e := d.r.Engine
	if d.finished || d.pending[c.Kind] != c.ID || d.queued[c.ID] {
		e.Fail(fmt.Errorf("unknown/duplicate worker completion"))
		return
	}
	d.queued[c.ID] = true
	d.inbox = append(d.inbox, c)
	e.Emit(Record{Name: "driver_completion_queued", Operation: d.operation, Source: c.ID, Reason: c.Kind})
	if d.waiting && !d.signalled {
		x := d.take()
		d.signal = &x
		d.signalled = true
		e.Emit(Record{Name: "driver_wait_signal", Operation: d.operation, Source: x.ID, Reason: "completion"})
		e.Kick()
	}
}
func (d *controlDriver) released() {
	if d.decision.Done {
		d.stage("driver_done", nil, func() { d.finished = true; d.out.CompleteNS = d.r.Engine.NowNS })
		return
	}
	prepare := func() {
		active := 0.0
		if len(d.pending) > 0 {
			active = 1
		}
		d.stage("driver_wait_prepare", ControlFeatures{"active": active}, d.wait)
	}
	if d.s.HoldNS != nil {
		ns := d.s.HoldNS(len(d.out.Cycles)-1, len(d.decision.Control.Commands))
		if ns < 0 {
			d.r.Engine.Fail(fmt.Errorf("negative diagnostic hold"))
			return
		}
		if ns > 0 {
			d.r.Engine.MustAdd(Task{Name: "driver_diagnostic_hold", Operation: d.operation, Resource: "protocol_driver", DurationNS: ns, After: prepare})
			return
		}
	}
	prepare()
}
func (d *controlDriver) resume() { d.stage("driver_loop_resume", nil, d.beginDrain) }
func (d *controlDriver) wait() {
	e := d.r.Engine
	deadline := d.decision.NextEventNS
	if len(d.pending) > 0 && len(d.inbox) > 0 {
		x := d.take()
		d.stage("driver_get_ready", nil, func() { d.received = append(d.received, x); d.resume() })
		return
	}
	if deadline >= 0 && deadline <= e.NowNS {
		name := "driver_expired_idle"
		if len(d.pending) > 0 {
			name = "driver_get_expired"
		}
		d.stage(name, nil, d.resume)
		return
	}
	if len(d.pending) == 0 && deadline < 0 {
		e.Fail(fmt.Errorf("driver deadlock without work or policy timer"))
		return
	}
	queueWait := len(d.pending) > 0
	watchdog := deadline < 0
	if watchdog {
		deadline = e.NowNS + 30_000_000_000
	}
	d.waitGeneration++
	version := d.waitGeneration
	d.waiting = true
	d.signalled = false
	d.signal = nil
	valid := func() bool { return d.waiting && !d.signalled && version == d.waitGeneration }
	gate := e.MustAdd(Task{Name: "driver_wait_condition", Operation: d.operation, Resource: fmt.Sprintf("driver_wait_gate/%d", version), Ready: func() bool { return d.signalled }})
	e.MustAdd(Task{Name: "driver_blocking_wait", Operation: d.operation, Resource: "protocol_driver", WaitFor: []int64{gate}, After: func() {
		d.waiting = false
		d.waitGeneration++
		if x := d.signal; x != nil {
			d.stage(x.Kind+"_receive_wake", nil, func() { d.received = append(d.received, *x); d.resume() })
		} else {
			name := "driver_sleep_return"
			if queueWait {
				name = "driver_queue_timeout"
			}
			d.stage(name, nil, d.resume)
		}
	}})
	if err := e.atIf(deadline, func() {
		if watchdog {
			e.Fail(fmt.Errorf("worker failed to notify within native 30-second watchdog"))
			return
		}
		d.signalled = true
		e.Emit(Record{Name: "driver_wait_signal", Operation: d.operation, Reason: "timer"})
		e.Kick()
	}, valid); err != nil {
		e.Fail(err)
	}
}
