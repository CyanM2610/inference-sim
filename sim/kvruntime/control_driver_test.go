package kvruntime

import "testing"

func driverFixture(t *testing.T) (*Runtime, *ControlProfile) {
	t.Helper()
	e := NewEngine(nil)
	m, _ := NewMemory(e, 8, map[string]int64{"unused": 8})
	r, _ := NewRuntime(e, m, 8)
	p := &ControlProfile{Models: map[string]ControlCost{}}
	for _, name := range []string{"python_send", "request_available", "go_decode", "go_poll", "go_encode", "go_write", "reply_available", "python_decode", "driver_initial", "driver_drain_begin", "driver_drain_get", "driver_drain_advance", "driver_drain_empty", "driver_clock", "driver_ack_pack", "driver_dispatch_finish", "driver_done", "driver_wait_prepare", "driver_loop_resume", "driver_get_ready", "driver_expired_idle", "driver_get_expired", "driver_sleep_return", "driver_queue_timeout"} {
		p.Models[name] = ControlCost{Coefficients: map[string]float64{"base": 10}}
	}
	for _, kind := range []string{"copy", "batch"} {
		for _, suffix := range []string{"dispatch", "queue_put", "wake", "entry", "exit", "enqueue", "put_tail", "receive_wake"} {
			p.Models[kind+"_"+suffix] = ControlCost{Coefficients: map[string]float64{"base": 10}}
		}
	}
	return r, p
}

func TestControlDriverAccumulatesCompletionsWhileBusyAndUsesSnapshotClock(t *testing.T) {
	r, p := driverFixture(t)
	e := r.Engine
	p.Models["go_poll"] = ControlCost{Coefficients: map[string]float64{"base": 10, "slow": 1000}}
	var calls int
	var decisionsAt []int64
	out, err := r.ControlDriver(ControlDriverSpec{Profile: p, MaxRounds: 10, Prepare: func(DriverObservation) (ControlFeatures, error) { return nil, nil }, Decide: func(o DriverObservation) (*DriverDecision, error) {
		calls++
		decisionsAt = append(decisionsAt, e.NowNS)
		d := &DriverDecision{Control: &ControlDecision{PolicyFeatures: ControlFeatures{"slow": 0}}, NextEventNS: -1}
		switch calls {
		case 1:
			d.NextEventNS = 200
			for _, kind := range []string{"copy", "batch"} {
				kind := kind
				d.Control.Commands = append(d.Control.Commands, ControlCommand{ID: kind + "-1", Kind: kind, Start: func() (int64, error) {
					return e.MustAdd(Task{Name: "physical_body", Resource: "body/" + kind, DurationNS: 200}), nil
				}})
			}
		case 2:
			if o.TimeNS >= 275 || e.NowNS <= 275 || len(o.Completed) != 0 {
				t.Fatalf("arrival incorrectly entered the captured observation: %+v decision=%d", o, e.NowNS)
			}
			d.Control.PolicyFeatures["slow"] = 1
			d.NextEventNS = 275
		case 3:
			if o.TimeNS < 275 || len(o.Completed) != 2 {
				t.Fatalf("busy completions were not coalesced: %+v", o)
			}
			for _, c := range o.Completed {
				if c.QueuedNS >= decisionsAt[1]+1010 {
					t.Fatal("completion did not actually arrive during busy control service")
				}
			}
			d.Done = true
		default:
			t.Fatal("unexpected extra Poll")
		}
		return d, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if calls != 3 || len(out.Cycles) != 3 || out.CompleteNS <= out.Cycles[2].Observation.TimeNS {
		t.Fatal("driver failed to complete through one combined acknowledgement")
	}
}

func TestControlDriverCompletionCancelsTimerAndHoldsWaitingThread(t *testing.T) {
	r, p := driverFixture(t)
	e := r.Engine
	calls := 0
	out, err := r.ControlDriver(ControlDriverSpec{Profile: p, MaxRounds: 5, Prepare: func(DriverObservation) (ControlFeatures, error) { return nil, nil }, Decide: func(o DriverObservation) (*DriverDecision, error) {
		calls++
		if calls == 1 {
			return &DriverDecision{NextEventNS: 1_000_000, Control: &ControlDecision{Commands: []ControlCommand{{ID: "copy-1", Kind: "copy", Start: func() (int64, error) {
				return e.MustAdd(Task{Name: "actual_copy", Resource: "dma", DurationNS: 1000}), nil
			}}}}}, nil
		}
		if len(o.Completed) != 1 || o.Completed[0].Kind != "copy" {
			t.Fatal("copy completion lost")
		}
		return &DriverDecision{NextEventNS: -1, Done: true, Control: &ControlDecision{}}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(500); err != nil {
		t.Fatal(err)
	}
	if active := e.resources["protocol_driver"].active; active == nil || active.Name != "driver_blocking_wait" {
		t.Fatal("driver thread was released while queue.get was blocked")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if calls != 2 || e.NowNS >= 1_000_000 || out.CompleteNS != e.NowNS {
		t.Fatal("stale timer advanced time or reentered completed driver")
	}
}

func TestControlDriverSleepsToPolicyTimerWithoutInventingWorkerCompletion(t *testing.T) {
	r, p := driverFixture(t)
	calls := 0
	_, err := r.ControlDriver(ControlDriverSpec{Profile: p, MaxRounds: 4, Prepare: func(DriverObservation) (ControlFeatures, error) { return nil, nil }, Decide: func(o DriverObservation) (*DriverDecision, error) {
		calls++
		if len(o.Completed) != 0 {
			t.Fatal("timer fabricated a worker acknowledgement")
		}
		if calls == 1 {
			return &DriverDecision{NextEventNS: 1000, Control: &ControlDecision{}}, nil
		}
		if o.TimeNS <= 1000 {
			t.Fatal("sleep return and drain costs omitted")
		}
		return &DriverDecision{NextEventNS: -1, Done: true, Control: &ControlDecision{}}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err = r.Engine.Run(); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatal("timer not observed exactly once")
	}
}
