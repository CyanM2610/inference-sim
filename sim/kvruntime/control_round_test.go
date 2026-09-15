package kvruntime

import "testing"

func controlTestProfile() *ControlProfile {
	p := &ControlProfile{Models: map[string]ControlCost{}}
	for _, name := range []string{"python_send", "request_available", "go_decode", "go_poll", "go_encode", "go_write", "reply_available", "python_decode", "copy_dispatch", "copy_queue_put", "copy_wake", "copy_entry", "copy_exit", "copy_notify"} {
		p.Models[name] = ControlCost{Coefficients: map[string]float64{"base": 10}}
	}
	return p
}

func TestControlRoundWaitsForReplyAndPhysicalCompletion(t *testing.T) {
	e := NewEngine(nil)
	m, _ := NewMemory(e, 8, map[string]int64{"gpu": 8, "host": 8})
	r, _ := NewRuntime(e, m, 8)
	m.Reserve("source", "gpu", "input", 8)
	m.Allocate("source")
	m.Fill("source", 123)
	m.Reserve("destination", "host", "output", 8)
	m.Allocate("destination")
	p := controlTestProfile()
	p.Models["go_write"] = ControlCost{Coefficients: map[string]float64{"base": 1000}}
	p.Models["reply_available"] = ControlCost{Coefficients: map[string]float64{"base": 200}}
	e.MustAdd(Task{Name: "physical_copy_backlog", Resource: "dma", DurationNS: 5000})
	var noticed int64
	var released bool
	round, err := r.ControlRound(ControlRoundSpec{ID: "round", Profile: p, Decide: func() (*ControlDecision, error) {
		return &ControlDecision{Commands: []ControlCommand{{ID: "copy1", Kind: "copy", Start: func() (int64, error) {
			tx, err := r.Transfer(TransferSpec{ID: "copy1", Source: "source", Destination: "destination", CPU: "copy_thread", DMA: "dma", Spans: []Span{{Bytes: 8}}, Cost: TransferCost{SubmitNS: 1, DMANS: 1, HoldCPUUntilDMA: true}})
			if err != nil {
				return 0, err
			}
			return tx.CompletionTask, nil
		}, Ready: func(at int64) { noticed = at }}}}, nil
	}, Released: func(int64) { released = true }})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.RunUntil(100); err != nil {
		t.Fatal(err)
	}
	if released || noticed != 0 || e.resources["protocol_driver"].active == nil || e.resources["protocol_driver"].active.Name != "driver_readline_wait" {
		t.Fatal("driver advanced before reply became available")
	}
	if err = e.RunUntil(2000); err != nil {
		t.Fatal(err)
	}
	if !released || noticed != 0 || m.Buffers["source"].Pins != 1 || m.Buffers["destination"].Writers != 1 {
		t.Fatal("control round conflated dispatch with physical completion")
	}
	if round.ReplyDecodedNS >= 1040 {
		t.Fatal("response incorrectly waited for producer write return")
	}
	if err = e.Run(); err != nil {
		t.Fatal(err)
	}
	if noticed <= 5001 || m.Buffers["destination"].Cells[0] != (Cell{123, 0, true}) {
		t.Fatal("notification preceded actual physical data completion")
	}
}

func TestControlProfileRejectsUnsupportedMessageShape(t *testing.T) {
	p := &ControlProfile{Models: map[string]ControlCost{"decode": {Coefficients: map[string]float64{"base": 1, "reply_kib": 10}, Bounds: map[string][2]float64{"reply_kib": {1, 4}}}}}
	if _, err := p.Predict("decode", ControlFeatures{"reply_kib": 5}); err == nil {
		t.Fatal("extrapolated an unsupported message shape")
	}
	if _, err := p.Predict("decode", nil); err == nil {
		t.Fatal("missing message size silently became zero")
	}
}

func TestControlCostPredictionIsDeterministic(t *testing.T) {
	p := &ControlProfile{Models: map[string]ControlCost{"stage": {Coefficients: map[string]float64{"a": 1e16, "b": 1, "c": 1}}}}
	features := ControlFeatures{"a": 1, "b": 1, "c": 1}
	want, err := p.Predict("stage", features)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < 1000; n++ {
		got, err := p.Predict("stage", features)
		if err != nil || got != want {
			t.Fatalf("prediction changed without a feature/profile change: %d vs %d (%v)", got, want, err)
		}
	}
}
