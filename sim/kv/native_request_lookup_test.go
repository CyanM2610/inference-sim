package kv

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type nativeRequestTestCost struct{ nativeControlTestCost }

func (nativeRequestTestCost) RequestRPCPlan(shape sim.NativeControlRPCShape) sim.NativeControlRPCPlan {
	return nativeControlTestCost{}.ControlRPCPlan(shape)
}

func TestNativeRequestLookupMissWaitsForStatusBeforeAllocation(t *testing.T) {
	for _, tokens := range []int{3, 9} { // zero complete chunks also uses RPCs
		t.Run(string(rune('0'+tokens)), func(t *testing.T) {
			h, s, e, _, _ := nativeControlHarnessWithCost(t, nativeRequestTestCost{})
			if err := s.EnableNativeRequestLookups(); err != nil {
				t.Fatal(err)
			}
			input := make([]sim.TokenID, tokens)
			for i := range input {
				input[i] = sim.TokenID(101 + i)
			}
			r := &sim.Request{ID: "miss", InputTokens: input, OutputTokens: []sim.TokenID{99}, MaxOutputLen: 1}
			scheduled := false
			plans := map[string]int{}
			e.EventObserver = func(event sim.Event, before bool) {
				x, ok := event.(*sim.NativeAsyncEvent)
				if !ok || before {
					return
				}
				if x.Name == "native_request_rpc_plan" {
					plans[x.ControlShape.Operation]++
					wantTokens, wantObjects := int64(0), int64(0)
					if x.ControlShape.Operation == "lookup" {
						wantTokens, wantObjects = int64(tokens), int64(tokens/4)
					}
					if x.ControlShape.Tokens != wantTokens || x.ControlShape.Objects != wantObjects {
						t.Fatal("lookup payload/alignment lost", x.ControlShape)
					}
				}
				if x.Name == "native_async_scheduled" && len(x.Shape) > 0 {
					scheduled = true
					if c := s.nativeRequestLookups[r.ID]; c == nil || !c.visible || !c.replied || !c.acked {
						t.Fatal("miss allocated before status adoption")
					}
				}
				if !scheduled && s.UsedBlocks() != 0 {
					t.Fatal("pending miss consumed HBM")
				}
			}
			e.InjectArrival(r)
			e.Run()
			var sequence []string
			for _, row := range h.records {
				if row.Request == r.ID && strings.HasPrefix(row.Name, "native_request_") {
					sequence = append(sequence, row.Name)
				}
			}
			want := []string{"native_request_lookup_submitted", "native_request_lookup_applied", "native_request_lookup_ack", "native_request_status_submitted", "native_request_status_applied", "native_request_status_reply", "native_request_lookup_adopted", "native_request_end_submitted", "native_request_end_applied", "native_request_end_reply", "native_request_lookup_retired"}
			if strings.Join(sequence, ",") != strings.Join(want, ",") || plans["lookup"] != 1 || plans["status"] != 1 || plans["end"] != 1 || !scheduled {
				t.Fatal("wrong request protocol", sequence, plans)
			}
			if s.nativeRequestRPCs != 0 || len(s.nativeRequestLookups) != 0 || s.UsedBlocks() != 0 || e.Metrics.CompletedRequests != 1 {
				t.Fatal("request/RPC ownership did not drain")
			}
		})
	}
}

func TestNativeRequestLookupCapacityRetryKeepsResultAndSources(t *testing.T) {
	h, s, e, prefix, _ := nativeControlHarnessWithCost(t, nativeRequestTestCost{})
	if err := s.EnableNativeRequestLookups(); err != nil {
		t.Fatal(err)
	}
	r := &sim.Request{ID: "held", InputTokens: append(append([]sim.TokenID(nil), prefix.tokens...), 99), MaxOutputLen: 1}
	// Poll manually with no frontend request: a ready RPC is not adoption.
	if s.NativeRequestLookupReady(r, false) {
		t.Fatal("first asynchronous call ready")
	}
	e.Run()
	if s.nativeRequestLookups[r.ID].visible || s.nativeRequestRPCSerial != 1 {
		t.Fatal("ACK auto-adopted")
	}
	if s.NativeRequestLookupReady(r, false) {
		t.Fatal("status RPC auto-adopted")
	}
	e.Run()
	if s.nativeRequestLookups[r.ID].visible {
		t.Fatal("status reply auto-adopted")
	}
	if !s.NativeRequestLookupReady(r, false) {
		t.Fatal("available result not adopted")
	}
	serial := s.nativeRequestRPCSerial
	for range 10 {
		if !s.NativeRequestLookupReady(r, false) {
			t.Fatal("cached lookup result disappeared")
		}
	}
	if s.nativeRequestRPCSerial != serial {
		t.Fatal("capacity retry resubmitted lookup")
	}
	for _, object := range h.f.hostChunks.lookups[r.ID].prefix {
		if object.lookupReaders != 1 || object.evictable() {
			t.Fatal("waiting lookup lost source lease")
		}
	}
	s.ClearDeferred(r.ID)
	e.Run()
	if s.nativeRequestRPCs != 0 || len(s.nativeRequestLookups) != 0 || len(h.f.hostChunks.lookups) != 0 {
		t.Fatal("END did not retire pending admission")
	}
}

type slowRequestLookupCost struct{ nativeRequestTestCost }

func (slowRequestLookupCost) RequestRPCPlan(shape sim.NativeControlRPCShape) sim.NativeControlRPCPlan {
	p := nativeRequestTestCost{}.RequestRPCPlan(shape)
	if shape.Operation == "lookup" {
		p.Stages[0].After = []string{"network"}
		p.Stages = append([]sim.NativeStage{{Name: "network", DurationUS: 100, Delay: true}}, p.Stages...)
	}
	return p
}

func TestNativeRequestENDOvertakesLookupWithoutResurrection(t *testing.T) {
	h, s, e, prefix, _ := nativeControlHarnessWithCost(t, slowRequestLookupCost{})
	if err := s.EnableNativeRequestLookups(); err != nil {
		t.Fatal(err)
	}
	r := &sim.Request{ID: "cancelled", InputTokens: prefix.tokens, MaxOutputLen: 1}
	s.NativeRequestLookupReady(r, false)
	s.ClearDeferred(r.ID)
	s.ClearDeferred(r.ID)
	e.Run()
	for _, row := range h.records {
		if row.Request == r.ID && (row.Name == "host_lookup_reserved" || row.Name == "native_request_lookup_adopted") {
			t.Fatal("late lookup resurrected END", row)
		}
		if row.Request == r.ID && row.Name == "native_request_lookup_ack" && row.Counters["accepted"] != 0 {
			t.Fatal("late lookup reported accepted after END", row)
		}
	}
	if s.nativeRequestRPCSerial != 2 || s.nativeRequestRPCs != 0 || len(s.nativeRequestLookups) != 0 || len(h.f.hostChunks.lookups) != 0 {
		t.Fatal("cancel leaked or duplicated RPC")
	}
}

func TestNativeRequestLookupRequiresExplicitCostsAndPreemptionBypasses(t *testing.T) {
	_, s, _, _, _ := nativeControlHarness(t)
	if s.EnableNativeRequestLookups() == nil {
		t.Fatal("missing request RPC cost silently accepted")
	}
	_, s, _, prefix, _ := nativeControlHarnessWithCost(t, nativeRequestTestCost{})
	if err := s.EnableNativeRequestLookups(); err != nil {
		t.Fatal(err)
	}
	if !s.NativeRequestLookupReady(&sim.Request{ID: "preempted", InputTokens: prefix.tokens}, true) || s.nativeRequestRPCSerial != 0 {
		t.Fatal("preemption repeated external lookup")
	}
}

type slowRequestStatusCost struct{ nativeRequestTestCost }

func (slowRequestStatusCost) RequestRPCPlan(shape sim.NativeControlRPCShape) sim.NativeControlRPCPlan {
	p := nativeRequestTestCost{}.RequestRPCPlan(shape)
	if shape.Operation == "status" {
		p.Stages[0].After = []string{"network"}
		p.Stages = append([]sim.NativeStage{{Name: "network", DurationUS: 100, Delay: true}}, p.Stages...)
	}
	return p
}

func TestNativeRequestENDOvertakesStatusAndClearsReply(t *testing.T) {
	h, s, e, prefix, _ := nativeControlHarnessWithCost(t, slowRequestStatusCost{})
	if err := s.EnableNativeRequestLookups(); err != nil {
		t.Fatal(err)
	}
	r := &sim.Request{ID: "status_cancel", InputTokens: prefix.tokens, MaxOutputLen: 1}
	s.NativeRequestLookupReady(r, false)
	e.Run()
	if s.nativeRequestLookups[r.ID].matched != 8 {
		t.Fatal("fixture needs positive lookup")
	}
	s.NativeRequestLookupReady(r, false)
	s.ClearDeferred(r.ID)
	e.Run()
	status := false
	for _, row := range h.records {
		if row.Request == r.ID && row.Name == "native_request_status_applied" {
			status = true
			if row.Counters["matched_tokens"] != 0 {
				t.Fatal("status retained hit after END", row)
			}
		}
		if row.Request == r.ID && row.Name == "native_request_lookup_adopted" {
			t.Fatal("cancelled request adopted status")
		}
	}
	if !status || s.nativeRequestRPCs != 0 || len(s.nativeRequestLookups) != 0 || len(h.f.hostChunks.lookups) != 0 {
		t.Fatal("status cancellation did not drain")
	}
}

type selectiveRequestCost struct{ nativeRequestTestCost }

func (selectiveRequestCost) RequestRPCPlan(shape sim.NativeControlRPCShape) sim.NativeControlRPCPlan {
	p := nativeRequestTestCost{}.RequestRPCPlan(shape)
	if shape.Operation == "lookup" && shape.Tokens == 9 {
		p.Stages[0].After = []string{"network"}
		p.Stages = append([]sim.NativeStage{{Name: "network", DurationUS: 1000, Delay: true}}, p.Stages...)
	}
	return p
}

func TestNativePendingLookupLetsYoungerReadyRequestRun(t *testing.T) {
	_, s, e, _, _ := nativeControlHarnessWithCost(t, selectiveRequestCost{})
	if err := s.EnableNativeRequestLookups(); err != nil {
		t.Fatal(err)
	}
	older := &sim.Request{ID: "older", InputTokens: []sim.TokenID{101, 102, 103, 104, 105, 106, 107, 108, 109}, OutputTokens: []sim.TokenID{200}, MaxOutputLen: 1}
	younger := &sim.Request{ID: "younger", InputTokens: []sim.TokenID{201, 202, 203, 204, 205}, OutputTokens: []sim.TokenID{300}, MaxOutputLen: 1}
	var first string
	e.EventObserver = func(event sim.Event, before bool) {
		x, ok := event.(*sim.NativeAsyncEvent)
		if !ok || before {
			return
		}
		if x.Name == "native_async_scheduled" && len(x.Shape) > 0 && first == "" {
			first = x.Shape[0].Request
		}
	}
	e.InjectArrival(older)
	e.InjectArrival(younger)
	e.Run()
	if first != "younger" || e.Metrics.CompletedRequests != 2 || s.nativeRequestRPCs != 0 || len(s.nativeRequestLookups) != 0 {
		t.Fatal("pending lookup stopped waiting scan or leaked", first)
	}
}

func TestNativeRequestTimeoutSubmitsENDAtActualDeadline(t *testing.T) {
	for _, deadline := range []int64{7, 95} {
		h, s, e, prefix, _ := nativeControlHarnessWithCost(t, nativeRequestTestCost{})
		if err := s.EnableNativeRequestLookups(); err != nil {
			t.Fatal(err)
		}
		r := &sim.Request{ID: "timeout", InputTokens: prefix.tokens, OutputTokens: []sim.TokenID{99}, MaxOutputLen: 1, Deadline: deadline}
		var last int64
		e.EventObserver = func(event sim.Event, before bool) {
			if before {
				return
			}
			if event.Timestamp() < last {
				t.Fatal("event clock regressed", last, event.Timestamp())
			}
			last = event.Timestamp()
		}
		e.InjectArrival(r)
		e.Run()
		ended := false
		for _, row := range h.records {
			if row.Request == r.ID && row.Name == "native_request_end_submitted" {
				ended = true
				if row.Time != deadline {
					t.Fatal("END used stale KV clock", row.Time, deadline)
				}
			}
		}
		if !ended || r.State != sim.StateTimedOut || e.Metrics.TimedOutRequests != 1 || s.nativeRequestRPCs != 0 || len(s.nativeRequestLookups) != 0 || s.UsedBlocks() != 0 {
			t.Fatal("timed-out request did not drain")
		}
	}
}

func TestNativeRequestCancelKeepsOtherLookupReaderThroughLOAD(t *testing.T) {
	h, s, e, prefix, _ := nativeControlHarnessWithCost(t, nativeRequestTestCost{})
	if err := s.EnableNativeRequestLookups(); err != nil {
		t.Fatal(err)
	}
	a := &sim.Request{ID: "cancel_a", InputTokens: append(append([]sim.TokenID(nil), prefix.tokens...), 99), MaxOutputLen: 1}
	b := &sim.Request{ID: "keep_b", InputTokens: append(append([]sim.TokenID(nil), prefix.tokens...), 99), OutputTokens: []sim.TokenID{100}, MaxOutputLen: 1}
	s.NativeRequestLookupReady(a, false)
	s.NativeRequestLookupReady(b, false)
	e.Run()
	s.NativeRequestLookupReady(a, false)
	s.NativeRequestLookupReady(b, false)
	e.Run()
	if !s.NativeRequestLookupReady(a, false) || !s.NativeRequestLookupReady(b, false) {
		t.Fatal("shared lookups not ready")
	}
	s.ClearDeferred(a.ID)
	e.Run()
	for _, object := range h.f.hostChunks.lookups[b.ID].prefix {
		if object.lookupReaders != 1 || object.evictable() {
			t.Fatal("cancel released another reader")
		}
	}
	b.ArrivalTime = e.Clock
	e.InjectArrival(b)
	e.Run()
	loads := 0
	for _, row := range h.records {
		if row.Request == b.ID && row.Name == "transfer_end" && row.Destination == "hbm" {
			loads++
		}
	}
	if loads == 0 || e.Metrics.CompletedRequests != 1 || s.nativeRequestRPCs != 0 || len(s.nativeRequestLookups) != 0 || s.UsedBlocks() != 0 {
		t.Fatal("surviving reader did not complete/drain", loads)
	}
}

type badRequestCost struct {
	nativeRequestTestCost
	mode string
}

func (c badRequestCost) RequestRPCPlan(shape sim.NativeControlRPCShape) sim.NativeControlRPCPlan {
	p := c.nativeRequestTestCost.RequestRPCPlan(shape)
	switch c.mode {
	case "scope":
		p.CostScope = ""
	case "zero":
		p.Stages[0].DurationUS = 0
	case "unjoined":
		p.Stages = append(p.Stages, sim.NativeStage{Name: "orphan", Resource: "mp", DurationUS: 1})
	case "reply_before_apply":
		p.Stages[1].After = nil
	}
	return p
}
func TestNativeRequestRPCRejectsUncostedOrUnjoinedPlans(t *testing.T) {
	for _, mode := range []string{"scope", "zero", "unjoined", "reply_before_apply"} {
		t.Run(mode, func(t *testing.T) {
			_, s, _, prefix, _ := nativeControlHarnessWithCost(t, badRequestCost{mode: mode})
			if err := s.EnableNativeRequestLookups(); err != nil {
				t.Fatal(err)
			}
			defer func() {
				if recover() == nil {
					t.Fatal("malformed request RPC plan accepted")
				}
			}()
			s.NativeRequestLookupReady(&sim.Request{ID: "invalid", InputTokens: prefix.tokens}, false)
		})
	}
}
