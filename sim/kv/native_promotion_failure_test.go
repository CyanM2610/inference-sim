package kv

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type partialNativeCost struct {
	nativeControlTestCost
	failureUS int64
	loseJoin  bool
	seen      *sim.NativeTransferFailure
}

type earlyErrorACKCost struct{ partialNativeCost }

func (c earlyErrorACKCost) FailedTransferPlan(f sim.NativeTransferFailure) sim.NativeTransferPlan {
	p := c.partialNativeCost.FailedTransferPlan(f)
	for i := range p.Stages {
		if p.Stages[i].Name == "model" {
			p.Stages[i].DurationUS = 10000
		}
	}
	reply := p.Stages[len(p.Stages)-1]
	reply.After = []string{"copy", "error_ack", "failure"}
	p.Stages = append(p.Stages[:len(p.Stages)-1], sim.NativeStage{Name: "error_ack", Resource: "mp", DurationUS: 2, After: []string{"copy", "failure"}}, reply)
	p.ModelComplete = "error_ack"
	return p
}

func TestNativePartialFailureCanAdoptModelErrorBeforeScheduledServiceDeadline(t *testing.T) {
	h, s, e, p, cached := nativeControlHarnessWithCost(t, earlyErrorACKCost{partialNativeCost{failureUS: 10}})
	s.invalidate(s.Blocks[cached[1]])
	if err := s.ConfigureNativePromotionPartialFailure(1); err != nil {
		t.Fatal(err)
	}
	id, err := s.QueueNativePromotion(p, 3)
	if err != nil {
		t.Fatal(err)
	}
	deadline := int64(-1)
	e.EventObserver = func(event sim.Event, before bool) {
		if before {
			return
		}
		x, ok := event.(*sim.NativeAsyncEvent)
		if ok && x.Name == "native_phase_end" && strings.HasSuffix(x.Phase, "/model") {
			deadline = x.At
		}
	}
	e.Run()
	adopted := int64(-1)
	for _, r := range h.records {
		if r.Request == id && r.Name == "native_control_adopted" {
			adopted = r.Time
		}
	}
	if adopted < 0 || deadline <= adopted {
		t.Fatal("failed ACK unnecessarily waited for successful modeled deadline", adopted, deadline)
	}
	nativeControlDrained(t, s)
}

func (c partialNativeCost) FailedTransferPlan(f sim.NativeTransferFailure) sim.NativeTransferPlan {
	if c.seen != nil {
		*c.seen = f
	}
	deps := []string{"copy", "model", "failure"}
	if c.loseJoin {
		deps = []string{"copy", "model"}
	}
	return sim.NativeTransferPlan{Stages: []sim.NativeStage{
		{Name: "submit", Resource: "mp", DurationUS: 1},
		{Name: "enqueued", After: []string{"submit"}},
		{Name: "copy", Resource: "copy", Stream: "native-kv-stream", DurationUS: int64(f.SubmittedObjects) * 200, After: []string{"enqueued"}},
		{Name: "model", Delay: true, DurationUS: 500, After: []string{"enqueued"}},
		{Name: "failure", Resource: "mp", DurationUS: c.failureUS, After: []string{"enqueued"}},
		{Name: "reply", Resource: "mp", DurationUS: 1, After: deps},
	}, ProducerWait: "enqueued", GPUComplete: "copy", ModelComplete: "model", Published: "reply", FailureObserved: "failure",
		CostScope: "artificial submit1/copy200-per-submitted-object/model500/failure-explicit/reply1; not calibrated"}
}

func TestNativePartialFailureRequiresExplicitCostBeforeOwnership(t *testing.T) {
	_, s, _, _, _ := nativeControlHarness(t)
	before := nativeControlHBMState(s)
	if err := s.ConfigureNativePromotionPartialFailure(1); err == nil {
		t.Fatal("accepted missing failure cost")
	}
	if s.nativePartialFailure != 0 || !reflect.DeepEqual(before, nativeControlHBMState(s)) {
		t.Fatal("rejection changed ownership")
	}
}

func TestNativePartialFailureHoldsBothSourcesAndAllTargetsUntilTerminal(t *testing.T) {
	for _, cancel := range []bool{false, true} {
		t.Run(map[bool]string{false: "failed", true: "cancelled-and-failed"}[cancel], func(t *testing.T) {
			var shape sim.NativeTransferFailure
			h, s, e, prefix, cached := nativeControlHarnessWithCost(t, partialNativeCost{failureUS: 10, seen: &shape})
			// First object now has one cached and one missing block; second has two missing.
			s.invalidate(s.Blocks[cached[1]])
			if err := s.ConfigureNativePromotionPartialFailure(1); err != nil {
				t.Fatal(err)
			}
			id, err := s.QueueNativePromotion(prefix, 3)
			if err != nil {
				t.Fatal(err)
			}
			var policyEvents []sim.DecisionEvent
			s.BindDecisionEvents(func(event sim.DecisionEvent) { policyEvents = append(policyEvents, event) })
			observed, captured := false, false
			var targets []int64
			e.EventObserver = func(event sim.Event, before bool) {
				if before {
					return
				}
				x, ok := event.(*sim.NativeAsyncEvent)
				if !ok {
					return
				}
				c := s.nativeControl
				if c != nil && len(c.ids) > 0 && !captured {
					targets = append([]int64(nil), c.ids...)
					captured = true
					if cancel {
						s.CancelNativePromotion(id)
					}
				}
				if x.Name == "native_phase_end" && strings.HasSuffix(x.Phase, "/failure") {
					observed = true
					if c == nil || !c.failed || c.claimed || c.settled != 0 {
						t.Fatal("failure skipped physical drain")
					}
					for i, bid := range c.ids {
						if s.Blocks[bid].RefCount != 1 || i >= c.cached && s.Blocks[bid].Hash != "" {
							t.Fatal("failure released/published pending target")
						}
					}
					for _, object := range s.fabric.hostChunks.objects {
						if object.transferRefs != 1 || object.evictable() {
							t.Fatal("uncopied source released before operation terminal")
						}
					}
				}
			}
			e.Run()
			if !observed || !captured || len(shape.ReservedShape.Objects) != 2 || shape.SubmittedObjects != 1 {
				t.Fatal("partial attempt not observed", shape)
			}
			if shape.ReservedShape.Objects[0].SkipTokens != 2 || shape.ReservedShape.Objects[0].KernelBytes != 100 {
				t.Fatal("cached prefix not skipped", shape)
			}
			var start, end, failed, receipt, adopted int64 = -1, -1, -1, -1, -1
			notSubmitted, physicalReleased := 0, 0
			for _, r := range h.records {
				if r.Request != id {
					continue
				}
				switch r.Name {
				case "transfer_start":
					start = r.Time
					if r.Bytes != 200 || r.EffectiveBytes != 100 || len(r.NativeObjects) != 1 || len(r.NativeFailure.ReservedShape.Objects) != 2 {
						t.Fatal("charged uncopied second object", r)
					}
				case "native_transfer_failure_observed":
					failed = r.Time
				case "transfer_end":
					end = r.Time
					if r.Outcome != "failed" {
						t.Fatal("failure became success")
					}
				case "native_control_receipt":
					receipt = r.Time
					if r.Reason != "worker_poll_failed" {
						t.Fatal("worker lost failed outcome")
					}
				case "native_control_adopted":
					adopted = r.Time
					if r.Reason != "partial_submission_failed" {
						t.Fatal("adoption lost failed outcome")
					}
				case "native_load_not_submitted":
					notSubmitted++
				case "host_transfer_released":
					physicalReleased++
					if end < 0 || r.Time < end {
						t.Fatal("source released before terminal")
					}
				case "hbm_publish":
					t.Fatal("failed prefix published")
				}
			}
			if start < 0 || failed <= start || failed >= end || receipt < end || adopted < receipt || notSubmitted != 2 || physicalReleased != 2 {
				t.Fatal("failure boundaries/cleanup incomplete", start, failed, end, receipt, adopted, notSubmitted, physicalReleased)
			}
			if len(policyEvents) != 1 || policyEvents[0].Kind != "transfer_failed" || policyEvents[0].OccurredUS != end || policyEvents[0].Transfer.Bytes != 200 {
				t.Fatal("policy was told a failed transfer was adopted, or learned it before terminal", policyEvents)
			}
			for _, bid := range targets[1:] {
				if s.Blocks[bid].Hash != "" || s.Blocks[bid].RefCount != 0 {
					t.Fatal("failed target remained published/held")
				}
			}
			if len(s.Blocks[targets[1]].Tokens) != 0 || len(s.Blocks[targets[2]].Tokens) != 0 {
				t.Fatal("failed adoption retained partial contents")
			}
			nativeControlDrained(t, s)
		})
	}
}

func TestNativePartialFailureRejectsSingleRetainedObjectWithoutHBMMutation(t *testing.T) {
	_, s, e, prefix, _ := nativeControlHarnessWithCost(t, partialNativeCost{failureUS: 10})
	if err := s.ConfigureNativePromotionPartialFailure(1); err != nil {
		t.Fatal(err)
	}
	before := nativeControlHBMState(s)
	if _, err := s.QueueNativePromotion(prefix, 2); err != nil {
		t.Fatal(err)
	}
	e.Run()
	if !reflect.DeepEqual(before, nativeControlHBMState(s)) || s.nativePartialFailure != 1 {
		t.Fatal("invalid partial geometry consumed fault or changed HBM")
	}
	nativeControlDrained(t, s)
}

func TestNativePartialFailureRejectsFreeOrUnjoinedExceptionHandling(t *testing.T) {
	for _, cost := range []partialNativeCost{{failureUS: 0}, {failureUS: 10, loseJoin: true}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("accepted uncosted/unjoined failure handling")
				}
			}()
			_, s, e, p, cached := nativeControlHarnessWithCost(t, cost)
			s.invalidate(s.Blocks[cached[1]])
			if err := s.ConfigureNativePromotionPartialFailure(1); err != nil {
				t.Fatal(err)
			}
			if _, err := s.QueueNativePromotion(p, 3); err != nil {
				t.Fatal(err)
			}
			e.Run()
		}()
	}
}

func TestNativePartialFailureFollowedByIndependentFullForegroundRestore(t *testing.T) {
	h, s, e, p, cached := nativeControlHarnessWithCost(t, partialNativeCost{failureUS: 10})
	s.invalidate(s.Blocks[cached[1]])
	if err := s.ConfigureNativePromotionPartialFailure(1); err != nil {
		t.Fatal(err)
	}
	id, err := s.QueueNativePromotion(p, 3)
	if err != nil {
		t.Fatal(err)
	}
	r := &sim.Request{ID: "after-failure", InputTokens: []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}, OutputTokens: []sim.TokenID{10}, MaxOutputLen: 1}
	injected := false
	e.EventObserver = func(event sim.Event, before bool) {
		if before || injected {
			return
		}
		x, ok := event.(*sim.NativeAsyncEvent)
		if !ok {
			return
		}
		if s.nativeControl == nil && s.nativeControlSerial == 1 && len(s.GetCachedBlocks(r.InputTokens)) == 1 {
			r.ArrivalTime = x.At
			injected = true
			e.InjectArrival(r)
		}
	}
	e.Run()
	loads := map[string]PeerRecord{}
	for _, row := range h.records {
		if row.Name == "transfer_end" && row.Destination == "hbm" {
			loads[row.Request] = row
		}
	}
	if !injected || r.State != sim.StateCompleted || loads[id].Bytes != 200 || loads[r.ID].Bytes != 400 || loads[r.ID].EffectiveBytes != 300 || loads[r.ID].Outcome != "" {
		t.Fatal("foreground did not fully restore after failure", loads, r.State)
	}
	nativeControlDrained(t, s)
}
