package policylab

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestRequestEDFUsesAbsoluteDeadlineAndStableTies(t *testing.T) {
	q := []*sim.Request{{ID: "no-target", ArrivalTime: 0}, {ID: "loose", ArrivalTime: 1}, {ID: "tie-b", ArrivalTime: 2}, {ID: "tie-a", ArrivalTime: 2}, {ID: "urgent", ArrivalTime: 3}}
	s := &requestScheduler{name: "edf", deadlines: map[string]int64{"loose": 100, "tie-a": 40, "tie-b": 40, "urgent": 20}}
	s.OrderQueue(q, 10)
	ids := []string{}
	for _, r := range q {
		ids = append(ids, r.ID)
	}
	if !reflect.DeepEqual(ids, []string{"urgent", "tie-a", "tie-b", "loose", "no-target"}) {
		t.Fatal(ids)
	}
}

func TestExplicitFCFSRetainsPriorBehavior(t *testing.T) {
	c := labConfig(2, "both")
	for i := range c.Requests {
		c.Requests[i].At = int64(i) * 1000
	}
	a, e := Run(c)
	if e != nil {
		t.Fatal(e)
	}
	c.RequestScheduler = "fcfs"
	b, e := Run(c)
	if e != nil {
		t.Fatal(e)
	}
	filtered := []kv.PeerRecord{}
	for _, event := range b.Events {
		if event.Name != "request_queue_order" {
			filtered = append(filtered, event)
		}
	}
	if !reflect.DeepEqual(a.Events, filtered) || !reflect.DeepEqual(a.Requests, b.Requests) {
		t.Fatal("explicit FCFS changed execution")
	}
}

func TestRequestOrderingAffectsActualAdmission(t *testing.T) {
	c := labConfig(1, "hbm")
	c.MaxSequences = 1
	c.Instances[0].HBMBlocks = 32
	c.Requests = []RequestConfig{
		{ID: "blocker", At: 0, Input: make([]sim.TokenID, 65), Output: make([]sim.TokenID, 16), TTFTSLOUS: 100000},
		{ID: "long", At: 1, Input: make([]sim.TokenID, 129), Output: make([]sim.TokenID, 2), TTFTSLOUS: 200000},
		{ID: "short", At: 2, Input: make([]sim.TokenID, 17), Output: make([]sim.TokenID, 2), TTFTSLOUS: 10000},
	}
	for _, policy := range []string{"fcfs", "sjf", "edf"} {
		c.RequestScheduler = policy
		r, e := Run(c)
		if e != nil {
			t.Fatal(e)
		}
		order := []string{}
		for _, v := range r.Events {
			if v.Name == "sim.ScheduledEvent" {
				order = append(order, v.Request)
			}
		}
		want := []string{"blocker", "long", "short"}
		if policy != "fcfs" {
			want = []string{"blocker", "short", "long"}
		}
		if !reflect.DeepEqual(order, want) {
			t.Fatalf("%s: %v want %v", policy, order, want)
		}
	}
}

func TestRequestSchedulerValidation(t *testing.T) {
	c := labConfig(1, "hbm")
	c.RequestScheduler = "not-a-policy"
	if c.Validate() == nil {
		t.Fatal("unknown request scheduler accepted")
	}
}
