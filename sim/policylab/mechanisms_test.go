package policylab

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestMechanismsDisabledParity(t *testing.T) {
	c := labConfig(2, "both")
	a, e := Run(c)
	if e != nil {
		t.Fatal(e)
	}
	c.Mechanisms = &kv.PeerMechanisms{}
	b, e := Run(c)
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(a.Events, b.Events) || !reflect.DeepEqual(a.Requests, b.Requests) {
		t.Fatal("disabled mechanisms changed baseline")
	}
}
func TestMechanismsCombinedWindows(t *testing.T) {
	for _, late := range []bool{false, true} {
		for _, window := range []int{1, 4} {
			for _, policy := range []string{"fifo", "read_first", "dependency_edf"} {
				c := labConfig(4, "both")
				c.Mechanisms = &kv.PeerMechanisms{DirectoryUS: 25, CompletionUS: 15, ControlWorkers: 2, RestoreWindow: window, ReserveAtDispatch: late, TransferPolicy: policy, QueueAware: true, CancelQueuedPromotions: true}
				for i := range c.Requests {
					c.Requests[i].TTFTSLOUS = 20000
					c.Requests[i].At = int64(i) * 1500
				}
				r, e := Run(c)
				if e != nil {
					t.Fatalf("late=%v window=%d policy=%s: %v", late, window, policy, e)
				}
				if r.Counts["control_start"] == 0 {
					t.Fatal("control mechanism unexercised")
				}
				for _, p := range r.Pools {
					if p["reserved"] != 0 || p["read_pins"] != 0 {
						t.Fatal("L2 leak")
					}
				}
			}
		}
	}
}
func TestMechanismsOnlineHistoryAndBudget(t *testing.T) {
	c := labConfig(2, "both")
	c.Mechanisms = &kv.PeerMechanisms{RestoreWindow: 4, ReserveAtDispatch: true, Online: &kv.PeerOnlinePromotion{PeriodUS: 50000, EndUS: 550000, HalfLifeUS: 1000000, MinFrequency: 0.5, BlocksPerTick: 2, BytesPerSecond: 40 * 1024 * 1024, BurstBlocks: 2}}
	r, e := Run(c)
	if e != nil {
		t.Fatal(e)
	}
	promoted := int64(0)
	first := int64(0)
	for _, v := range r.Events {
		if v.Name == "transfer_start" && v.Reason == "promotion" {
			promoted += v.Bytes
			if first == 0 {
				first = v.Time
			}
		}
	}
	if promoted == 0 || first < 200000 {
		t.Fatalf("promotion absent or used future history: bytes=%d first=%d", promoted, first)
	}
	limit := int64(len(c.Instances)) * (2*r.BlockBytes + 40*1024*1024*550000/1000000)
	if promoted > limit {
		t.Fatal("byte budget exceeded")
	}
}

func TestMechanismsRandomPressure(t *testing.T) {
	for seed := int64(0); seed < 6; seed++ {
		c := labConfig(2, "both")
		c.Requests = nil
		rng := rand.New(rand.NewSource(seed))
		c.Mechanisms = &kv.PeerMechanisms{DirectoryUS: 20, CompletionUS: 30, PolicyUS: 10, ControlWorkers: 1, RestoreWindow: 4, ReserveAtDispatch: true, TransferPolicy: "dependency_edf", QueueAware: true, CancelQueuedPromotions: true, Online: &kv.PeerOnlinePromotion{PeriodUS: 5000, EndUS: 100000, HalfLifeUS: 50000, MinFrequency: 0.5, BlocksPerTick: 2, BytesPerSecond: 40 * 1024 * 1024, BurstBlocks: 2}}
		for i := 0; i < 60; i++ {
			n := 17 + rng.Intn(65)
			group := rng.Intn(4)
			tokens := make([]sim.TokenID, n)
			for j := range tokens {
				tokens[j] = sim.TokenID(group*1000 + j)
			}
			c.Requests = append(c.Requests, RequestConfig{ID: fmt.Sprint(i), At: int64(i) * 1000, Input: tokens, Output: make([]sim.TokenID, 1+rng.Intn(24)), TTFTSLOUS: 50000})
		}
		r, e := Run(c)
		if e != nil {
			t.Fatalf("seed %d: %v; final HBM: %v", seed, e, r.HBM)
		}
		for _, p := range r.Pools {
			if p["read_pins"] != 0 || p["reserved"] != 0 {
				t.Fatal("L2 leak")
			}
		}
	}
}
