package policylab

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestRankingObservationSurvivesPostOutputPreemption(t *testing.T) {
	c := liveConfig()
	tokens := func(start, count int) []sim.TokenID {
		out := make([]sim.TokenID, count)
		for i := range out {
			out[i] = sim.TokenID(start + i)
		}
		return out
	}
	c.Requests = []RequestConfig{
		{ID: "warmup", Input: tokens(1000, 17), Output: []sim.TokenID{9}},
		{ID: "primary", At: 100000, Input: tokens(100, 79), Output: []sim.TokenID{10, 11, 12, 13}},
	}
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	var firstComputeEnd int64
	for _, event := range r.Events {
		if event.Name == "compute" && len(event.Requests) == 1 && event.Requests[0] == "primary" {
			firstComputeEnd = event.Time + event.Duration
			break
		}
	}
	if firstComputeEnd == 0 || r.FirstTokenUS["primary"] != firstComputeEnd {
		t.Fatalf("first output must come from the initial full prefill: observed=%d first_compute=%d", r.FirstTokenUS["primary"], firstComputeEnd)
	}
	// The legacy core metric reflects the last re-prefill. This fixture must
	// exercise that difference, so the assertion cannot pass without preemption.
	for _, q := range r.Requests {
		if q.ID == "primary" && int64(q.TTFT*1000)+100000 <= firstComputeEnd {
			t.Fatal("fixture did not reach post-output re-prefill")
		}
	}
	if r.FinishedUS["primary"] <= firstComputeEnd {
		t.Fatal("missing final completion")
	}
}
