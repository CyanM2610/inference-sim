package policylab

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestLiveFirstOutputSurvivesSchedulerPreemption(t *testing.T) {
	c := liveConfig()
	makeTokens := func(start, count int) []sim.TokenID {
		tokens := make([]sim.TokenID, count)
		for i := range tokens {
			tokens[i] = sim.TokenID(start + i)
		}
		return tokens
	}
	// Five active pages and one cached warmup page occupy the six-slot arena.
	// Crossing the next decode boundary starts a victim STORE. The real batch
	// formation path preempts the running request while that source is pinned.
	c.Requests = []RequestConfig{
		{ID: "warmup", Input: makeTokens(1000, 17), Output: []sim.TokenID{0}},
		{ID: "primary", At: 1000, Input: makeTokens(100, 79), Output: []sim.TokenID{0, 0, 0, 0}},
	}
	l, err := NewLiveController(c)
	if err != nil {
		t.Fatal(err)
	}
	inputs := map[string]int64{"warmup": 17, "primary": 79}
	first := map[string]int64{}
	observed := map[string]int64{}
	stores, repeated := 0, 0
	poll := LivePoll{}
	for round := 0; round < 2000; round++ {
		reply, err := l.Poll(poll)
		if err != nil {
			t.Fatal(err)
		}
		if reply.Done {
			if stores == 0 || repeated == 0 || observed["primary"] != 4 {
				t.Fatalf("fixture missed real preemption: stores=%d repeated=%d outputs=%v", stores, repeated, observed)
			}
			for _, request := range reply.Requests {
				if request.FirstTokenUS != first[request.ID] {
					t.Fatalf("%s first output overwritten by recomputation: got %d, first observed %d (reobservations=%d)", request.ID, request.FirstTokenUS, first[request.ID], repeated)
				}
			}
			return
		}
		poll = LivePoll{NowUS: poll.NowUS + 100}
		if reply.Batch != nil {
			command := reply.Batch
			poll.Batch = command.ID
			index := command.Prefix + command.Query - inputs[command.Request]
			if index >= 0 {
				token := sim.TokenID(9000 + index)
				poll.Token = &token
				if index < observed[command.Request] {
					repeated++
				} else if index == observed[command.Request] {
					if index == 0 {
						first[command.Request] = poll.NowUS
					}
					observed[command.Request]++
				} else {
					t.Fatal("scheduler skipped an output")
				}
			}
		}
		for _, transfer := range reply.Copies {
			poll.Copies = append(poll.Copies, transfer.ID)
			if transfer.Reason == "store" {
				stores++
			}
		}
		if poll.Batch == 0 && len(poll.Copies) == 0 && reply.NextEventUS > poll.NowUS {
			poll.NowUS = reply.NextEventUS
		}
	}
	t.Fatal("controller recovery did not finish")
}
