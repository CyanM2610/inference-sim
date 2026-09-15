package main

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/policylab"
)

// These registries are the command's policy extension point. An algorithm adds
// its implementation and registration here without replacing experiment I/O,
// event execution, cache ownership or TTFT accounting.
var requestPlugins = map[string]func() sim.DecisionPolicy{
	"decode_fill": func() sim.DecisionPolicy {
		base, _ := sim.NewBalancedBatchPolicy(100, 0, 0, 0)
		p, _ := sim.NewDecodeFillPolicy(base)
		return p
	},
	"budget_queues": func() sim.DecisionPolicy { p, _ := sim.NewBudgetQueuePolicy(1, 0, 4, 1); return p },
	"prefill_sjf":   func() sim.DecisionPolicy { return &sim.PrefillSJFPolicy{} },
	"demand_fcfs":   func() sim.DecisionPolicy { p, _ := sim.NewPrefixQueuePolicy("fcfs", false); return p },
	"prefetch_fcfs": func() sim.DecisionPolicy { p, _ := sim.NewPrefetchQueuePolicy("fcfs"); return p },
	"fcfs":          func() sim.DecisionPolicy { p, _ := sim.NewQueueDecisionPolicy("fcfs", 0); return p },
	"sjf":           func() sim.DecisionPolicy { p, _ := sim.NewQueueDecisionPolicy("sjf", 0); return p },
	"edf":           func() sim.DecisionPolicy { p, _ := sim.NewQueueDecisionPolicy("edf", 0); return p },
}

var peerPlugins = map[string]func() kv.PeerPolicy{
	"drop_lru":       func() kv.PeerPolicy { return kv.BuiltinPeerPolicy{Name: "lru_drop"} },
	"shortest_store": func() kv.PeerPolicy { return shortestStorePolicy{} },
}

var poolPlugins = map[string]func() kv.PoolEvictionPolicy{
	"time_lru":   func() kv.PoolEvictionPolicy { return kv.TimestampLRU{} },
	"prefix_lru": func() kv.PoolEvictionPolicy { return &kv.PrefixLRU{} },
}

var transferPlugins = map[string]func() kv.TransferSubmissionPolicy{
	"fifo":     func() kv.TransferSubmissionPolicy { p, _ := kv.NewTransferSubmissionPolicy("fifo"); return p },
	"deadline": func() kv.TransferSubmissionPolicy { p, _ := kv.NewTransferSubmissionPolicy("deadline"); return p },
}

// shortestStorePolicy is an example local reclaim policy, not a paper baseline
// or active placement optimizer. Ready copies avoid writes; otherwise select
// the lowest exposed store + queue estimate, with stable target-order ties.
// Queue estimates are only populated when the backend enables QueueAware.
type shortestStorePolicy struct{}

func (shortestStorePolicy) Choose(c kv.PeerReclaimContext) kv.PeerDecision {
	for _, candidate := range c.Candidates {
		if len(candidate.ReadyCopies) > 0 {
			return kv.PeerDecision{BlockID: candidate.ID}
		}
	}
	d := kv.PeerDecision{BlockID: c.Candidates[0].ID}
	best := math.Inf(1)
	for _, target := range c.Targets {
		cost := float64(target.StoreUS) + float64(target.QueueUS)
		if target.Available && cost < best {
			best, d.Pool = cost, target.Pool
		}
	}
	return d
}

func policyFactories(request, peer, pool string, transfer ...string) (policylab.PolicyFactories, error) {
	factories := policylab.PolicyFactories{}
	if len(transfer) > 1 {
		return factories, fmt.Errorf("at most one transfer submission plugin is supported")
	}
	if len(transfer) == 1 && transfer[0] != "" {
		build, ok := transferPlugins[transfer[0]]
		if !ok {
			return factories, fmt.Errorf("unknown transfer submission plugin %q", transfer[0])
		}
		factories.TransferSubmission = func(string) kv.TransferSubmissionPolicy { return build() }
	}
	if request != "" {
		build, ok := requestPlugins[request]
		if !ok {
			return factories, fmt.Errorf("unknown request plugin %q", request)
		}
		factories.Decision = func(string) sim.DecisionPolicy { return build() }
	}
	if peer != "" {
		build, ok := peerPlugins[peer]
		if !ok {
			return factories, fmt.Errorf("unknown KV plugin %q", peer)
		}
		factories.Peer = func(string) kv.PeerPolicy { return build() }
	}
	if pool != "" {
		build, ok := poolPlugins[pool]
		if !ok {
			return factories, fmt.Errorf("unknown pool plugin %q", pool)
		}
		factories.PoolEviction = func(string) kv.PoolEvictionPolicy { return build() }
	}
	return factories, nil
}
