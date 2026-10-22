package main

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/kv"
)

func TestShortestStoreUsesReadyCopiesAvailabilityAndQueue(t *testing.T) {
	c := kv.PeerReclaimContext{
		Candidates: []kv.PeerCandidate{{ID: 1}, {ID: 2}},
		Targets: []kv.PeerTarget{{Pool: "busy", Available: true, StoreUS: 2, QueueUS: 100},
			{Pool: "quick", Available: true, StoreUS: 10}, {Pool: "unavailable", StoreUS: 1}},
	}
	p := shortestStorePolicy{}
	if got := p.Choose(c); got.BlockID != 1 || got.Pool != "quick" {
		t.Fatal("ignored queue or selected unavailable target", got)
	}
	c.Candidates[1].ReadyCopies = []string{"busy"}
	if got := p.Choose(c); got.BlockID != 2 || got.Pool != "" {
		t.Fatal("ready copy caused redundant write", got)
	}
}

func TestPolicyPluginRegistrationRejectsUnknownNames(t *testing.T) {
	if _, err := policyFactories("", "", "", "unknown"); err == nil {
		t.Fatal("unknown submission plugin accepted")
	}
	if _, err := policyFactories("", "", "", "fifo", "deadline"); err == nil {
		t.Fatal("ambiguous submission plugins accepted")
	}
	for _, names := range [][3]string{{"unknown", "", ""}, {"", "unknown", ""}, {"", "", "unknown"}} {
		if _, err := policyFactories(names[0], names[1], names[2]); err == nil {
			t.Fatal("unknown plugin silently fell back", names)
		}
	}
	f, err := policyFactories("sjf", "shortest_store", "prefix_lru", "deadline")
	if err != nil || f.Decision == nil || f.Peer == nil || f.PoolEviction == nil || f.TransferSubmission == nil {
		t.Fatal("registered combination unavailable", err)
	}
	if f.PoolEviction("dram") == f.PoolEviction("cxl") {
		t.Fatal("stateful pool policy shared between factories")
	}
	order := f.TransferSubmission("instance_0").Order(kv.TransferSubmissionContext{Candidates: []kv.TransferSubmissionCandidate{
		{Transaction: 1}, {Transaction: 2, DeadlineKnown: true, DeadlineUS: 30}, {Transaction: 3, DeadlineKnown: true, DeadlineUS: 20},
	}})
	if len(order) != 3 || order[0] != 3 || order[1] != 2 || order[2] != 1 {
		t.Fatal("registered deadline policy not used", order)
	}
}
