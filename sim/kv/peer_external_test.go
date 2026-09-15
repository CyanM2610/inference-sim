package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

func TestExternalStoreCompletionIsObservedThroughEventQueue(t *testing.T) {
	h := &peerHarness{}
	f, err := NewPeerFabric([]PeerResource{{ID: "pcie", BytesPerUS: 23000, LatencyUS: 3}}, []PeerPoolConfig{{ID: "dram", CapacityBlocks: 4}}, 917504, func(r PeerRecord) { h.records = append(h.records, r) })
	if err != nil {
		t.Fatal(err)
	}
	h.f = f
	s, err := NewPeerCache("instance_0", 4, 16, f, []PeerAccess{{Pool: "dram", ReadPath: []string{"pcie"}, WritePath: []string{"pcie"}}}, BuiltinPeerPolicy{Name: "lfu_store", MinFrequency: 1})
	if err != nil {
		t.Fatal(err)
	}
	s.BindEvents(func(e sim.Event) { h.events = append(h.events, e) }, func(int64) {})
	if err = f.ConfigureExternalTransfers(); err != nil {
		t.Fatal(err)
	}
	tokens := make([]sim.TokenID, 64)
	for i := range tokens {
		tokens[i] = sim.TokenID(i + 1)
	}
	seedPeer(t, s, "seed", tokens)
	for _, b := range s.Blocks {
		s.frequency[b.Hash] = 1
	}
	if s.makeSpace(2, nil, "wait") {
		t.Fatal("pending stores made space immediately")
	}
	commands := f.DrainExternalTransfers()
	if len(commands) != 1 {
		t.Fatal("single transfer gate was bypassed")
	}
	first := commands[0]
	for i := 0; i < 4; i++ {
		s.SetClock(int64(1000 + i))
		if s.makeSpace(2, nil, "wait") || reclaimCount(h) != 2 {
			t.Fatal("elapsed time/retry appended extra victims or completed hardware")
		}
	}
	if len(h.events) != 0 || f.Snapshot()["dram"]["ready"] != 0 {
		t.Fatal("synthetic completion emitted")
	}
	if err = f.CompleteExternalTransfer(first.ID, 2000); err != nil {
		t.Fatal(err)
	}
	if f.Snapshot()["dram"]["ready"] != 0 || len(h.events) != 1 {
		t.Fatal("acknowledgement bypassed timestamp-ordered publication")
	}
	h.next(t)
	if f.Snapshot()["dram"]["ready"] != 1 {
		t.Fatal("actual completion did not publish DRAM")
	}
	if err = f.CompleteExternalTransfer(first.ID, 2001); err == nil {
		t.Fatal("duplicate acknowledgement accepted")
	}
	commands = f.DrainExternalTransfers()
	if len(commands) != 1 {
		t.Fatal("next copy not dispatched after actual completion")
	}
	if err = f.CompleteExternalTransfer(commands[0].ID, 3000); err != nil {
		t.Fatal(err)
	}
	h.next(t)
	if !s.makeSpace(2, nil, "wait") || reclaimCount(h) != 2 {
		t.Fatal("two acknowledged stores did not release exactly the requested space")
	}
	assertPeerConservation(t, s)
}

func TestExternalAcknowledgementRequiresActualTransactionCompletion(t *testing.T) {
	x, req := pagedExecutorFixture(t)
	h := &peerHarness{}
	f, err := NewPeerFabric([]PeerResource{{ID: "pcie", BytesPerUS: 23000, LatencyUS: 3}}, []PeerPoolConfig{{ID: "dram", CapacityBlocks: 64}}, 917504, func(r PeerRecord) { h.records = append(h.records, r) })
	if err != nil {
		t.Fatal(err)
	}
	h.f = f
	s, err := NewPeerCache("instance_0", 1, 16, f, []PeerAccess{{Pool: "dram", ReadPath: []string{"pcie"}, WritePath: []string{"pcie"}}}, BuiltinPeerPolicy{Name: "lfu_store", MinFrequency: 1})
	if err != nil {
		t.Fatal(err)
	}
	s.BindEvents(func(e sim.Event) { h.events = append(h.events, e) }, func(int64) {})
	if err = f.ConfigureExternalTransfers(); err != nil {
		t.Fatal(err)
	}
	if err = f.AttachExternalMemory(x); err != nil {
		t.Fatal(err)
	}
	if !s.AllocateKVBlocks(req, 0, 16, nil) {
		t.Fatal("allocation failed")
	}
	if _, err = x.StartBatch(1, req.ID, 0, 16, s.RequestMap[req.ID], req.InputTokens); err != nil {
		t.Fatal(err)
	}
	if err = x.Runtime.Engine.Run(); err != nil {
		t.Fatal(err)
	}
	req.ProgressIndex = 16
	s.MirrorToCPU([]*sim.Request{req})
	s.ReleaseKVBlocks(req)
	for _, block := range s.Blocks {
		s.frequency[block.Hash] = 1
	}
	if s.makeSpace(1, nil, "waiting") {
		t.Fatal("space freed before store")
	}
	commands := f.DrainExternalTransfers()
	if len(commands) != 1 {
		t.Fatal("missing physical store")
	}
	command := commands[0]
	if err = f.CompleteExternalTransfer(command.ID, 2000); err == nil {
		t.Fatal("fabric accepted ACK without physical copy")
	}
	if len(h.events) != 0 || f.Pending() == 0 {
		t.Fatal("failed ACK changed pending policy state")
	}
	// Even identical old bytes cannot acknowledge a different transaction.
	old := command
	old.ID = 1000
	if _, err = x.StartTransfer(old); err != nil {
		t.Fatal(err)
	}
	if err = x.Runtime.Engine.Run(); err != nil {
		t.Fatal(err)
	}
	if err = x.CheckContents(command.Hash, command.Destination); err != nil {
		t.Fatal(err)
	}
	if err = f.CompleteExternalTransfer(command.ID, 2000); err == nil {
		t.Fatal("old matching bytes acknowledged an unexecuted transaction")
	}
	if _, err = x.StartTransfer(command); err != nil {
		t.Fatal(err)
	}
	if err = x.Runtime.Engine.Run(); err != nil {
		t.Fatal(err)
	}
	if err = f.CompleteExternalTransfer(command.ID, 2000); err != nil {
		t.Fatal(err)
	}
	if f.Snapshot()["dram"]["ready"] != 0 {
		t.Fatal("ACK bypassed policy event publication")
	}
	h.next(t)
	if err = x.CheckContents(command.Hash, command.Source); err == nil {
		t.Fatal("eviction did not invalidate the physical HBM slot")
	}
	if err = f.ValidateExternalMemory(); err != nil {
		t.Fatal(err)
	}
	if err = x.ValidateDrained(); err != nil {
		t.Fatal(err)
	}
}
