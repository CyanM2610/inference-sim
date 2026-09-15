package kv

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestCapacityReservationViewIsPureAndTracksRelease(t *testing.T) {
	_, s, _ := newPeerHarness(t, false)
	if err := s.EnableDecodeCapacityReservation(); err != nil {
		t.Fatal(err)
	}
	a := &sim.Request{ID: "a", InputTokens: []sim.TokenID{1, 2, 3, 4}, MaxOutputLen: 4}
	b := &sim.Request{ID: "b", InputTokens: []sim.TokenID{10, 11}, MaxOutputLen: 1}
	if !s.AllocateKVBlocks(a, 0, 2, nil) {
		t.Fatal("seed reservation failed")
	}
	q := []sim.DecisionKVQuery{{ID: b.ID, Input: b.InputTokens, InputTokens: b.InputLen(), ClientOutputLimit: 1, CapacityReservation: true}}
	before, failure := s.PeerSnapshot(), s.LastAllocationFailure()
	view := s.DecisionState(q)
	preview := view.Requests[0].CapacityReservation
	if preview == nil || preview.Fits || !reflect.DeepEqual(before, s.PeerSnapshot()) || failure != s.LastAllocationFailure() {
		t.Fatal("preview changed ownership/failure or missed capacity pressure", preview)
	}
	if s.AllocateKVBlocks(b, 0, 2, nil) {
		t.Fatal("actual reservation should also fail")
	}
	if f := s.LastAllocationFailure(); f.NeededBlocks != preview.NeededBlocks || f.FreeBlocks != preview.FreeBlocks {
		t.Fatal("preview and actual reservation disagree", f, preview)
	}
	preview.Fits = true
	if s.DecisionState(q).Requests[0].CapacityReservation.Fits {
		t.Fatal("view aliases runtime state")
	}
	s.ReleaseKVBlocks(a)
	if !s.DecisionState(q).Requests[0].CapacityReservation.Fits || !s.AllocateKVBlocks(b, 0, 2, nil) {
		t.Fatal("released capacity remained blocked")
	}
	s.ReleaseKVBlocks(b)
	assertPeerConservation(t, s)
}
