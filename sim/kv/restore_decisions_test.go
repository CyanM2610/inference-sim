package kv

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func restoreDecisionFixture(t *testing.T) (*peerHarness, *PeerCache, *sim.Request) {
	t.Helper()
	h, s, tokens := phaseFixture(t, EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, PollUS: 1000, TailUS: 2})
	// The CPU fixture contains both blocks; target HBM can fit exactly the prompt.
	s.KVCacheState = NewKVCacheState(2, 2)
	s.SetClock(0)
	if err := s.EnableRestoreDecisions(); err != nil {
		t.Fatal(err)
	}
	return h, s, &sim.Request{ID: "reader", InputTokens: append([]sim.TokenID(nil), tokens[:4]...)}
}

func finishRestoreStep(t *testing.T, h *peerHarness, s *PeerCache) {
	t.Helper()
	var end int64
	if ok, err := s.BeginBatch(s.clock, nil, func(at int64) { end = at }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for n := 0; end == 0; n++ {
		if n > 100 {
			t.Fatal("phase did not finish")
		}
		phaseNext(t, h)
	}
}

func TestRestoreChoicesChangePhysicalCopiesAndRecompute(t *testing.T) {
	for _, limit := range []int64{0, 1, 2} {
		t.Run(fmt.Sprint(limit), func(t *testing.T) {
			h, s, r := restoreDecisionFixture(t)
			updates := []sim.DecisionRestore{{Request: r.ID, MaxPrefixBlocks: limit}}
			if err := s.ValidateRestoreDecisions(updates); err != nil {
				t.Fatal(err)
			}
			s.ApplyRestoreDecisions(updates)
			initial := s.DecisionState([]sim.DecisionKVQuery{{ID: r.ID, Input: r.InputTokens}}).Requests[0]
			if initial.RecoverablePrefixBlocks != 2 || initial.RecoverablePrefixTokens != 3 || initial.LocalPrefixTokens != 0 || !initial.RestoreLimitSet {
				t.Fatalf("physical/compute view mixed: %+v", initial)
			}
			admitted := s.AllocateKVBlocks(r, 0, 4, nil)
			if admitted != (limit == 0) {
				t.Fatal("incorrect first admission", admitted)
			}
			if limit > 0 {
				pending := h.f.Pending()
				for retry := 0; retry < 3; retry++ {
					if err := s.ValidateRestoreDecisions(updates); err != nil {
						t.Fatal(err)
					}
					s.ApplyRestoreDecisions(updates)
					if s.AllocateKVBlocks(r, 0, 4, nil) || h.f.Pending() != pending {
						t.Fatal("retry duplicated load or admitted early")
					}
				}
				if len(s.GetRequestCachedPrefix(r).Blocks) != 0 {
					t.Fatal("unadopted KV visible")
				}
				finishRestoreStep(t, h, s)
				prefix := s.GetRequestCachedPrefix(r)
				wantTokens := min(limit*2, int64(3))
				if int64(len(prefix.Blocks)) != limit || prefix.Tokens != wantTokens {
					t.Fatal("wrong restored prefix", prefix)
				}
				if !s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks) {
					t.Fatal("restored request cannot run")
				}
				if r.ProgressIndex != 0 || r.TTFTSet {
					t.Fatal("admission fabricated computation/output")
				}
			}
			var copied int64
			for _, record := range h.records {
				if record.Name == "transfer_start" && record.Request == r.ID && record.Destination == "hbm" {
					copied += record.Bytes
				}
			}
			if copied != limit*h.f.bytes || len(s.RequestMap[r.ID]) != 2 || s.FreeBlockCnt != 0 {
				t.Fatalf("wrong copies/allocation: bytes=%d blocks=%v free=%d", copied, s.RequestMap[r.ID], s.FreeBlockCnt)
			}
			s.ReleaseKVBlocks(r)
			assertPeerConservation(t, s)
			if s.FreeBlockCnt != 2 || len(s.holds) != 0 || len(s.restoreLimits) != 0 || len(s.restoredPrefixBlocks) != 0 || h.f.Pending() != 0 {
				t.Fatal("restore state leaked")
			}
		})
	}
}

func TestRestoreLimitChangeRejectedAtomicallyWhilePending(t *testing.T) {
	h, s, r := restoreDecisionFixture(t)
	s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: r.ID, MaxPrefixBlocks: 2}})
	s.AllocateKVBlocks(r, 0, 4, nil)
	before := s.PeerSnapshot()
	updates := []sim.DecisionRestore{{Request: "other", MaxPrefixBlocks: 0}, {Request: r.ID, MaxPrefixBlocks: 0}}
	if err := s.ValidateRestoreDecisions(updates); err == nil {
		t.Fatal("changed pending DMA range")
	}
	if !reflect.DeepEqual(before, s.PeerSnapshot()) || len(s.restoreLimits) != 1 || s.restoreLimits[r.ID] != 2 {
		t.Fatal("validation mutated another action")
	}
	s.ClearDeferred(r.ID)
	if s.FreeBlockCnt != 0 {
		t.Fatal("cancel released active DMA targets")
	}
	finishRestoreStep(t, h, s)
	if s.FreeBlockCnt != 2 || len(s.holds) != 0 || len(s.restoreLimits) != 0 || len(s.restoredPrefixBlocks) != 0 || h.f.Snapshot()["cxl"]["read_pins"] != 0 {
		t.Fatal("cancel/adoption leaked ownership")
	}
	assertPeerConservation(t, s)
}

func TestLowerRestoreLimitKeepsCompletedHitsAndAPCBoundary(t *testing.T) {
	h, s, r := restoreDecisionFixture(t)
	s.AllocateKVBlocks(r, 0, 4, nil) // unspecified means automatic full restore
	finishRestoreStep(t, h, s)
	u := []sim.DecisionRestore{{Request: r.ID, MaxPrefixBlocks: 0}}
	if err := s.ValidateRestoreDecisions(u); err != nil {
		t.Fatal(err)
	}
	s.ApplyRestoreDecisions(u)
	prefix := s.GetRequestCachedPrefix(r)
	if prefix.Tokens != 3 || len(prefix.Blocks) != 2 {
		t.Fatal("lower limit discarded completed load")
	}
	other := &sim.Request{ID: "apc-only", InputTokens: r.InputTokens}
	if p := s.GetRequestCachedPrefix(other); p.Tokens != 2 || len(p.Blocks) != 1 {
		t.Fatal("ordinary APC incorrectly reused full prompt")
	}
	if !s.AllocateKVBlocks(r, prefix.Tokens, 4, prefix.Blocks) {
		t.Fatal("completed full hit could not replay")
	}
	s.ReleaseKVBlocks(r)
	assertPeerConservation(t, s)
	s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: other.ID, MaxPrefixBlocks: 0}})
	local := s.GetRequestCachedPrefix(other)
	if local.Tokens != 2 || !s.AllocateKVBlocks(other, local.Tokens, 4, local.Blocks) || h.f.Pending() != 0 {
		t.Fatal("zero loading disabled ordinary HBM reuse or submitted a new load")
	}
	s.ReleaseKVBlocks(other)
	assertPeerConservation(t, s)
}

func TestRestoreViewDoesNotRevealPhysicalCompletion(t *testing.T) {
	h, s, r := restoreDecisionFixture(t)
	s.fabric.phases.model = fixedEnginePhases{EngineStepTiming{PreForwardUS: 2, PostForwardUS: 10, GPUReadyUS: 30, OutputReadyUS: 30, PollUS: 1, TailUS: 2}, 3}
	s.AllocateKVBlocks(r, 0, 4, nil)
	finishRestoreStep(t, h, s) // worker poll precedes completion
	q := []sim.DecisionKVQuery{{ID: r.ID, Input: r.InputTokens}}
	s.EnableHashWorkAccounting()
	s.TakeHashWork()
	a := s.DecisionState(q)
	for h.f.active > 0 {
		phaseNext(t, h)
	}
	b := s.DecisionState(q)
	if !reflect.DeepEqual(a, b) || b.Requests[0].LocalPrefixTokens != 0 || s.TakeHashWork().Calls != 0 {
		t.Fatal("snapshot exposed future/unadopted state or mutated hash accounting")
	}
	s.SetClock(1000)
	finishRestoreStep(t, h, s)
	if p := s.GetRequestCachedPrefix(r); p.Tokens != 3 {
		t.Fatal("adoption did not unlock prefix")
	}
	s.ClearDeferred(r.ID)
	assertPeerConservation(t, s)
}

func TestRestoreCanExtendReceivedPrefixBeforeCompute(t *testing.T) {
	for _, cancel := range []bool{false, true} {
		t.Run(fmt.Sprint(cancel), func(t *testing.T) {
			h, s, r := restoreDecisionFixture(t)
			s.ApplyRestoreDecisions([]sim.DecisionRestore{{Request: r.ID, MaxPrefixBlocks: 1}})
			if s.AllocateKVBlocks(r, 0, 4, nil) {
				t.Fatal("admitted before first load")
			}
			finishRestoreStep(t, h, s)
			first := s.GetRequestCachedPrefix(r)
			if first.Tokens != 2 || len(first.Blocks) != 1 || len(s.RequestMap[r.ID]) != 0 {
				t.Fatal("first restore not held separately from computation")
			}
			u := []sim.DecisionRestore{{Request: r.ID, MaxPrefixBlocks: 2}}
			if err := s.ValidateRestoreDecisions(u); err != nil {
				t.Fatal(err)
			}
			s.ApplyRestoreDecisions(u)
			if s.AllocateKVBlocks(r, first.Tokens, 4, first.Blocks) {
				t.Fatal("admitted before second load")
			}
			if p := s.GetRequestCachedPrefix(r); p.Tokens != 2 || p.Blocks[0] != first.Blocks[0] {
				t.Fatal("second load replaced or prematurely extended first prefix")
			}
			if cancel {
				s.ClearDeferred(r.ID)
			}
			finishRestoreStep(t, h, s)
			var loads, bytes int64
			for _, e := range h.records {
				if e.Name == "transfer_start" && e.Request == r.ID && e.Destination == "hbm" {
					loads++
					bytes += e.Bytes
				}
			}
			if loads != 2 || bytes != 2*h.f.bytes || r.ProgressIndex != 0 || r.TTFTSet {
				t.Fatal("duplicate bytes or fabricated compute", loads, bytes)
			}
			if !cancel {
				p := s.GetRequestCachedPrefix(r)
				if p.Tokens != 3 || p.Blocks[0] != first.Blocks[0] || !s.AllocateKVBlocks(r, p.Tokens, 4, p.Blocks) {
					t.Fatal("extended prefix unusable")
				}
				if s.ValidateRestoreDecisions(u) == nil {
					t.Fatal("committed computation permits another restore update")
				}
				s.ReleaseKVBlocks(r)
			}
			assertPeerConservation(t, s)
			if s.FreeBlockCnt != 2 || len(s.holds) != 0 || h.f.Pending() != 0 || h.f.Snapshot()["cxl"]["read_pins"] != 0 {
				t.Fatal("continued restore leaked resources")
			}
		})
	}
}
