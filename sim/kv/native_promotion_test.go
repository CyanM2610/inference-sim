package kv

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Separate server directory service from reply visibility. All numbers are
// artificial; this fixture tests ownership and asynchronous ordering only.
type nativeControlTestCost struct{ nativePageAccessCost }

func (nativeControlTestCost) ControlRPCPlan(shape sim.NativeControlRPCShape) sim.NativeControlRPCPlan {
	return sim.NativeControlRPCPlan{Stages: []sim.NativeStage{
		{Name: "directory", Resource: "mp", DurationUS: 30},
		{Name: "reply", DurationUS: 40, Delay: true, After: []string{"directory"}},
	}, Applied: "directory", Replied: "reply", CostScope: "artificial directory30/reply40us; no CPU calibration"}
}

func nativeControlHarness(t *testing.T) (*peerHarness, *PeerCache, *sim.Simulator, NativePromotionPrefix, []int64) {
	return nativeControlHarnessWithCost(t, nativeControlTestCost{})
}

func nativeControlHarnessWithCost(t *testing.T, cost sim.NativeHostCost) (*peerHarness, *PeerCache, *sim.Simulator, NativePromotionPrefix, []int64) {
	t.Helper()
	h, s := hostChunkHarness(t, 2, 4)
	if err := s.ConfigureNativeHostGeometry(1, 64<<20); err != nil {
		t.Fatal(err)
	}
	input := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}
	r := seedPeer(t, s, "seed", input)
	drainHost(t, h, s)
	ids := append([]int64(nil), s.GetCachedBlocks(input)...)
	for _, id := range ids[2:] {
		s.invalidate(s.Blocks[id])
	}
	e, err := sim.NewSimulator(sim.SimConfig{Horizon: 100000, Seed: 1, BatchCompletionEvents: true,
		KVCacheConfig: sim.KVCacheConfig{TotalKVBlocks: 24, BlockSizeTokens: 2},
		BatchConfig:   sim.BatchConfig{MaxNumSeqs: 2, MaxNumBatchedTokens: 64}}, s, nativeTestCost{})
	if err != nil {
		t.Fatal(err)
	}
	if err = e.EnableNativeAsync(2, cost); err != nil {
		t.Fatal(err)
	}
	prefix, err := s.SnapshotNativePromotionPrefix(r, 8)
	if err != nil {
		t.Fatal(err)
	}
	h.records = nil
	s.SetClock(0)
	return h, s, e, prefix, ids[:2]
}

type delayedNativeLookupCost struct{ nativeControlTestCost }

func (delayedNativeLookupCost) ControlRPCPlan(shape sim.NativeControlRPCShape) sim.NativeControlRPCPlan {
	p := nativeControlTestCost{}.ControlRPCPlan(shape)
	if shape.Operation == "lookup" {
		p.Stages[0].After = []string{"network"}
		p.Stages = append([]sim.NativeStage{{Name: "network", DurationUS: 100, Delay: true}}, p.Stages...)
	}
	return p
}

func TestNativeControlENDTombstoneRejectsLateLookupWithoutResurrection(t *testing.T) {
	h, s, e, prefix, _ := nativeControlHarnessWithCost(t, delayedNativeLookupCost{})
	before := nativeControlHBMState(s)
	id, err := s.QueueNativePromotion(prefix, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.CancelNativePromotion(id)
	e.Run()
	retired, lookup := int64(-1), int64(-1)
	for _, row := range h.records {
		if row.Request != id {
			continue
		}
		switch row.Name {
		case "host_lookup_reserved", "native_control_captured":
			t.Fatal("late lookup resurrected ended control", row)
		case "native_control_retired":
			retired = row.Time
		case "native_control_lookup_ready":
			lookup = row.Time
		}
	}
	if retired < 0 || retired >= lookup || !reflect.DeepEqual(before, nativeControlHBMState(s)) {
		t.Fatal("fixture did not let guarded END overtake LOOKUP", retired, lookup)
	}
	nativeControlDrained(t, s)
}

func TestNativeControlRequiresRPCModelBeforeCreatingOwnership(t *testing.T) {
	_, s, _, prefix, _ := nativeControlHarnessWithCost(t, nativePageAccessCost{})
	before := nativeControlHBMState(s)
	if id, err := s.QueueNativePromotion(prefix, 2); err == nil || id != "" {
		t.Fatal("accepted uncosted control protocol")
	}
	if s.nativeControlSerial != 0 || !reflect.DeepEqual(before, nativeControlHBMState(s)) {
		t.Fatal("unsupported control mutated ownership")
	}
	nativeControlDrained(t, s)
}

func nativeControlDrained(t *testing.T, s *PeerCache) {
	t.Helper()
	if s.NativeControlPending() || s.nativeControlRPCs != 0 || s.fabric.Pending() != 0 || s.UsedBlocks() != 0 || len(s.restoring) != 0 {
		t.Fatal("control did not drain", s.PeerSnapshot())
	}
	for _, o := range s.fabric.hostChunks.objects {
		if o.lookupReaders != 0 || o.transferRefs != 0 {
			t.Fatal("control retained host ownership", o.key, o.lookupReaders, o.transferRefs)
		}
	}
}

func nativeControlHBMState(s *PeerCache) []any {
	var state []any
	for _, b := range s.Blocks {
		state = append(state, []any{b.ID, b.Hash, b.RefCount, b.InUse})
	}
	var free []int64
	for b := s.FreeHead; b != nil; b = b.NextFree {
		free = append(free, b.ID)
	}
	return append(state, free)
}

func TestNativeControlBudgetAndCapacityRejectionPreserveEntireHBM(t *testing.T) {
	for _, mode := range []string{"budget", "capacity"} {
		t.Run(mode, func(t *testing.T) {
			h, s, e, prefix, cached := nativeControlHarness(t)
			budget := 1 // two missing pages cannot be truncated to this budget
			var holder *sim.Request
			if mode == "capacity" {
				budget = 2
				holder = &sim.Request{ID: "held", InputTokens: make([]sim.TokenID, 44)}
				if !s.ReserveNativeRange(holder, 0, 44, nil) || s.FreeBlockCnt != 2 {
					t.Fatal("capacity fixture did not leave just two cached free pages")
				}
				for _, id := range cached {
					if s.Blocks[id].RefCount != 0 || s.Blocks[id].Hash == "" {
						t.Fatal("fixture evicted the prefix before control admission")
					}
				}
			}
			before := nativeControlHBMState(s)
			id, err := s.QueueNativePromotion(prefix, budget)
			if id == "" || err != nil || !reflect.DeepEqual(before, nativeControlHBMState(s)) {
				t.Fatal("enqueue allocated HBM", id, err)
			}
			e.Run()
			if !reflect.DeepEqual(before, nativeControlHBMState(s)) {
				t.Fatal("rejected whole prefix changed references, hashes or free order")
			}
			for _, row := range h.records {
				if row.Request == id && (row.Name == "native_control_captured" || row.Name == "transfer_start" || row.Name == "hbm_drop") {
					t.Fatal("failed admission performed destructive or partial work", row)
				}
			}
			if holder != nil {
				s.ReleaseKVBlocks(holder)
			}
			nativeControlDrained(t, s)
		})
	}
}

func TestNativeControlCancellationBeforeLookupReplyDrainsBackgroundEND(t *testing.T) {
	h, s, e, prefix, _ := nativeControlHarness(t)
	before := nativeControlHBMState(s)
	id, err := s.QueueNativePromotion(prefix, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.CancelNativePromotion(id)
	if s.NativeControlPending() || s.nativeControlRPCs != 2 {
		t.Fatal("cancel discarded lookup/END obligations or kept the scheduler spinning")
	}
	e.Run()
	if !reflect.DeepEqual(before, nativeControlHBMState(s)) {
		t.Fatal("cancel before capture allocated or reordered HBM")
	}
	lookup, released := int64(-1), int64(-1)
	for _, row := range h.records {
		if row.Request == id && row.Name == "native_control_lookup_ready" {
			lookup = row.Time
		}
		if row.Request == id && row.Name == "host_lookup_released" {
			released = row.Time
		}
	}
	if lookup < 0 || released < 0 || released >= lookup {
		t.Fatal("guarded END unnecessarily waited for the outstanding lookup reply", lookup, released)
	}
	nativeControlDrained(t, s)
}

func TestNativeControlENDDoesNotWaitForOldReplyOrReleaseAnotherReader(t *testing.T) {
	h, s, e, prefix, _ := nativeControlHarness(t)
	first, err := s.QueueNativePromotion(prefix, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.CancelNativePromotion(first)
	second, err := s.QueueNativePromotion(prefix, 2)
	if err != nil || second == "" {
		t.Fatal("cancelled selection retained the pending bound", err)
	}
	e.Run()
	firstReply, firstRelease, secondDirectory := int64(-1), int64(-1), int64(-1)
	for _, row := range h.records {
		if row.Request == first && row.Name == "native_control_lookup_ready" {
			firstReply = row.Time
		}
		if row.Request == second && row.Name == "host_lookup_reserved" {
			secondDirectory = row.Time
		}
		if row.Request == first && row.Name == "host_lookup_released" {
			firstRelease = row.Time
		}
	}
	if secondDirectory < 0 || firstRelease < 0 || firstRelease >= firstReply {
		t.Fatal("guarded cleanup incorrectly waited for first lookup reply", secondDirectory, firstRelease, firstReply)
	}
	nativeControlDrained(t, s)
}

func TestNativeControlWholeReservationStaysOwnedUntilSuccessOrCancelAdoption(t *testing.T) {
	for _, scenario := range []struct {
		name, cancelAt string
		cached         int
	}{
		{"success", "never", 2}, {"cancel_capture", "capture", 2}, {"cancel_terminal", "terminal", 2},
		{"success_partial_host_chunk", "never", 1}, {"cancel_partial_host_chunk", "terminal", 1},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			cancelAt := scenario.cancelAt
			h, s, e, prefix, old := nativeControlHarness(t)
			if scenario.cached == 1 {
				s.invalidate(s.Blocks[old[1]])
				old = old[:1]
			}
			missing := 4 - len(old)
			id, err := s.QueueNativePromotion(prefix, missing)
			if err != nil || id == "" {
				t.Fatal(id, err)
			}
			if duplicate, err := s.QueueNativePromotion(prefix, 4); duplicate != "" || err != nil {
				t.Fatal("pending bound did not reject a second control")
			}
			var captured []int64
			replySeen, adopted := false, false
			e.EventObserver = func(event sim.Event, before bool) {
				if before {
					return
				}
				c := s.nativeControl
				if c != nil && len(c.ids) != 0 && captured == nil {
					captured = append([]int64(nil), c.ids...)
					if len(captured) != 4 || !reflect.DeepEqual(captured[:len(old)], old) {
						t.Fatal("missing existing prefix in independent reservation", captured, old)
					}
					if cancelAt == "capture" {
						if c.submitted {
							t.Fatal("fixture missed pre-submit cancellation")
						}
						s.CancelNativePromotion(id)
					}
				}
				if c != nil && len(c.ids) != 0 {
					for i, bid := range c.ids {
						block := s.Blocks[bid]
						if block.RefCount != 1 || i >= c.cached && block.Hash != "" {
							t.Fatal("reserved prefix freed or suffix hashed before adoption", i, block)
						}
					}
					if c.settled == missing {
						replySeen = true
						if cancelAt == "terminal" {
							s.CancelNativePromotion(id)
						}
					}
				}
				if c == nil && captured != nil {
					adopted = true
				}
			}
			e.Run()
			if !replySeen || !adopted {
				t.Fatal("captured copy did not reach its terminal adoption")
			}
			loads := 0
			for _, row := range h.records {
				if row.Request == id && row.Name == "transfer_start" {
					loads++
					if len(row.HBMBlocks) != missing || row.Bytes != int64(2-len(old)/2)*200 {
						t.Fatal("control recopied its already cached host chunk", row)
					}
				}
			}
			if loads != 1 {
				t.Fatal("captured cancellation must still drain exactly one copy", loads)
			}
			for i, hash := range prefix.hashes {
				_, exists := s.lookup[hash]
				if exists != (i < len(old) || cancelAt == "never") {
					t.Fatal("success/cancellation published the wrong prefix", i, exists)
				}
			}
			nativeControlDrained(t, s)
		})
	}
}

func TestNativeControlFrontendAllocatesBeforePendingPromotion(t *testing.T) {
	h, s, e, prefix, _ := nativeControlHarness(t)
	id, err := s.QueueNativePromotion(prefix, 2)
	if err != nil {
		t.Fatal(err)
	}
	injected := false
	request := &sim.Request{ID: "foreground-priority", InputTokens: make([]sim.TokenID, 44), OutputTokens: []sim.TokenID{20}, MaxOutputLen: 1}
	e.EventObserver = func(event sim.Event, before bool) {
		x, ok := event.(*sim.NativeAsyncEvent)
		if before || !ok || injected || x.Name != "native_phase_end" || !strings.HasSuffix(x.Phase, "/lookup/reply") {
			return
		}
		injected = true
		request.ArrivalTime = x.At
		e.InjectArrival(request)
	}
	e.Run()
	if !injected || request.State != sim.StateCompleted {
		t.Fatal("foreground did not complete")
	}
	for _, row := range h.records {
		if row.Request == id && row.Name == "native_control_captured" {
			t.Fatal("promotion reserved before the native frontend allocation")
		}
	}
	nativeControlDrained(t, s)
}

func TestNativeControlSnapshotRejectsHiddenOutputAndCopiesVisibleInput(t *testing.T) {
	_, s, _, _, _ := nativeControlHarness(t)
	r := &sim.Request{ID: "visible", InputTokens: []sim.TokenID{1, 2, 3, 4}, OutputTokens: []sim.TokenID{5, 6, 7, 8}}
	if _, err := s.SnapshotNativePromotionPrefix(r, 8); err == nil {
		t.Fatal("snapshot exposed not-yet-delivered output")
	}
	prefix, err := s.SnapshotNativePromotionPrefix(r, 4)
	if err != nil {
		t.Fatal(err)
	}
	r.InputTokens[0] = 999
	if prefix.tokens[0] != 1 || len(prefix.hashes) != 2 {
		t.Fatal("snapshot aliases input or incorrectly excludes the last complete block")
	}
}

func TestNativeControlForegroundRestoreCannotBorrowUnpublishedSuffix(t *testing.T) {
	for _, point := range []string{"submitted", "terminal"} {
		t.Run(point, func(t *testing.T) {
			h, s, e, prefix, _ := nativeControlHarness(t)
			id, err := s.QueueNativePromotion(prefix, 2)
			if err != nil {
				t.Fatal(err)
			}
			r := &sim.Request{ID: "overlap", InputTokens: []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8, 9}, OutputTokens: []sim.TokenID{10}, MaxOutputLen: 1}
			injected := false
			e.EventObserver = func(event sim.Event, before bool) {
				x, ok := event.(*sim.NativeAsyncEvent)
				if before || !ok || injected {
					return
				}
				trigger := point == "submitted" && x.Name == "native_transfer_plan" ||
					point == "terminal" && x.Name == "native_phase_end" && strings.HasPrefix(x.Phase, "kv_") && strings.HasSuffix(x.Phase, "/reply")
				if trigger {
					if len(s.GetCachedBlocks(r.InputTokens)) != 2 {
						t.Fatal("fixture missed unpublished suffix")
					}
					r.ArrivalTime = x.At
					injected = true
					e.InjectArrival(r)
				}
			}
			e.Run()
			loads := map[string]PeerRecord{}
			for _, row := range h.records {
				if row.Name == "transfer_start" && row.Destination == "hbm" {
					loads[row.Request] = row
				}
			}
			if !injected || len(loads) != 2 || len(loads[id].HBMBlocks) != 2 || len(loads[r.ID].HBMBlocks) != 2 || r.State != sim.StateCompleted {
				t.Fatal("foreground did not independently restore its unpublished suffix", loads)
			}
			for _, a := range loads[id].HBMBlocks {
				for _, b := range loads[r.ID].HBMBlocks {
					if a == b {
						t.Fatal("foreground borrowed a control target")
					}
				}
			}
			nativeControlDrained(t, s)
		})
	}
}
