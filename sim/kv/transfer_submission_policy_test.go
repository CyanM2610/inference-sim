package kv

import (
	"math"
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type submissionFunc func(TransferSubmissionContext) []int64

func (f submissionFunc) Order(c TransferSubmissionContext) []int64 { return f(c) }

func reverseSubmission(c TransferSubmissionContext) []int64 {
	ids := make([]int64, len(c.Candidates))
	for i, job := range c.Candidates {
		ids[len(ids)-1-i] = job.Transaction
	}
	return ids
}

func twoSubmissionFixture(t *testing.T, load bool, policy TransferSubmissionPolicy, cost *TransferSubmissionCost, observe func(TransferSubmissionRecord), directional ...bool) (*peerHarness, *PeerCache, []*sim.Request) {
	t.Helper()
	var h *peerHarness
	var s *PeerCache
	var reqs []*sim.Request
	if load {
		var tokens [][]sim.TokenID
		h, s, tokens = concurrentRestoreFixture(t, 6)
		for i, tok := range tokens {
			reqs = append(reqs, &sim.Request{ID: string(rune('a' + i)), InputTokens: tok})
		}
	} else {
		h, s, _ = newPeerHarness(t, false)
		reqs = []*sim.Request{seedPeer(t, s, "writer", []sim.TokenID{30, 31, 32, 33, 34})}
		if err := h.f.ConfigureMechanisms(PeerMechanisms{BackgroundStorePool: "cxl", GroupTransfers: true, ConcurrentRestores: true, RestoreWindow: 16}); err != nil {
			t.Fatal(err)
		}
	}
	h.f.stores = []*PeerCache{s}
	poll := int64(30)
	if load {
		poll = 50
	}
	if err := s.ConfigureEnginePhases(fixedEnginePhases{EngineStepTiming{PostForwardUS: 2, PollUS: poll, TailUS: 1}, 3}, nil); err != nil {
		t.Fatal(err)
	}
	if len(directional) > 0 && directional[0] {
		if err := s.EnableDirectionalTransferOrder(); err != nil {
			t.Fatal(err)
		}
	}
	if policy != nil {
		if err := s.SetTransferSubmissionPolicy(policy, cost, observe); err != nil {
			t.Fatal(err)
		}
	}
	h.records = nil
	if load {
		h.f.SetRequestDeadline("a", 400)
		h.f.SetRequestDeadline("b", 300)
		for _, req := range reqs {
			if s.AllocateKVBlocks(req, 0, 2, nil) {
				t.Fatal("load did not stage")
			}
		}
	} else {
		for _, id := range s.GetCachedBlocks(reqs[0].InputTokens) {
			if !s.store(s.Blocks[id], "cxl", "writer") {
				t.Fatal("store did not stage")
			}
		}
	}
	if len(h.f.phases.staged) != 2 {
		t.Fatalf("need two real jobs: %d", len(h.f.phases.staged))
	}
	return h, s, reqs
}

func submissionStep(t *testing.T, h *peerHarness, s *PeerCache, now int64) int64 {
	t.Helper()
	end := int64(-1)
	if ok, err := s.BeginBatch(now, nil, func(at int64) { end = at }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for end < 0 {
		phaseNext(t, h)
	}
	return end
}

func TestTransferSubmissionActualOrderAndLeases(t *testing.T) {
	for _, load := range []bool{false, true} {
		var original []PeerRecord
		for _, mode := range []string{"default", "fifo", "reverse"} {
			t.Run(map[bool]string{true: "load", false: "store"}[load]+"/"+mode, func(t *testing.T) {
				var policy TransferSubmissionPolicy
				if mode == "fifo" {
					policy, _ = NewTransferSubmissionPolicy("fifo")
				}
				if mode == "reverse" {
					policy = submissionFunc(reverseSubmission)
				}
				var records []TransferSubmissionRecord
				h, s, reqs := twoSubmissionFixture(t, load, policy, nil, func(r TransferSubmissionRecord) { records = append(records, r) })
				first, second := h.f.phases.staged[0].record.Transaction, h.f.phases.staged[1].record.Transaction
				if mode == "reverse" {
					first, second = second, first
				}
				if load && (h.f.Snapshot()["cxl"]["read_pins"] != 4 || s.FreeBlockCnt != 2) {
					t.Fatal("missing load reservations/pins")
				}
				submissionStep(t, h, s, 0)
				var submits, starts, adopted []int64
				for _, r := range h.records {
					switch r.Name {
					case "transfer_submit_begin":
						submits = append(submits, r.Transaction)
					case "transfer_start":
						starts = append(starts, r.Transaction)
					case "transfer_adopted":
						adopted = append(adopted, r.Transaction)
					}
				}
				want := []int64{first, second}
				if !reflect.DeepEqual(submits, want) || !reflect.DeepEqual(starts, want) || !reflect.DeepEqual(adopted, want[:1]) {
					t.Fatalf("order did not reach execution: submit %v start %v adopt %v want %v", submits, starts, adopted, want)
				}
				if load {
					index := 0
					if mode == "reverse" {
						index = 1
					}
					if len(s.GetCachedBlocks(reqs[index].InputTokens)) != 2 || len(s.GetCachedBlocks(reqs[1-index].InputTokens)) != 0 || h.f.Snapshot()["cxl"]["read_pins"] != 2 {
						t.Fatal("publication or pending source lease incorrect")
					}
					// Cancel the still pending request. Its DMA destination must remain
					// protected until the later worker adoption, even after physical end.
					s.ClearDeferred(reqs[1-index].ID)
					if s.FreeBlockCnt != 2 {
						t.Fatal("cancellation freed an in-flight destination")
					}
				}
				for len(h.events) > 0 {
					phaseNext(t, h)
				}
				if h.f.Pending() == 0 {
					t.Fatal("physical completion bypassed adoption")
				}
				submissionStep(t, h, s, 200)
				for _, req := range reqs {
					s.ClearDeferred(req.ID)
					s.ReleaseKVBlocks(req)
				}
				assertPeerConservation(t, s)
				for _, pool := range h.f.Snapshot() {
					if pool["read_pins"] != 0 || pool["reserved"] != 0 {
						t.Fatal("pool lease leak")
					}
				}
				if h.f.Pending() != 0 {
					t.Fatal("undrained transfer")
				}
				if mode == "default" {
					original = append([]PeerRecord(nil), h.records...)
				}
				if mode == "fifo" && !reflect.DeepEqual(h.records, original) {
					t.Fatal("FIFO changed default events or timestamps")
				}
				if mode != "default" && len(records) != 1 {
					t.Fatal("called for an empty or already submitted phase", len(records))
				}
			})
		}
	}
}

func TestTransferSubmissionDeadlineAndExplicitService(t *testing.T) {
	for _, extra := range []int64{0, 11} {
		var records []TransferSubmissionRecord
		policy, _ := NewTransferSubmissionPolicy("deadline")
		var cost *TransferSubmissionCost
		if extra != 0 {
			cost = &TransferSubmissionCost{FixedUS: 5, PerJobUS: 3, Provenance: "synthetic fixture"}
		}
		h, s, _ := twoSubmissionFixture(t, true, policy, cost, func(r TransferSubmissionRecord) { records = append(records, r) })
		if cost != nil {
			cost.FixedUS = 999
		} // installation owns its cost snapshot
		end := submissionStep(t, h, s, 0)
		if end != 59+extra || len(records) != 1 {
			t.Fatal("decision service not applied exactly once", end, records)
		}
		r := records[0]
		if r.ServiceEndUS != 2+extra || r.SubmitEndUS != 8+extra || !r.View.Candidates[0].DeadlineKnown || r.View.Candidates[1].DeadlineUS != 300 {
			t.Fatal("wrong observable cost/deadline", r)
		}
		for _, e := range h.records {
			if e.Name == "transfer_submit_begin" && e.Transaction == r.Order[0] && (e.Time != 2+extra || e.Request != "b") {
				t.Fatal("deadline/cost did not affect actual submit", e)
			}
		}
		for len(h.events) > 0 {
			phaseNext(t, h)
		}
		submissionStep(t, h, s, 200)
	}
}

func TestTransferSubmissionStoreServiceDelaysForwardPhase(t *testing.T) {
	policy, _ := NewTransferSubmissionPolicy("fifo")
	cost := &TransferSubmissionCost{FixedUS: 5, PerJobUS: 3, Provenance: "synthetic STORE phase"}
	var records []TransferSubmissionRecord
	h, s, _ := twoSubmissionFixture(t, false, policy, cost, func(r TransferSubmissionRecord) { records = append(records, r) })
	if end := submissionStep(t, h, s, 0); end != 50 {
		t.Fatal("STORE decision service missing from post/poll/adoption", end)
	}
	if len(records) != 1 || records[0].View.Direction != "d2h" || records[0].ServiceEndUS != 11 || records[0].View.Candidates[0].DeadlineKnown {
		t.Fatal("wrong STORE phase record", records)
	}
	post := int64(-1)
	for _, e := range h.records {
		if e.Name == "engine_post_forward" {
			post = e.Time
		}
	}
	if post != 19 {
		t.Fatal("STORE cost did not delay forward/post", post)
	}
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	submissionStep(t, h, s, 200)
	if len(records) != 1 {
		t.Fatal("charged an empty phase")
	}
}

func TestTransferSubmissionGPUFenceRetainsExistingWireArbitration(t *testing.T) {
	// Host order is a separate control point. When both jobs reach the fabric
	// before a common GPU fence, its existing FIFO uses creation IDs. Preserve
	// this counterexample: it needs a native stream-order model before ranking
	// reordered policies against vLLM, whose handler chains submitted events.
	h, s, _ := twoSubmissionFixture(t, true, submissionFunc(reverseSubmission), nil, nil)
	a, b := h.f.phases.staged[0].record.Transaction, h.f.phases.staged[1].record.Transaction
	h.f.phases.lastGPUReady = 200
	submissionStep(t, h, s, 0)
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	var host, wire []int64
	for _, e := range h.records {
		if e.Name == "transfer_submit_begin" {
			host = append(host, e.Transaction)
		}
		if e.Name == "transfer_start" {
			wire = append(wire, e.Transaction)
		}
	}
	if !reflect.DeepEqual(host, []int64{b, a}) || !reflect.DeepEqual(wire, []int64{a, b}) {
		t.Fatal("phase/wire boundary changed", host, wire)
	}
	submissionStep(t, h, s, 400)
}

func TestTransferSubmissionRejectsBeforeConsumingWork(t *testing.T) {
	for _, mode := range []string{"missing", "duplicate", "foreign", "mutated", "cost-overflow", "clock-overflow"} {
		t.Run(mode, func(t *testing.T) {
			calls := 0
			policy := submissionFunc(func(c TransferSubmissionContext) []int64 {
				calls++
				a, b := c.Candidates[0].Transaction, c.Candidates[1].Transaction
				switch mode {
				case "missing":
					return []int64{a}
				case "duplicate":
					return []int64{a, a}
				case "foreign":
					return []int64{a, 9999}
				case "mutated":
					c.Candidates[1].Transaction = 9999
					return []int64{a, c.Candidates[1].Transaction}
				}
				return []int64{a, b}
			})
			var cost *TransferSubmissionCost
			if mode == "cost-overflow" {
				cost = &TransferSubmissionCost{PerJobUS: math.MaxInt64, Provenance: "overflow fixture"}
			}
			h, s, _ := twoSubmissionFixture(t, true, policy, cost, nil)
			before := append([]*peerJob(nil), h.f.phases.staged...)
			count := len(h.records)
			for i := 0; i < 5; i++ {
				h.f.EstimateQueueUS(0, []string{"cxl"})
			}
			if calls != 0 {
				t.Fatal("estimator consumed policy history")
			}
			now := int64(0)
			if mode == "clock-overflow" {
				now = math.MaxInt64 - 4
			}
			panicked := func() (bad bool) { defer func() { bad = recover() != nil }(); h.f.phases.submit(now, 0, true); return }()
			if !panicked || !reflect.DeepEqual(before, h.f.phases.staged) || len(h.events) != 0 || len(h.records) != count || len(h.f.queue) != 0 {
				t.Fatal("invalid policy partially submitted/consumed work")
			}
			if s.FreeBlockCnt != 2 || h.f.Snapshot()["cxl"]["read_pins"] != 4 {
				t.Fatal("invalid result corrupted existing leases")
			}
			// A failed plan has not lost jobs: repair the fixture's callback and drain.
			h.f.phases.submissionPolicy, _ = NewTransferSubmissionPolicy("fifo")
			h.f.phases.submissionCost = nil
			submissionStep(t, h, s, 0)
			for len(h.events) > 0 {
				phaseNext(t, h)
			}
			submissionStep(t, h, s, 200)
		})
	}
}

func TestTransferSubmissionInstallationAndSnapshot(t *testing.T) {
	policy, _ := NewTransferSubmissionPolicy("fifo")
	_, raw, _ := newPeerHarness(t, false)
	if err := raw.SetTransferSubmissionPolicy(policy, nil, nil); err == nil {
		t.Fatal("accepted absent phase backend")
	}
	h, s, _ := phaseFixture(t, EngineStepTiming{PollUS: 1, TailUS: 1})
	for _, invalid := range []TransferSubmissionPolicy{nil, submissionFunc(nil)} {
		if err := s.SetTransferSubmissionPolicy(invalid, nil, nil); err == nil {
			t.Fatal("accepted nil/typed nil")
		}
	}
	if err := s.SetTransferSubmissionPolicy(policy, &TransferSubmissionCost{FixedUS: -1, Provenance: "test"}, nil); err == nil {
		t.Fatal("accepted negative cost")
	}
	finished := false
	if ok, err := s.BeginBatch(0, []sim.BatchWork{{Request: &sim.Request{ID: "compute"}, NewTokens: 1}}, func(int64) { finished = true }); !ok || err != nil {
		t.Fatal(ok, err)
	}
	for !finished {
		phaseNext(t, h)
	}
	if err := s.SetTransferSubmissionPolicy(policy, nil, nil); err == nil {
		t.Fatal("accepted policy after an engine step without transfers")
	}
	var observed TransferSubmissionRecord
	mutator := submissionFunc(func(c TransferSubmissionContext) []int64 {
		ids := reverseSubmission(c)
		c.Candidates[0].Request = "forged"
		c.Candidates[0].DeadlineUS = 1
		return ids
	})
	h, s, _ = twoSubmissionFixture(t, true, mutator, nil, func(r TransferSubmissionRecord) {
		observed = r
		r.Order[0] = 9999
		r.View.Candidates[0].Transaction = 9999
	})
	if err := s.SetTransferSubmissionPolicy(policy, nil, nil); err == nil {
		t.Fatal("replaced active policy")
	}
	submissionStep(t, h, s, 0)
	if observed.View.Candidates[0].Request != "a" || observed.View.Candidates[0].DeadlineUS != 400 {
		t.Fatal("policy mutated authoritative observation")
	}
	for _, e := range h.records {
		if e.Transaction == 9999 || e.Request == "forged" {
			t.Fatal("snapshot mutation reached runtime")
		}
	}
}
