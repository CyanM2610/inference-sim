package kv

import (
	"github.com/inference-sim/inference-sim/sim"
	"testing"
)

func TestPoolPolicyReceivesRealStoreCompletionAndRequestLookup(t *testing.T) {
	h, s := reuseFixture(t)
	if err := h.f.SetPoolEvictionPolicy("cpu", &PrefixLRU{}); err != nil {
		t.Fatal(err)
	}
	old := seedPeer(t, s, "old", []sim.TokenID{1, 2, 3, 4})
	var end int64
	s.BeginBatch(0, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	for len(h.events) > 0 {
		phaseNext(t, h)
	}
	end = 0
	s.BeginBatch(300, nil, func(at int64) { end = at })
	for end == 0 {
		phaseNext(t, h)
	}
	first := s.hashes(old.InputTokens)[0]
	if h.f.pools["cpu"].policy.(*PrefixLRU).serial != 2 {
		t.Fatal("store completion did not notify cache policy")
	}
	r := &sim.Request{ID: "hit", InputTokens: []sim.TokenID{1, 2, 9}}
	s.clock = end
	if !s.AllocateKVBlocks(r, 2, 3, s.GetCachedBlocks(r.InputTokens)) {
		t.Fatal("prefix hit rejected")
	}
	if h.f.pools["cpu"].evictionVictim().hash == first {
		t.Fatal("request lookup did not protect shared prefix recency")
	}
	s.ReleaseKVBlocks(r)
	assertPeerConservation(t, s)
}

func TestPrefixLRUUsesOrderAndKeepsSharedPrefix(t *testing.T) {
	f, err := NewPeerFabric(nil, []PeerPoolConfig{{ID: "cpu", CapacityBlocks: 4, EvictionPolicy: "prefix_lru"}}, 100, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The lexicographically small shared key must not become a victim merely
	// because all these accesses/completions occur at the same timestamp.
	for _, key := range []string{"a-prefix", "b-tail", "c-prefix", "d-tail"} {
		e, _ := f.reserve("cpu", key, nil, 10)
		e.ready = true
		f.poolPolicyEvent("cpu", "ready", "", []string{key}, 10)
	}
	f.poolPolicyEvent("cpu", "touch", "reuse", []string{"a-prefix"}, 10)
	f.reserve("cpu", "new-tail", nil, 10)
	if f.pools["cpu"].entries["a-prefix"] == nil || f.pools["cpu"].entries["b-tail"] != nil {
		t.Fatal("shared prefix was evicted instead of untouched tail")
	}
	// A full prefix touch is reverse-recency ordered, so its tail is cheaper
	// to evict than a head whose loss would invalidate every later hit.
	f.poolPolicyEvent("cpu", "touch", "full", []string{"c-prefix", "d-tail"}, 10)
	if got := f.pools["cpu"].evictionVictim().hash; got != "a-prefix" {
		t.Fatal(got)
	}
	f.poolPolicyEvent("cpu", "touch", "reuse", []string{"a-prefix"}, 10)
	if got := f.pools["cpu"].evictionVictim().hash; got != "d-tail" {
		t.Fatal("prefix order was lost", got)
	}
}

type invalidPoolPolicy struct{}

func (invalidPoolPolicy) Observe(PoolPolicyEvent)                 {}
func (invalidPoolPolicy) Victim(c []PoolEvictionCandidate) string { c[0].Hash = "busy"; return "busy" }

func TestPoolPolicyCannotEvictProtectedCopyOrMutateCandidates(t *testing.T) {
	f, _ := NewPeerFabric(nil, []PeerPoolConfig{{ID: "cpu", CapacityBlocks: 2}}, 100, nil)
	if err := f.SetPoolEvictionPolicy("cpu", invalidPoolPolicy{}); err != nil {
		t.Fatal(err)
	}
	a, _ := f.reserve("cpu", "idle", nil, 0)
	a.ready = true
	b, _ := f.reserve("cpu", "busy", nil, 0)
	b.ready = true
	b.readers = 1
	defer func() {
		if recover() == nil {
			t.Error("invalid candidate was accepted")
		}
		if len(f.pools["cpu"].entries) != 2 || b.readers != 1 || f.pools["cpu"].entries["idle"] != a {
			t.Error("invalid decision mutated storage")
		}
	}()
	f.reserve("cpu", "new", nil, 0)
}

func TestPoolPolicyTouchesOnlyEvictableCopiesAndKeepsPoolsIndependent(t *testing.T) {
	f, _ := NewPeerFabric(nil, []PeerPoolConfig{{ID: "a", CapacityBlocks: 3, EvictionPolicy: "prefix_lru"}, {ID: "b", CapacityBlocks: 3, EvictionPolicy: "prefix_lru"}}, 100, nil)
	for _, id := range []string{"a", "b"} {
		for _, key := range []string{"first", "last"} {
			e, _ := f.reserve(id, key, nil, 0)
			e.ready = true
			f.poolPolicyEvent(id, "ready", "", []string{key}, 0)
		}
	}
	f.pools["a"].entries["first"].readers = 1
	f.poolPolicyEvent("a", "touch", "", []string{"first", "missing"}, 0)
	f.pools["a"].entries["first"].readers = 0
	if f.pools["a"].evictionVictim().hash != "first" {
		t.Fatal("protected touch changed recency")
	}
	f.poolPolicyEvent("a", "ready", "", []string{"first"}, 0)
	if f.pools["a"].evictionVictim().hash != "last" || f.pools["b"].evictionVictim().hash != "first" {
		t.Fatal("completion order or pool isolation lost")
	}
}
