package kv

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// PoolEvictionCandidate is an idle, ready copy. The runtime owns eligibility,
// reservation and reclamation; a policy never receives peerEntry pointers.
type PoolEvictionCandidate struct {
	Hash         string
	LastAccessUS int64
}

// PoolPolicyEvent contains only currently evictable keys. Touch keys are in
// logical prefix order; Ready keys follow completion adoption order.
type PoolPolicyEvent struct {
	Kind    string // "ready", "touch", "remove"
	Keys    []string
	TimeUS  int64
	Request string
}

// PoolEvictionPolicy chooses an optional victim and can maintain access history.
// Empty means decline reclamation. The runtime rejects non-candidate results
// before releasing memory. Use a separate policy instance for each pool.
type PoolEvictionPolicy interface {
	Observe(PoolPolicyEvent)
	Victim([]PoolEvictionCandidate) string
}

// PoolAdmissionPolicy is optional. It receives an independent snapshot before
// an incoming block can evict a copy or reserve capacity. Existing hits/joined
// stores bypass admission; they require no new physical allocation.
type PoolAdmissionContext struct {
	Hash                       string
	CapacityBlocks, UsedBlocks int64
	Candidates                 []PoolEvictionCandidate
}
type PoolAdmissionDecision struct {
	Accept bool
	Reason string
}
type PoolAdmissionPolicy interface {
	Admit(PoolAdmissionContext) PoolAdmissionDecision
}

// TimestampLRU preserves the original time/hash tie-break for frozen baselines.
type TimestampLRU struct{}

func (TimestampLRU) Observe(PoolPolicyEvent) {}
func (TimestampLRU) Victim(c []PoolEvictionCandidate) string {
	if len(c) == 0 {
		return ""
	}
	best := c[0]
	for _, v := range c[1:] {
		if v.LastAccessUS < best.LastAccessUS || v.LastAccessUS == best.LastAccessUS && v.Hash < best.Hash {
			best = v
		}
	}
	return best.Hash
}

// PrefixLRU preserves access order even when simulated timestamps are equal.
// Touching a logical prefix makes earlier blocks more recent than its tail,
// matching the native CPU cache's prefix-aware LRU convention.
type PrefixLRU struct {
	serial uint64
	order  map[string]uint64
}

func (p *PrefixLRU) Observe(e PoolPolicyEvent) {
	if p.order == nil {
		p.order = map[string]uint64{}
	}
	visit := func(key string) { p.serial++; p.order[key] = p.serial }
	switch e.Kind {
	case "ready":
		for _, k := range e.Keys {
			visit(k)
		}
	case "touch":
		for i := len(e.Keys) - 1; i >= 0; i-- {
			visit(e.Keys[i])
		}
	case "remove":
		for _, k := range e.Keys {
			delete(p.order, k)
		}
	default:
		panic("unknown pool policy event")
	}
}
func (p *PrefixLRU) Victim(c []PoolEvictionCandidate) string {
	if len(c) == 0 {
		return ""
	}
	best := c[0].Hash
	for _, v := range c[1:] {
		if p.order[v.Hash] < p.order[best] || p.order[v.Hash] == p.order[best] && v.Hash < best {
			best = v.Hash
		}
	}
	return best
}

func NewPoolEvictionPolicy(name string) (PoolEvictionPolicy, error) {
	switch name {
	case "", "time_lru":
		return TimestampLRU{}, nil
	case "prefix_lru":
		return &PrefixLRU{}, nil
	default:
		return nil, fmt.Errorf("unknown pool eviction policy %q", name)
	}
}

// SetPoolEvictionPolicy registers a programmatic policy before pool use.
func (f *PeerFabric) SetPoolEvictionPolicy(id string, policy PoolEvictionPolicy) error {
	p := f.pools[id]
	if p == nil || policy == nil || len(p.entries) > 0 || f.Pending() != 0 {
		return fmt.Errorf("pool policy requires a known unused pool and nonnil policy")
	}
	p.policy, p.trackAccess = policy, true
	return nil
}

func (f *PeerFabric) poolPolicyEvent(pool, kind, request string, keys []string, now int64) {
	p := f.pools[pool]
	if !p.trackAccess {
		return
	}
	visible := make([]string, 0, len(keys))
	for _, key := range keys {
		e := p.entries[key]
		if kind == "remove" || e != nil && e.ready && e.readers == 0 {
			visible = append(visible, key)
		}
	}
	if len(visible) == 0 {
		return
	}
	// Retaining or mutating a policy's event must not change the trace/runtime.
	p.policy.Observe(PoolPolicyEvent{Kind: kind, Keys: append([]string(nil), visible...), TimeUS: now, Request: request})
	f.emit(PeerRecord{Time: now, Name: "l2_policy_event", Request: request, Destination: pool, Hashes: visible, Reason: kind})
}

func (p *peerPool) evictionVictim() *peerEntry {
	candidates := p.evictionCandidates()
	allowed := map[string]bool{}
	for _, c := range candidates {
		allowed[c.Hash] = true
	}
	if len(candidates) == 0 {
		return nil
	}
	key := p.policy.Victim(candidates)
	if key == "" {
		return nil
	}
	if !allowed[key] {
		panic("pool policy selected a non-evictable copy")
	}
	return p.entries[key]
}

func (p *peerPool) evictionCandidates() []PoolEvictionCandidate {
	var candidates []PoolEvictionCandidate
	for _, e := range p.entries {
		if e.ready && e.readers == 0 {
			candidates = append(candidates, PoolEvictionCandidate{Hash: e.hash, LastAccessUS: e.last})
		}
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Hash < candidates[j].Hash })
	return candidates
}

// touchRequestPools reports currently resident input-prefix accesses, not
// future requests or outputs. Full input blocks include the final logits block;
// restore eligibility separately excludes its last token.
func (s *PeerCache) touchRequestPools(req string, tokens []sim.TokenID) {
	var keys []string
	for _, a := range s.access {
		if !s.fabric.pools[a.Pool].trackAccess {
			continue
		}
		if keys == nil {
			previous := ""
			for i := int64(0); i+s.BlockSizeTokens <= int64(len(tokens)); i += s.BlockSizeTokens {
				previous = s.hashBlock(previous, tokens[i:i+s.BlockSizeTokens])
				keys = append(keys, previous)
			}
		}
		s.fabric.poolPolicyEvent(a.Pool, "touch", req, keys, s.clock)
	}
}
