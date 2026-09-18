package kv

import (
	"fmt"
	"sort"
)

// HotPrefixConfig defines a block-granular placement policy. It deliberately
// uses exact metadata, not the paper's approximate token-radix cuckoo filter.
type HotPrefixConfig struct {
	ShadowTTLUS           *int64 `json:"shadow_ttl_us,omitempty"` // nil = 60 seconds; zero disables shadow
	AgingIntervalRequests int64  `json:"aging_interval_requests"`
	AdmissionThreshold    int64  `json:"admission_threshold"`
	MaxAge                int64  `json:"max_age,omitempty"`
	PromotionBlocks       int64  `json:"promotion_blocks_per_step"`
	PlannerUS             int64  `json:"planner_us,omitempty"`
}

func (c HotPrefixConfig) Validate() error {
	if c.ShadowTTLUS != nil && (*c.ShadowTTLUS < 0 || *c.ShadowTTLUS > 1<<62) {
		return fmt.Errorf("invalid HotPrefix shadow_ttl_us")
	}
	if c.AgingIntervalRequests <= 0 || c.AdmissionThreshold < 0 || c.AdmissionThreshold > 255 || c.MaxAge < 0 || c.MaxAge > 255 || c.PromotionBlocks < 0 || c.PlannerUS < 0 {
		return fmt.Errorf("invalid HotPrefix aging, threshold, clock, promotion budget or planner cost")
	}
	return nil
}

type hotPrefixNode struct {
	parent           string
	frequency, clock int64
	depth            int64
}

// HotPrefixPolicy owns history and pure decisions, never resource references.
// Zero-frequency records identify arrived input only; first publication makes
// them hotness records. The runtime expires shadow heat independently from
// the structural prefix identities used by resident descendants.
type HotPrefixPolicy struct {
	config      HotPrefixConfig
	blockTokens int64
	nodes       map[string]*hotPrefixNode
	requests    int64
}

func NewHotPrefixPolicy(c HotPrefixConfig, blockTokens int64) (*HotPrefixPolicy, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if blockTokens <= 0 {
		return nil, fmt.Errorf("HotPrefix requires positive block tokens")
	}
	if c.MaxAge == 0 {
		c.MaxAge = 255
	}
	return &HotPrefixPolicy{config: c, blockTokens: blockTokens, nodes: map[string]*hotPrefixNode{}}, nil
}

func (p *HotPrefixPolicy) remember(keys []string) {
	parent := ""
	for i, key := range keys {
		n := p.nodes[key]
		if n == nil {
			n = &hotPrefixNode{parent: parent, depth: int64(i + 1)}
			p.nodes[key] = n
		} else if n.parent != parent {
			panic("inconsistent HotPrefix identity")
		}
		parent = key
	}
}

func (p *HotPrefixPolicy) observe(keys []string) {
	p.remember(keys)
	p.requests++
	if p.requests%p.config.AgingIntervalRequests == 0 {
		for _, n := range p.nodes {
			n.clock = max(0, n.clock-1)
		}
	}
}

// reuse is invoked by completed execution, not a speculative READY lookup.
func (p *HotPrefixPolicy) reuse(hash string, shadowFrequency int64) {
	n := p.nodes[hash]
	if n == nil {
		panic("HotPrefix reuse lacks arrived prefix identity")
	}
	n.frequency = min(255, max(n.frequency, shadowFrequency)+1)
	n.clock = p.config.MaxAge
}

func (p *HotPrefixPolicy) published(hash string) {
	n := p.nodes[hash]
	if n == nil {
		panic("HotPrefix publication lacks arrived prefix identity")
	}
	if n.frequency == 0 {
		n.frequency, n.clock = 1, p.config.MaxAge
	}
}

func (p *HotPrefixPolicy) hotness(hash string) int64 {
	if n := p.nodes[hash]; n != nil {
		return n.frequency * n.clock
	}
	return 0
}

func (p *HotPrefixPolicy) parent(hash string) string {
	if n := p.nodes[hash]; n != nil {
		return n.parent
	}
	return ""
}

// Leaves are defined by all resident descendants, including non-evictable
// children. A missing intermediate block must not make an ancestor a leaf.
func (p *HotPrefixPolicy) nonLeaves(resident []string) map[string]bool {
	parents := map[string]bool{}
	for _, hash := range resident {
		for parent := p.parent(hash); parent != "" && !parents[parent]; parent = p.parent(parent) {
			parents[parent] = true
		}
	}
	return parents
}

func (p *HotPrefixPolicy) Choose(c PeerReclaimContext) PeerDecision {
	parents := p.nonLeaves(c.ResidentHashes)
	copies := map[string]int{}
	for _, hash := range c.ResidentHashes {
		copies[hash]++
	}
	best := -1
	var score float64
	for i, candidate := range c.Candidates {
		n := p.nodes[candidate.Hash]
		if n == nil || n.frequency == 0 || parents[candidate.Hash] && copies[candidate.Hash] < 2 {
			continue
		}
		value := float64(n.frequency) + float64(n.clock)/float64(p.blockTokens)
		if best < 0 || value < score { // retain native free-list/LRU order for ties
			best, score = i, value
		}
	}
	if best < 0 {
		return PeerDecision{Decline: true}
	}
	v := c.Candidates[best]
	d := PeerDecision{BlockID: v.ID}
	if len(v.ReadyCopies) == 0 && copies[v.Hash] < 2 {
		for _, target := range c.Targets {
			if target.Available {
				d.Pool = target.Pool // actual admission is checked before reservation
				break
			}
		}
	}
	return d
}

type hotPrefixPoolPolicy struct{ state *HotPrefixPolicy }

func (hotPrefixPoolPolicy) Observe(PoolPolicyEvent) {} // one request observation owns heat

func (p hotPrefixPoolPolicy) Victim(c []PoolEvictionCandidate) string {
	best := ""
	for _, v := range c {
		if best == "" || p.state.hotness(v.Hash) < p.state.hotness(best) ||
			p.state.hotness(v.Hash) == p.state.hotness(best) && v.Hash < best {
			best = v.Hash
		}
	}
	return best
}

func (p hotPrefixPoolPolicy) Admit(c PoolAdmissionContext) PoolAdmissionDecision {
	n := p.state.nodes[c.Hash]
	if n == nil || n.frequency == 0 || n.frequency < p.state.config.AdmissionThreshold {
		return PoolAdmissionDecision{Reason: "frequency_below_threshold"}
	}
	if c.UsedBlocks < c.CapacityBlocks {
		return PoolAdmissionDecision{Accept: true, Reason: "capacity_available"}
	}
	victim := p.Victim(c.Candidates)
	if victim == "" {
		return PoolAdmissionDecision{Reason: "no_evictable_copy"}
	}
	if p.state.hotness(c.Hash) <= p.state.hotness(victim) {
		return PoolAdmissionDecision{Reason: "not_hotter_than_replacement"}
	}
	return PoolAdmissionDecision{Accept: true, Reason: "replace_colder_copy"}
}

type hotPrefixPromotion struct {
	hash   string
	victim int64 // -1 means an empty physical block
}

// planPromotions considers only roots with currently resident parents. It does
// not pretend that a target planned earlier in this batch is already READY.
func (p *HotPrefixPolicy) planPromotions(resident []string, idle []PeerCandidate, host []string, empty, budget int64) []hotPrefixPromotion {
	counts := map[string]int{}
	for _, h := range resident {
		counts[h]++
	}
	parents := p.nonLeaves(resident)
	cold := make([]PeerCandidate, 0, len(idle))
	for _, v := range idle {
		if !parents[v.Hash] {
			cold = append(cold, v)
		}
	}
	sort.SliceStable(cold, func(i, j int) bool { return p.hotness(cold[i].Hash) < p.hotness(cold[j].Hash) })
	host = append([]string(nil), host...)
	sort.Slice(host, func(i, j int) bool {
		if p.hotness(host[i]) != p.hotness(host[j]) {
			return p.hotness(host[i]) > p.hotness(host[j])
		}
		return host[i] < host[j]
	})
	var plan []hotPrefixPromotion
	protected := map[string]bool{}
	for _, h := range host {
		if int64(len(plan)) == budget {
			break
		}
		parent := p.parent(h)
		if counts[h] > 0 || p.hotness(h) == 0 || parent != "" && counts[parent] == 0 {
			continue
		}
		victim := -1
		if empty == 0 {
			for i, v := range cold {
				if v.Hash != parent && !protected[v.Hash] && p.hotness(h) > p.hotness(v.Hash) {
					victim = i
					break
				}
			}
			if victim < 0 {
				continue
			}
		}
		action := hotPrefixPromotion{hash: h, victim: -1}
		if victim >= 0 {
			action.victim = cold[victim].ID
			counts[cold[victim].Hash]--
			cold = append(cold[:victim], cold[victim+1:]...)
		} else {
			empty--
		}
		protected[parent] = true
		plan = append(plan, action)
	}
	return plan
}
