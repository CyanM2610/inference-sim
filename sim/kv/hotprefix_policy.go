package kv

import (
	"fmt"
	"sort"
)

// HotPrefixConfig controls block placement or the explicitly named resident
// logical-segment adaptation. Both use exact per-prefix heat metadata.
type HotPrefixConfig struct {
	Benefit               *HotPrefixBenefitConfig     `json:"benefit,omitempty"`
	HBMEvictionUnit       string                      `json:"hbm_eviction_unit,omitempty"`
	LengthReferenceTokens int64                       `json:"length_reference_tokens,omitempty"`
	HBMScore              string                      `json:"hbm_score,omitempty"`
	HBMCandidates         string                      `json:"hbm_candidates,omitempty"`
	Diagnostics           *HotPrefixDiagnosticsConfig `json:"diagnostics,omitempty"`
	ShadowTTLUS           *int64                      `json:"shadow_ttl_us,omitempty"` // nil = 60 seconds; zero disables shadow
	AgingIntervalRequests int64                       `json:"aging_interval_requests"`
	AdmissionThreshold    int64                       `json:"admission_threshold"`
	MaxAge                int64                       `json:"max_age,omitempty"`
	PromotionBlocks       int64                       `json:"promotion_blocks_per_step"`
	PlannerUS             int64                       `json:"planner_us,omitempty"`
}

func (c HotPrefixConfig) Validate() error {
	if c.Benefit != nil {
		if err := c.Benefit.Validate(); err != nil {
			return err
		}
		if c.PromotionBlocks != 0 {
			return fmt.Errorf("benefit experiments require promotion disabled")
		}
	}
	switch c.HBMScore {
	case "", "paper", "paper_fixed", "frequency", "clock", "lru", "benefit_next":
	default:
		return fmt.Errorf("invalid HotPrefix hbm_score %q", c.HBMScore)
	}
	if c.HBMScore == "benefit_next" && (c.Benefit == nil || c.HBMEvictionUnit != "logical_segment") {
		return fmt.Errorf("benefit_next requires an explicit forecast and logical segments")
	}
	switch c.HBMEvictionUnit {
	case "", "block", "logical_segment":
	default:
		return fmt.Errorf("invalid HotPrefix hbm_eviction_unit %q", c.HBMEvictionUnit)
	}
	if c.HBMEvictionUnit == "logical_segment" && (c.PromotionBlocks != 0 || c.HBMCandidates == "all_idle") {
		return fmt.Errorf("HotPrefix logical segments require leaf candidates and promotion disabled")
	}
	if c.HBMScore == "paper_fixed" && (c.HBMEvictionUnit != "logical_segment" || c.LengthReferenceTokens <= 0) {
		return fmt.Errorf("paper_fixed requires logical segments and an explicit positive length reference")
	}
	if c.LengthReferenceTokens < 0 {
		return fmt.Errorf("invalid HotPrefix length_reference_tokens")
	}
	switch c.HBMCandidates {
	case "", "leaf", "all_idle":
	default:
		return fmt.Errorf("invalid HotPrefix hbm_candidates %q", c.HBMCandidates)
	}
	if c.PromotionBlocks > 0 && ((c.HBMScore != "" && c.HBMScore != "paper") || c.HBMCandidates == "all_idle") {
		return fmt.Errorf("HotPrefix HBM ablations require promotion disabled")
	}
	if c.ShadowTTLUS != nil && (*c.ShadowTTLUS < 0 || *c.ShadowTTLUS > 1<<62) {
		return fmt.Errorf("invalid HotPrefix shadow_ttl_us")
	}
	if c.AgingIntervalRequests <= 0 || c.AdmissionThreshold < 0 || c.AdmissionThreshold > 255 || c.MaxAge < 0 || c.MaxAge > 255 || c.PromotionBlocks < 0 || c.PlannerUS < 0 {
		return fmt.Errorf("invalid HotPrefix aging, threshold, clock, promotion budget or planner cost")
	}
	return nil
}

type hotPrefixNode struct {
	references       hotPrefixReferenceHistory
	parent           string
	frequency, clock int64
	depth            int64
	children         int
	inputEndpoint    bool
}

// HotPrefixPolicy owns history and pure decisions, never resource references.
// Zero-frequency records identify arrived input only; first publication makes
// them hotness records. The runtime expires shadow heat independently from
// the structural prefix identities used by resident descendants.
type HotPrefixPolicy struct {
	lastSegments []hotPrefixLogicalSegment
	config       HotPrefixConfig
	blockTokens  int64
	nodes        map[string]*hotPrefixNode
	requests     int64
	diagnostics  *HotPrefixDiagnostics
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
	p := &HotPrefixPolicy{config: c, blockTokens: blockTokens, nodes: map[string]*hotPrefixNode{}}
	if c.Diagnostics != nil {
		p.diagnostics = &HotPrefixDiagnostics{Schema: "hotprefix_choices_v1", MeasureCPU: c.Diagnostics.MeasureCPU, CPU: map[string]HotPrefixCPU{},
			Coverage: "local Go history, policy and snapshot methods; inclusive method timings; trace construction/export excluded; not simulated service or native CPU calibration"}
	}
	return p, nil
}

func (p *HotPrefixPolicy) remember(keys []string) {
	parent := ""
	for i, key := range keys {
		n := p.nodes[key]
		if n == nil {
			n = &hotPrefixNode{parent: parent, depth: int64(i + 1)}
			p.nodes[key] = n
			if parent != "" {
				p.nodes[parent].children++
			}
		} else if n.parent != parent {
			panic("inconsistent HotPrefix identity")
		}
		parent = key
	}
}

func (p *HotPrefixPolicy) observe(keys []string) {
	started := p.cpuStart()
	defer p.cpuEnd("observe_history", started)
	p.remember(keys)
	if len(keys) > 0 {
		p.nodes[keys[len(keys)-1]].inputEndpoint = true
	}
	p.requests++
	if p.requests%p.config.AgingIntervalRequests == 0 {
		for _, n := range p.nodes {
			n.clock = max(0, n.clock-1)
		}
	}
}

// reuse is invoked by completed execution, not a speculative READY lookup.
func (p *HotPrefixPolicy) reuse(hash string, shadowFrequency int64) {
	started := p.cpuStart()
	defer p.cpuEnd("completed_reuse", started)
	n := p.nodes[hash]
	if n == nil {
		panic("HotPrefix reuse lacks arrived prefix identity")
	}
	n.frequency = min(255, max(n.frequency, shadowFrequency)+1)
	n.clock = p.config.MaxAge
}

func (p *HotPrefixPolicy) published(hash string) {
	started := p.cpuStart()
	defer p.cpuEnd("publish_history", started)
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
	started := p.cpuStart()
	d := p.choose(c)
	p.cpuEnd("hbm_choose", started)
	p.recordChoice(c, d)
	return d
}

func (p *HotPrefixPolicy) score(hash string) float64 {
	n := p.nodes[hash]
	if n == nil {
		return 0
	}
	switch p.config.HBMScore {
	case "frequency":
		return float64(n.frequency)
	case "clock":
		return float64(n.clock)
	case "lru":
		return 0 // stable candidate order is the shared tiebreaker
	default:
		return float64(n.frequency) + float64(n.clock)/float64(p.blockTokens)
	}
}

func (p *HotPrefixPolicy) eligible(v PeerCandidate, parents map[string]bool, copies map[string]int) bool {
	if p.config.HBMCandidates == "all_idle" {
		return true
	}
	n := p.nodes[v.Hash]
	return n != nil && n.frequency > 0 && (!parents[v.Hash] || copies[v.Hash] > 1)
}

func (p *HotPrefixPolicy) choose(c PeerReclaimContext) PeerDecision {
	if p.config.HBMEvictionUnit == "logical_segment" {
		return p.chooseLogicalSegment(c)
	}
	parents := p.nonLeaves(c.ResidentHashes)
	copies := map[string]int{}
	for _, hash := range c.ResidentHashes {
		copies[hash]++
	}
	best := -1
	var score float64
	for i, candidate := range c.Candidates {
		if !p.eligible(candidate, parents, copies) {
			continue
		}
		value := p.score(candidate.Hash)
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
	started := p.state.cpuStart()
	defer p.state.cpuEnd("dram_victim", started)
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
	started := p.state.cpuStart()
	defer p.state.cpuEnd("dram_admit", started)
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
