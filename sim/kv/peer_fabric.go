package kv

import (
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// PeerResource is an exclusive, non-preemptive transfer server. Reusing an ID
// on several paths models a shared bottleneck, independently of pool capacity.
type PeerResource struct {
	ID         string  `json:"id"`
	BytesPerUS float64 `json:"bytes_per_us"`
	LatencyUS  int64   `json:"latency_us"`
}
type PeerPoolConfig struct {
	ID             string `json:"id"`
	CapacityBlocks int64  `json:"capacity_blocks"`
	EvictionPolicy string `json:"eviction_policy,omitempty"`
}
type PeerAccess struct {
	Pool      string   `json:"pool"`
	ReadPath  []string `json:"read_path"`
	WritePath []string `json:"write_path"`
}
type PeerRecord struct {
	Time        int64            `json:"time_us"`
	Name        string           `json:"name"`
	Instance    string           `json:"instance,omitempty"`
	Request     string           `json:"request,omitempty"`
	Requests    []string         `json:"requests,omitempty"`
	Transaction int64            `json:"transaction,omitempty"`
	Source      string           `json:"source,omitempty"`
	Destination string           `json:"destination,omitempty"`
	Hash        string           `json:"hash,omitempty"`
	Hashes      []string         `json:"hashes,omitempty"`
	HBMBlocks   []int64          `json:"hbm_blocks,omitempty"`
	Bytes       int64            `json:"bytes,omitempty"`
	Duration    int64            `json:"duration_us,omitempty"`
	Reason      string           `json:"reason,omitempty"`
	Value       int64            `json:"value,omitempty"`
	Counters    map[string]int64 `json:"counters,omitempty"`
}
type PeerSink func(PeerRecord)
type peerBlocker struct {
	request string
	active  func() bool
}
type peerEntry struct {
	nativeBuffer string
	hash         string
	tokens       []sim.TokenID
	ready        bool
	readers      int
	last         int64
	waiters      []func(int64)
	blockers     []peerBlocker
}
type peerPool struct {
	config      PeerPoolConfig
	entries     map[string]*peerEntry
	policy      PoolEvictionPolicy
	trackAccess bool
}
type peerJob struct {
	overwrites        []sourceOverwrite
	notBefore         int64
	units             int64
	nativeSource      func() string
	nativeDestination func() string
	record            PeerRecord
	path              []string
	schedule          func(sim.Event)
	done              func(int64)
	prepare           func(int64) bool
	deadline          func() int64
	blockedRequests   func() []string
	cancel            func(int64)
}
type PeerFabric struct {
	phases        *PeerEnginePhases
	groupDepth    int
	groupBuffer   []*peerJob
	external      *externalPeerRuntime
	native        *NativePeerRuntime
	resources     map[string]PeerResource
	busy          map[string]bool
	pools         map[string]*peerPool
	queue         []*peerJob
	stores        []*PeerCache
	serial        int64
	active        int
	bytes         int64
	sink          PeerSink
	mechanisms    PeerMechanisms
	deadlines     map[string]int64
	busyUntil     map[string]int64
	dispatching   bool
	controlQueue  []*peerControlJob
	controlActive int
	controlBusy   map[int]bool
}

func NewPeerFabric(resources []PeerResource, pools []PeerPoolConfig, blockBytes int64, sink PeerSink) (*PeerFabric, error) {
	f := &PeerFabric{resources: map[string]PeerResource{}, busy: map[string]bool{}, pools: map[string]*peerPool{}, bytes: blockBytes, sink: sink, deadlines: map[string]int64{}, busyUntil: map[string]int64{}, controlBusy: map[int]bool{}}
	if blockBytes <= 0 {
		return nil, fmt.Errorf("block bytes must be positive")
	}
	for _, r := range resources {
		if _, ok := f.resources[r.ID]; ok || r.ID == "" || r.BytesPerUS <= 0 || math.IsNaN(r.BytesPerUS) || math.IsInf(r.BytesPerUS, 0) || r.LatencyUS < 0 {
			return nil, fmt.Errorf("invalid or duplicate resource %q", r.ID)
		}
		f.resources[r.ID] = r
	}
	for _, p := range pools {
		if _, ok := f.pools[p.ID]; ok || p.ID == "" || p.CapacityBlocks <= 0 {
			return nil, fmt.Errorf("invalid or duplicate pool %q", p.ID)
		}
		policy, err := NewPoolEvictionPolicy(p.EvictionPolicy)
		if err != nil {
			return nil, err
		}
		f.pools[p.ID] = &peerPool{config: p, entries: map[string]*peerEntry{}, policy: policy, trackAccess: p.EvictionPolicy != ""}
	}
	return f, nil
}
func (f *PeerFabric) emit(r PeerRecord) {
	if f.sink != nil {
		f.sink(r)
	}
}
func (f *PeerFabric) duration(path []string) int64 {
	return f.durationBytes(path, f.bytes)
}
func (f *PeerFabric) durationBytes(path []string, bytes int64) int64 {
	if f.native != nil {
		return max(f.native.estimate("d2h"), f.native.estimate("h2d"))
	}
	var latency int64
	var transfer float64
	for _, id := range path {
		r := f.resources[id]
		latency += r.LatencyUS
		transfer = math.Max(transfer, float64(bytes)/r.BytesPerUS)
	}
	return max(1, latency+int64(math.Ceil(transfer)))
}
func (f *PeerFabric) validateAccess(access []PeerAccess) error {
	seen := map[string]bool{}
	for _, a := range access {
		if f.pools[a.Pool] == nil || seen[a.Pool] {
			return fmt.Errorf("unknown or duplicate pool %q", a.Pool)
		}
		seen[a.Pool] = true
		for _, path := range [][]string{a.ReadPath, a.WritePath} {
			if len(path) == 0 {
				return fmt.Errorf("pool %s needs nonempty read/write paths", a.Pool)
			}
			ids := map[string]bool{}
			for _, id := range path {
				if _, ok := f.resources[id]; !ok || ids[id] {
					return fmt.Errorf("unknown or repeated resource %q", id)
				}
				ids[id] = true
			}
		}
	}
	return nil
}
func (f *PeerFabric) submit(now int64, j *peerJob) {
	if f.groupDepth > 0 {
		f.groupBuffer = append(f.groupBuffer, j)
		return
	}
	f.serial++
	j.record.Transaction = f.serial
	j.record.Time = now
	j.record.Bytes = max(1, j.units) * f.bytes
	if f.phases != nil {
		f.phases.stage(now, j)
		return
	}
	j.record.Name = "transfer_queued"
	f.emit(j.record)
	f.queue = append(f.queue, j)
	f.dispatch(now)
}
func (f *PeerFabric) dispatch(now int64) {
	if f.dispatching {
		return
	}
	f.dispatching = true
	defer func() { f.dispatching = false }()
	attempted := map[int64]bool{}
	for {
		sort.SliceStable(f.queue, func(i, j int) bool { return f.jobLess(f.queue[i], f.queue[j]) })
		index := -1
		for i, j := range f.queue {
			if !attempted[j.record.Transaction] {
				index = i
				break
			}
		}
		if index < 0 {
			break
		}
		j := f.queue[index]
		attempted[j.record.Transaction] = true
		if j.notBefore > now {
			continue
		}
		if f.phases != nil && !f.phases.submissionReady(j) {
			continue
		}
		available := true
		for _, id := range j.path {
			if f.busy[id] {
				available = false
				break
			}
		}
		if !available {
			continue
		}
		if j.prepare != nil && !j.prepare(now) {
			continue
		}
		// prepare may append STORE dependencies; submit is non-reentrant.
		f.queue = append(f.queue[:index], f.queue[index+1:]...)
		for _, id := range j.path {
			f.busy[id] = true
		}
		f.active++
		d := f.durationBytes(j.path, max(1, j.units)*f.bytes)
		if f.native != nil {
			direction := "d2h"
			if j.record.Destination == "hbm" {
				direction = "h2d"
			}
			d = f.native.estimate(direction)
		}
		for _, id := range j.path {
			f.busyUntil[id] = now + d
		}
		j.record.Time = now
		j.record.Duration = d
		if f.external != nil {
			j.record.Duration = 0
			j.record.Counters = map[string]int64{"estimated_duration_us": d}
		}
		j.record.Name = "transfer_start"
		if f.mechanisms.TransferPolicy == "dependency_edf" && j.blockedRequests != nil {
			j.record.Requests = j.blockedRequests()
			if len(j.record.Requests) > 0 {
				r := j.record
				r.Name = "dependency_inherit"
				d := j.effectiveDeadline(f)
				if d < math.MaxInt64 {
					r.Counters = map[string]int64{"effective_deadline_us": d}
				}
				f.emit(r)
			}
		}
		f.emit(j.record)
		if f.native != nil {
			f.native.submit(now, j)
		} else if f.external != nil {
			f.external.submit(j)
		} else {
			j.schedule(&peerTransferEvent{at: now + d, fabric: f, job: j})
		}
	}
}

type peerTransferEvent struct {
	at     int64
	fabric *PeerFabric
	job    *peerJob
}

func (e *peerTransferEvent) Timestamp() int64 { return e.at }
func (e *peerTransferEvent) Priority() int    { return sim.PriorityAdapterLoad }
func (e *peerTransferEvent) Execute(_ *sim.Simulator) {
	f, j := e.fabric, e.job
	for _, id := range j.path {
		f.busy[id] = false
	}
	f.active--
	if f.external != nil {
		j.record.Duration = e.at - j.record.Time
	}
	j.record.Time = e.at
	j.record.Name = "transfer_end"
	f.emit(j.record)
	if f.phases != nil {
		f.phases.physicalComplete(e.at, j)
		f.dispatch(e.at)
		return
	}
	finish := func(now int64) { j.done(now); f.notifyDecisionTransfer(now, j); f.dispatch(now); f.wakeStores(now) }
	if f.mechanisms.CompletionUS > 0 {
		f.controlSubmit(e.at, &peerControlJob{record: PeerRecord{Instance: j.record.Instance, Request: j.record.Request, Transaction: j.record.Transaction, Reason: "completion"}, service: f.mechanisms.CompletionUS, schedule: j.schedule, done: finish})
		f.dispatch(e.at)
	} else {
		finish(e.at)
	}
}

// One notification per completed job (a group calls all member callbacks
// before this point). No physical block identity or predicted time is exposed.
func (f *PeerFabric) notifyDecisionTransfer(now int64, j *peerJob) {
	for _, s := range f.stores {
		if s.id == j.record.Instance && s.decisionEvents != nil {
			s.decisionEvents(sim.DecisionEvent{Kind: "transfer_adopted", OccurredUS: now, Request: j.record.Request,
				Transfer: &sim.DecisionTransfer{ID: j.record.Transaction, Request: j.record.Request, Source: j.record.Source, Destination: j.record.Destination, Bytes: j.record.Bytes}})
		}
	}
}
func (f *PeerFabric) reserve(pool, hash string, tokens []sim.TokenID, now int64) (*peerEntry, bool) {
	p := f.pools[pool]
	if e := p.entries[hash]; e != nil {
		return e, false
	}
	if policy, ok := p.policy.(PoolAdmissionPolicy); ok {
		decision := policy.Admit(PoolAdmissionContext{Hash: hash, CapacityBlocks: p.config.CapacityBlocks,
			UsedBlocks: int64(len(p.entries)), Candidates: p.evictionCandidates()})
		accepted := int64(0)
		if decision.Accept {
			accepted = 1
		}
		f.emit(PeerRecord{Time: now, Name: "l2_admission_decision", Destination: pool, Hash: hash,
			Reason: decision.Reason, Counters: map[string]int64{"accepted": accepted}})
		if !decision.Accept {
			return nil, false
		}
	}
	if int64(len(p.entries)) == p.config.CapacityBlocks {
		victim := p.evictionVictim()
		if victim == nil {
			return nil, false
		}
		if f.native != nil {
			f.native.releaseEntry(victim)
		} else if f.external != nil {
			f.external.release(victim.nativeBuffer)
		}
		delete(p.entries, victim.hash)
		f.poolPolicyEvent(pool, "remove", "", []string{victim.hash}, now)
		f.emit(PeerRecord{Time: now, Name: "l2_evict", Source: pool, Hash: victim.hash, Bytes: f.bytes})
	}
	e := &peerEntry{hash: hash, tokens: append([]sim.TokenID(nil), tokens...), last: now}
	if f.native != nil {
		e.nativeBuffer = f.native.reserveEntry(pool)
	} else if f.external != nil {
		e.nativeBuffer = f.external.reserve()
	}
	p.entries[hash] = e
	f.emit(PeerRecord{Time: now, Name: "l2_reserve", Destination: pool, Hash: hash, Bytes: f.bytes})
	return e, true
}

// Snapshot counts physical resident/reserved blocks once per pool, not per reader.
func (f *PeerFabric) Snapshot() map[string]map[string]int64 {
	out := map[string]map[string]int64{}
	for id, p := range f.pools {
		m := map[string]int64{"capacity": p.config.CapacityBlocks, "ready": 0, "reserved": 0, "read_pins": 0}
		for _, e := range p.entries {
			if e.ready {
				m["ready"]++
			} else {
				m["reserved"]++
			}
			m["read_pins"] += int64(e.readers)
		}
		out[id] = m
	}
	return out
}
func (f *PeerFabric) Pending() int {
	if f.phases != nil {
		return len(f.phases.jobs)
	}
	return f.active + len(f.queue) + f.controlActive + len(f.controlQueue)
}
func sortedPeerIDs(m map[string]bool) []string {
	ids := make([]string, 0, len(m))
	for id, v := range m {
		if v {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}
