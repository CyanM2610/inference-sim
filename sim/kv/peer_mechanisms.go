package kv

import (
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// PeerMechanisms contains opt-in control/scheduling experiments. Zero values
// preserve the original peer-L2 behavior and deterministic event stream.
type PeerMechanisms struct {
	// Empty/continuous preserves eager background saving. on_preemption and
	// on_reclaim publish completed HBM blocks but skip eager copies. The former
	// allows explicit request-spill decisions; the latter stores on reclamation.
	BackgroundStoreMode string `json:"background_store_mode,omitempty"`
	// BackgroundStorePool saves computed full input blocks without evicting the
	// HBM copy. Reclaim remains a separate policy decision; empty preserves the
	// original store-on-reclaim path. Transfer costs still come from resources.
	BackgroundStorePool string `json:"background_store_pool,omitempty"`
	GroupTransfers      bool   `json:"group_transfers,omitempty"`
	// ConcurrentRestores admits complete restore ranges against HBM capacity
	// and the remaining prefill reservations of other in-flight requests.
	ConcurrentRestores     bool                 `json:"concurrent_restores,omitempty"`
	DirectoryUS            int64                `json:"directory_us,omitempty"`
	CompletionUS           int64                `json:"completion_us,omitempty"`
	PolicyUS               int64                `json:"policy_us,omitempty"`
	ControlWorkers         int                  `json:"control_workers,omitempty"`
	TransferPolicy         string               `json:"transfer_policy,omitempty"`
	QueueAware             bool                 `json:"queue_aware,omitempty"`
	RestoreWindow          int                  `json:"restore_window_blocks,omitempty"`
	ReserveAtDispatch      bool                 `json:"reserve_at_dispatch,omitempty"`
	CancelQueuedPromotions bool                 `json:"cancel_queued_promotions,omitempty"`
	MaxPromotionQueueUS    int64                `json:"max_promotion_queue_us,omitempty"`
	Online                 *PeerOnlinePromotion `json:"online_promotion,omitempty"`
}
type PeerOnlinePromotion struct {
	PeriodUS       int64   `json:"period_us"`
	EndUS          int64   `json:"end_us"`
	HalfLifeUS     int64   `json:"half_life_us"`
	MinFrequency   float64 `json:"min_frequency"`
	BlocksPerTick  int     `json:"blocks_per_tick"`
	BytesPerSecond int64   `json:"bytes_per_second"`
	BurstBlocks    int     `json:"burst_blocks"`
}

func (m PeerMechanisms) Validate() error {
	switch m.BackgroundStoreMode {
	case "", "continuous":
	case "on_preemption", "on_reclaim":
		if m.BackgroundStorePool == "" {
			return fmt.Errorf("%s storage requires a background store pool", m.BackgroundStoreMode)
		}
	default:
		return fmt.Errorf("unknown background store mode %q", m.BackgroundStoreMode)
	}
	if m.ConcurrentRestores && (!m.GroupTransfers || m.ReserveAtDispatch || m.Online != nil) {
		return fmt.Errorf("concurrent restores require grouped immediate reservations without online promotion")
	}
	if m.GroupTransfers && (m.BackgroundStorePool == "" || m.ReserveAtDispatch) {
		return fmt.Errorf("grouped transfers require a background pool and immediate reservations")
	}
	if m.DirectoryUS < 0 || m.CompletionUS < 0 || m.PolicyUS < 0 || m.ControlWorkers < 0 || m.RestoreWindow < 0 || m.MaxPromotionQueueUS < 0 {
		return fmt.Errorf("peer mechanism costs/windows must be nonnegative")
	}
	switch m.TransferPolicy {
	case "", "fifo", "read_first", "dependency_edf":
	default:
		return fmt.Errorf("unknown transfer policy %q", m.TransferPolicy)
	}
	if p := m.Online; p != nil {
		if p.PeriodUS <= 0 || p.EndUS < p.PeriodUS || p.HalfLifeUS < 0 || p.MinFrequency < 0 || math.IsNaN(p.MinFrequency) || math.IsInf(p.MinFrequency, 0) || p.BlocksPerTick <= 0 || p.BytesPerSecond <= 0 || p.BurstBlocks <= 0 {
			return fmt.Errorf("invalid online promotion period, horizon, frequency or byte budget")
		}
	}
	return nil
}

// ConfigureMechanisms is normally called before constructing caches. Online
// tracking must be enabled before NewPeerCache; the runner enforces this order.
func (f *PeerFabric) ConfigureMechanisms(m PeerMechanisms) error {
	if err := m.Validate(); err != nil {
		return err
	}
	f.mechanisms = m
	return nil
}
func (f *PeerFabric) SetRequestDeadline(id string, at int64) { f.deadlines[id] = at }
func (f *PeerFabric) deadline(id string) int64 {
	if d := f.deadlines[id]; d > 0 {
		return d
	}
	return math.MaxInt64
}
func (j *peerJob) effectiveDeadline(f *PeerFabric) int64 {
	if j.deadline != nil {
		return j.deadline()
	}
	if j.record.Reason == "restore" {
		return f.deadline(j.record.Request)
	}
	return math.MaxInt64
}
func (f *PeerFabric) jobLess(a, b *peerJob) bool {
	rank := func(j *peerJob) int {
		switch j.record.Reason {
		case "restore":
			return 0
		case "store":
			return 1
		default:
			return 2
		}
	}
	switch f.mechanisms.TransferPolicy {
	case "read_first":
		if rank(a) != rank(b) {
			return rank(a) < rank(b)
		}
	case "dependency_edf":
		x, y := a.effectiveDeadline(f), b.effectiveDeadline(f)
		if x != y {
			return x < y
		}
		if rank(a) != rank(b) {
			return rank(a) < rank(b)
		}
	}
	return a.record.Transaction < b.record.Transaction
}

// EstimateQueueUS projects currently submitted data work without mutating it.
// It excludes unresolved HBM dependencies, control work and future arrivals.
func (f *PeerFabric) EstimateQueueUS(now int64, path []string) int64 {
	if f.phases != nil && f.phases.submissionTails != nil {
		return f.phases.estimateQueueUS(now, path)
	}
	until := map[string]int64{}
	for id, t := range f.busyUntil {
		until[id] = max(now, t)
	}
	q := append([]*peerJob(nil), f.queue...)
	sort.SliceStable(q, func(i, j int) bool { return f.jobLess(q[i], q[j]) })
	for _, j := range q {
		start := now
		for _, id := range j.path {
			start = max(start, until[id])
		}
		end := start + f.durationBytes(j.path, max(1, j.units)*f.bytes)
		for _, id := range j.path {
			until[id] = end
		}
	}
	start := now
	for _, id := range path {
		start = max(start, until[id])
	}
	return start - now
}

type peerControlJob struct {
	record   PeerRecord
	service  int64
	schedule func(sim.Event)
	done     func(int64)
	worker   int
}

func (f *PeerFabric) controlSubmit(now int64, j *peerControlJob) {
	j.record.Time = now
	j.record.Name = "control_queued"
	f.emit(j.record)
	f.controlQueue = append(f.controlQueue, j)
	f.controlDispatch(now)
}
func (f *PeerFabric) controlDispatch(now int64) {
	workers := max(1, f.mechanisms.ControlWorkers)
	for f.controlActive < workers && len(f.controlQueue) > 0 {
		j := f.controlQueue[0]
		f.controlQueue = f.controlQueue[1:]
		f.controlActive++
		for w := 0; w < workers; w++ {
			if !f.controlBusy[w] {
				j.worker = w
				f.controlBusy[w] = true
				break
			}
		}
		j.record.Source = fmt.Sprintf("control_worker_%d", j.worker)
		j.record.Time = now
		j.record.Name = "control_start"
		j.record.Duration = j.service
		f.emit(j.record)
		j.schedule(&peerControlEvent{at: now + j.service, f: f, j: j})
	}
}

type peerControlEvent struct {
	at int64
	f  *PeerFabric
	j  *peerControlJob
}

func (e *peerControlEvent) Timestamp() int64 { return e.at }
func (e *peerControlEvent) Priority() int    { return sim.PriorityAdapterLoad }
func (e *peerControlEvent) Execute(_ *sim.Simulator) {
	e.f.controlActive--
	delete(e.f.controlBusy, e.j.worker)
	e.j.record.Time = e.at
	e.j.record.Name = "control_end"
	e.f.emit(e.j.record)
	e.j.done(e.at)
	e.f.controlDispatch(e.at)
}
func (f *PeerFabric) wakeStores(now int64) {
	for _, s := range f.stores {
		s.clock = now
		s.waiting = map[string]bool{}
		s.wake(now)
	}
}

func (f *PeerFabric) CancelQueuedPromotions(now int64, instance string) int {
	count := 0
	remaining := f.queue[:0]
	for _, j := range f.queue {
		if j.record.Instance == instance && j.record.Reason == "promotion" {
			j.cancel(now)
			count++
		} else {
			remaining = append(remaining, j)
		}
	}
	f.queue = remaining
	return count
}
func (f *PeerFabric) ControlSnapshot() map[string]int64 {
	return map[string]int64{"active": int64(f.controlActive), "queued": int64(len(f.controlQueue)), "workers": int64(max(1, f.mechanisms.ControlWorkers))}
}
func (s *PeerCache) directoryReady(req string) bool {
	if s.fabric.mechanisms.DirectoryUS == 0 {
		return true
	}
	ready, started := s.directory[req]
	if started {
		return ready
	}
	s.directory[req] = false
	s.fabric.controlSubmit(s.clock, &peerControlJob{record: PeerRecord{Instance: s.id, Request: req, Reason: "directory"}, service: s.fabric.mechanisms.DirectoryUS, schedule: s.schedule, done: func(now int64) {
		s.directory[req] = true
		s.clock = now
		s.emit("directory_reply", req, "", "", "", "lookup_permitted")
		s.wake(now)
	}})
	return false
}
