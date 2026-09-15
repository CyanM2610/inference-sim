package policylab

import (
	"fmt"
	"math"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/kv"
)

// LiveController executes the existing instance scheduler and PeerCache on
// externally observed hardware time. It never predicts completion of a copy or
// model step. All calls must be serialized on one goroutine.
type LiveController struct {
	instance *cluster.InstanceSimulator
	store    *liveStore
	fabric   *kv.PeerFabric
	requests []*sim.Request
	records  []kv.PeerRecord
	previous int64
	finished map[string]LiveRequestResult
}

func (l *LiveController) AttachPhysicalMemory(memory kv.ExternalMemory) error {
	if l.previous != -1 {
		return fmt.Errorf("physical memory must attach before the first Poll")
	}
	return l.fabric.AttachExternalMemory(memory)
}
func (l *LiveController) ValidatePhysicalMemory() error { return l.fabric.ValidateExternalMemory() }

func (l *LiveController) EnableHashWorkAccounting() { l.store.EnableHashWorkAccounting() }
func (l *LiveController) TakeHashWork() kv.HashWork { return l.store.TakeHashWork() }

type LiveBatch struct {
	Tokens   []sim.TokenID `json:"tokens"`
	ID       int64         `json:"id"`
	IssuedUS int64         `json:"issued_us"`
	Request  string        `json:"request"`
	Prefix   int64         `json:"prefix"`
	Query    int64         `json:"query"`
	Slots    []int64       `json:"slots"`
}
type LiveRequestResult struct {
	Outputs      []sim.TokenID `json:"output_tokens"`
	ID           string        `json:"id"`
	ArrivalUS    int64         `json:"arrival_us"`
	FirstTokenUS int64         `json:"first_token_us"`
	FinishedUS   int64         `json:"finished_us"`
	Progress     int64         `json:"progress"`
}
type LivePoll struct {
	Token  *sim.TokenID `json:"observed_token,omitempty"`
	NowUS  int64        `json:"now_us"`
	Copies []int64      `json:"completed_copies,omitempty"`
	Batch  int64        `json:"completed_batch,omitempty"`
}
type LiveReply struct {
	MetricSchema    string                      `json:"metric_schema"`
	Done            bool                        `json:"done"`
	NextEventUS     int64                       `json:"next_event_us"`
	Copies          []kv.ExternalTransfer       `json:"copies"`
	Batch           *LiveBatch                  `json:"batch,omitempty"`
	Records         []kv.PeerRecord             `json:"records"`
	Requests        []LiveRequestResult         `json:"completed_requests"`
	ControlWallNS   int64                       `json:"control_wall_ns"`
	EventsProcessed int                         `json:"events_processed"`
	HBM             map[string]int64            `json:"hbm"`
	Pools           map[string]map[string]int64 `json:"pools"`
}
type liveStore struct {
	*kv.PeerCache
	serial    int64
	batch     *LiveBatch
	delivered bool
	complete  func(int64)
	request   *sim.Request
	generated map[string]int64
	// Request.FirstTokenTime tracks the latest prefill in the legacy simulator.
	// Hardware-visible output progress survives KV loss, so its first observed
	// timestamp must survive the scheduler's prefill-state reset as well.
	firstObservedUS map[string]int64
}

func (*liveStore) AsyncBatchEnabled() bool { return true }
func (s *liveStore) BeginBatch(now int64, work []sim.BatchWork, done func(int64)) (bool, error) {
	if len(work) != 1 || s.batch != nil {
		return false, fmt.Errorf("live backend requires one idle batch slot")
	}
	w := work[0]
	slots := append([]int64(nil), s.RequestMap[w.Request.ID]...)
	if int64(len(slots)) < (w.PrefixTokens+w.NewTokens+15)/16 {
		return false, fmt.Errorf("live batch missing physical pages")
	}
	s.serial++
	s.batch = &LiveBatch{ID: s.serial, IssuedUS: now, Request: w.Request.ID, Prefix: w.PrefixTokens, Query: w.NewTokens, Slots: slots}
	s.request = w.Request
	if w.PrefixTokens < w.Request.InputLen() {
		s.batch.Tokens = append([]sim.TokenID(nil), w.Request.InputTokenSlice(w.PrefixTokens, w.PrefixTokens+w.NewTokens)...)
	} else {
		index := w.PrefixTokens - w.Request.InputLen()
		if index >= s.generated[w.Request.ID] {
			return false, fmt.Errorf("decode attempted to consume an unobserved token")
		}
		s.batch.Tokens = append([]sim.TokenID(nil), w.Request.OutputTokens[index:index+w.NewTokens]...)
	}
	s.complete = done
	s.delivered = false
	return true, nil
}

type liveLatency struct{}

func (*liveLatency) StepTime([]*sim.Request) int64 {
	panic("live backend attempted synthetic computation")
}
func (*liveLatency) QueueingTime(*sim.Request) int64  { return 0 }
func (*liveLatency) OutputTokenProcessingTime() int64 { return 0 }
func (*liveLatency) PostDecodeFixedOverhead() int64   { return 0 }

func NewLiveController(c Config) (*LiveController, error) {
	c.Native = nil // Hardware supplies service completion, never profile delays.
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if len(c.Instances) != 1 || c.MaxSequences != 1 || c.BlockTokens != 16 || c.RoutingUS != 0 || c.Model.NumLayers != 28 || c.Model.NumKVHeads != 4 || c.Model.EffectiveHeadDim() != 128 || c.Model.EffectiveKVBytesPerParam() != 2 {
		return nil, fmt.Errorf("live controller requires one batch-one native 7B instance")
	}
	l := &LiveController{previous: -1, finished: map[string]LiveRequestResult{}}
	sink := func(r kv.PeerRecord) { l.records = append(l.records, r) }
	f, err := kv.NewPeerFabric(c.Resources, c.Pools, 917504, sink)
	if err != nil {
		return nil, err
	}
	l.fabric = f
	if c.Mechanisms != nil {
		if err = f.ConfigureMechanisms(*c.Mechanisms); err != nil {
			return nil, err
		}
		if c.Mechanisms.Online != nil {
			return nil, fmt.Errorf("live controller requires explicit promotion events")
		}
	}
	p, err := kv.NewPeerCache("instance_0", c.Instances[0].HBMBlocks, 16, f, c.Instances[0].Access, kv.BuiltinPeerPolicy{Name: c.Policy.Name, MinFrequency: c.Policy.MinFrequency, MaxStoreUS: c.Policy.MaxStoreUS})
	if err != nil {
		return nil, err
	}
	if err = f.ConfigureExternalTransfers(); err != nil {
		return nil, err
	}
	l.store = &liveStore{PeerCache: p, generated: map[string]int64{}, firstObservedUS: map[string]int64{}}
	sc := sim.SimConfig{Horizon: math.MaxInt64, Seed: c.Seed, BatchCompletionEvents: true,
		KVCacheConfig: sim.NewKVCacheConfig(c.Instances[0].HBMBlocks, 16, 0, 0, 0, 0),
		BatchConfig:   sim.NewBatchConfig(1, c.MaxBatchTokens, c.PrefillChunk), LatencyCoeffs: sim.NewLatencyCoeffs(nil, c.Alpha),
		ModelHardwareConfig: sim.NewModelHardwareConfig(c.Model, c.Hardware, "policy-model", "configured-gpu", 1, 1, false, "", "roofline", 0)}
	l.instance = cluster.NewInstanceSimulatorWithBackends("instance_0", sc, l.store, &liveLatency{})
	deadlines := map[string]int64{}
	for _, r := range c.Requests {
		if r.TTFTSLOUS > 0 {
			deadlines[r.ID] = r.At + r.TTFTSLOUS
			f.SetRequestDeadline(r.ID, r.At+r.TTFTSLOUS)
		}
	}
	l.instance.SetInstanceScheduler(&requestScheduler{name: c.RequestScheduler, instance: "instance_0", deadlines: deadlines, sink: sink})
	for _, r := range c.Requests {
		q := &sim.Request{ID: r.ID, ArrivalTime: r.At, InputTokens: append([]sim.TokenID(nil), r.Input...), OutputTokens: append([]sim.TokenID(nil), r.Output...), State: sim.StateQueued}
		l.requests = append(l.requests, q)
		l.instance.InjectRequest(q)
	}
	for _, x := range c.Promotions {
		p.SchedulePromotion(x.At, x.Tokens, x.Budget)
	}
	return l, nil
}
func (l *LiveController) Poll(p LivePoll) (*LiveReply, error) {
	start := time.Now()
	if p.NowUS < 0 || p.NowUS < l.previous {
		return nil, fmt.Errorf("live observation clock moved backwards")
	}
	l.previous = p.NowUS
	for _, id := range p.Copies {
		if err := l.fabric.CompleteExternalTransfer(id, p.NowUS); err != nil {
			return nil, err
		}
	}
	if p.Batch != 0 {
		s := l.store
		if s.batch == nil || !s.delivered || s.batch.ID != p.Batch || p.NowUS <= s.batch.IssuedUS {
			return nil, fmt.Errorf("unknown or premature live batch completion")
		}
		index := s.batch.Prefix + s.batch.Query - s.request.InputLen()
		if index >= 0 {
			if p.Token == nil || *p.Token < 0 || index >= int64(len(s.request.OutputTokens)) || index > s.generated[s.request.ID] {
				return nil, fmt.Errorf("missing or out-of-order hardware output token")
			}
			if index < s.generated[s.request.ID] && s.request.OutputTokens[index] != *p.Token {
				return nil, fmt.Errorf("recomputation changed an already observed token")
			}
			s.request.OutputTokens[index] = *p.Token
			if index == s.generated[s.request.ID] {
				if index == 0 {
					s.firstObservedUS[s.request.ID] = p.NowUS
				}
				s.generated[s.request.ID]++
			}
		} else if p.Token != nil {
			return nil, fmt.Errorf("internal prefill has no request-visible token")
		}
		complete := s.complete
		s.complete = nil
		s.batch = nil
		s.delivered = false
		complete(p.NowUS)
	}
	reply := &LiveReply{NextEventUS: -1, MetricSchema: "absolute-first-token-v2"}
	for l.instance.HasPendingEvents() && l.instance.PeekNextEventTime() <= p.NowUS {
		e := l.instance.ProcessNextEvent()
		reply.EventsProcessed++
		for _, r := range l.requests {
			if r.State == sim.StateCompleted {
				if _, ok := l.finished[r.ID]; !ok {
					// Keep the historical no-output convention, but requests with
					// visible outputs use their first unique hardware acknowledgement.
					first := r.ArrivalTime + r.FirstTokenTime
					if len(r.OutputTokens) > 0 {
						var observed bool
						first, observed = l.store.firstObservedUS[r.ID]
						if !observed || l.store.generated[r.ID] != int64(len(r.OutputTokens)) {
							return nil, fmt.Errorf("request completed without its unique output observations")
						}
					}
					l.finished[r.ID] = LiveRequestResult{ID: r.ID, ArrivalUS: r.ArrivalTime, FirstTokenUS: first, FinishedUS: e.Timestamp(), Progress: r.ProgressIndex, Outputs: append([]sim.TokenID(nil), r.OutputTokens...)}
				}
			}
		}
		if reply.EventsProcessed > 100000 {
			return nil, fmt.Errorf("live controller did not yield")
		}
	}
	if l.instance.HasPendingEvents() {
		reply.NextEventUS = l.instance.PeekNextEventTime()
	}
	if l.store.batch != nil && !l.store.delivered {
		b := *l.store.batch
		b.Slots = append([]int64(nil), b.Slots...)
		reply.Batch = &b
		l.store.delivered = true
	}
	reply.Copies = l.fabric.DrainExternalTransfers()
	reply.Records = l.records
	l.records = nil
	for _, r := range l.requests {
		if done, ok := l.finished[r.ID]; ok {
			reply.Requests = append(reply.Requests, done)
		}
	}
	reply.HBM = l.store.PeerSnapshot()
	reply.Pools = l.fabric.Snapshot()
	reply.Done = len(l.finished) == len(l.requests) && l.fabric.Pending() == 0 && l.store.batch == nil && !l.instance.HasPendingEvents()
	reply.ControlWallNS = time.Since(start).Nanoseconds()
	return reply, nil
}
