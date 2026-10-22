package kvruntime

import (
	"fmt"
	"sort"
)

type Span struct {
	SourceOffset      int64 `json:"source_offset"`
	DestinationOffset int64 `json:"destination_offset"`
	Bytes             int64 `json:"bytes"`
}

// TransferCost describes primitive service costs, not a synthetic completion
// timestamp. CPU issue and DMA evolve independently under finite queue credits.
type TransferCost struct {
	// HoldCPUUntilCompletion submits the entire transaction asynchronously,
	// then holds the host thread on its last DMA (e.g. cudaEventSynchronize).
	HoldCPUUntilCompletion bool `json:"hold_cpu_until_completion,omitempty"`
	// HoldCPUUntilDMA models a synchronous host wait on actual device completion,
	// including queue delays and service-rate changes, after async submission.
	HoldCPUUntilDMA  bool   `json:"hold_cpu_until_dma,omitempty"`
	PrepareName      string `json:"prepare_stage_name,omitempty"`
	PlanNS           int64  `json:"plan_ns"`
	SubmitNS         int64  `json:"submit_per_copy_ns"`
	DMANS            int64  `json:"dma_per_copy_ns"`
	SetupNS          int64  `json:"event_setup_ns"`
	ObserveNS        int64  `json:"completion_observe_ns"`
	StageBeforeNS    int64  `json:"staging_before_ns"`
	StageAfterNS     int64  `json:"staging_after_ns"`
	BlockingAPI      bool   `json:"blocking_api"`
	PreparePerCopyNS int64  `json:"prepare_per_copy_ns"`
	CopyAfterNS      int64  `json:"staging_after_copy_ns"`
	WaitForDMA       bool   `json:"wait_for_dma"`
	ReturnLeadNS     int64  `json:"return_before_dma_end_ns"`
}

func (c TransferCost) Validate() error {
	if c.HoldCPUUntilDMA && c.HoldCPUUntilCompletion {
		return fmt.Errorf("choose per-copy or transaction-end synchronization")
	}
	if (c.HoldCPUUntilDMA || c.HoldCPUUntilCompletion) && (c.WaitForDMA || c.BlockingAPI || c.ReturnLeadNS != 0) {
		return fmt.Errorf("actual host synchronization cannot be combined with predicted/blocking return modes")
	}
	if c.PlanNS < 0 || c.SubmitNS < 0 || c.DMANS <= 0 || c.SetupNS < 0 || c.ObserveNS < 0 || c.StageBeforeNS < 0 || c.StageAfterNS < 0 || c.PreparePerCopyNS < 0 || c.CopyAfterNS < 0 || c.ReturnLeadNS < 0 {
		return fmt.Errorf("invalid transfer service costs")
	}
	return nil
}

type TransferSpec struct {
	// HostServiceClass covers measured host work excluding actual device waits.
	// SubmitServiceClass, when nonempty, overrides it for CUDA submissions.
	HostServiceClass                    string
	Activity                            func(bool) error // successful plan start / fully published completion
	StreamGroup                         string
	SubmitServiceClass, DMAServiceClass string
	ID, Source, Destination             string
	CPU, DMA                            string
	Spans                               []Span
	// Endpoints overrides Source/Destination per span for real tensor lists.
	// An empty list uses the single-buffer defaults.
	Endpoints []CopyEndpoints
	Cost      TransferCost
	Parents   []int64
	Done      func(*TransferResult)
}
type CopyEndpoints struct{ Source, Destination string }
type TransferResult struct {
	ID              string `json:"id"`
	StartNS         int64  `json:"start_ns"`
	CPUFinishedNS   int64  `json:"cpu_finished_ns"`
	FirstDMANS      int64  `json:"first_dma_ns"`
	LastDMANS       int64  `json:"last_dma_ns"`
	CompleteNS      int64  `json:"complete_ns"`
	Bytes           int64  `json:"bytes"`
	Copies          int    `json:"copies"`
	CompletionTask  int64  `json:"completion_task"`
	PeakOutstanding int    `json:"peak_outstanding"`
}
type Runtime struct {
	Engine      *Engine
	Memory      *Memory
	QueueDepth  int
	outstanding int
	operations  map[string]*TransferResult
	hostTail    map[string]int64
}

func NewRuntime(e *Engine, m *Memory, queueDepth int) (*Runtime, error) {
	if e == nil || m == nil || queueDepth <= 0 {
		return nil, fmt.Errorf("runtime requires memory, engine and finite queue depth")
	}
	return &Runtime{Engine: e, Memory: m, QueueDepth: queueDepth, operations: map[string]*TransferResult{}, hostTail: map[string]int64{}}, nil
}
func (r *Runtime) Transfer(s TransferSpec) (*TransferResult, error) {
	if s.ID == "" || r.operations[s.ID] != nil || s.CPU == "" || s.DMA == "" || len(s.Spans) == 0 {
		return nil, fmt.Errorf("invalid/duplicate transfer")
	}
	if err := s.Cost.Validate(); err != nil {
		return nil, err
	}
	if s.DMAServiceClass != "" && s.Cost.WaitForDMA {
		return nil, fmt.Errorf("progress-based DMA requires asynchronous or completion-dependent host return; predicted pageable return lead is not calibrated under contention")
	}
	endpoints := append([]CopyEndpoints(nil), s.Endpoints...)
	if len(endpoints) == 0 {
		for range s.Spans {
			endpoints = append(endpoints, CopyEndpoints{s.Source, s.Destination})
		}
	}
	if len(endpoints) != len(s.Spans) {
		return nil, fmt.Errorf("copy endpoint count mismatch")
	}
	sourceSet, targetSet := map[string]bool{}, map[string]bool{}
	for i, p := range s.Spans {
		ep := endpoints[i]
		a, b := r.Memory.Buffers[ep.Source], r.Memory.Buffers[ep.Destination]
		if a == nil || b == nil || a == b || a.Opaque || b.Opaque {
			return nil, fmt.Errorf("invalid transfer buffers")
		}
		sourceSet[ep.Source] = true
		targetSet[ep.Destination] = true
		if a.CellBytes != b.CellBytes || p.Bytes <= 0 || p.SourceOffset < 0 || p.DestinationOffset < 0 || p.SourceOffset > a.Bytes-p.Bytes || p.DestinationOffset > b.Bytes-p.Bytes || p.Bytes%a.CellBytes != 0 || p.SourceOffset%a.CellBytes != 0 || p.DestinationOffset%b.CellBytes != 0 {
			return nil, fmt.Errorf("invalid transfer span")
		}
	}
	var sources, targets []string
	for id := range sourceSet {
		if targetSet[id] {
			return nil, fmt.Errorf("source/target alias in transaction")
		}
		sources = append(sources, id)
	}
	for id := range targetSet {
		targets = append(targets, id)
	}
	sort.Strings(sources)
	sort.Strings(targets)
	e := r.Engine
	result := &TransferResult{ID: s.ID, Copies: len(s.Spans), FirstDMANS: -1}
	r.operations[s.ID] = result
	add := func(name, resource string, duration int64, parents []int64, start, finish func() error) int64 {
		class := ""
		group := ""
		if name == "dma_copy" {
			class = s.DMAServiceClass
			group = s.StreamGroup
		} else if resource == s.CPU {
			class = s.HostServiceClass
		}
		return e.MustAdd(Task{Name: name, Operation: s.ID, Resource: resource, DurationNS: duration, Parents: parents, Start: start, Finish: finish, ServiceClass: class, ResourceGroup: group})
	}
	planParents := append([]int64(nil), s.Parents...)
	if tail := r.hostTail[s.CPU]; tail != 0 {
		found := false
		for _, id := range planParents {
			if id == tail {
				found = true
			}
		}
		if !found {
			planParents = append(planParents, tail)
		}
	}
	begin := add("copy_plan", s.CPU, s.Cost.PlanNS, planParents, func() error {
		result.StartNS = e.NowNS
		// Precheck every endpoint before mutating any reference. A failed plan
		// must not strand the first tensor pin when a later tensor is unavailable.
		for _, id := range sources {
			b := r.Memory.Buffers[id]
			if b == nil || b.State != "ready" || b.Writers > 0 {
				return fmt.Errorf("source unavailable: %s", id)
			}
		}
		for _, id := range targets {
			b := r.Memory.Buffers[id]
			if b == nil || b.State == "reserved" || b.State == "declared" || b.Pins > 0 || b.Writers > 0 || b.Holds > 0 {
				return fmt.Errorf("target unavailable: %s", id)
			}
		}
		for _, id := range sources {
			if err := r.Memory.Pin(id); err != nil {
				return err
			}
		}
		for _, id := range targets {
			if err := r.Memory.BeginWrite(id); err != nil {
				return err
			}
		}
		if s.Activity != nil {
			return s.Activity(true)
		}
		return nil
	}, nil)
	previous := begin
	if s.Cost.StageBeforeNS > 0 {
		previous = add("pageable_staging_before", s.CPU, s.Cost.StageBeforeNS, []int64{previous}, nil, nil)
	}
	previous = add("event_record_start", s.CPU, s.Cost.SetupNS, []int64{previous}, nil, nil)
	var dmaIDs []int64
	for i, p := range s.Spans {
		ep := endpoints[i]
		if s.Cost.PreparePerCopyNS > 0 {
			name := s.Cost.PrepareName
			if name == "" {
				name = "pageable_prepare_copy"
			}
			previous = add(name, s.CPU, s.Cost.PreparePerCopyNS, []int64{previous}, nil, nil)
		}
		submitClass := s.SubmitServiceClass
		if submitClass == "" {
			submitClass = s.HostServiceClass
		}
		api := e.MustAdd(Task{Name: "cuda_submit", Operation: s.ID, Resource: s.CPU, ServiceClass: submitClass, DurationNS: s.Cost.SubmitNS, Parents: []int64{previous}, Ready: func() bool { return r.outstanding < r.QueueDepth }, Start: func() error {
			r.outstanding++
			if r.outstanding > result.PeakOutstanding {
				result.PeakOutstanding = r.outstanding
			}
			return nil
		}, Finish: func() error { result.CPUFinishedNS = e.NowNS; return nil }})
		parents := []int64{api}
		if i > 0 {
			parents = append(parents, dmaIDs[i-1])
		} // stream ordering
		var due int64
		dma := add("dma_copy", s.DMA, s.Cost.DMANS, parents, func() error {
			due = e.NowNS + s.Cost.DMANS
			if s.Cost.WaitForDMA {
				_ = e.At(e.NowNS, e.Kick)
			}
			if result.FirstDMANS < 0 {
				result.FirstDMANS = e.NowNS
			}
			e.Emit(Record{Name: "dma_start", Operation: s.ID, Resource: s.DMA, Bytes: p.Bytes, Source: ep.Source, Destination: ep.Destination, SourceOffset: p.SourceOffset, DestinationOffset: p.DestinationOffset, DurationNS: s.Cost.DMANS})
			return nil
		}, func() error {
			if err := r.Memory.Copy(ep.Source, ep.Destination, p.SourceOffset, p.DestinationOffset, p.Bytes); err != nil {
				return err
			}
			r.outstanding--
			result.Bytes += p.Bytes
			result.LastDMANS = e.NowNS
			if s.Cost.BlockingAPI && i == len(s.Spans)-1 && s.Cost.StageAfterNS == 0 {
				result.CPUFinishedNS = e.NowNS
				e.Emit(Record{Name: "host_api_return", Operation: s.ID, Resource: s.CPU, Reason: "blocking_copy"})
			}
			e.Emit(Record{Name: "dma_end", Operation: s.ID, Resource: s.DMA, Bytes: p.Bytes})
			return nil
		})
		dmaIDs = append(dmaIDs, dma)
		previous = api
		if s.Cost.HoldCPUUntilDMA || s.Cost.BlockingAPI {
			name := "host_stream_synchronize"
			if s.Cost.BlockingAPI {
				name = "host_blocking_copy_wait"
			}
			// A DMA dependency alone would serialize calls but leave the host
			// resource free for unrelated work while this blocking call runs.
			previous = e.MustAdd(Task{Name: name, Operation: s.ID, Resource: s.CPU, Parents: []int64{api}, WaitFor: []int64{dma}, Finish: func() error { result.CPUFinishedNS = e.NowNS; return nil }})
		}
		if s.Cost.WaitForDMA {
			previous = e.MustAdd(Task{Name: "host_copy_wait", Operation: s.ID, Resource: s.CPU, Parents: []int64{api}, Ready: func() bool { return due > 0 }, Duration: func() int64 { return max(0, due-s.Cost.ReturnLeadNS-e.NowNS) }, Finish: func() error { result.CPUFinishedNS = e.NowNS; return nil }})
		}
		if s.Cost.CopyAfterNS > 0 {
			previous = add("pageable_copy_after_dma", s.CPU, s.Cost.CopyAfterNS, []int64{previous}, nil, func() error { result.CPUFinishedNS = e.NowNS; return nil })
		}
	}
	if s.Cost.HoldCPUUntilCompletion {
		previous = e.MustAdd(Task{Name: "host_transaction_synchronize", Operation: s.ID, Resource: s.CPU, Parents: []int64{previous}, WaitFor: []int64{dmaIDs[len(dmaIDs)-1]}, Finish: func() error { result.CPUFinishedNS = e.NowNS; return nil }})
	}
	parents := []int64{dmaIDs[len(dmaIDs)-1]}
	if previous != parents[0] {
		parents = append(parents, previous)
	}
	if s.Cost.StageAfterNS > 0 {
		stage := add("pageable_staging_after", s.CPU, s.Cost.StageAfterNS, parents, nil, func() error {
			if s.Cost.BlockingAPI {
				result.CPUFinishedNS = e.NowNS
			}
			return nil
		})
		parents = []int64{stage}
	}
	result.CompletionTask = e.MustAdd(Task{Name: "completion_observe_publish", Operation: s.ID, Resource: s.CPU, ServiceClass: s.HostServiceClass, DurationNS: s.Cost.ObserveNS, Parents: parents, Finish: func() error {
		for _, id := range targets {
			if err := r.Memory.Publish(id); err != nil {
				return err
			}
		}
		for _, id := range sources {
			if err := r.Memory.Unpin(id); err != nil {
				return err
			}
		}
		result.CompleteNS = e.NowNS
		return nil
	}, After: func() {
		if s.Activity != nil {
			if err := s.Activity(false); err != nil {
				e.Fail(err)
				return
			}
		}
		if s.Done != nil {
			s.Done(result)
		}
	}})
	r.hostTail[s.CPU] = previous
	if s.Cost.BlockingAPI || s.Cost.HoldCPUUntilDMA || s.Cost.HoldCPUUntilCompletion {
		r.hostTail[s.CPU] = result.CompletionTask
	}
	if err := e.Error(); err != nil {
		return nil, err
	}
	return result, nil
}

// NativeSpans matches the CUDA profiling layouts exactly (28 K/V layer pairs,
// 16-token pages, GQA 4x128 BF16). The reverse path swaps offsets, not the data.
func NativeSpans(layout string, blocks int64, restore bool) ([]Span, int64, int64, error) {
	const page = int64(16 * 4 * 128 * 2)
	const layers = int64(56)
	if blocks <= 0 || blocks > 1<<20 {
		return nil, 0, 0, fmt.Errorf("invalid block count")
	}
	bytes := layers * blocks * page
	gpuBytes := bytes
	var spans []Span
	switch layout {
	case "packed":
		spans = []Span{{0, 0, bytes}}
	case "layers":
		for l := int64(0); l < layers; l++ {
			spans = append(spans, Span{l * blocks * page, l * blocks * page, blocks * page})
		}
	case "paged":
		gpuBytes = 2 * bytes
		// The hardware fixture's permutation must be bijective. Reject counts
		// sharing a factor with 37 instead of silently aliasing physical pages.
		if blocks%37 == 0 {
			return nil, 0, 0, fmt.Errorf("profiling permutation aliases this block count")
		}
		for l := int64(0); l < layers; l++ {
			for b := int64(0); b < blocks; b++ {
				spans = append(spans, Span{(l*2*blocks + 2*((37*b)%blocks)) * page, (l*blocks + b) * page, page})
			}
		}
	default:
		return nil, 0, 0, fmt.Errorf("unknown native layout %q", layout)
	}
	if restore {
		for i := range spans {
			spans[i].SourceOffset, spans[i].DestinationOffset = spans[i].DestinationOffset, spans[i].SourceOffset
		}
	}
	return spans, gpuBytes, bytes, nil
}
