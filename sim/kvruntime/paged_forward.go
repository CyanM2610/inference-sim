package kvruntime

import (
	"fmt"
	"math"
)

// PagedServiceClasses opt existing tasks into progress-preserving rate changes.
// Forward still denotes the measured forward envelope, not pure kernel cycles.
// Empty classes retain fixed solo costs. Rate selection belongs to a separately
// calibrated interference model; these bindings do not invent slowdown factors.
type PagedServiceClasses struct {
	Forward                      string
	GatherSubmit, GatherKernel   string
	ScatterSubmit, ScatterKernel string
}

// PagedStepSpec describes one real batch-one gather/forward/delta-scatter step.
// TokenOrigins identify the full causal input prefix, including reference output
// tokens already consumed by decode. They are not numerical model predictions.
type PagedStepSpec struct {
	PhaseObserver       func(string, bool) error
	ServiceClasses      PagedServiceClasses
	Inputs              *RequestInputs
	HostOutputBuffer    string
	OutputCopyResource  string
	OutputStreamGroup   string
	OutputValue         uint64
	OutputReferenceKind string
	Request             string
	ID                  string
	Slots               []string
	Prefix, Query       int64
	TokenOrigins        []uint64
	Weights             []float64
	Point               PagedForwardPoint
	WorkspacePool       string
	Parents             []int64
	Done                func(*PagedStepResult)
}
type PagedStepResult struct {
	OutputIndex        *int64           `json:"output_index,omitempty"`
	RecomputedOutput   bool             `json:"recomputed_output,omitempty"`
	OutputObservedNS   int64            `json:"output_observed_ns,omitempty"`
	OutputBytes        int64            `json:"output_bytes,omitempty"`
	ID                 string           `json:"id"`
	StartNS            int64            `json:"start_ns"`
	CompleteNS         int64            `json:"complete_ns"`
	CompletionTask     int64            `json:"completion_task"`
	StagesNS           map[string]int64 `json:"stages_ns"`
	MappingTasks       int              `json:"mapping_tasks"`
	AllocatedPeakDelta int64            `json:"allocated_peak_delta_bytes"`
	CellsVerified      int64            `json:"cells_verified"`
}

func (r *Runtime) PagedStep(s PagedStepSpec) (*PagedStepResult, error) {
	if err := s.Point.Validate(); err != nil {
		return nil, err
	}
	n := s.Prefix + s.Query
	if s.ID == "" || s.Prefix < 0 || s.Query <= 0 || n <= s.Prefix || int64(len(s.Slots)) < (n+15)/16 || int64(len(s.TokenOrigins)) != n || len(s.Weights) != 30 || s.Point.Prefix != s.Prefix || s.Point.Query != s.Query {
		return nil, fmt.Errorf("invalid paged forward shape or lineage")
	}
	var weightSum float64
	for _, w := range s.Weights {
		if w < 0 || math.IsNaN(w) || math.IsInf(w, 0) {
			return nil, fmt.Errorf("invalid paged forward structural weight")
		}
		weightSum += w
	}
	if math.Abs(weightSum-1) > 1e-9 {
		return nil, fmt.Errorf("paged forward weights must sum to one")
	}
	for _, o := range s.TokenOrigins {
		if o == 0 {
			return nil, fmt.Errorf("undefined token origin")
		}
	}
	if s.WorkspacePool == "" {
		return nil, fmt.Errorf("workspace pool required")
	}
	// A location-specific cost is valid only when that phase actually executes.
	// In particular, first-output next-input costs cannot silently disappear on
	// an internal prefill or final-output step using the same numerical shape.
	if s.Point.PhaseGapsNS != nil {
		phases := map[string]bool{"forward": true, "scatter_delta": true, "hf_release": true, "complete": true, "gather_prefix": s.Prefix > 0, "observe_token": s.Point.Output != nil, "index_bind": s.Inputs != nil}
		if s.Inputs != nil {
			if req := s.Inputs.requests[s.Request]; req != nil {
				binding := InputBinding{request: req, prefix: s.Prefix, query: s.Query}
				phases["next_input_d2d"] = binding.NeedsNextInput()
			}
		}
		for phase := range s.Point.PhaseGapsNS {
			if !phases[phase] {
				return nil, fmt.Errorf("profiled phase gap %s has no executing phase", phase)
			}
		}
	}
	e, m := r.Engine, r.Memory
	outputID := s.ID + "/output_token"
	if o := s.Point.Output; o != nil {
		b := m.Buffers[s.HostOutputBuffer]
		if b == nil || b.Bytes < o.HostStorageBytes || b.CellBytes != 8 || s.OutputCopyResource == "" || s.OutputReferenceKind == "" {
			return nil, fmt.Errorf("native output requires bound host storage, copy resource and reference kind")
		}
		if _, err := m.Declare(outputID, s.WorkspacePool, "gpu_output_token_storage", o.GPUStorageBytes, 8); err != nil {
			return nil, err
		}
	}
	before := m.Used[s.WorkspacePool]
	result := &PagedStepResult{ID: s.ID, StartNS: e.NowNS, StagesNS: map[string]int64{}}
	e.Emit(Record{Name: "paged_step_begin", Request: s.Request, Operation: s.ID, PrefixTokens: s.Prefix, QueryTokens: s.Query, PageSlots: append([]string(nil), s.Slots...)})
	if s.Point.Output != nil {
		e.Emit(Record{Name: "paged_output_expected", Request: s.Request, Operation: s.ID, Origin: s.OutputValue, PrefixTokens: n, Reason: s.OutputReferenceKind, Bytes: 8})
	}
	allocatedPeak := before
	alloc := func(id, kind string, bytes int64, opaque bool) error {
		var err error
		if opaque {
			_, err = m.ReserveOpaque(id, s.WorkspacePool, kind, bytes)
		} else {
			_, err = m.Reserve(id, s.WorkspacePool, kind, bytes)
		}
		if err != nil {
			return err
		}
		allocatedPeak = max(allocatedPeak, m.Used[s.WorkspacePool])
		return m.Allocate(id)
	}
	unique := map[string]bool{}
	for _, id := range s.Slots {
		b := m.Buffers[id]
		if b == nil || b.Bytes != 56*16384 || unique[id] {
			return nil, fmt.Errorf("invalid physical page binding")
		}
		unique[id] = true
	}
	var parents = append([]int64(nil), s.Parents...)
	beginPhase := func(name string) int64 {
		if gap := s.Point.PhaseGapsNS[name]; gap > 0 {
			parents = []int64{e.MustAdd(Task{Name: "python_phase_gap", Operation: s.ID + "/" + name, Resource: "cpu", DurationNS: gap, Parents: parents})}
		}
		return e.MustAdd(Task{Name: "phase_begin", Operation: s.ID + "/" + name, Resource: "phase_control", Parents: parents, Finish: func() error {
			result.StagesNS[name] = -e.NowNS
			if s.PhaseObserver != nil {
				return s.PhaseObserver(name, true)
			}
			return nil
		}})
	}
	endPhase := func(name string, tail []int64, observe int64, action func() error) int64 {
		id := e.MustAdd(Task{Name: "phase_complete", Operation: s.ID + "/" + name, Resource: "cpu", DurationNS: observe, Parents: tail, Finish: func() error {
			if action != nil {
				if err := action(); err != nil {
					return err
				}
			}
			result.StagesNS[name] += e.NowNS
			if s.PhaseObserver != nil {
				return s.PhaseObserver(name, false)
			}
			return nil
		}})
		parents = []int64{id}
		return id
	}
	inputs, outputs := make([]string, 56), make([]string, 56)
	var binding *InputBinding
	if s.Inputs != nil {
		if s.Point.Output == nil {
			return nil, fmt.Errorf("physical request inputs require output observation")
		}
		begin := beginPhase("index_bind")
		var tail int64
		var err error
		binding, tail, err = s.Inputs.Bind(s.ID, s.Request, s.Prefix, s.Query, s.Slots, []int64{begin})
		if err != nil {
			return nil, err
		}
		endPhase("index_bind", []int64{tail}, 0, nil)
	}
	for i := range inputs {
		inputs[i] = fmt.Sprintf("%s/input/%d", s.ID, i)
		outputs[i] = fmt.Sprintf("%s/output/%d", s.ID, i)
	}
	if s.Prefix > 0 {
		cost, ok := s.Point.Stages["gather_prefix"]
		if !ok || cost.SubmitNS <= 0 || cost.ServiceNS <= 0 {
			return nil, fmt.Errorf("missing gather calibration")
		}
		cpuTail := beginPhase("gather_prefix")
		var gpuTail int64
		kernel := func(k MappingKernel) error {
			k.SubmitServiceClass = s.ServiceClasses.GatherSubmit
			k.KernelServiceClass = s.ServiceClasses.GatherKernel
			k.Parents = []int64{cpuTail}
			k.StreamParent = gpuTail
			k.SubmitNS = cost.SubmitNS
			k.ServiceNS = cost.ServiceNS
			var err error
			cpuTail, gpuTail, err = r.MapKernel(s.ID+"/gather", "cpu", "gpu_compute", k)
			result.MappingTasks++
			return err
		}
		pages := (s.Prefix + 15) / 16
		full, tail := s.Prefix/16, s.Prefix%16
		for tensor := 0; tensor < 56; tensor++ {
			temporary := fmt.Sprintf("%s/select/%d", s.ID, tensor)
			destination := inputs[tensor]
			owner := destination + "/gather"
			var spans []Span
			var endpoints []CopyEndpoints
			for page := int64(0); page < pages; page++ {
				spans = append(spans, Span{int64(tensor) * 16384, page * 16384, 16384})
				endpoints = append(endpoints, CopyEndpoints{s.Slots[page], temporary})
			}
			if err := kernel(MappingKernel{Name: "index_select_pages", Spans: spans, Endpoints: endpoints, AllowUndefined: true, Prepare: func() error { return alloc(temporary, "index_select_temporary", pages*16384, false) }}); err != nil {
				return nil, err
			}
			firstCopy := true
			copyPart := func(name string, spans []Span, last bool) error {
				var prepare func() error
				if firstCopy {
					prepare = func() error {
						if err := alloc(destination, "hf_gathered_prefix", s.Prefix*1024, false); err != nil {
							return err
						}
						return m.BeginWriteOwned(destination, owner)
					}
					firstCopy = false
				}
				var finish func() error
				if last {
					finish = func() error {
						if err := m.PublishOwned(destination, owner); err != nil {
							return err
						}
						return m.Free(temporary)
					}
				}
				return kernel(MappingKernel{Name: name, Source: temporary, Destination: destination, Spans: spans, WriteOwner: owner, Prepare: prepare, Complete: finish})
			}
			if full > 0 {
				var spans []Span
				for page := int64(0); page < full; page++ {
					for head := int64(0); head < 4; head++ {
						spans = append(spans, Span{page*16384 + head*4096, (head*s.Prefix + page*16) * 256, 4096})
					}
				}
				if err := copyPart("transpose_full_pages", spans, tail == 0); err != nil {
					return nil, err
				}
			}
			if tail > 0 {
				var spans []Span
				for head := int64(0); head < 4; head++ {
					spans = append(spans, Span{full*16384 + head*4096, (head*s.Prefix + full*16) * 256, tail * 256})
				}
				if err := copyPart("trim_partial_page", spans, true); err != nil {
					return nil, err
				}
			}
		}
		endPhase("gather_prefix", []int64{cpuTail, gpuTail}, cost.ObserveNS, nil)
	}
	forward, ok := s.Point.Stages["forward"]
	if !ok || forward.WallNS <= 0 {
		return nil, fmt.Errorf("missing forward calibration")
	}
	forwardBegin := beginPhase("forward")
	tailID := forwardBegin
	if binding != nil {
		tailID = e.MustAdd(Task{Name: "request_query_acquire", Operation: s.ID + "/forward", Resource: "cpu", Parents: []int64{tailID}, Finish: binding.AcquireQuery})
	}
	// A measured phase peak supplies a conservative, explicitly opaque
	// activation envelope. KV and logits remain separately accounted objects.
	knownKVPeak := n*56*1024 + 2*s.Prefix*1024
	workspaceBytes := max(0, s.Point.PhasePeakDelta["forward"]-knownKVPeak)
	workspaceBytes = (workspaceBytes + 255) / 256 * 256
	workspaceID := s.ID + "/activation_envelope"
	logitsID := s.ID + "/logits"
	logitsBytes := max(0, s.Point.PhaseEndDelta["forward"]-n*56*1024)
	if s.Point.Output != nil {
		logitsBytes = max(0, logitsBytes-s.Point.Output.GPUStorageBytes)
	}
	if workspaceBytes > 0 {
		tailID = e.MustAdd(Task{Name: "activation_envelope_acquire", Operation: s.ID + "/forward", Resource: "cpu", Parents: []int64{tailID}, Finish: func() error { return alloc(workspaceID, "opaque_unlocalized_forward_workspace", workspaceBytes, true) }})
	}
	var assigned int64
	for segment, w := range s.Weights {
		if w < 0 {
			return nil, fmt.Errorf("negative structural weight")
		}
		ns := int64(float64(forward.WallNS) * w)
		if segment == 29 {
			ns = forward.WallNS - assigned
		}
		assigned += ns
		layer := segment - 1
		tailID = e.MustAdd(Task{Name: fmt.Sprintf("forward_segment_%02d", segment), Operation: s.ID + "/forward", Resource: "gpu_compute", ServiceClass: s.ServiceClasses.Forward, DurationNS: ns, Parents: []int64{tailID}, Start: func() error {
			if layer < 0 || layer >= 28 {
				return nil
			}
			for j := 0; j < 2; j++ {
				tensor := 2*layer + j
				if s.Prefix > 0 {
					if err := m.Pin(inputs[tensor]); err != nil {
						return err
					}
					b := m.Buffers[inputs[tensor]]
					for head := int64(0); head < 4; head++ {
						for token := int64(0); token < s.Prefix; token++ {
							want := Cell{s.TokenOrigins[token], int64(tensor*4+int(head)) * 256, true}
							if b.Cells[head*s.Prefix+token] != want {
								return fmt.Errorf("gathered prefix lineage mismatch tensor=%d token=%d", tensor, token)
							}
						}
					}
				}
				if err := alloc(outputs[tensor], "hf_forward_kv", n*1024, false); err != nil {
					return err
				}
				if err := m.BeginWrite(outputs[tensor]); err != nil {
					return err
				}
			}
			return nil
		}, Finish: func() error {
			if layer < 0 || layer >= 28 {
				return nil
			}
			for j := 0; j < 2; j++ {
				tensor := 2*layer + j
				out := outputs[tensor]
				for head := int64(0); head < 4; head++ {
					if s.Prefix > 0 {
						if err := m.Copy(inputs[tensor], out, head*s.Prefix*256, head*n*256, s.Prefix*256); err != nil {
							return err
						}
					}
					for token := s.Prefix; token < n; token++ {
						if err := m.ComputeRange(out, (head*n+token)*256, 256, s.TokenOrigins[token], int64(tensor*4+int(head))*256); err != nil {
							return err
						}
					}
				}
				if err := m.Publish(out); err != nil {
					return err
				}
				if s.Prefix > 0 {
					if err := m.Unpin(inputs[tensor]); err != nil {
						return err
					}
					if err := m.Free(inputs[tensor]); err != nil {
						return err
					}
				}
			}
			if layer == 27 && workspaceBytes > 0 {
				return m.Free(workspaceID)
			}
			return nil
		}})
	}
	endPhase("forward", []int64{tailID}, 0, func() error {
		if binding != nil {
			if err := binding.ReleaseQuery(); err != nil {
				return err
			}
		}
		if logitsBytes > 0 {
			if err := alloc(logitsID, "opaque_logits_and_retained_forward_output", logitsBytes, true); err != nil {
				return err
			}
		}
		if s.Point.Output != nil {
			if err := m.Allocate(outputID); err != nil {
				return err
			}
			allocatedPeak = max(allocatedPeak, m.Used[s.WorkspacePool])
			if err := m.BeginWrite(outputID); err != nil {
				return err
			}
			if err := m.ComputeRange(outputID, 0, 8, s.OutputValue, 0); err != nil {
				return err
			}
			return m.Publish(outputID)
		}
		return nil
	})
	if o := s.Point.Output; o != nil {
		observeBegin := beginPhase("observe_token")
		tx, err := r.Transfer(TransferSpec{ID: s.ID + "/output", Source: outputID, Destination: s.HostOutputBuffer, CPU: "cpu", DMA: s.OutputCopyResource, StreamGroup: s.OutputStreamGroup, Parents: []int64{observeBegin}, Spans: []Span{{SourceOffset: 0, DestinationOffset: 0, Bytes: 8}}, Cost: o.Cost})
		if err != nil {
			return nil, err
		}
		endPhase("observe_token", []int64{tx.CompletionTask}, 0, func() error {
			b := m.Buffers[s.HostOutputBuffer]
			if b.State != "ready" || b.Cells[0] != (Cell{s.OutputValue, 0, true}) {
				return fmt.Errorf("incorrect/unpublished output token")
			}
			result.OutputObservedNS = e.NowNS
			result.OutputBytes = tx.Bytes
			if binding != nil {
				if index := binding.OutputIndex(); index >= 0 {
					result.OutputIndex = &index
					result.RecomputedOutput = binding.PreviouslyObservedOutput()
				}
			}
			e.Emit(Record{Name: "paged_output_observed", Request: s.Request, Operation: s.ID, Source: outputID, Destination: s.HostOutputBuffer, Origin: s.OutputValue, Bytes: tx.Bytes, Reason: s.OutputReferenceKind, OutputIndex: result.OutputIndex, Recomputed: result.RecomputedOutput})
			if binding != nil {
				return nil
			} // The downstream input write/release still owns the GPU token.
			return m.Free(outputID)
		})
		if binding != nil {
			if binding.NeedsNextInput() {
				begin := beginPhase("next_input_d2d")
				tail, err := binding.ObserveOutput(outputID, s.OutputValue, []int64{begin})
				if err != nil {
					return nil, err
				}
				endPhase("next_input_d2d", []int64{tail}, 0, nil)
			} else {
				tail, err := binding.ObserveOutput(outputID, s.OutputValue, parents)
				if err != nil {
					return nil, err
				}
				if tail != 0 {
					parents = []int64{tail}
				}
			}
			parents = []int64{e.MustAdd(Task{Name: "gpu_output_tensor_release", Operation: s.ID, Resource: "cpu", Parents: parents, Finish: func() error { return m.Free(outputID) }})}
		}
	}
	scatter, ok := s.Point.Stages["scatter_delta"]
	if !ok || scatter.SubmitNS <= 0 || scatter.ServiceNS <= 0 {
		return nil, fmt.Errorf("missing delta scatter calibration")
	}
	scatterBegin := beginPhase("scatter_delta")
	owner := s.ID + "/scatter"
	firstPage, lastPage := s.Prefix/16, (n-1)/16
	cpuTail := e.MustAdd(Task{Name: "scatter_write_scope", Operation: s.ID + "/scatter", Resource: "cpu", Parents: []int64{scatterBegin}, Finish: func() error {
		for page := firstPage; page <= lastPage; page++ {
			if err := m.BeginWriteOwned(s.Slots[page], owner); err != nil {
				return err
			}
		}
		return nil
	}})
	var gpuTail int64
	scatterKernel := func(tensor int, name string, ranges [][2]int64) error {
		var spans []Span
		var endpoints []CopyEndpoints
		for _, v := range ranges {
			lo, hi := v[0], v[1]
			page := lo / 16
			for head := int64(0); head < 4; head++ {
				spans = append(spans, Span{(head*n + lo) * 256, int64(tensor)*16384 + head*4096 + (lo%16)*256, (hi - lo) * 256})
				endpoints = append(endpoints, CopyEndpoints{outputs[tensor], s.Slots[page]})
			}
		}
		var err error
		cpuTail, gpuTail, err = r.MapKernel(s.ID+"/scatter", "cpu", "gpu_compute", MappingKernel{Name: name, Spans: spans, Endpoints: endpoints, WriteOwner: owner, Parents: []int64{cpuTail}, StreamParent: gpuTail, SubmitNS: scatter.SubmitNS, ServiceNS: scatter.ServiceNS, SubmitServiceClass: s.ServiceClasses.ScatterSubmit, KernelServiceClass: s.ServiceClasses.ScatterKernel})
		result.MappingTasks++
		return err
	}
	firstFull, lastFull := (s.Prefix+15)/16, n/16
	for tensor := 0; tensor < 56; tensor++ {
		if lastFull > firstFull {
			var ranges [][2]int64
			for page := firstFull; page < lastFull; page++ {
				ranges = append(ranges, [2]int64{page * 16, (page + 1) * 16})
			}
			if err := scatterKernel(tensor, "index_copy_full_delta", ranges); err != nil {
				return nil, err
			}
		}
		partial := []int64{firstPage}
		if lastPage != firstPage {
			partial = append(partial, lastPage)
		}
		for _, page := range partial {
			lo, hi := max(s.Prefix, page*16), min(n, (page+1)*16)
			if lo < hi && (lo != page*16 || hi != (page+1)*16) {
				if err := scatterKernel(tensor, "copy_partial_delta", [][2]int64{{lo, hi}}); err != nil {
					return nil, err
				}
			}
		}
	}
	endPhase("scatter_delta", []int64{cpuTail, gpuTail}, scatter.ObserveNS, func() error {
		for page := firstPage; page <= lastPage; page++ {
			if err := m.PublishOwned(s.Slots[page], owner); err != nil {
				return err
			}
		}
		// Independent coordinate reference verifies every meaningful cell, including
		// old prefix cells that a malformed delta scatter might have overwritten.
		for token := int64(0); token < n; token++ {
			page := token / 16
			b := m.Buffers[s.Slots[page]]
			for tensor := 0; tensor < 56; tensor++ {
				for head := int64(0); head < 4; head++ {
					index := int64(tensor)*64 + head*16 + token%16
					want := Cell{s.TokenOrigins[token], int64(tensor*4+int(head)) * 256, true}
					if b.Cells[index] != want {
						return fmt.Errorf("paged delta lineage mismatch token=%d tensor=%d", token, tensor)
					}
					result.CellsVerified++
				}
			}
		}
		return nil
	})
	cleanup, ok := s.Point.Stages["hf_release"]
	if !ok || cleanup.WallNS < 0 {
		return nil, fmt.Errorf("missing HF release calibration")
	}
	beginPhaseID := beginPhase("hf_release")
	endPhase("hf_release", []int64{beginPhaseID}, cleanup.WallNS, func() error {
		for _, id := range outputs {
			if err := m.Free(id); err != nil {
				return err
			}
		}
		if logitsBytes > 0 {
			if err := m.Free(logitsID); err != nil {
				return err
			}
		}
		if binding != nil {
			return binding.ReleaseIndices()
		}
		return nil
	})
	residual := s.Point.ResidualNS
	if s.Point.PhaseGapsNS != nil {
		residual = s.Point.PhaseGapsNS["complete"]
	}
	result.CompletionTask = e.MustAdd(Task{Name: "python_and_measurement_bookkeeping", Operation: s.ID, Resource: "cpu", DurationNS: residual, Parents: parents, Finish: func() error {
		result.CompleteNS = e.NowNS
		result.AllocatedPeakDelta = allocatedPeak - before
		e.Emit(Record{Name: "paged_step_complete", Request: s.Request, Operation: s.ID, PrefixTokens: s.Prefix, QueryTokens: s.Query})
		return m.Check()
	}, After: func() {
		if s.Done != nil {
			s.Done(result)
		}
	}})
	return result, e.Error()
}
