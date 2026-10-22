package kvruntime

import "fmt"

// InputPipelineProfile describes independently measured input primitives.
// Costs remain separate from the request's token readiness and physical bytes.
type InputPipelineProfile struct {
	GPUIndexBytes         int64             `json:"gpu_index_bytes"`
	HostIndexBytes        int64             `json:"host_index_bytes"`
	GPUAllocationQuantum  int64             `json:"gpu_allocation_quantum"`
	ReplacedGPUInputBytes int64             `json:"replaced_gpu_input_bytes"`
	Index                 []InputIndexPoint `json:"index"`
	NextInput             TransferCost      `json:"next_input"`
	Evidence              string            `json:"evidence"`
}
type InputIndexPoint struct {
	Count     int64        `json:"count"`
	PrepareNS int64        `json:"prepare_ns"`
	Cost      TransferCost `json:"cost"`
}

func (p InputPipelineProfile) Validate() error {
	if p.Evidence == "" || p.HostIndexBytes <= 0 || p.HostIndexBytes%8 != 0 || p.GPUIndexBytes < p.HostIndexBytes || p.GPUIndexBytes%8 != 0 || p.GPUAllocationQuantum <= 0 || p.GPUAllocationQuantum%8 != 0 || p.ReplacedGPUInputBytes <= 0 || len(p.Index) == 0 {
		return fmt.Errorf("invalid input pipeline storage or evidence")
	}
	seen := map[int64]bool{}
	for _, x := range p.Index {
		if x.Count <= 0 || x.Count > p.HostIndexBytes/8 || seen[x.Count] || x.PrepareNS <= 0 || !x.Cost.HoldCPUUntilDMA {
			return fmt.Errorf("invalid/duplicate index input shape")
		}
		seen[x.Count] = true
		if err := x.Cost.Validate(); err != nil {
			return err
		}
	}
	if !p.NextInput.HoldCPUUntilDMA {
		return fmt.Errorf("next-input copy requires actual host completion wait")
	}
	return p.NextInput.Validate()
}

type InputRequest struct {
	ID             string
	Prompt, Output []uint64
}
type inputRequestState struct {
	InputRequest
	buffer   string
	produced int64
}

// RequestInputs owns persistent input tensors and the single worker's index
// tensors. Output-budget zeros are physical data, not generated-token readiness.
type RequestInputs struct {
	// Optional class for index preparation and host-side H2D work; DMA waits
	// remain dependent on actual device completion.
	IndexHostServiceClass         string
	r                             *Runtime
	profile                       InputPipelineProfile
	requests                      map[string]*inputRequestState
	pages                         map[string]uint64
	indexGPU, indexHost, hostPool string
}

func NewRequestInputs(r *Runtime, p InputPipelineProfile, gpuPool, hostPool string, pages []string, requests []InputRequest) (*RequestInputs, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if r == nil || gpuPool == "" || hostPool == "" || len(pages) == 0 || len(requests) == 0 {
		return nil, fmt.Errorf("missing request input bindings")
	}
	i := &RequestInputs{r: r, profile: p, requests: map[string]*inputRequestState{}, pages: map[string]uint64{}, indexGPU: "native/input_indices_gpu", indexHost: "native/input_indices_host", hostPool: hostPool}
	for n, id := range pages {
		if id == "" {
			return nil, fmt.Errorf("empty input page binding")
		}
		if _, exists := i.pages[id]; exists {
			return nil, fmt.Errorf("duplicate input page binding")
		}
		i.pages[id] = uint64(n)
	}
	allocate := func(id, pool, kind string, bytes int64) error {
		if _, err := r.Memory.ReserveGranular(id, pool, kind, bytes, 8); err != nil {
			return err
		}
		if err := r.Memory.Allocate(id); err != nil {
			return err
		}
		return nil // torch.empty index storage and allocator padding are undefined.
	}
	if err := allocate(i.indexGPU, gpuPool, "warm_input_index_gpu", p.GPUIndexBytes); err != nil {
		return nil, err
	}
	if err := allocate(i.indexHost, hostPool, "warm_pinned_input_indices", p.HostIndexBytes); err != nil {
		return nil, err
	}
	for _, req := range requests {
		if req.ID == "" || i.requests[req.ID] != nil || len(req.Prompt) == 0 || len(req.Output) == 0 {
			return nil, fmt.Errorf("invalid/duplicate input request")
		}
		x := &inputRequestState{InputRequest: InputRequest{ID: req.ID, Prompt: append([]uint64(nil), req.Prompt...), Output: append([]uint64(nil), req.Output...)}, buffer: "native/request_input/" + req.ID}
		i.requests[req.ID] = x
		bytes := int64(len(req.Prompt)+len(req.Output)) * 8
		bytes = (bytes + p.GPUAllocationQuantum - 1) / p.GPUAllocationQuantum * p.GPUAllocationQuantum
		if err := allocate(x.buffer, gpuPool, "warm_request_input_and_output_budget", bytes); err != nil {
			return nil, err
		}
		if err := r.Memory.BeginWrite(x.buffer); err != nil {
			return nil, err
		}
		// Initialize the logical prompt/budget only. The rounded allocator tail
		// is reserved physical storage, not a tensor containing known zero tokens.
		for n := int64(0); n < int64(len(req.Prompt)+len(req.Output)); n++ {
			value := uint64(0)
			if n < int64(len(req.Prompt)) {
				value = req.Prompt[n]
			}
			if err := r.Memory.ComputeRange(x.buffer, n*8, 8, value, 0); err != nil {
				return nil, err
			}
		}
		if err := r.Memory.Publish(x.buffer); err != nil {
			return nil, err
		}
		r.Engine.Emit(Record{Name: "request_input_initialized", Request: req.ID, Destination: x.buffer, PrefixTokens: int64(len(req.Prompt)), QueryTokens: int64(len(req.Output)), Bytes: bytes, Reason: "warm_gpu_prompt_and_zero_output_budget; generation_readiness_is_separate"})
	}
	return i, nil
}

type InputBinding struct {
	priorProduced              int64
	owner                      *RequestInputs
	request                    *inputRequestState
	operation                  string
	prefix, query              int64
	values                     []uint64
	queryPinned, indicesPinned bool
}

func (i *RequestInputs) Bind(operation, request string, prefix, query int64, pages []string, parents []int64) (*InputBinding, int64, error) {
	x := i.requests[request]
	if x == nil || operation == "" || prefix < 0 || query <= 0 || prefix+query <= prefix || prefix+query > int64(len(x.Prompt)+len(x.Output)-1) {
		return nil, 0, fmt.Errorf("invalid request input range")
	}
	var point *InputIndexPoint
	for n := range i.profile.Index {
		if i.profile.Index[n].Count == int64(len(pages)) {
			point = &i.profile.Index[n]
			break
		}
	}
	if point == nil {
		return nil, 0, fmt.Errorf("unmeasured input index count %d", len(pages))
	}
	b := &InputBinding{owner: i, request: x, operation: operation, prefix: prefix, query: query, priorProduced: x.produced}
	seen := map[string]bool{}
	for _, page := range pages {
		value, ok := i.pages[page]
		if !ok || seen[page] {
			return nil, 0, fmt.Errorf("unknown/duplicate physical input page")
		}
		seen[page] = true
		b.values = append(b.values, value)
	}
	e, m := i.r.Engine, i.r.Memory
	temporary := operation + "/index_list_cpu"
	prepare := e.MustAdd(Task{Name: "input_index_prepare", Operation: operation + "/index_bind", Resource: "cpu", ServiceClass: i.IndexHostServiceClass, DurationNS: point.PrepareNS, Parents: parents, Start: func() error {
		if _, err := m.ReserveGranular(temporary, i.hostPool, "temporary_cpu_index_list", int64(len(pages))*8, 8); err != nil {
			return err
		}
		if err := m.Allocate(temporary); err != nil {
			return err
		}
		return m.Fill(temporary, 0)
	}, Finish: func() error {
		for n, value := range b.values {
			if err := m.SeedRange(temporary, int64(n)*8, 8, value, 0); err != nil {
				return err
			}
		}
		if err := m.Pin(temporary); err != nil {
			return err
		}
		if err := m.BeginWrite(i.indexHost); err != nil {
			return err
		}
		if err := m.Copy(temporary, i.indexHost, 0, 0, int64(len(pages))*8); err != nil {
			return err
		}
		if err := m.Publish(i.indexHost); err != nil {
			return err
		}
		if err := m.Unpin(temporary); err != nil {
			return err
		}
		return m.Free(temporary)
	}})
	tx, err := i.r.Transfer(TransferSpec{ID: operation + "/index_h2d", Source: i.indexHost, Destination: i.indexGPU, CPU: "cpu", DMA: "copy_h2d", StreamGroup: "request_stream", Spans: []Span{{Bytes: int64(len(pages)) * 8}}, Cost: point.Cost, Parents: []int64{prepare}, HostServiceClass: i.IndexHostServiceClass})
	if err != nil {
		return nil, 0, err
	}
	ready := e.MustAdd(Task{Name: "input_indices_ready", Operation: operation + "/index_bind", Resource: "cpu", Parents: []int64{tx.CompletionTask}, Finish: func() error {
		if err := m.Pin(i.indexGPU); err != nil {
			return err
		}
		b.indicesPinned = true
		for n, value := range b.values {
			if m.Buffers[i.indexGPU].Cells[n] != (Cell{value, 0, true}) {
				return fmt.Errorf("input index H2D content mismatch")
			}
		}
		e.Emit(Record{Name: "request_indices_ready", Request: request, Operation: operation, Source: i.indexGPU, PageSlots: append([]string(nil), pages...), Bytes: int64(len(pages)) * 8})
		return nil
	}})
	return b, ready, nil
}

func (b *InputBinding) AcquireQuery() error {
	i, x := b.owner, b.request
	if b.queryPinned || b.prefix+b.query > int64(len(x.Prompt))+x.produced {
		return fmt.Errorf("request consumed input before token production")
	}
	if err := i.r.Memory.Pin(x.buffer); err != nil {
		return err
	}
	b.queryPinned = true
	for n := b.prefix; n < b.prefix+b.query; n++ {
		value := uint64(0)
		if n < int64(len(x.Prompt)) {
			value = x.Prompt[n]
		} else {
			value = x.Output[n-int64(len(x.Prompt))]
		}
		if i.r.Memory.Buffers[x.buffer].Cells[n] != (Cell{value, 0, true}) {
			return fmt.Errorf("GPU query token differs from independently supplied workload")
		}
	}
	i.r.Engine.Emit(Record{Name: "request_input_read", Request: x.ID, Operation: b.operation, Source: x.buffer, SourceOffset: b.prefix * 8, Bytes: b.query * 8, PrefixTokens: b.prefix, QueryTokens: b.query})
	return nil
}
func (b *InputBinding) ReleaseQuery() error {
	if !b.queryPinned {
		return fmt.Errorf("request query release without acquisition")
	}
	b.queryPinned = false
	return b.owner.r.Memory.Unpin(b.request.buffer)
}

// OutputIndex is negative for internal prefill observations. Generated-output
// progress survives KV eviction; it is distinct from the computed KV frontier.
func (b *InputBinding) OutputIndex() int64 {
	return b.prefix + b.query - int64(len(b.request.Prompt))
}
func (b *InputBinding) PreviouslyObservedOutput() bool {
	return b.OutputIndex() >= 0 && b.OutputIndex() < b.priorProduced
}
func (b *InputBinding) NeedsNextInput() bool {
	index := b.OutputIndex()
	return index >= 0 && index+1 < int64(len(b.request.Output))
}

// ObserveOutput creates the real D2D dependency for nonfinal visible tokens.
// Internal prefill outputs and the final output are not next-input writes.
func (b *InputBinding) ObserveOutput(source string, value uint64, parents []int64) (int64, error) {
	i, x := b.owner, b.request
	index := b.OutputIndex()
	if index < 0 {
		return 0, nil
	}
	if index >= int64(len(x.Output)) || value != x.Output[index] {
		return 0, fmt.Errorf("invalid output reference for request input")
	}
	complete := func() error {
		if index > x.produced {
			return fmt.Errorf("output input write skipped a token")
		}
		if index+1 < int64(len(x.Output)) && i.r.Memory.Buffers[x.buffer].Cells[int64(len(x.Prompt))+index] != (Cell{value, 0, true}) {
			return fmt.Errorf("next input D2D did not publish expected value")
		}
		x.produced = max(x.produced, index+1)
		i.r.Engine.Emit(Record{Name: "request_output_input_ready", Request: x.ID, Operation: b.operation, Source: source, Destination: x.buffer, DestinationOffset: (int64(len(x.Prompt)) + index) * 8, Origin: value, Bytes: 8, PrefixTokens: index, Reason: map[bool]string{true: "final_output_no_input_write", false: "actual_gpu_d2d"}[index+1 == int64(len(x.Output))]})
		return nil
	}
	e := i.r.Engine
	if index+1 == int64(len(x.Output)) {
		return e.MustAdd(Task{Name: "final_output_observed", Operation: b.operation, Resource: "cpu", Parents: parents, Finish: complete}), nil
	}
	tx, err := i.r.Transfer(TransferSpec{ID: b.operation + "/next_input_d2d", Source: source, Destination: x.buffer, CPU: "cpu", DMA: "copy_d2d", StreamGroup: "request_stream", Spans: []Span{{DestinationOffset: (int64(len(x.Prompt)) + index) * 8, Bytes: 8}}, Cost: i.profile.NextInput, Parents: parents})
	if err != nil {
		return 0, err
	}
	return e.MustAdd(Task{Name: "next_input_ready", Operation: b.operation + "/next_input_d2d", Resource: "cpu", Parents: []int64{tx.CompletionTask}, Finish: complete}), nil
}
func (b *InputBinding) ReleaseIndices() error {
	if !b.indicesPinned || b.queryPinned {
		return fmt.Errorf("invalid input binding release")
	}
	b.indicesPinned = false
	return b.owner.r.Memory.Unpin(b.owner.indexGPU)
}
