package kvruntime

import "fmt"

// MappingKernel is one GPU transformation launch with many logical byte spans.
// It distinguishes an index/scatter kernel from hundreds of memcpy API calls.
type MappingKernel struct {
	Name                string
	Source, Destination string
	Spans               []Span
	Parents             []int64
	StreamParent        int64
	SubmitNS, ServiceNS int64
	Prepare             func() error
	Complete            func() error
}

// MapKernel shares the host and GPU resource queues with the rest of the DES.
// The caller supplies observed kernel-envelope costs and explicit stream/data
// predecessors. Contents change only at completion; publication is independent
// of CPU launch return. CUDA internal instructions are not numerically executed.
func (r *Runtime) MapKernel(operation, cpu, gpu string, k MappingKernel) (submit, complete int64, err error) {
	if operation == "" || cpu == "" || gpu == "" || k.SubmitNS < 0 || k.ServiceNS <= 0 || len(k.Spans) == 0 || k.Source == k.Destination {
		return 0, 0, fmt.Errorf("invalid mapping kernel")
	}
	e, m := r.Engine, r.Memory
	submit = e.MustAdd(Task{Name: k.Name + "_submit", Operation: operation, Resource: cpu, DurationNS: k.SubmitNS, Parents: k.Parents, Start: func() error {
		if k.Prepare != nil {
			if err := k.Prepare(); err != nil {
				return err
			}
		}
		a, b := m.Buffers[k.Source], m.Buffers[k.Destination]
		if a == nil || b == nil || a.Opaque || b.Opaque || a.State == "reserved" || b.State == "reserved" || b.Pins > 0 || b.Writers > 0 || b.Holds > 0 {
			return fmt.Errorf("mapping buffers unavailable")
		}
		for _, p := range k.Spans {
			if p.Bytes <= 0 || p.SourceOffset < 0 || p.DestinationOffset < 0 || p.SourceOffset > a.Bytes-p.Bytes || p.DestinationOffset > b.Bytes-p.Bytes || p.SourceOffset%m.Unit != 0 || p.DestinationOffset%m.Unit != 0 || p.Bytes%m.Unit != 0 {
				return fmt.Errorf("mapping span out of bounds")
			}
		}
		if err := m.Hold(k.Source); err != nil {
			return err
		}
		return m.BeginWrite(k.Destination)
	}})
	parents := []int64{submit}
	if k.StreamParent != 0 {
		parents = append(parents, k.StreamParent)
	}
	complete = e.MustAdd(Task{Name: k.Name, Operation: operation, Resource: gpu, DurationNS: k.ServiceNS, Parents: parents, Start: func() error {
		if err := m.Pin(k.Source); err != nil {
			return err
		}
		return m.Unhold(k.Source)
	}, Finish: func() error {
		for _, p := range k.Spans {
			if err := m.Copy(k.Source, k.Destination, p.SourceOffset, p.DestinationOffset, p.Bytes); err != nil {
				return err
			}
		}
		if err := m.Publish(k.Destination); err != nil {
			return err
		}
		if err := m.Unpin(k.Source); err != nil {
			return err
		}
		if k.Complete != nil {
			return k.Complete()
		}
		return nil
	}})
	return submit, complete, e.Error()
}

// HFScatterSpans maps [head,token,dim] into physical paged slots. One mapping
// kernel consumes the entire list; the list length is not a DMA/API count.
func HFScatterSpans(tokens int64) ([]Span, error) {
	if tokens <= 0 || tokens%16 != 0 {
		return nil, fmt.Errorf("full native pages required")
	}
	blocks := tokens / 16
	if blocks%37 == 0 {
		return nil, fmt.Errorf("non-bijective physical page permutation")
	}
	var spans []Span
	for b := int64(0); b < blocks; b++ {
		slot := 2 * ((37 * b) % blocks)
		for h := int64(0); h < 4; h++ {
			spans = append(spans, Span{SourceOffset: (h*tokens + b*16) * 256, DestinationOffset: (slot*4*16 + h*16) * 256, Bytes: 16 * 256})
		}
	}
	return spans, nil
}
