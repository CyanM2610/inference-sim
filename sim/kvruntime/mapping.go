package kvruntime

import (
	"fmt"
	"sort"
)

// MappingKernel is one GPU transformation launch with many logical byte spans.
// It distinguishes an index/scatter kernel from hundreds of memcpy API calls.
type MappingKernel struct {
	SubmitServiceClass, KernelServiceClass string
	Name                                   string
	Source, Destination                    string
	Spans                                  []Span
	Endpoints                              []CopyEndpoints
	// Nonempty WriteOwner uses an already-open compound destination writer.
	// The caller publishes that writer after all owned kernels complete.
	WriteOwner          string
	AllowUndefined      bool
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
	if operation == "" || cpu == "" || gpu == "" || k.SubmitNS < 0 || k.ServiceNS <= 0 || len(k.Spans) == 0 {
		return 0, 0, fmt.Errorf("invalid mapping kernel")
	}
	e, m := r.Engine, r.Memory
	if k.StreamParent != 0 && e.tasks[k.StreamParent] == nil {
		return 0, 0, fmt.Errorf("unknown mapping stream predecessor")
	}
	endpoints := append([]CopyEndpoints(nil), k.Endpoints...)
	if len(endpoints) == 0 {
		for range k.Spans {
			endpoints = append(endpoints, CopyEndpoints{k.Source, k.Destination})
		}
	}
	if len(endpoints) != len(k.Spans) {
		return 0, 0, fmt.Errorf("mapping endpoint count mismatch")
	}
	srcSet, dstSet := map[string]bool{}, map[string]bool{}
	for _, p := range endpoints {
		if p.Source == "" || p.Destination == "" || p.Source == p.Destination {
			return 0, 0, fmt.Errorf("mapping endpoint invalid")
		}
		srcSet[p.Source] = true
		dstSet[p.Destination] = true
	}
	var sources, targets []string
	for id := range srcSet {
		if dstSet[id] {
			return 0, 0, fmt.Errorf("mapping source/target alias")
		}
		sources = append(sources, id)
	}
	for id := range dstSet {
		targets = append(targets, id)
	}
	sort.Strings(sources)
	sort.Strings(targets)
	submit = e.MustAdd(Task{Name: k.Name + "_submit", Operation: operation, Resource: cpu, ServiceClass: k.SubmitServiceClass, DurationNS: k.SubmitNS, Parents: k.Parents, Start: func() error {
		if k.Prepare != nil {
			if err := k.Prepare(); err != nil {
				return err
			}
		}
		for _, id := range sources {
			a := m.Buffers[id]
			if a == nil || a.Opaque || a.State == "reserved" {
				return fmt.Errorf("mapping source unavailable")
			}
		}
		for _, id := range targets {
			b := m.Buffers[id]
			if b == nil || b.Opaque || b.State == "reserved" || b.Pins > 0 {
				return fmt.Errorf("mapping target unavailable")
			}
			if k.WriteOwner != "" {
				if b.Writers != 1 || b.WriterOwner != k.WriteOwner {
					return fmt.Errorf("mapping destination writer ownership mismatch")
				}
			} else if b.Writers > 0 || b.Holds > 0 {
				return fmt.Errorf("mapping target busy")
			}
		}
		for i, p := range k.Spans {
			a, b := m.Buffers[endpoints[i].Source], m.Buffers[endpoints[i].Destination]
			if a.CellBytes != b.CellBytes || p.Bytes <= 0 || p.SourceOffset < 0 || p.DestinationOffset < 0 || p.SourceOffset > a.Bytes-p.Bytes || p.DestinationOffset > b.Bytes-p.Bytes || p.SourceOffset%a.CellBytes != 0 || p.DestinationOffset%b.CellBytes != 0 || p.Bytes%a.CellBytes != 0 {
				return fmt.Errorf("mapping span out of bounds")
			}
		}
		for _, id := range sources {
			if err := m.Hold(id); err != nil {
				return err
			}
		}
		if k.WriteOwner == "" {
			for _, id := range targets {
				if err := m.BeginWrite(id); err != nil {
					return err
				}
			}
		}
		return nil
	}})
	parents := []int64{submit}
	if k.StreamParent != 0 {
		parents = append(parents, k.StreamParent)
	}
	complete = e.MustAdd(Task{Name: k.Name, Operation: operation, Resource: gpu, ServiceClass: k.KernelServiceClass, DurationNS: k.ServiceNS, Parents: parents, Start: func() error {
		for _, id := range sources {
			if err := m.Pin(id); err != nil {
				return err
			}
			if err := m.Unhold(id); err != nil {
				return err
			}
		}
		return nil
	}, Finish: func() error {
		for i, p := range k.Spans {
			copy := m.Copy
			if k.AllowUndefined {
				copy = m.CopyPadding
			}
			if err := copy(endpoints[i].Source, endpoints[i].Destination, p.SourceOffset, p.DestinationOffset, p.Bytes); err != nil {
				return err
			}
		}
		if k.WriteOwner == "" {
			for _, id := range targets {
				if err := m.Publish(id); err != nil {
					return err
				}
			}
		}
		for _, id := range sources {
			if err := m.Unpin(id); err != nil {
				return err
			}
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
