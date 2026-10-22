package sim

import "fmt"

// NativeStage describes predicted service on one exclusive resource. Stream
// ordering is separate from shared-resource arbitration. After names must refer
// to earlier stages in the same program; no native absolute timestamps enter it.
type NativeStage struct {
	Name       string   `json:"name"`
	Resource   string   `json:"resource,omitempty"`
	Stream     string   `json:"stream,omitempty"`
	DurationUS int64    `json:"duration_us"`
	After      []string `json:"after,omitempty"`
	Delay      bool     `json:"delay,omitempty"`
}

// NativeExecutionPlan is supplied by the trusted execution model, not a policy.
// Milestones describe native protocol boundaries, not interchangeable costs.
type NativeExecutionPlan struct {
	Stages        []NativeStage `json:"stages"`
	KVAccess      string        `json:"kv_access"`
	Computed      string        `json:"computed"`
	ProducerReady string        `json:"producer_ready"`
	PostForward   string        `json:"post_forward"`
	Poll          string        `json:"poll"`
	HostReturned  string        `json:"host_returned"`
	OutputReady   string        `json:"output_ready"`
}

type NativePhaseCost interface {
	NativeHostCost
	ExecutionResources() []string
	CopyResource() string
	ExecutionPlan([]NativeBatchShape) NativeExecutionPlan
}

// SubmitService lets the native KV adapter use the same copy resource as output
// D2H. Duration is a service prediction, never an observed completion timestamp.
func (e *NativeExecutor) SubmitService(now int64, name, resource, stream string, duration int64, producer *NativeFence, begin, done func(int64)) {
	if _, ok := e.busy[resource]; !ok || resource == "" || name == "" || stream == "" || duration <= 0 || producer == nil {
		panic("invalid native shared service")
	}
	j := &nativeStageRun{spec: NativeStage{Name: name, Resource: resource, Stream: stream, DurationUS: duration}, fence: &NativeFence{}, dependencies: []*NativeFence{producer}, begin: begin, end: done}
	if previous := e.streams[stream]; previous != nil {
		j.dependencies = append(j.dependencies, previous)
	}
	e.streams[stream] = j.fence
	e.jobs = append(e.jobs, j)
	e.pump(now)
}

// NativeFence is a simulation completion dependency. Registering an observer
// does not add service or change the native resource order.
type NativeFence struct {
	ready   bool
	at      int64
	waiters []func(int64)
	batch   int64
	phase   string
}

func (f *NativeFence) Ready() bool { return f.ready }
func (f *NativeFence) WhenReady(done func(int64)) {
	if f.ready {
		done(f.at)
	} else {
		f.waiters = append(f.waiters, done)
	}
}
func (f *NativeFence) finish(at int64) {
	if f.ready {
		panic("native fence completed twice")
	}
	f.ready, f.at = true, at
	callbacks := f.waiters
	f.waiters = nil
	for _, done := range callbacks {
		done(at)
	}
}

type nativeStageRun struct {
	spec         NativeStage
	batch        *nativeQueuedBatch
	dependencies []*NativeFence
	started      bool
	fence        *NativeFence
	begin, end   func(int64)
}

// NativeExecutor owns stream FIFO, shared-resource exclusion and event wakeups.
// Programs can branch across streams while waiting on explicit producer fences.
type NativeExecutor struct {
	n       *nativeAsyncRuntime
	busy    map[string]bool
	streams map[string]*NativeFence
	jobs    []*nativeStageRun
	pumping bool
}

func newNativeExecutor(n *nativeAsyncRuntime, resources []string) (*NativeExecutor, error) {
	e := &NativeExecutor{n: n, busy: map[string]bool{}, streams: map[string]*NativeFence{}}
	for _, id := range resources {
		if _, exists := e.busy[id]; id == "" || exists {
			return nil, fmt.Errorf("invalid native execution resource %q", id)
		}
		e.busy[id] = false
	}
	return e, nil
}

func validateNativeStages(stages []NativeStage, resources map[string]bool) (map[string]map[string]bool, error) {
	ancestors := map[string]map[string]bool{}
	streams := map[string]string{}
	for _, s := range stages {
		if s.Name == "" || ancestors[s.Name] != nil || s.DurationUS < 0 || s.Resource == "" && s.DurationUS != 0 && !s.Delay || s.Delay && s.Resource != "" {
			return nil, fmt.Errorf("invalid native execution stage %q", s.Name)
		}
		if s.Resource != "" {
			if _, ok := resources[s.Resource]; !ok {
				return nil, fmt.Errorf("unknown native resource %q", s.Resource)
			}
		}
		parents := append([]string(nil), s.After...)
		if previous := streams[s.Stream]; s.Stream != "" && previous != "" {
			parents = append(parents, previous)
		}
		ancestors[s.Name] = map[string]bool{s.Name: true}
		for _, dep := range parents {
			if dep == s.Name || ancestors[dep] == nil {
				return nil, fmt.Errorf("native stages must be topologically ordered: %s -> %s", dep, s.Name)
			}
			for name := range ancestors[dep] {
				ancestors[s.Name][name] = true
			}
		}
		if s.Stream != "" {
			streams[s.Stream] = s.Name
		}
	}
	return ancestors, nil
}

func validateNativePlan(p NativeExecutionPlan, resources map[string]bool) error {
	ancestors, err := validateNativeStages(p.Stages, resources)
	if err != nil {
		return err
	}
	for _, name := range []string{p.KVAccess, p.Computed, p.ProducerReady, p.PostForward, p.Poll, p.HostReturned, p.OutputReady} {
		if ancestors[name] == nil {
			return fmt.Errorf("missing native milestone %q", name)
		}
	}
	for _, edge := range [][2]string{{p.KVAccess, p.Computed}, {p.KVAccess, p.PostForward}, {p.Computed, p.OutputReady}, {p.Computed, p.ProducerReady}, {p.PostForward, p.Poll}, {p.Poll, p.HostReturned}} {
		if !ancestors[edge[1]][edge[0]] {
			return fmt.Errorf("missing native dependency %s -> %s", edge[0], edge[1])
		}
	}
	return nil
}

func (e *NativeExecutor) HasTransferModel() bool { _, ok := e.n.cost.(NativeTransferCost); return ok }

// SubmitTransfer keeps modeled service parallel to physical CUDA work. Stream
// queues are shared with compute/output programs; a model delay holds no GPU.
func (e *NativeExecutor) SubmitTransfer(now, transaction int64, shape NativeTransferShape, producer *NativeFence, begin, gpuDone, done func(int64)) *NativeFence {
	model, ok := e.n.cost.(NativeTransferCost)
	if !ok || producer == nil || len(shape.Objects) == 0 {
		panic("native transfer geometry/model missing")
	}
	p := model.TransferPlan(shape)
	return e.submitTransferPlan(now, transaction, shape, p, nil, producer, begin, gpuDone, nil, done)
}

func (e *NativeExecutor) submitTransferPlan(now, transaction int64, shape NativeTransferShape, p NativeTransferPlan, failure *NativeTransferFailure, producer *NativeFence, begin, gpuDone, failed, done func(int64)) *NativeFence {
	ancestors, err := validateNativeStages(p.Stages, e.busy)
	if err != nil {
		panic(err)
	}
	for _, name := range []string{p.ProducerWait, p.GPUComplete, p.ModelComplete, p.Published} {
		if ancestors[name] == nil {
			panic("missing native transfer milestone")
		}
	}
	if !ancestors[p.GPUComplete][p.ProducerWait] || !ancestors[p.Published][p.GPUComplete] || !ancestors[p.Published][p.ModelComplete] {
		panic("native transfer publication must join CUDA and model")
	}
	if failure != nil {
		if p.CostScope == "" || ancestors[p.FailureObserved] == nil || !ancestors[p.Published][p.FailureObserved] || failed == nil {
			panic("failed transfer requires an explicit failure handler and terminal dependency")
		}
		for _, stage := range p.Stages {
			if stage.Name == p.FailureObserved && (stage.Resource == "" || stage.Delay || stage.DurationUS <= 0) {
				panic("failure handling requires positive host service")
			}
		}
	} else if p.FailureObserved != "" {
		panic("successful transfer cannot silently execute a failure plan")
	}
	prefix := fmt.Sprintf("kv_%d/", transaction)
	fences := map[string]*NativeFence{}
	for _, spec := range p.Stages {
		fences[spec.Name] = &NativeFence{}
	}
	for _, spec := range p.Stages {
		j := &nativeStageRun{spec: spec, fence: fences[spec.Name]}
		j.spec.Name = prefix + spec.Name
		for _, dep := range spec.After {
			j.dependencies = append(j.dependencies, fences[dep])
		}
		if spec.Name == p.ProducerWait {
			j.dependencies = append(j.dependencies, producer)
			j.begin = begin
		}
		if spec.Name == p.GPUComplete {
			j.end = gpuDone
		}
		if failure != nil && spec.Name == p.FailureObserved {
			previous := j.end
			j.end = func(at int64) {
				if previous != nil {
					previous(at)
				}
				failed(at)
			}
		}
		if spec.Name == p.Published {
			previous := j.end
			j.end = func(at int64) {
				if previous != nil {
					previous(at)
				}
				done(at)
			}
		}
		if spec.Stream != "" {
			if previous := e.streams[spec.Stream]; previous != nil {
				j.dependencies = append(j.dependencies, previous)
			}
			e.streams[spec.Stream] = j.fence
		}
		e.jobs = append(e.jobs, j)
	}
	record := e.n.event(now, "native_transfer_plan", nil, nil)
	record.Transaction = transaction
	record.Transfer = &p
	record.TransferShape = &shape
	record.TransferFailure = failure
	record.ProducerBatch, record.ProducerPhase = producer.batch, producer.phase
	e.pump(now)
	return fences[p.Published]
}

func (e *NativeExecutor) submit(now int64, b *nativeQueuedBatch, p NativeExecutionPlan, callbacks map[string]func(int64), fences map[string]*NativeFence, access []NativeKVAccessDependency) {
	if err := validateNativePlan(p, e.busy); err != nil {
		panic(err)
	}
	for _, spec := range p.Stages {
		if fences[spec.Name] == nil {
			fences[spec.Name] = &NativeFence{}
		}
		fences[spec.Name].batch, fences[spec.Name].phase = b.id, spec.Name
	}
	for _, spec := range p.Stages {
		j := &nativeStageRun{spec: spec, batch: b, fence: fences[spec.Name], end: callbacks[spec.Name]}
		if spec.Name == p.KVAccess {
			for _, dep := range access {
				if dep.Fence == nil || dep.Transaction <= 0 || len(dep.HBMBlocks) == 0 {
					panic("invalid native physical page dependency")
				}
				j.dependencies = append(j.dependencies, dep.Fence)
			}
		}
		for _, dep := range spec.After {
			j.dependencies = append(j.dependencies, fences[dep])
		}
		if spec.Stream != "" {
			if previous := e.streams[spec.Stream]; previous != nil {
				j.dependencies = append(j.dependencies, previous)
			}
			e.streams[spec.Stream] = j.fence
		}
		e.jobs = append(e.jobs, j)
	}
	e.pump(now)
}

func (e *NativeExecutor) pump(now int64) {
	if e.pumping {
		return
	}
	e.pumping = true
	defer func() { e.pumping = false }()
	for _, j := range e.jobs {
		if j.started || j.spec.Resource != "" && e.busy[j.spec.Resource] {
			continue
		}
		ready := true
		for _, f := range j.dependencies {
			if !f.ready {
				ready = false
				break
			}
		}
		if !ready {
			continue
		}
		end := nativeAfter(now, j.spec.DurationUS, false)
		j.started = true
		if j.spec.Resource != "" {
			e.busy[j.spec.Resource] = true
		}
		startEvent := e.n.event(now, "native_phase_start", j.batch, func() {
			if j.begin != nil {
				j.begin(now)
			}
		})
		startEvent.Phase, startEvent.Resource, startEvent.DurationUS = j.spec.Name, j.spec.Resource, j.spec.DurationUS
		endEvent := e.n.event(end, "native_phase_end", j.batch, func() {
			if j.spec.Resource != "" {
				e.busy[j.spec.Resource] = false
			}
			if j.end != nil {
				j.end(end)
			}
			j.fence.finish(end)
			var keep []*nativeStageRun
			for _, item := range e.jobs {
				if !item.fence.ready {
					keep = append(keep, item)
				}
			}
			e.jobs = keep
			e.pump(end)
		})
		endEvent.Phase, endEvent.Resource, endEvent.DurationUS = j.spec.Name, j.spec.Resource, j.spec.DurationUS
	}
}
