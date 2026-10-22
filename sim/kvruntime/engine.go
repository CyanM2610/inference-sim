// Package kvruntime models native KV operations with nanosecond events and
// explicit resource/dependency state. Native fidelity experiments use this
// runtime, including the BLIS cache and asynchronous batch adapters. Profiles
// supply costs, not outcomes.
package kvruntime

import (
	"container/heap"
	"fmt"
	"sort"
)

type Record struct {
	OutputIndex         *int64   `json:"output_index,omitempty"`
	Recomputed          bool     `json:"recomputed,omitempty"`
	WaitFor             []int64  `json:"wait_for,omitempty"`
	ResourceGroup       string   `json:"resource_group,omitempty"`
	CellBytes           int64    `json:"cell_bytes,omitempty"`
	ServiceClass        string   `json:"service_class,omitempty"`
	ServiceRatePPM      int64    `json:"service_rate_ppm,omitempty"`
	RemainingNS         int64    `json:"remaining_service_ns,omitempty"`
	FractionPPM         int64    `json:"completed_fraction_ppm,omitempty"`
	TimeNS              int64    `json:"time_ns"`
	Name                string   `json:"name"`
	Task                int64    `json:"task,omitempty"`
	Operation           string   `json:"operation,omitempty"`
	Resource            string   `json:"resource,omitempty"`
	DurationNS          int64    `json:"duration_ns,omitempty"`
	Bytes               int64    `json:"bytes,omitempty"`
	Source              string   `json:"source,omitempty"`
	Destination         string   `json:"destination,omitempty"`
	SourceOffset        int64    `json:"source_offset,omitempty"`
	DestinationOffset   int64    `json:"destination_offset,omitempty"`
	Reason              string   `json:"reason,omitempty"`
	Parents             []int64  `json:"parents,omitempty"`
	Origin              uint64   `json:"origin,omitempty"`
	OriginOffset        int64    `json:"origin_offset,omitempty"`
	ContentSHA256       string   `json:"content_sha256,omitempty"`
	PhysicalLayerStride int64    `json:"physical_layer_stride_bytes,omitempty"`
	PrefixTokens        int64    `json:"prefix_tokens,omitempty"`
	QueryTokens         int64    `json:"query_tokens,omitempty"`
	PayloadKey          string   `json:"payload_key,omitempty"`
	AllowUndefined      bool     `json:"allow_undefined_padding,omitempty"`
	Request             string   `json:"request,omitempty"`
	PageSlots           []string `json:"page_slots,omitempty"`
}
type Task struct {
	// WaitFor holds the resource until these tasks actually complete. Unlike
	// Parents, it does not delay resource acquisition. Used for a host thread
	// blocked in stream synchronization, including device queue wait time.
	WaitFor          []int64
	waiters          []int64
	waitEndScheduled bool
	ResourceGroup    string
	// ServiceClass opts into progress-based repricing. DurationNS is then solo
	// service work, not a promise of elapsed wall time. Empty preserves fixed
	// duration semantics, including event-dependent host return waits.
	ServiceClass              string
	service                   *serviceProgress
	ID                        int64
	Name, Operation, Resource string
	DurationNS                int64
	Duration                  func() int64
	Parents                   []int64
	// Ready represents finite driver credits or memory availability. A blocked
	// head retains its place on a serialized resource (host-thread HOL blocking).
	Ready    func() bool
	Start    func() error
	Finish   func() error
	After    func()
	pending  int
	children []int64
	state    string
	queuedAt int64
	blocked  bool
}
type resource struct {
	policy, owner string
	name          string
	queue         []*Task
	busy          bool
	active        *Task
}
type event struct {
	at, serial int64
	run        func()
	valid      func() bool
}
type events []event

func (q events) Len() int { return len(q) }
func (q events) Less(i, j int) bool {
	if q[i].at != q[j].at {
		return q[i].at < q[j].at
	}
	return q[i].serial < q[j].serial
}
func (q events) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *events) Push(x any)   { *q = append(*q, x.(event)) }
func (q *events) Pop() any     { old := *q; x := old[len(old)-1]; *q = old[:len(old)-1]; return x }

type Engine struct {
	serviceRates       map[string]int64
	actionTask         *Task
	NowNS              int64
	queue              events
	serial, taskSerial int64
	tasks              map[int64]*Task
	resources          map[string]*resource
	resourceNames      []string
	unfinished         int
	kicking            bool
	err                error
	Sink               func(Record)
}

func NewEngine(sink func(Record)) *Engine {
	return &Engine{tasks: map[int64]*Task{}, resources: map[string]*resource{}, Sink: sink}
}
func (e *Engine) Emit(r Record) {
	r.TimeNS = e.NowNS
	if r.Task == 0 && e.actionTask != nil {
		r.Task = e.actionTask.ID
		if r.Operation == "" {
			r.Operation = e.actionTask.Operation
		}
		if r.Resource == "" {
			r.Resource = e.actionTask.Resource
		}
	}
	if e.Sink != nil {
		e.Sink(r)
	}
}
func (e *Engine) Fail(err error) {
	if e.err == nil {
		e.err = err
	}
}
func (e *Engine) Error() error { return e.err }
func (e *Engine) At(at int64, run func()) error {
	return e.atIf(at, run, nil)
}
func (e *Engine) atIf(at int64, run func(), valid func() bool) error {
	if at < e.NowNS {
		return fmt.Errorf("event in past: %d < %d", at, e.NowNS)
	}
	e.serial++
	heap.Push(&e.queue, event{at: at, serial: e.serial, run: run, valid: valid})
	return nil
}
func (e *Engine) NextNS() (int64, bool) {
	e.pruneCancelled()
	if len(e.queue) == 0 {
		return 0, false
	}
	return e.queue[0].at, true
}
func (e *Engine) Add(t Task) (int64, error) {
	if t.DurationNS < 0 {
		return 0, fmt.Errorf("negative duration for %s", t.Name)
	}
	if len(t.WaitFor) > 0 && (t.DurationNS != 0 || t.Duration != nil || t.ServiceClass != "") {
		return 0, fmt.Errorf("completion wait cannot also have service duration")
	}
	waitSeen := map[int64]bool{}
	for _, id := range t.WaitFor {
		if waitSeen[id] || e.tasks[id] == nil {
			return 0, fmt.Errorf("invalid/duplicate completion target")
		}
		waitSeen[id] = true
	}
	seen := map[int64]bool{}
	for _, id := range t.Parents {
		if seen[id] || e.tasks[id] == nil {
			return 0, fmt.Errorf("invalid/duplicate parent %d", id)
		}
		seen[id] = true
	}
	e.taskSerial++
	t.ID = e.taskSerial
	t.state = "waiting"
	t.Parents = append([]int64(nil), t.Parents...)
	t.WaitFor = append([]int64(nil), t.WaitFor...)
	e.tasks[t.ID] = &t
	e.unfinished++
	for _, id := range t.WaitFor {
		e.tasks[id].waiters = append(e.tasks[id].waiters, t.ID)
	}
	for _, id := range t.Parents {
		p := e.tasks[id]
		if p.state != "complete" {
			t.pending++
			p.children = append(p.children, t.ID)
		}
	}
	e.Emit(Record{Name: "task_created", Task: t.ID, Operation: t.Operation, Resource: t.Resource, Reason: t.Name, Parents: t.Parents, ServiceClass: t.ServiceClass, ResourceGroup: t.ResourceGroup, WaitFor: t.WaitFor})
	if t.pending == 0 {
		e.ready(&t)
	}
	return t.ID, nil
}
func (e *Engine) MustAdd(t Task) int64 {
	id, err := e.Add(t)
	if err != nil {
		e.Fail(err)
	}
	return id
}
func (e *Engine) ready(t *Task) {
	t.state = "ready"
	t.queuedAt = e.NowNS
	r := e.resources[t.Resource]
	if r == nil {
		r = &resource{name: t.Resource}
		e.resources[t.Resource] = r
		e.resourceNames = append(e.resourceNames, t.Resource)
		sort.Strings(e.resourceNames)
	}
	r.queue = append(r.queue, t)
	e.Emit(Record{Name: "task_ready", Task: t.ID, Operation: t.Operation, Resource: t.Resource, Reason: t.Name})
	e.Kick()
}
func (e *Engine) Kick() {
	if e.kicking || e.err != nil {
		return
	}
	e.kicking = true
	defer func() { e.kicking = false }()
	for _, name := range e.resourceNames {
		r := e.resources[name]
		if r.busy || len(r.queue) == 0 {
			continue
		}
		selected := 0
		if r.policy == "stream_drain" && r.owner != "" {
			for i, t := range r.queue {
				if t.ResourceGroup == r.owner {
					selected = i
					break
				}
			}
		}
		t := r.queue[selected]
		if t.Ready != nil && !t.Ready() {
			if !t.blocked {
				e.Emit(Record{Name: "resource_blocked", Task: t.ID, Operation: t.Operation, Resource: name, Reason: t.Name})
				t.blocked = true
			}
			continue
		}
		if selected == 0 {
			r.queue = r.queue[1:]
		} else {
			r.queue = append(r.queue[:selected], r.queue[selected+1:]...)
		}
		r.busy = true
		r.active = t
		r.owner = t.ResourceGroup
		t.state = "running"
		if t.Start != nil {
			previous := e.actionTask
			e.actionTask = t
			err := t.Start()
			e.actionTask = previous
			if err != nil {
				e.Fail(fmt.Errorf("%s start: %w", t.Name, err))
				return
			}
		}
		if t.Duration != nil {
			t.DurationNS = t.Duration()
			if t.DurationNS < 0 {
				e.Fail(fmt.Errorf("negative dynamic duration for %s", t.Name))
				return
			}
		}
		e.Emit(Record{Name: "task_start", Task: t.ID, Operation: t.Operation, Resource: name, DurationNS: t.DurationNS, Reason: t.Name, Parents: t.Parents, ServiceClass: t.ServiceClass, ResourceGroup: t.ResourceGroup, WaitFor: t.WaitFor})
		var err error
		if len(t.WaitFor) > 0 {
			e.finishWaitWhenReady(t)
		} else if t.ServiceClass != "" {
			t.service = &serviceProgress{remainingNS: t.DurationNS, updatedAt: e.NowNS, ratePPM: e.ServiceRate(t.ServiceClass)}
			err = e.scheduleService(t)
		} else {
			err = e.At(e.NowNS+t.DurationNS, func() { e.completeTask(t) })
		}
		if err != nil {
			e.Fail(err)
			return
		}
	}
}
func (e *Engine) completeTask(t *Task) {
	if t.service != nil {
		e.advanceService(t)
		if t.service.remainingNS != 0 {
			e.Fail(fmt.Errorf("premature service completion %s", t.Name))
			return
		}
	}
	if t.Finish != nil {
		previous := e.actionTask
		e.actionTask = t
		err := t.Finish()
		e.actionTask = previous
		if err != nil {
			e.Fail(fmt.Errorf("%s finish: %w", t.Name, err))
			return
		}
	}
	t.state = "complete"
	e.resources[t.Resource].busy = false
	e.resources[t.Resource].active = nil
	e.unfinished--
	e.Emit(Record{Name: "task_end", Task: t.ID, Operation: t.Operation, Resource: t.Resource, Reason: t.Name, ServiceClass: t.ServiceClass})
	for _, id := range t.waiters {
		e.finishWaitWhenReady(e.tasks[id])
	}
	for _, id := range t.children {
		c := e.tasks[id]
		c.pending--
		if c.pending == 0 {
			e.ready(c)
		}
	}
	if t.After != nil {
		t.After()
	}
	e.Kick()
}

func (e *Engine) finishWaitWhenReady(t *Task) {
	if t.state != "running" || t.waitEndScheduled || len(t.WaitFor) == 0 {
		return
	}
	for _, id := range t.WaitFor {
		if e.tasks[id].state != "complete" {
			return
		}
	}
	t.waitEndScheduled = true
	if err := e.At(e.NowNS, func() { e.completeTask(t) }); err != nil {
		e.Fail(err)
	}
}
func (e *Engine) pruneCancelled() {
	for len(e.queue) > 0 && e.queue[0].valid != nil && !e.queue[0].valid() {
		heap.Pop(&e.queue)
	}
}
func (e *Engine) RunUntil(until int64) error {
	if until < e.NowNS {
		return fmt.Errorf("cannot reverse time")
	}
	for e.err == nil {
		e.pruneCancelled()
		if len(e.queue) == 0 || e.queue[0].at > until {
			break
		}
		x := heap.Pop(&e.queue).(event)
		e.NowNS = x.at
		x.run()
	}
	return e.err
}
func (e *Engine) Run() error {
	for e.err == nil {
		e.pruneCancelled()
		if len(e.queue) == 0 {
			break
		}
		x := heap.Pop(&e.queue).(event)
		e.NowNS = x.at
		x.run()
	}
	if e.err != nil {
		return e.err
	}
	if e.unfinished != 0 {
		return fmt.Errorf("deadlock: %d unfinished tasks without a completion event", e.unfinished)
	}
	return nil
}
func (e *Engine) Pending() int {
	n := e.unfinished
	for _, x := range e.queue {
		if x.valid == nil || x.valid() {
			n++
		}
	}
	return n
}
