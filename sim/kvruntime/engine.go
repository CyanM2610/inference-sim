// Package kvruntime models native KV operations with nanosecond events and
// explicit resource/dependency state. Native fidelity experiments use this
// runtime; BLIS cache integration is pending. Profiles supply costs, not outcomes.
package kvruntime

import (
	"container/heap"
	"fmt"
	"sort"
)

type Record struct {
	TimeNS              int64   `json:"time_ns"`
	Name                string  `json:"name"`
	Task                int64   `json:"task,omitempty"`
	Operation           string  `json:"operation,omitempty"`
	Resource            string  `json:"resource,omitempty"`
	DurationNS          int64   `json:"duration_ns,omitempty"`
	Bytes               int64   `json:"bytes,omitempty"`
	Source              string  `json:"source,omitempty"`
	Destination         string  `json:"destination,omitempty"`
	SourceOffset        int64   `json:"source_offset,omitempty"`
	DestinationOffset   int64   `json:"destination_offset,omitempty"`
	Reason              string  `json:"reason,omitempty"`
	Parents             []int64 `json:"parents,omitempty"`
	Origin              uint64  `json:"origin,omitempty"`
	OriginOffset        int64   `json:"origin_offset,omitempty"`
	ContentSHA256       string  `json:"content_sha256,omitempty"`
	PhysicalLayerStride int64   `json:"physical_layer_stride_bytes,omitempty"`
	PrefixTokens        int64   `json:"prefix_tokens,omitempty"`
	QueryTokens         int64   `json:"query_tokens,omitempty"`
	PayloadKey          string  `json:"payload_key,omitempty"`
}
type Task struct {
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
	name  string
	queue []*Task
	busy  bool
}
type event struct {
	at, serial int64
	run        func()
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
	if at < e.NowNS {
		return fmt.Errorf("event in past: %d < %d", at, e.NowNS)
	}
	e.serial++
	heap.Push(&e.queue, event{at, e.serial, run})
	return nil
}
func (e *Engine) NextNS() (int64, bool) {
	if len(e.queue) == 0 {
		return 0, false
	}
	return e.queue[0].at, true
}
func (e *Engine) Add(t Task) (int64, error) {
	if t.DurationNS < 0 {
		return 0, fmt.Errorf("negative duration for %s", t.Name)
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
	e.tasks[t.ID] = &t
	e.unfinished++
	for _, id := range t.Parents {
		p := e.tasks[id]
		if p.state != "complete" {
			t.pending++
			p.children = append(p.children, t.ID)
		}
	}
	e.Emit(Record{Name: "task_created", Task: t.ID, Operation: t.Operation, Resource: t.Resource, Reason: t.Name, Parents: t.Parents})
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
		t := r.queue[0]
		if t.Ready != nil && !t.Ready() {
			if !t.blocked {
				e.Emit(Record{Name: "resource_blocked", Task: t.ID, Operation: t.Operation, Resource: name, Reason: t.Name})
				t.blocked = true
			}
			continue
		}
		r.queue = r.queue[1:]
		r.busy = true
		t.state = "running"
		if t.Start != nil {
			if err := t.Start(); err != nil {
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
		e.Emit(Record{Name: "task_start", Task: t.ID, Operation: t.Operation, Resource: name, DurationNS: t.DurationNS, Reason: t.Name, Parents: t.Parents})
		err := e.At(e.NowNS+t.DurationNS, func() {
			if t.Finish != nil {
				if err := t.Finish(); err != nil {
					e.Fail(fmt.Errorf("%s finish: %w", t.Name, err))
					return
				}
			}
			t.state = "complete"
			r.busy = false
			e.unfinished--
			e.Emit(Record{Name: "task_end", Task: t.ID, Operation: t.Operation, Resource: name, Reason: t.Name})
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
		})
		if err != nil {
			e.Fail(err)
			return
		}
	}
}
func (e *Engine) RunUntil(until int64) error {
	if until < e.NowNS {
		return fmt.Errorf("cannot reverse time")
	}
	for len(e.queue) > 0 && e.queue[0].at <= until && e.err == nil {
		x := heap.Pop(&e.queue).(event)
		e.NowNS = x.at
		x.run()
	}
	return e.err
}
func (e *Engine) Run() error {
	for len(e.queue) > 0 && e.err == nil {
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
func (e *Engine) Pending() int { return e.unfinished + len(e.queue) }
