package kvruntime

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

type ControlFeatures map[string]float64
type ControlCost struct {
	Coefficients map[string]float64    `json:"coefficients_ns"`
	Bounds       map[string][2]float64 `json:"feature_bounds"`
}
type ControlProfile struct {
	Version int                    `json:"schema_version"`
	Models  map[string]ControlCost `json:"models"`
	Sources map[string]string      `json:"source_sha256"`
}

func LoadControlProfile(path string) (*ControlProfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p ControlProfile
	if err = json.Unmarshal(data, &p); err != nil {
		return nil, err
	}
	if p.Version != 1 || len(p.Sources) == 0 {
		return nil, fmt.Errorf("control profile lacks version or provenance")
	}
	for name, cost := range p.Models {
		if name == "" || len(cost.Coefficients) == 0 {
			return nil, fmt.Errorf("empty control stage")
		}
		for feature, value := range cost.Coefficients {
			if feature == "" || value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("invalid control coefficient")
			}
		}
		for feature, bounds := range cost.Bounds {
			if feature == "" || bounds[0] < 0 || bounds[1] < bounds[0] || math.IsNaN(bounds[0]) || math.IsNaN(bounds[1]) || math.IsInf(bounds[0], 0) || math.IsInf(bounds[1], 0) {
				return nil, fmt.Errorf("invalid control feature bounds")
			}
		}
	}
	return &p, nil
}
func (p *ControlProfile) Predict(name string, features ControlFeatures) (int64, error) {
	cost, ok := p.Models[name]
	if !ok {
		return 0, fmt.Errorf("unmeasured control stage %s", name)
	}
	var ns float64
	keys := make([]string, 0, len(cost.Coefficients))
	for key := range cost.Coefficients {
		keys = append(keys, key)
	}
	sort.Strings(keys) // floating-point summation must not depend on Go map iteration.
	for _, key := range keys {
		coefficient := cost.Coefficients[key]
		value, exists := features[key]
		if key == "base" {
			value, exists = 1, true
		}
		if !exists || value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, fmt.Errorf("missing/invalid control feature %s for %s", key, name)
		}
		if bounds, ok := cost.Bounds[key]; ok && (value < bounds[0]-1e-9 || value > bounds[1]+1e-9) {
			return 0, fmt.Errorf("unmeasured control feature %s=%g for %s", key, value, name)
		}
		ns += coefficient * value
	}
	if ns < 0 || math.IsNaN(ns) || math.IsInf(ns, 0) || ns >= float64(math.MaxInt64) {
		return 0, fmt.Errorf("control duration overflow")
	}
	return int64(math.Ceil(ns)), nil
}

// ControlCommand starts a physical DAG and returns its actual completion task.
// Ready is a notification to an inbox, not permission to publish cache state.
type ControlCommand struct {
	ID     string
	Kind   string // copy or batch: one command per worker in the current protocol.
	Start  func() (int64, error)
	Ready  func(int64)
	Queued func(int64) // completion committed to the queue, before driver receive
}
type ControlDecision struct {
	PolicyFeatures ControlFeatures
	ReplyFeatures  ControlFeatures
	Commands       []ControlCommand
}
type ControlRoundSpec struct {
	DriverProtocol bool // measured queue commit/put tail and dispatch-finish boundaries
	ID             string
	Profile        *ControlProfile
	InputFeatures  ControlFeatures
	Parents        []int64
	Decide         func() (*ControlDecision, error)
	Released       func(int64) // driver finished enqueuing; hardware may still be active.
}
type ControlRoundResult struct {
	RequestAvailableTask, ReplyAvailableTask, DriverWaitTask int64
	DecisionNS, ReplyDecodedNS, ReleasedNS                   int64
	WorkerStartedNS                                          map[string]int64
	NotificationsNS                                          map[string]int64
}

// ControlRound separates CPU occupancy, pipe availability, worker wakeup and
// physical completion. In particular, the driver owns its thread while waiting
// for the actual reply, and a callback never substitutes a predicted DMA finish.
func (r *Runtime) ControlRound(s ControlRoundSpec) (*ControlRoundResult, error) {
	if s.ID == "" || s.Profile == nil || s.Decide == nil {
		return nil, fmt.Errorf("invalid control round")
	}
	e := r.Engine
	out := &ControlRoundResult{WorkerStartedNS: map[string]int64{}, NotificationsNS: map[string]int64{}}
	var decision *ControlDecision
	input := func() ControlFeatures { return s.InputFeatures }
	reply := func() ControlFeatures { return decision.ReplyFeatures }
	policy := func() ControlFeatures { return decision.PolicyFeatures }
	stage := func(name, resource string, parents []int64, features func() ControlFeatures, start, finish func() error, after func()) int64 {
		return e.MustAdd(Task{Name: name, Operation: s.ID, Resource: resource, Parents: parents, Start: start, Duration: func() int64 {
			ns, err := s.Profile.Predict(name, features())
			if err != nil {
				e.Fail(err)
				return 0
			}
			return ns
		}, Finish: finish, After: after})
	}
	send := stage("python_send", "protocol_driver", s.Parents, input, nil, nil, nil)
	request := stage("request_available", "request_pipe", s.Parents, input, nil, nil, nil)
	out.RequestAvailableTask = request
	decode := stage("go_decode", "protocol_controller", []int64{request}, input, nil, nil, nil)
	poll := stage("go_poll", "protocol_controller", []int64{decode}, policy, func() error {
		var err error
		decision, err = s.Decide()
		if err != nil {
			return err
		}
		if decision == nil {
			return fmt.Errorf("controller returned no decision")
		}
		seen := map[string]bool{}
		ids := map[string]bool{}
		for _, command := range decision.Commands {
			if command.ID == "" || command.Start == nil || (command.Kind != "copy" && command.Kind != "batch") || seen[command.Kind] || ids[command.ID] || (command.Ready != nil && command.Queued != nil) || (s.DriverProtocol && command.Queued == nil) {
				return fmt.Errorf("control round requires at most one command per native worker")
			}
			seen[command.Kind] = true
			ids[command.ID] = true
		}
		out.DecisionNS = e.NowNS
		return nil
	}, nil, nil)
	encoded := stage("go_encode", "protocol_controller", []int64{poll}, reply, nil, nil, nil)
	stage("go_write", "protocol_controller", []int64{encoded}, reply, nil, nil, nil)
	available := stage("reply_available", "reply_pipe", []int64{encoded}, reply, nil, nil, nil)
	out.ReplyAvailableTask = available
	wait := e.MustAdd(Task{Name: "driver_readline_wait", Operation: s.ID, Resource: "protocol_driver", Parents: []int64{send}, WaitFor: []int64{available}})
	out.DriverWaitTask = wait
	stage("python_decode", "protocol_driver", []int64{wait}, reply, nil, func() error { out.ReplyDecodedNS = e.NowNS; return nil }, func() {
		var previous int64
		for _, command := range decision.Commands {
			command := command
			features := func() ControlFeatures { return nil }
			var parents []int64
			if previous != 0 {
				parents = []int64{previous}
			}
			dispatch := stage(command.Kind+"_dispatch", "protocol_driver", parents, features, nil, nil, nil)
			previous = stage(command.Kind+"_queue_put", "protocol_driver", []int64{dispatch}, features, func() error {
				worker := "copy_thread"
				if command.Kind == "batch" {
					worker = "cpu"
				}
				wake := stage(command.Kind+"_wake", "worker_wakeup/"+command.ID, nil, features, nil, nil, nil)
				stage(command.Kind+"_entry", worker, []int64{wake}, features, nil, nil, func() {
					out.WorkerStartedNS[command.ID] = e.NowNS
					body, err := command.Start()
					if err != nil {
						e.Fail(err)
						return
					}
					if e.tasks[body] == nil {
						e.Fail(fmt.Errorf("worker returned unknown physical completion task"))
						return
					}
					exit := stage(command.Kind+"_exit", worker, []int64{body}, features, nil, nil, nil)
					if s.DriverProtocol {
						commit := stage(command.Kind+"_enqueue", worker, []int64{exit}, features, nil, func() error {
							out.NotificationsNS[command.ID] = e.NowNS
							command.Queued(e.NowNS)
							return nil
						}, nil)
						stage(command.Kind+"_put_tail", worker, []int64{commit}, features, nil, nil, nil)
						return
					}
					stage(command.Kind+"_notify", "completion_queue/"+command.ID, []int64{exit}, features, nil, nil, func() {
						out.NotificationsNS[command.ID] = e.NowNS
						if command.Ready != nil {
							command.Ready(e.NowNS)
						}
					})
				})
				return nil
			}, nil, nil)
		}
		var parents []int64
		if previous != 0 {
			parents = []int64{previous}
		}
		if s.DriverProtocol {
			parents = []int64{stage("driver_dispatch_finish", "protocol_driver", parents, reply, nil, nil, nil)}
		}
		e.MustAdd(Task{Name: "driver_round_released", Operation: s.ID, Resource: "protocol_driver", Parents: parents, Finish: func() error { out.ReleasedNS = e.NowNS; return nil }, After: func() {
			if s.Released != nil {
				s.Released(e.NowNS)
			}
		}})
	})
	return out, e.Error()
}
