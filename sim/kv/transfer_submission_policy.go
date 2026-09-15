package kv

import (
	"fmt"
	"math"
	"reflect"
	"sort"
)

// TransferSubmissionCandidate describes an already staged job. It contains no
// predicted physical completion, future arrival, or mutable cache ownership.
type TransferSubmissionCandidate struct {
	Transaction   int64  `json:"transaction"`
	Request       string `json:"request,omitempty"`
	Reason        string `json:"reason,omitempty"`
	Source        string `json:"source"`
	Destination   string `json:"destination"`
	Blocks        int64  `json:"blocks"`
	Bytes         int64  `json:"bytes"`
	DeadlineUS    int64  `json:"deadline_us"`
	DeadlineKnown bool   `json:"deadline_known"`
}

type TransferSubmissionContext struct {
	NowUS      int64                         `json:"now_us"`
	Instance   string                        `json:"instance"`
	Direction  string                        `json:"direction"`
	Candidates []TransferSubmissionCandidate `json:"candidates"`
}

// Order is called once per nonempty submission phase, never by a comparator or
// queue estimator. Return a complete permutation of the supplied transaction IDs.
// The runtime validates before consuming staged jobs or scheduling submissions.
type TransferSubmissionPolicy interface {
	Order(TransferSubmissionContext) []int64
}

// BuiltinTransferSubmissionPolicy orders host submissions, not wire arbitration.
// FIFO retains staging order; deadline uses known deadlines first, with stable ties.
type BuiltinTransferSubmissionPolicy struct{ name string }

func NewTransferSubmissionPolicy(name string) (TransferSubmissionPolicy, error) {
	if name != "fifo" && name != "deadline" {
		return nil, fmt.Errorf("unknown transfer submission policy %q", name)
	}
	return BuiltinTransferSubmissionPolicy{name: name}, nil
}

func (p BuiltinTransferSubmissionPolicy) Order(c TransferSubmissionContext) []int64 {
	candidates := append([]TransferSubmissionCandidate(nil), c.Candidates...)
	if p.name == "deadline" {
		sort.SliceStable(candidates, func(i, j int) bool {
			a, b := candidates[i], candidates[j]
			if a.DeadlineKnown != b.DeadlineKnown {
				return a.DeadlineKnown
			}
			return a.DeadlineKnown && a.DeadlineUS < b.DeadlineUS
		})
	}
	ids := make([]int64, len(candidates))
	for i, candidate := range candidates {
		ids[i] = candidate.Transaction
	}
	return ids
}

// TransferSubmissionCost is extra simulated CPU service before the first host
// submission. It covers snapshot/order/validation work, separately from the
// existing per-job host and DMA costs. Declared values are not a calibration.
type TransferSubmissionCost struct {
	FixedUS    int64  `json:"fixed_us"`
	PerJobUS   int64  `json:"per_job_us"`
	Provenance string `json:"provenance"`
}

func (c *TransferSubmissionCost) Validate() error {
	if c != nil && (c.FixedUS < 0 || c.PerJobUS < 0 || c.Provenance == "") {
		return fmt.Errorf("transfer submission cost requires nonnegative service and provenance")
	}
	return nil
}

func (c *TransferSubmissionCost) Coverage() string {
	if c == nil {
		return "extra_submission_decision_service_unmeasured_zero"
	}
	return "declared_extra_submission_service_not_native_validated"
}

func (c *TransferSubmissionCost) service(jobs int) (int64, error) {
	if c == nil || jobs == 0 {
		return 0, nil
	}
	if err := c.Validate(); err != nil {
		return 0, err
	}
	if c.PerJobUS > (math.MaxInt64-c.FixedUS)/int64(jobs) {
		return 0, fmt.Errorf("transfer submission decision service overflow")
	}
	return c.FixedUS + int64(jobs)*c.PerJobUS, nil
}

type TransferSubmissionRecord struct {
	View         TransferSubmissionContext `json:"view"`
	Order        []int64                   `json:"order"`
	ServiceEndUS int64                     `json:"service_end_us"`
	SubmitEndUS  int64                     `json:"submit_end_us"`
	CostCoverage string                    `json:"cost_coverage"`
	Cost         *TransferSubmissionCost   `json:"cost,omitempty"`
}

// SetTransferSubmissionPolicy installs a policy before this phase backend has
// staged or executed work. Cost and observation are optional. Callbacks receive
// detached records; source leases, reservations and adoption remain runtime-owned.
func (s *PeerCache) SetTransferSubmissionPolicy(policy TransferSubmissionPolicy, cost *TransferSubmissionCost, observe func(TransferSubmissionRecord)) error {
	p := s.fabric.phases
	if p == nil || p.store != s || p.started || p.active || s.fabric.Pending() != 0 || p.submissionPolicy != nil {
		return fmt.Errorf("transfer submission policy requires unused configured engine phases")
	}
	if policy == nil {
		return fmt.Errorf("transfer submission policy must be nonnil")
	}
	v := reflect.ValueOf(policy)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		if v.IsNil() {
			return fmt.Errorf("transfer submission policy must be nonnil")
		}
	}
	if err := cost.Validate(); err != nil {
		return err
	}
	if cost != nil {
		copy := *cost
		p.submissionCost = &copy
	}
	p.submissionPolicy, p.observeSubmission = policy, observe
	return nil
}

func (p *PeerEnginePhases) orderSubmissions(now int64, direction string, jobs []*peerJob) ([]*peerJob, *TransferSubmissionRecord, error) {
	if p.submissionPolicy == nil || len(jobs) == 0 {
		return jobs, nil, nil
	}
	view := TransferSubmissionContext{NowUS: now, Instance: p.store.id, Direction: direction}
	eligible := make(map[int64]*peerJob, len(jobs))
	for _, j := range jobs {
		id := j.record.Transaction
		if _, exists := eligible[id]; exists {
			return nil, nil, fmt.Errorf("duplicate staged transfer %d", id)
		}
		eligible[id] = j
		d := j.effectiveDeadline(p.store.fabric)
		known := d > 0 && d != math.MaxInt64
		if !known {
			d = 0
		}
		view.Candidates = append(view.Candidates, TransferSubmissionCandidate{
			Transaction: id, Request: j.record.Request, Reason: j.record.Reason,
			Source: j.record.Source, Destination: j.record.Destination,
			Blocks: max(1, j.units), Bytes: j.record.Bytes, DeadlineUS: d, DeadlineKnown: known,
		})
	}
	extra, err := p.submissionCost.service(len(jobs))
	if err != nil || now < 0 || extra > math.MaxInt64-now {
		return nil, nil, fmt.Errorf("invalid transfer submission decision service at %d: %v", now, err)
	}
	input := view
	input.Candidates = append([]TransferSubmissionCandidate(nil), view.Candidates...)
	ids := append([]int64(nil), p.submissionPolicy.Order(input)...)
	if len(ids) != len(jobs) {
		return nil, nil, fmt.Errorf("transfer submission policy must return a complete permutation")
	}
	ordered := make([]*peerJob, 0, len(jobs))
	for _, id := range ids {
		j, ok := eligible[id]
		if !ok {
			return nil, nil, fmt.Errorf("transfer submission policy returned duplicate or foreign transaction %d", id)
		}
		ordered = append(ordered, j)
		delete(eligible, id)
	}
	record := &TransferSubmissionRecord{View: view, Order: ids, ServiceEndUS: now + extra, CostCoverage: p.submissionCost.Coverage()}
	if p.submissionCost != nil {
		copy := *p.submissionCost
		record.Cost = &copy
	}
	return ordered, record, nil
}
