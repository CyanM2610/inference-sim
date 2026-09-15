package policylab

import (
	"fmt"
	"reflect"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
)

// PolicyFactories replaces decision logic while retaining the configured
// execution backend. Nil factories retain the corresponding configured policy.
// Factories run once per instance/pool before any simulated events. Return fresh
// policy state per ID unless cross-instance sharing is deliberately intended.
// The input Config remains the validated fallback configuration, not a record
// of the injected implementation; Result.InjectedPolicies records replacements.
type PolicyFactories struct {
	Decision           func(instance string) sim.DecisionPolicy
	Peer               func(instance string) kv.PeerPolicy
	PoolEviction       func(pool string) kv.PoolEvictionPolicy
	TransferSubmission func(instance string) kv.TransferSubmissionPolicy
}

// RunWithPolicies composes custom request, HBM reclaim/save and pool eviction
// policies and same-phase transfer submission ordering. It does not grant active
// placement or wire transfer-priority capabilities.
// Custom KV callback costs are not calibrated by this injection mechanism.
func RunWithPolicies(c Config, factories PolicyFactories) (*Result, error) {
	if factories.Decision != nil && c.DecisionPolicy == nil {
		return nil, fmt.Errorf("custom decision factory requires decision_policy")
	}
	return run(c, factories)
}

// An interface containing a typed nil is also an invalid factory result.
func nilPolicy(policy any) bool {
	if policy == nil {
		return true
	}
	v := reflect.ValueOf(policy)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}

func (r *Result) recordInjectedPolicy(role, id string, policy any) {
	if r.InjectedPolicies == nil {
		r.InjectedPolicies = map[string]map[string]string{}
	}
	if r.InjectedPolicies[role] == nil {
		r.InjectedPolicies[role] = map[string]string{}
	}
	r.InjectedPolicies[role][id] = fmt.Sprintf("%T", policy)
	if role != "decision" && role != "transfer_submission" {
		r.KVPolicyCostCoverage = "injected_kv_callbacks_not_independently_calibrated"
	}
}
