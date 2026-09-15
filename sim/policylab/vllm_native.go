package policylab

import (
	"fmt"
	"reflect"
)

func (c Config) validateVLLMNative() error {
	if len(c.Instances) != 1 || len(c.Pools) != 1 || c.Pools[0].ID != "dram" || len(c.Instances[0].Access) != 1 || c.Instances[0].Access[0].Pool != "dram" || c.Native != nil || !c.RestoreControl || c.EnginePhases == nil {
		return fmt.Errorf("vllm_native requires one abstract HBM-DRAM instance, engine_phases and restore_control for output-history recomputation")
	}
	if c.DecodeCapacityReservation || c.BatchPrefixReuse || c.PromotionControl {
		return fmt.Errorf("vllm_native does not support custom output reservation, batch prefix donation or request promotion controls")
	}
	if c.DecisionPolicy != nil {
		d := *c.DecisionPolicy
		if d.QueueOrder != "" && d.QueueOrder != "fcfs" {
			return fmt.Errorf("vllm_native fixes FCFS queue order")
		}
		// Observation, event plumbing and a declared extra fee do not change
		// the native decisions. Other controllers would create a hybrid baseline.
		d.QueueOrder, d.ControlSteps, d.TraceEvents, d.ExecutionWork, d.ExtraCost = "", false, false, false, nil
		if !reflect.DeepEqual(d, DecisionPolicyConfig{}) {
			return fmt.Errorf("vllm_native permits only FCFS observation/control_steps/execution_work/extra_cost in decision_policy; remove custom controllers and old fee profiles")
		}
	}
	return nil
}

func (c Config) decisionQueueOrder() string {
	if c.RequestScheduler == "vllm_native" {
		return "fcfs"
	}
	return c.RequestScheduler
}
