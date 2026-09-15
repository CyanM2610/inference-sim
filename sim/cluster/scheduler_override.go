package cluster

import "github.com/inference-sim/inference-sim/sim"

func (i *InstanceSimulator) SetDecisionEstimator(model sim.DecisionEstimator) error {
	return i.sim.SetDecisionEstimator(model)
}

// SetInstanceScheduler configures ordering before the instance starts.
func (i *InstanceSimulator) SetInstanceScheduler(policy sim.InstanceScheduler) {
	i.sim.SetInstanceScheduler(policy)
}

func (i *InstanceSimulator) SetDecisionPolicy(policy sim.DecisionPolicy, observe func(sim.DecisionRecord)) error {
	return i.sim.SetDecisionPolicy(policy, observe)
}

func (i *InstanceSimulator) SetDecisionCostModel(model sim.DecisionCostModel) error {
	return i.sim.SetDecisionCostModel(model)
}

func (i *InstanceSimulator) EnableExecutionWorkObservation() error {
	return i.sim.EnableExecutionWorkObservation()
}

func (i *InstanceSimulator) SetExecutionCostModel(model sim.ExecutionCostModel) error {
	return i.sim.SetExecutionCostModel(model)
}

func (i *InstanceSimulator) SetExecutionOutcomeCostModel(model sim.ExecutionOutcomeCostModel) error {
	return i.sim.SetExecutionOutcomeCostModel(model)
}

func (i *InstanceSimulator) SetHostServiceCosts(model sim.HostServiceCostModel, completion, registration bool, observe func(sim.HostServiceRecord)) error {
	return i.sim.SetHostServiceCosts(model, completion, registration, observe)
}

func (i *InstanceSimulator) SetPostStepCostModel(model sim.PostStepCostModel, inputRequests int, observe func(sim.PostStepRecord)) error {
	return i.sim.SetPostStepCostModel(model, inputRequests, observe)
}

func (i *InstanceSimulator) EnableDecisionControlSteps() error {
	return i.sim.EnableDecisionControlSteps()
}

func (i *InstanceSimulator) EnableDecisionPrefixProducers() error {
	return i.sim.EnableDecisionPrefixProducers()
}

func (i *InstanceSimulator) EnableDecisionPrefixSharing() error {
	return i.sim.EnableDecisionPrefixSharing()
}

func (i *InstanceSimulator) EnableDecisionPrefillPreemption() error {
	return i.sim.EnableDecisionPrefillPreemption()
}

func (i *InstanceSimulator) EnableDecisionRequestSpill() error {
	return i.sim.EnableDecisionRequestSpill()
}

func (i *InstanceSimulator) EnableDecisionPrefillWaits() error {
	return i.sim.EnableDecisionPrefillWaits()
}

func (i *InstanceSimulator) EnableDecisionPrefillDeferrals() error {
	return i.sim.EnableDecisionPrefillDeferrals()
}

func (i *InstanceSimulator) EnableDecisionCapacityPreemption() error {
	return i.sim.EnableDecisionCapacityPreemption()
}

func (i *InstanceSimulator) EnableDecisionCapacityReservationView() error {
	return i.sim.EnableDecisionCapacityReservationView()
}

func (i *InstanceSimulator) SetDecisionEventObserver(observe func(sim.DecisionEventRecord)) error {
	return i.sim.SetDecisionEventObserver(observe)
}
