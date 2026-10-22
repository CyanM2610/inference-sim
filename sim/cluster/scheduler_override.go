package cluster

import "github.com/inference-sim/inference-sim/sim"

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

func (i *InstanceSimulator) SetDecisionEventObserver(observe func(sim.DecisionEventRecord)) error {
	return i.sim.SetDecisionEventObserver(observe)
}
