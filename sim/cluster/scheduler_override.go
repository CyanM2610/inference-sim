package cluster

import "github.com/inference-sim/inference-sim/sim"

// SetInstanceScheduler configures ordering before the instance starts.
func (i *InstanceSimulator) SetInstanceScheduler(policy sim.InstanceScheduler) {
	i.sim.SetInstanceScheduler(policy)
}

func (i *InstanceSimulator) SetDecisionPolicy(policy sim.DecisionPolicy, observe func(sim.DecisionRecord)) error {
	return i.sim.SetDecisionPolicy(policy, observe)
}
