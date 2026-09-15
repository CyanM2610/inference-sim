package cluster

import "github.com/inference-sim/inference-sim/sim"

func (cs *ClusterSimulator) newInstance(id InstanceID, cfg sim.SimConfig) *InstanceSimulator {
	if cs.config.InstanceFactory != nil {
		return cs.config.InstanceFactory(id, cfg)
	}
	return NewInstanceSimulator(id, cfg)
}
