package sim

// SetInstanceScheduler injects an experiment's request ordering policy.
// Call during construction, before processing events. It does not change batch
// formation, preemption, or transfer ordering. A nil policy is a caller error.
func (s *Simulator) SetInstanceScheduler(policy InstanceScheduler) {
	if policy == nil {
		panic("nil instance scheduler")
	}
	s.scheduler = policy
}
