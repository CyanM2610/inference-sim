package policylab

import "fmt"

func (c Config) validateHotPrefix() error {
	for _, r := range c.Requests {
		if r.ID == "@promotion" {
			return fmt.Errorf("@promotion is a reserved HotPrefix background actor")
		}
	}
	if c.HotPrefix == nil || c.Policy.Name != "hotprefix" {
		return fmt.Errorf("policy.name=hotprefix and hotprefix config must be specified together")
	}
	if err := c.HotPrefix.Validate(); err != nil {
		return err
	}
	if len(c.Instances) != 1 || len(c.Pools) != 1 || c.Pools[0].ID != "dram" || len(c.Instances[0].Access) != 1 || c.Instances[0].Access[0].Pool != "dram" {
		return fmt.Errorf("HotPrefix scope is one HBM instance and one DRAM pool")
	}
	if c.Native != nil || c.EnginePhases == nil || c.PrefixCopySelection != "first_published" || c.Mechanisms == nil || c.Mechanisms.BackgroundStoreMode != "on_reclaim" || c.Mechanisms.BackgroundStorePool != "dram" {
		return fmt.Errorf("HotPrefix requires abstract engine phases, first_published copies and on_reclaim DRAM storage")
	}
	if len(c.Promotions) != 0 || c.PromotionControl || c.Mechanisms.Online != nil || c.DirectionalTransferOrder || c.TransferSubmissionPolicy != "" || c.TransferSubmissionCost != nil || c.BatchPrefixReuse || c.DecodeCapacityReservation {
		return fmt.Errorf("HotPrefix cannot combine with other promotion, transfer, shared-producer or output-reservation mechanisms")
	}
	if c.Policy.MinFrequency != 0 || c.Policy.MaxStoreUS != 0 {
		return fmt.Errorf("HotPrefix thresholds belong in hotprefix config, not generic policy limits")
	}
	if d := c.DecisionPolicy; d != nil && (d.QueueServiceCost != nil || d.DecodeFillServiceCost != nil || d.RequestSpill) {
		return fmt.Errorf("HotPrefix cannot inherit old policy fee calibration or request-spill placement")
	}
	return nil
}
