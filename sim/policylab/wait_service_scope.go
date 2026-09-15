package policylab

import (
	"fmt"
	"strings"
)

// Shared execution geometry for explicitly measured wait-service components.
func validateWaitServiceScope(c Config, family string, shape RestoreServiceShape, hbm, input int64, output int, provenance string) error {
	if c.DecisionPolicy == nil || !c.DecisionPolicy.PrefillWaits || !c.DecisionPolicy.ControlSteps || !c.RestoreControl ||
		c.EnginePhases == nil || c.BatchPrefixReuse || c.PromotionControl || len(c.Instances) != 1 {
		return fmt.Errorf("wait service requires resident waits, control steps and one restore engine-phase instance")
	}
	want := "basic"
	if c.DecisionPolicy.CapacityPreemption {
		want = "capacity"
	}
	if !validServiceShape(shape) || family != want || hbm <= 0 || input <= 0 || output <= 0 || strings.TrimSpace(provenance) == "" {
		return fmt.Errorf("invalid wait service family, geometry or coverage")
	}
	report := c.profile("wait_service", provenance)
	report.equal("hbm_blocks", c.Instances[0].HBMBlocks, hbm)
	report.shape(shape, serviceShape(c))
	return report.requests(c, input, output)
}
