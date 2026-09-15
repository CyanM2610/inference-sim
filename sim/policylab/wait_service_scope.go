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
	if family != want || hbm != c.Instances[0].HBMBlocks || shape != serviceShape(c) || input <= 0 || output <= 0 || strings.TrimSpace(provenance) == "" {
		return fmt.Errorf("invalid wait service family, geometry or coverage")
	}
	for _, r := range c.Requests {
		if int64(len(r.Input)) > input || r.MaxOutputTokens <= 0 || r.MaxOutputTokens > output {
			return fmt.Errorf("request exceeds wait service profile coverage")
		}
	}
	return nil
}
