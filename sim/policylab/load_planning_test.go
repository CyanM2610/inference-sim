package policylab

import (
	"math"
	"testing"
)

func TestLoadPlanningProfileBoundsAndZero(t *testing.T) {
	c := labConfig(1, "dram")
	phaseTestConfig(&c)
	p := &EngineLoadPlanningCost{Coefficients: []float64{1, 2, 3}, MaxLoads: 3, MaxBlocks: 48, Provenance: "behavior fixture", Coverage: "synthetic"}
	c.EnginePhases.LoadPlanning = p
	c.EnginePhases.warnings = &profileWarnings{}
	if err := c.EnginePhases.Validate(); err != nil {
		t.Fatal(err)
	}
	if v, err := c.EnginePhases.PredictLoadPlanning(2, 8); err != nil || v != 29 {
		t.Fatal(v, err)
	}
	if v, err := c.EnginePhases.PredictLoadPlanning(0, 0); err != nil || v != 0 {
		t.Fatal(v, err)
	}
	for _, counts := range [][2]int64{{4, 8}, {1, 49}} {
		if v, err := c.EnginePhases.PredictLoadPlanning(counts[0], counts[1]); err != nil || v != 1+2*counts[0]+3*counts[1] {
			t.Fatal("extrapolation clipped or rejected work", counts, v, err)
		}
	}
	requireProfileWarning(t, c.EnginePhases.warnings.snapshot(), "load_planning", "loads")
	requireProfileWarning(t, c.EnginePhases.warnings.snapshot(), "load_planning", "blocks")
	for _, counts := range [][2]int64{{2, 1}, {0, 1}, {-1, 0}} {
		if _, err := c.EnginePhases.PredictLoadPlanning(counts[0], counts[1]); err == nil {
			t.Fatal("invalid shape accepted", counts)
		}
	}
	p.Coefficients = []float64{0, 0, 0}
	if v, err := c.EnginePhases.PredictLoadPlanning(3, 48); err != nil || v != 0 {
		t.Fatal(v, err)
	}
	p.Coefficients[0] = math.NaN()
	if c.EnginePhases.Validate() == nil {
		t.Fatal("NaN profile accepted")
	}
}
