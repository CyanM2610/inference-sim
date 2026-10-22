package sim

import "testing"

func TestNativeExecutionRejectsUnknownCostResourcesAndMissingDependencies(t *testing.T) {
	p := NativeExecutionPlan{Stages: []NativeStage{{Name: "compute", Resource: "gpu", DurationUS: 10}, {Name: "host"}, {Name: "out", After: []string{"compute", "host"}}}, Computed: "compute", ProducerReady: "compute", PostForward: "host", Poll: "host", HostReturned: "host", OutputReady: "out"}
	if err := validateNativePlan(p, map[string]bool{"gpu": false}); err != nil {
		t.Fatal(err)
	}
	if err := validateNativePlan(p, map[string]bool{}); err == nil {
		t.Fatal("unconfigured GPU service accepted")
	}
	p.Stages[2].After = []string{"host"}
	if err := validateNativePlan(p, map[string]bool{"gpu": false}); err == nil {
		t.Fatal("output can precede computation")
	}
	p.Stages[2].After = []string{"compute", "host"}
	p.Stages[0].After = []string{"out"}
	if err := validateNativePlan(p, map[string]bool{"gpu": false}); err == nil {
		t.Fatal("cycle/forward reference accepted")
	}
}
