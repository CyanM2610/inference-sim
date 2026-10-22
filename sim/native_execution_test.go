package sim

import "testing"

func TestNativeExecutionRejectsUnknownCostResourcesAndMissingDependencies(t *testing.T) {
	p := NativeExecutionPlan{Stages: []NativeStage{{Name: "access"}, {Name: "compute", Resource: "gpu", DurationUS: 10, After: []string{"access"}}, {Name: "host", After: []string{"access"}}, {Name: "out", After: []string{"compute", "host"}}}, KVAccess: "access", Computed: "compute", ProducerReady: "compute", PostForward: "host", Poll: "host", HostReturned: "host", OutputReady: "out"}
	if err := validateNativePlan(p, map[string]bool{"gpu": false}); err != nil {
		t.Fatal(err)
	}
	if err := validateNativePlan(p, map[string]bool{}); err == nil {
		t.Fatal("unconfigured GPU service accepted")
	}
	p.Stages[3].After = []string{"host"}
	if err := validateNativePlan(p, map[string]bool{"gpu": false}); err == nil {
		t.Fatal("output can precede computation")
	}
	p.Stages[3].After = []string{"compute", "host"}
	p.Stages[1].After = []string{"out"}
	if err := validateNativePlan(p, map[string]bool{"gpu": false}); err == nil {
		t.Fatal("cycle/forward reference accepted")
	}
	p.Stages[1].After = nil
	if err := validateNativePlan(p, map[string]bool{"gpu": false}); err == nil {
		t.Fatal("compute bypassed physical access fence")
	}
	p.Stages[1].After = []string{"access"}
	p.Stages[2].After = nil
	if err := validateNativePlan(p, map[string]bool{"gpu": false}); err == nil {
		t.Fatal("host post/poll bypassed physical access fence")
	}
}
