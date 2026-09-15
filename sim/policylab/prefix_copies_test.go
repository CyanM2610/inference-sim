package policylab

import "testing"

func TestPrefixCopySelectionConfigurationAndPressureCompletion(t *testing.T) {
	c := pressureBudgetConfig()
	c.PrefixCopySelection = "unknown"
	if c.Validate() == nil {
		t.Fatal("unknown copy selection accepted")
	}
	c.PrefixCopySelection = "first_published"
	r, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.FinishedUS) != len(c.Requests) || r.PrefixCopyCostCoverage != "duplicate_prefix_index_not_independently_calibrated" || r.Config.PrefixCopySelection != c.PrefixCopySelection {
		t.Fatal("configured mechanism failed to complete or lost its provenance")
	}
	for _, state := range r.HBM {
		if state["active_or_pinned"] != 0 || state["decode_reserved_blocks"] != 0 {
			t.Fatal("pressure workload leaked capacity")
		}
	}
}
