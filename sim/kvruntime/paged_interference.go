package kvruntime

import "fmt"

// PagedInterferenceRates are effective phase/host-work rates, not pure device
// bandwidths. Absence of a phase means overlap there has not been calibrated.
type PagedInterferenceRates struct {
	ComputePPM  map[string]int64 `json:"compute_rate_ppm"`
	CopyHostPPM map[string]int64 `json:"copy_host_rate_ppm"`
	Evidence    string           `json:"evidence"`
}

func (r PagedInterferenceRates) Validate() error {
	if r.Evidence == "" || len(r.ComputePPM) == 0 || len(r.ComputePPM) != len(r.CopyHostPPM) {
		return fmt.Errorf("missing interference rate evidence")
	}
	for phase, rate := range r.ComputePPM {
		other, ok := r.CopyHostPPM[phase]
		if (phase != "index_bind" && phase != "gather_prefix" && phase != "forward") || rate <= 0 || rate > FullServicePPM || !ok || other <= 0 || other > FullServicePPM {
			return fmt.Errorf("invalid measured phase rate %s", phase)
		}
	}
	return nil
}

// PagedInterference tracks actual phase and transfer lifetimes. Callbacks run
// on the event thread; service changes preserve work already executed.
type PagedInterference struct {
	e       *Engine
	rates   PagedInterferenceRates
	phase   string
	copying bool
}

const CopyHostClass = "paged_contention/copy_host"

func PhaseServiceClass(phase string) string { return "paged_contention/" + phase }

func NewPagedInterference(e *Engine, rates PagedInterferenceRates) (*PagedInterference, error) {
	if e == nil {
		return nil, fmt.Errorf("interference requires an engine")
	}
	if err := rates.Validate(); err != nil {
		return nil, err
	}
	return &PagedInterference{e: e, rates: rates}, nil
}

func (c *PagedInterference) Phase(name string, active bool) error {
	if active {
		if c.phase != "" {
			return fmt.Errorf("overlapping batch phases")
		}
		c.phase = name
	} else {
		if c.phase != name {
			return fmt.Errorf("unmatched batch phase end")
		}
		c.phase = ""
	}
	reason := "end"
	if active {
		reason = "begin"
	}
	c.e.Emit(Record{Name: "interference_phase", Operation: name, Reason: reason})
	return c.update()
}

func (c *PagedInterference) Copy(active bool) error {
	if c.copying == active {
		return fmt.Errorf("unmatched/overlapping copy activity")
	}
	c.copying = active
	reason := "end"
	if active {
		reason = "begin"
	}
	c.e.Emit(Record{Name: "interference_copy", Reason: reason})
	return c.update()
}

func (c *PagedInterference) update() error {
	if c.copying && c.phase != "" {
		if _, ok := c.rates.ComputePPM[c.phase]; !ok {
			return fmt.Errorf("unmeasured paged interference phase %s", c.phase)
		}
	}
	for _, phase := range []string{"index_bind", "gather_prefix", "forward"} {
		rate := FullServicePPM
		if c.copying && c.phase == phase {
			rate = c.rates.ComputePPM[phase]
		}
		if err := c.e.SetServiceRate(PhaseServiceClass(phase), rate); err != nil {
			return err
		}
	}
	rate := FullServicePPM
	if measured, ok := c.rates.CopyHostPPM[c.phase]; ok {
		rate = measured
	}
	return c.e.SetServiceRate(CopyHostClass, rate)
}

func (c *PagedInterference) CheckIdle() error {
	if c.phase != "" || c.copying {
		return fmt.Errorf("interference activity did not drain")
	}
	return nil
}
