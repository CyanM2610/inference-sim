package policylab

import (
	"fmt"
	"sort"

	"github.com/sirupsen/logrus"
)

// ProfileWarning reports extrapolation of an existing cost formula. Bounds
// describe the original measurements; they never truncate work or capacity.
// Observations include predictor queries and configuration checks, not steps.
type ProfileWarning struct {
	Component    string `json:"component"`
	Parameter    string `json:"parameter"`
	ProfileMin   int64  `json:"profile_min"`
	ProfileMax   int64  `json:"profile_max"`
	ObservedMin  int64  `json:"observed_min"`
	ObservedMax  int64  `json:"observed_max"`
	Observations int64  `json:"observations"`
	Provenance   string `json:"provenance"`
	Action       string `json:"action"`
}

type profileWarnings struct{ items map[string]*ProfileWarning }

// Unexported state is deliberately absent from saved profile/config JSON.
type profileReporter struct {
	warnings   *profileWarnings
	component  string
	provenance string
}

func (c Config) profile(component, provenance string) profileReporter {
	return profileReporter{c.profileWarnings, component, provenance}
}

func (p profileReporter) above(parameter string, value, maximum int64) {
	p.outside(parameter, value, 0, maximum)
}

func (p profileReporter) equal(parameter string, value, measured int64) {
	p.outside(parameter, value, measured, measured)
}

func (p profileReporter) outside(parameter string, value, low, high int64) {
	if value >= low && value <= high {
		return
	}
	if p.warnings == nil {
		logrus.Warnf("profile extrapolation: %s.%s observed=%d measured=[%d,%d]; using unchanged cost formula", p.component, parameter, value, low, high)
		return
	}
	key := fmt.Sprintf("%s/%s/%d/%d/%s", p.component, parameter, low, high, p.provenance)
	if p.warnings.items == nil {
		p.warnings.items = map[string]*ProfileWarning{}
	}
	w := p.warnings.items[key]
	if w == nil {
		w = &ProfileWarning{Component: p.component, Parameter: parameter, ProfileMin: low, ProfileMax: high,
			ObservedMin: value, ObservedMax: value, Provenance: p.provenance,
			Action: "extrapolate_unchanged_formula_not_native_validated"}
		p.warnings.items[key] = w
	}
	w.ObservedMin, w.ObservedMax = min(w.ObservedMin, value), max(w.ObservedMax, value)
	w.Observations++
}

func (w *profileWarnings) snapshot() []ProfileWarning {
	keys := make([]string, 0, len(w.items))
	for key := range w.items {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var result []ProfileWarning
	for _, key := range keys {
		result = append(result, *w.items[key])
	}
	return result
}

func (p profileReporter) shape(measured, actual RestoreServiceShape) {
	p.equal("block_tokens", actual.BlockTokens, measured.BlockTokens)
	p.equal("max_batch_tokens", actual.MaxBatchTokens, measured.MaxBatchTokens)
	p.equal("max_sequences", actual.MaxSequences, measured.MaxSequences)
	p.equal("prefill_chunk", actual.PrefillChunk, measured.PrefillChunk)
	p.equal("prefill_token_cap", actual.PrefillTokenCap, measured.PrefillTokenCap)
	p.equal("restore_window_blocks", actual.RestoreWindowBlocks, measured.RestoreWindowBlocks)
}

func validServiceShape(s RestoreServiceShape) bool {
	return s.BlockTokens > 0 && s.MaxBatchTokens > 0 && s.MaxSequences > 0 &&
		s.PrefillChunk >= 0 && s.PrefillTokenCap >= 0 && s.RestoreWindowBlocks > 0 && s.MaxInputTokens >= 0
}

func (p profileReporter) requests(c Config, input int64, output int) error {
	for _, r := range c.Requests {
		if r.MaxOutputTokens <= 0 {
			return fmt.Errorf("profiled service requires a positive declared output limit")
		}
		p.above("input_tokens", int64(len(r.Input)), input)
		p.above("client_output_limit", int64(r.MaxOutputTokens), int64(output))
	}
	return nil
}
