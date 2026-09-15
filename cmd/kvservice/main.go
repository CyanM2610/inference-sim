// kvservice audits observed work counters; it never runs an inference policy.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/policylab"
)

type input struct {
	DecodeFill           bool               `json:"decode_fill"`
	Config               policylab.Config   `json:"config"`
	Record               sim.DecisionRecord `json:"record"`
	Quota                bool               `json:"quota"`
	DecisionOnly         bool               `json:"decision_only"`
	Spill                bool               `json:"spill"`
	SpillOnly            bool               `json:"spill_only"`
	MaxSpillSourceBlocks int64              `json:"max_spill_source_blocks"`
}

func main() {
	d := json.NewDecoder(os.Stdin)
	e := json.NewEncoder(os.Stdout)
	for {
		var in input
		if err := d.Decode(&in); err != nil {
			if err == io.EOF {
				return
			}
			panic(err)
		}
		if in.DecodeFill {
			prepare, execution, err := policylab.DecodeFillServiceFeatures(in.Record.View, in.Record.Plan, in.Record.Feedback.Grants)
			if err != nil {
				panic(err)
			}
			if err = e.Encode(map[string][]int64{"prepare": prepare, "execution": execution}); err != nil {
				panic(err)
			}
			continue
		}
		if in.SpillOnly {
			if in.Config.DecisionPolicy == nil || in.Config.DecisionPolicy.PreemptionStorage == nil {
				panic("missing spill policy")
			}
			counts, err := policylab.SpillDecisionCounts(in.Record.View, in.Record.Plan, in.Config.DecisionPolicy.PreemptionStorage.Mode)
			if err != nil {
				panic(err)
			}
			actual, err := policylab.SpillExecutionCounts(in.Record.View, in.Record.Plan, in.Record.Feedback, in.MaxSpillSourceBlocks)
			if err != nil {
				panic(err)
			}
			for key, value := range actual {
				counts[key] = value
			}
			if err := e.Encode(counts); err != nil {
				panic(err)
			}
			continue
		}
		counts, err := policylab.QueueDecisionCounts(in.Config, in.Record.View, in.Record.Plan)
		if err != nil {
			panic(err)
		}
		if in.Spill {
			if in.Config.DecisionPolicy == nil || in.Config.DecisionPolicy.PreemptionStorage == nil {
				panic("missing spill policy")
			}
			spill, err := policylab.SpillDecisionCounts(in.Record.View, in.Record.Plan, in.Config.DecisionPolicy.PreemptionStorage.Mode)
			if err != nil {
				panic(err)
			}
			for key, value := range spill {
				counts[key] = value
			}
		}
		if in.DecisionOnly {
			if err := e.Encode(counts); err != nil {
				panic(err)
			}
			continue
		}
		w := in.Record.ExecutionWork
		if w == nil && in.Record.ControlStep {
			w = &sim.BatchExecutionWork{TokenChecksKnown: true}
		}
		if w == nil {
			panic(fmt.Errorf("missing observed formation work"))
		}
		actual, err := policylab.QueueExecutionCounts(in.Record.View, *w, in.Record.Feedback, in.Quota)
		if err != nil {
			panic(err)
		}
		for key, value := range actual {
			counts[key] = value
		}
		if in.Spill {
			spill, err := policylab.SpillExecutionCounts(in.Record.View, in.Record.Plan, in.Record.Feedback, in.MaxSpillSourceBlocks)
			if err != nil {
				panic(err)
			}
			for key, value := range spill {
				counts[key] = value
			}
		}
		if err := e.Encode(counts); err != nil {
			panic(err)
		}
	}
}
