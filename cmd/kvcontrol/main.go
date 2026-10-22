// kvcontrol validates protocol-to-worker dispatch as an isolated component.
package main

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

type command struct {
	ID       string `json:"id"`
	Kind     string `json:"kind"`
	ActualNS int64  `json:"actual_start_delta_ns"`
}
type fixture struct {
	ID       string                    `json:"id"`
	Input    kvruntime.ControlFeatures `json:"input_features"`
	Policy   kvruntime.ControlFeatures `json:"policy_features"`
	Reply    kvruntime.ControlFeatures `json:"reply_features"`
	Commands []command                 `json:"commands"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	profile := flag.String("profile", "", "control profile")
	cases := flag.String("cases", "", "dispatch fixture JSON")
	out := flag.String("out", "", "new result directory")
	flag.Parse()
	p, err := kvruntime.LoadControlProfile(*profile)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(*cases)
	if err != nil {
		return err
	}
	var inputs []fixture
	if err = json.Unmarshal(data, &inputs); err != nil {
		return err
	}
	if err = os.Mkdir(*out, 0755); err != nil {
		return err
	}
	var rows []map[string]any
	for _, input := range inputs {
		input := input
		var records []kvruntime.Record
		e := kvruntime.NewEngine(func(r kvruntime.Record) { records = append(records, r) })
		m, _ := kvruntime.NewMemory(e, 8, map[string]int64{"unused": 8})
		runtime, _ := kvruntime.NewRuntime(e, m, 8)
		result, err := runtime.ControlRound(kvruntime.ControlRoundSpec{ID: input.ID, Profile: p, InputFeatures: input.Input, Decide: func() (*kvruntime.ControlDecision, error) {
			d := &kvruntime.ControlDecision{PolicyFeatures: input.Policy, ReplyFeatures: input.Reply}
			for _, c := range input.Commands {
				c := c
				d.Commands = append(d.Commands, kvruntime.ControlCommand{ID: c.ID, Kind: c.Kind, Start: func() (int64, error) {
					// Validation stops at worker operation entry. No GPU/body time
					// or data is supplied, predicted, or claimed by this fixture.
					return e.MustAdd(kvruntime.Task{Name: "dispatch_fixture_endpoint", Operation: input.ID, Resource: "fixture/" + c.ID}), nil
				}})
			}
			return d, nil
		}})
		if err == nil {
			err = e.Run()
		}
		if err != nil {
			if strings.Contains(err.Error(), "unmeasured control feature") {
				rows = append(rows, map[string]any{"id": input.ID, "status": "outside_calibration", "error": err.Error()})
				continue
			}
			return err
		}
		file, err := os.Create(filepath.Join(*out, input.ID+".jsonl.gz"))
		if err != nil {
			return err
		}
		z := gzip.NewWriter(file)
		encoder := json.NewEncoder(z)
		for _, record := range records {
			if err = encoder.Encode(record); err != nil {
				return err
			}
		}
		if err = z.Close(); err != nil {
			return err
		}
		if err = file.Close(); err != nil {
			return err
		}
		var predictions []map[string]any
		for _, c := range input.Commands {
			predictions = append(predictions, map[string]any{"id": c.ID, "kind": c.Kind, "actual_ns": c.ActualNS, "predicted_ns": result.WorkerStartedNS[c.ID]})
		}
		rows = append(rows, map[string]any{"id": input.ID, "status": "ok", "predictions": predictions, "trace": input.ID + ".jsonl.gz", "released_ns": result.ReleasedNS})
	}
	data, err = json.MarshalIndent(map[string]any{"rows": rows, "scope": "Held-out command dispatch component only. Decision features are fixtures, not a policy or workload prediction. No physical body duration is replayed. Unsupported feature ranges are reported, not extrapolated."}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(*out, "summary.json"), data, 0644)
}
