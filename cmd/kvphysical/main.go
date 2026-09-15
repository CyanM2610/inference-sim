// kvphysical validates LiveController against the shared physical executor.
package main

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/inference-sim/inference-sim/sim/kvruntime"
	"github.com/inference-sim/inference-sim/sim/policylab"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	input := flag.String("input", "", "independent request references and policy configuration")
	profile := flag.String("profile", "", "physical primitive calibration")
	controlPath := flag.String("control-profile", "", "measured full driver protocol; omitted means ideal-control functional validation")
	policy := flag.String("copy-policy", "fifo", "copy engine FIFO or stream_drain")
	out := flag.String("out", "", "new validation result directory")
	flag.Parse()
	data, err := os.ReadFile(*input)
	if err != nil {
		return err
	}
	var c policylab.Config
	if err = json.Unmarshal(data, &c); err != nil {
		return err
	}
	p, err := kvruntime.LoadProfile(*profile)
	if err != nil {
		return err
	}
	if err = os.Mkdir(*out, 0755); err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(*out, "input.json"), data, 0644); err != nil {
		return err
	}
	file, err := os.Create(filepath.Join(*out, "native.events.jsonl.gz"))
	if err != nil {
		return err
	}
	z := gzip.NewWriter(file)
	encoder := json.NewEncoder(z)
	var encodeErr error
	var events int64
	sink := func(r kvruntime.Record) {
		events++
		if encodeErr == nil {
			encodeErr = encoder.Encode(r)
		}
	}
	var result *policylab.PhysicalLiveResult
	var runErr error
	if *controlPath == "" {
		result, runErr = policylab.RunPhysicalLiveValidation(c, p, *policy, sink)
	} else {
		var control *kvruntime.ControlProfile
		control, runErr = kvruntime.LoadControlProfile(*controlPath)
		if runErr == nil {
			result, runErr = policylab.RunControlledLive(c, p, control, *policy, sink)
		}
	}
	zErr, fileErr := z.Close(), file.Close()
	for _, err := range []error{runErr, encodeErr, zErr, fileErr} {
		if err != nil {
			os.WriteFile(filepath.Join(*out, "failure.txt"), []byte(err.Error()+"\n"), 0644)
			return err
		}
	}
	data, err = json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(*out, "summary.json"), data, 0644); err != nil {
		return err
	}
	fmt.Printf("runtime complete (costed control=%t): %d requests, %d rounds, %d physical operations, %d events\n", *controlPath != "", len(result.Final.Requests), len(result.Rounds), len(result.Operations), events)
	return nil
}
