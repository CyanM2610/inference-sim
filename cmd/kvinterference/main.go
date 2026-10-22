package main

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

type direction struct {
	CopyCost kvruntime.TransferCost           `json:"copy_cost"`
	Enter    int64                            `json:"loop_enter_ns"`
	Exit     int64                            `json:"loop_exit_ns"`
	Gap      int64                            `json:"copy_gap_ns"`
	Rates    kvruntime.PagedInterferenceRates `json:"rates"`
}
type profile struct {
	Physical   kvruntime.Profile    `json:"physical_profile"`
	Directions map[string]direction `json:"directions"`
	Shapes     [][2]int64           `json:"shapes"`
	Rounds     []int                `json:"copy_rounds"`
	References map[string]uint64    `json:"references"`
	Offset     int64                `json:"copy_start_offset_ns"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	path := flag.String("profile", "", "frozen diagnostic profile")
	out := flag.String("out", "", "new output directory")
	only := flag.String("case", "", "optional exact case key")
	flag.Parse()
	data, err := os.ReadFile(*path)
	if err != nil {
		return err
	}
	var p profile
	if err = json.Unmarshal(data, &p); err != nil {
		return err
	}
	if err = p.Physical.Validate(); err != nil {
		return err
	}
	if err = os.Mkdir(*out, 0755); err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(*out, "profile.json"), data, 0644); err != nil {
		return err
	}
	var rows []map[string]any
	failed := 0
	for _, shape := range p.Shapes {
		for _, name := range []string{"d2h", "h2d"} {
			for _, rounds := range p.Rounds {
				for _, mode := range []string{"compute_only", "copy_only", "overlap"} {
					key := fmt.Sprintf("p%d-q%d-%s-r%d-%s", shape[0], shape[1], name, rounds, mode)
					if *only != "" && key != *only {
						continue
					}
					d, ok := p.Directions[name]
					if !ok {
						return fmt.Errorf("missing direction")
					}
					f, err := os.Create(filepath.Join(*out, key+".events.jsonl.gz"))
					if err != nil {
						return err
					}
					z := gzip.NewWriter(f)
					enc := json.NewEncoder(z)
					var encodeErr error
					var events int64
					result, runErr := kv.RunPagedInterferenceFixture(&p.Physical, kv.PagedInterferenceCase{Prefix: shape[0], Query: shape[1], Direction: name, Mode: mode, Rounds: rounds, ReferenceToken: p.References[fmt.Sprintf("%d/%d", shape[0], shape[1])], CopyCost: d.CopyCost, LoopEnterNS: d.Enter, LoopExitNS: d.Exit, CopyGapNS: d.Gap, CopyOffsetNS: p.Offset, Rates: d.Rates}, func(r kvruntime.Record) {
						events++
						if encodeErr == nil {
							encodeErr = enc.Encode(r)
						}
					})
					zErr, fErr := z.Close(), f.Close()
					row := map[string]any{"key": key, "events": events, "result": result}
					for _, e := range []error{runErr, encodeErr, zErr, fErr} {
						if e != nil {
							row["error"] = e.Error()
							failed++
							break
						}
					}
					rows = append(rows, row)
					encoded, _ := json.MarshalIndent(rows, "", "  ")
					if err = os.WriteFile(filepath.Join(*out, "progress.json"), encoded, 0644); err != nil {
						return err
					}
					fmt.Printf("%s events=%d error=%v\n", key, events, row["error"])
				}
			}
		}
	}
	if len(rows) == 0 {
		return fmt.Errorf("no case selected")
	}
	data, err = json.MarshalIndent(map[string]any{"rows": rows}, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(*out, "summary.json"), data, 0644); err != nil {
		return err
	}
	if failed != 0 {
		return fmt.Errorf("%d interference cases failed; partial traces and summary retained", failed)
	}
	return nil
}
