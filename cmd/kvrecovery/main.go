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

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	profilePath := flag.String("profile", "", "existing physical cost profile")
	referencePath := flag.String("references", "", "native input/internal/output references, without native timing")
	out := flag.String("out", "", "new output directory")
	only := flag.String("case", "", "optional exact case key")
	flag.Parse()
	p, err := kvruntime.LoadProfile(*profilePath)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(*referencePath)
	if err != nil {
		return err
	}
	var refs struct {
		Cases []kv.RecoveryCase `json:"cases"`
	}
	if err = json.Unmarshal(data, &refs); err != nil {
		return err
	}
	if err = os.Mkdir(*out, 0755); err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(*out, "references.json"), data, 0644); err != nil {
		return err
	}
	profileData, err := os.ReadFile(*profilePath)
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(*out, "profile.json"), profileData, 0644); err != nil {
		return err
	}
	var rows []map[string]any
	failed := 0
	for _, c := range refs.Cases {
		key := fmt.Sprintf("p%d-%s", c.Length, c.Mode)
		if *only != "" && key != *only {
			continue
		}
		f, err := os.Create(filepath.Join(*out, key+".events.jsonl.gz"))
		if err != nil {
			return err
		}
		z := gzip.NewWriter(f)
		encoder := json.NewEncoder(z)
		var encodeErr error
		var events int64
		result, runErr := kv.RunRecoveryFixture(p, c, func(r kvruntime.Record) {
			events++
			if encodeErr == nil {
				encodeErr = encoder.Encode(r)
			}
		})
		zErr, fErr := z.Close(), f.Close()
		row := map[string]any{"key": key, "events": events, "result": result}
		for _, err := range []error{runErr, encodeErr, zErr, fErr} {
			if err != nil {
				row["error"] = err.Error()
				failed++
				break
			}
		}
		rows = append(rows, row)
		progress, _ := json.MarshalIndent(rows, "", "  ")
		if err = os.WriteFile(filepath.Join(*out, "progress.json"), progress, 0644); err != nil {
			return err
		}
		fmt.Printf("%s events=%d error=%v\n", key, events, row["error"])
	}
	if len(rows) == 0 {
		return fmt.Errorf("no selected recovery cases")
	}
	data, err = json.MarshalIndent(map[string]any{"rows": rows}, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(*out, "summary.json"), data, 0644); err != nil {
		return err
	}
	if failed > 0 {
		return fmt.Errorf("%d recovery cases failed; traces retained", failed)
	}
	return nil
}
