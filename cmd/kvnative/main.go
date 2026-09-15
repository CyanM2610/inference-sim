// kvnative executes native-profile fixtures through the shared KV event runtime.
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

func writeJSON(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, append(b, '\n'), 0644)
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func run() error {
	path := flag.String("profile", "", "calibration JSON")
	out := flag.String("out", "", "new result directory")
	key := flag.String("case", "", "single case key (default all measured shapes)")
	model := flag.Bool("model", false, "execute measured real-model symbolic workflows")
	paged := flag.Bool("paged-forward", false, "execute native partial-page forward grid")
	torchCopy := flag.Bool("torch-copy", false, "execute native copies in the measured live Torch arena")
	flag.Parse()
	modes := 0
	for _, enabled := range []bool{*model, *paged, *torchCopy} {
		if enabled {
			modes++
		}
	}
	if modes > 1 {
		return fmt.Errorf("choose only one of -model, -paged-forward or -torch-copy")
	}
	if *path == "" || *out == "" {
		return fmt.Errorf("usage: kvnative -profile profile.json -out new-directory [-case key]")
	}
	if _, err := os.Stat(*out); !os.IsNotExist(err) {
		return fmt.Errorf("output already exists or cannot be checked")
	}
	p, err := kvruntime.LoadProfile(*path)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(*out, 0755); err != nil {
		return err
	}
	if err = writeJSON(filepath.Join(*out, "profile.json"), p); err != nil {
		return err
	}
	if *paged {
		return runPaged(p, *out, *key)
	}
	if *torchCopy {
		return runTorchCopies(p, *out, *key)
	}
	if *model {
		return runModels(p, *out, *key)
	}
	var results []*kvruntime.NativeResult
	var failures []map[string]string
	cases := 0
	for _, point := range p.Transfers {
		if *key != "" && *key != point.Key() {
			continue
		}
		cases++
		file, err := os.Create(filepath.Join(*out, strings.ReplaceAll(point.Key(), "/", "_")+".events.jsonl.gz"))
		if err != nil {
			return err
		}
		gz, _ := gzip.NewWriterLevel(file, gzip.BestSpeed)
		encoder := json.NewEncoder(gz)
		var encodeErr error
		r, e := kvruntime.RunNativeCase(p, point, func(event kvruntime.Record) {
			if encodeErr == nil {
				encodeErr = encoder.Encode(event)
			}
		})
		if err = gz.Close(); err != nil {
			return err
		}
		if err = file.Close(); err != nil {
			return err
		}
		if encodeErr != nil {
			return encodeErr
		}
		if e != nil {
			failures = append(failures, map[string]string{"case": point.Key(), "error": e.Error()})
			fmt.Println("FAILED", point.Key(), e)
			continue
		}
		results = append(results, r)
		fmt.Printf("%s wall_ns=%d bytes=%d copies=%d verified_cells=%d\n", r.Key, r.TransferWallNS, r.Transfer.Bytes, r.Transfer.Copies, r.CellsVerified)
	}
	if cases == 0 {
		return fmt.Errorf("no matching measured cases")
	}
	if err = writeJSON(filepath.Join(*out, "summary.json"), map[string]any{"cases": cases, "completed": len(results), "failures": failures, "results": results}); err != nil {
		return err
	}
	if len(failures) > 0 {
		return fmt.Errorf("%d functional cases failed", len(failures))
	}
	return nil
}
func runTorchCopies(p *kvruntime.Profile, out, key string) error {
	if p.TorchCopies == nil {
		return fmt.Errorf("missing in-process copy profile")
	}
	var results []*kvruntime.TorchCopyResult
	for _, point := range p.TorchCopies.Points {
		if key != "" && key != point.Key() {
			continue
		}
		file, err := os.Create(filepath.Join(out, strings.ReplaceAll(point.Key(), "/", "_")+".events.jsonl.gz"))
		if err != nil {
			return err
		}
		gz, _ := gzip.NewWriterLevel(file, gzip.BestSpeed)
		encoder := json.NewEncoder(gz)
		var encodeErr error
		result, runErr := kvruntime.RunTorchCopyCase(p, point, func(e kvruntime.Record) {
			if encodeErr == nil {
				encodeErr = encoder.Encode(e)
			}
		})
		gzErr := gz.Close()
		fileErr := file.Close()
		for _, err := range []error{runErr, encodeErr, gzErr, fileErr} {
			if err != nil {
				return err
			}
		}
		results = append(results, result)
		fmt.Printf("%s wall_ns=%d copies=%d verified_cells=%d\n", point.Key(), result.Transfer.CompleteNS-result.Transfer.StartNS, result.Transfer.Copies, result.CellsVerified)
	}
	if len(results) == 0 {
		return fmt.Errorf("no matching in-process transfer shape")
	}
	return writeJSON(filepath.Join(out, "summary.json"), map[string]any{"cases": len(results), "completed": len(results), "results": results})
}
func runPaged(p *kvruntime.Profile, out, key string) error {
	var results []*kvruntime.PagedFixtureResult
	for _, point := range p.PagedForward {
		name := fmt.Sprintf("paged/p%d/q%d", point.Prefix, point.Query)
		if key != "" && name != key {
			continue
		}
		file, err := os.Create(filepath.Join(out, strings.ReplaceAll(name, "/", "_")+".events.jsonl.gz"))
		if err != nil {
			return err
		}
		gz, _ := gzip.NewWriterLevel(file, gzip.BestSpeed)
		encoder := json.NewEncoder(gz)
		var encodeErr error
		result, runErr := kvruntime.RunPagedFixture(p, point, func(e kvruntime.Record) {
			if encodeErr == nil {
				encodeErr = encoder.Encode(e)
			}
		})
		gzErr := gz.Close()
		fileErr := file.Close()
		for _, err := range []error{runErr, encodeErr, gzErr, fileErr} {
			if err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
		}
		results = append(results, result)
		fmt.Printf("%s wall_ns=%d mappings=%d verified_cells=%d\n", name, result.Step.CompleteNS-result.Step.StartNS, result.Step.MappingTasks, result.Step.CellsVerified)
	}
	if len(results) == 0 {
		return fmt.Errorf("no matching paged forward shape")
	}
	return writeJSON(filepath.Join(out, "summary.json"), map[string]any{"cases": len(results), "completed": len(results), "results": results})
}
func runModels(p *kvruntime.Profile, out, key string) error {
	var results []*kvruntime.ModelResult
	for _, point := range p.Compute {
		name := fmt.Sprintf("model/%d/%s", point.Length, point.Mode)
		if key != "" && name != key {
			continue
		}
		file, err := os.Create(filepath.Join(out, strings.ReplaceAll(name, "/", "_")+".events.jsonl.gz"))
		if err != nil {
			return err
		}
		gz, _ := gzip.NewWriterLevel(file, gzip.BestSpeed)
		encoder := json.NewEncoder(gz)
		var encodingErr error
		result, runErr := kvruntime.RunModelCase(p, point, func(e kvruntime.Record) {
			if encodingErr == nil {
				encodingErr = encoder.Encode(e)
			}
		})
		gzErr := gz.Close()
		fileErr := file.Close()
		for _, err := range []error{runErr, encodingErr, gzErr, fileErr} {
			if err != nil {
				return err
			}
		}
		results = append(results, result)
		fmt.Printf("%s transaction_ns=%d resume_first_ns=%d verified_cells=%d\n", name, result.TransactionNS, result.ResumeFirstNS, result.KVCellsVerified)
	}
	if len(results) == 0 {
		return fmt.Errorf("no matching model case")
	}
	return writeJSON(filepath.Join(out, "summary.json"), map[string]any{"cases": len(results), "completed": len(results), "results": results})
}
