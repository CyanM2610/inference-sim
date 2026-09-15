// kvinput runs input primitives through the same runtime used by paged BLIS.
package main

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/inference-sim/inference-sim/sim/kvruntime"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	profilePath := flag.String("profile", "", "profile JSON")
	out := flag.String("out", "", "new result directory")
	flag.Parse()
	p, err := kvruntime.LoadProfile(*profilePath)
	if err != nil {
		return err
	}
	if p.InputPipeline == nil {
		return fmt.Errorf("missing input profile")
	}
	if err = os.Mkdir(*out, 0755); err != nil {
		return err
	}
	var rows []map[string]any
	for count := int64(0); count <= 40; count++ {
		var records []kvruntime.Record
		e := kvruntime.NewEngine(func(r kvruntime.Record) { records = append(records, r) })
		m, _ := kvruntime.NewMemory(e, 256, map[string]int64{"gpu": 1 << 20, "host": 640})
		r, _ := kvruntime.NewRuntime(e, m, 8)
		var pages []string
		for n := 0; n < 40; n++ {
			pages = append(pages, fmt.Sprintf("page/%d", n))
		}
		i, err := kvruntime.NewRequestInputs(r, *p.InputPipeline, "gpu", "host", pages, []kvruntime.InputRequest{{ID: "req", Prompt: []uint64{7}, Output: []uint64{12345, 54321}}})
		if err != nil {
			return err
		}
		m.ReserveGranular("token", "gpu", "fixture_producer", 512, 8)
		m.Allocate("token")
		m.Fill("token", 12345)
		var selected []string
		for n := int64(0); n < max(1, count); n++ {
			selected = append(selected, pages[(17*n)%40])
		}
		binding, _, err := i.Bind("step", "req", 0, 1, selected, nil)
		if err != nil {
			return err
		}
		if err = e.Run(); err != nil {
			return err
		}
		wall := e.NowNS
		kind := "index_bind"
		bytes := count * 8
		if err = binding.AcquireQuery(); err != nil {
			return err
		}
		if err = binding.ReleaseQuery(); err != nil {
			return err
		}
		if count == 0 {
			kind = "next_input_d2d"
			bytes = 8
			begin := e.NowNS
			if _, err = binding.ObserveOutput("token", 12345, nil); err != nil {
				return err
			}
			if err = e.Run(); err != nil {
				return err
			}
			wall = e.NowNS - begin
		}
		if err = binding.ReleaseIndices(); err != nil {
			return err
		}
		if err = m.Check(); err != nil {
			return err
		}
		name := fmt.Sprintf("%s-%02d", kind, count)
		f, err := os.Create(filepath.Join(*out, name+".events.jsonl.gz"))
		if err != nil {
			return err
		}
		z := gzip.NewWriter(f)
		enc := json.NewEncoder(z)
		for _, record := range records {
			if err = enc.Encode(record); err != nil {
				return err
			}
		}
		if err = z.Close(); err != nil {
			return err
		}
		if err = f.Close(); err != nil {
			return err
		}
		rows = append(rows, map[string]any{"kind": kind, "count": count, "bytes": bytes, "wall_ns": wall, "events": len(records), "trace": name + ".events.jsonl.gz"})
	}
	data, err := json.MarshalIndent(map[string]any{"rows": rows, "scope": "Warm physical request inputs and real index H2D/next-token D2D dependencies. All primitives use the same RequestInputs/Transfer implementation as BLIS. The token producer is a fixture, not neural execution."}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(*out, "summary.json"), data, 0644)
}
