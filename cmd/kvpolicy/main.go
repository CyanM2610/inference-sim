// kvpolicy is the standalone BLIS peer-L2 experiment command.
package main

import (
	"compress/gzip"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/inference-sim/inference-sim/sim/policylab"
	"github.com/sirupsen/logrus"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func writeJSON(path string, v any) error {
	f, e := os.Create(path)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
func run() error {
	config := flag.String("config", "", "experiment JSON")
	output := flag.String("out", "", "result directory")
	requestPlugin := flag.String("request-plugin", "", "registered request policy; empty retains config")
	peerPlugin := flag.String("kv-plugin", "", "registered HBM reclaim/save policy; empty retains config")
	poolPlugin := flag.String("pool-plugin", "", "registered pool eviction policy; empty retains config")
	transferPlugin := flag.String("transfer-plugin", "", "registered same-phase host submission policy; empty retains config")
	flag.Parse()
	if *config == "" || *output == "" {
		return fmt.Errorf("usage: kvpolicy -config experiment.json -out result-directory")
	}
	data, err := os.ReadFile(*config)
	if err != nil {
		return err
	}
	c, err := policylab.Decode(data)
	if err != nil {
		return err
	}
	logrus.SetLevel(logrus.ErrorLevel)
	factories, err := policyFactories(*requestPlugin, *peerPlugin, *poolPlugin, *transferPlugin)
	if err != nil {
		return err
	}
	r, runErr := policylab.RunWithPolicies(c, factories)
	if r == nil {
		return runErr
	}
	if err = os.MkdirAll(*output, 0755); err != nil {
		return err
	}
	if *requestPlugin != "" || *peerPlugin != "" || *poolPlugin != "" || *transferPlugin != "" {
		selection := map[string]string{"request": *requestPlugin, "kv": *peerPlugin, "pool": *poolPlugin}
		if *transferPlugin != "" {
			selection["transfer_submission"] = *transferPlugin
		}
		if err = writeJSON(filepath.Join(*output, "policy-selection.json"), selection); err != nil {
			return err
		}
	}
	for name, v := range map[string]any{"summary.json": r, "profile_cpu.json": r.CPU, "routing.json": r.RoutingTrace, "config.json": c} {
		if err = writeJSON(filepath.Join(*output, name), v); err != nil {
			return err
		}
	}
	if r.TransferSubmissionCostCoverage != "" {
		f, err := os.Create(filepath.Join(*output, "transfer-submissions.jsonl"))
		if err != nil {
			return err
		}
		enc := json.NewEncoder(f)
		for _, record := range r.TransferSubmissions {
			if err := enc.Encode(record); err != nil {
				f.Close()
				return err
			}
		}
		if err := f.Close(); err != nil {
			return err
		}
	}
	if c.DecisionPolicy != nil {
		if r.PostStepCostCoverage != "" {
			f, err := os.Create(filepath.Join(*output, "post-step-services.jsonl"))
			if err != nil {
				return err
			}
			enc := json.NewEncoder(f)
			for _, record := range r.PostStepServices {
				if err := enc.Encode(record); err != nil {
					f.Close()
					return err
				}
			}
			if err := f.Close(); err != nil {
				return err
			}
		}
		if r.HostServiceCostCoverage != "" {
			f, err := os.Create(filepath.Join(*output, "host-services.jsonl"))
			if err != nil {
				return err
			}
			enc := json.NewEncoder(f)
			for _, record := range r.HostServices {
				if err := enc.Encode(record); err != nil {
					f.Close()
					return err
				}
			}
			if err := f.Close(); err != nil {
				return err
			}
		}
		f, err := os.Create(filepath.Join(*output, "policy-decisions.jsonl"))
		if err != nil {
			return err
		}
		enc := json.NewEncoder(f)
		for _, record := range r.PolicyDecisions {
			if err = enc.Encode(record); err != nil {
				f.Close()
				return err
			}
		}
		if err = f.Close(); err != nil {
			return err
		}
	}
	if r.DecisionEventCostCoverage != "" {
		f, err := os.Create(filepath.Join(*output, "policy-events.jsonl"))
		if err != nil {
			return err
		}
		enc := json.NewEncoder(f)
		for _, event := range r.PolicyEvents {
			if err := enc.Encode(event); err != nil {
				f.Close()
				return err
			}
		}
		if err := f.Close(); err != nil {
			return err
		}
	}
	if c.Native != nil {
		file, err := os.Create(filepath.Join(*output, "native.events.jsonl.gz"))
		if err != nil {
			return err
		}
		gz, _ := gzip.NewWriterLevel(file, gzip.BestSpeed)
		encoder := json.NewEncoder(gz)
		for _, e := range r.NativeEvents {
			if err := encoder.Encode(e); err != nil {
				gz.Close()
				file.Close()
				return err
			}
		}
		if err := gz.Close(); err != nil {
			file.Close()
			return err
		}
		if err := file.Close(); err != nil {
			return err
		}
	}
	f, err := os.Create(filepath.Join(*output, "events.jsonl"))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	var chrome []map[string]any
	lanes := map[string]int{}
	laneID := func(name string) int {
		id, ok := lanes[name]
		if !ok {
			id = len(lanes) + 1
			lanes[name] = id
			chrome = append(chrome, map[string]any{"name": "thread_name", "ph": "M", "pid": 1, "tid": id, "args": map[string]string{"name": name}})
		}
		return id
	}
	for _, e := range r.Events {
		if err = enc.Encode(e); err != nil {
			f.Close()
			return err
		}
		tid := e.Instance
		if tid == "" {
			tid = "fabric"
		}
		ph := "i"
		name := e.Name
		x := map[string]any{"name": name, "ts": e.Time, "pid": 1, "tid": tid, "ph": ph, "s": "t", "args": e}
		if name == "compute" {
			x["ph"] = "X"
			x["dur"] = e.Duration
			delete(x, "s")
		}
		if name == "transfer_start" {
			x["ph"] = "X"
			x["dur"] = e.Duration
			x["tid"] = e.Instance + ":" + e.Source + "→" + e.Destination
			delete(x, "s")
		}
		if name == "control_start" {
			x["ph"] = "X"
			x["dur"] = e.Duration
			x["tid"] = e.Source
			delete(x, "s")
		}
		if name == "hbm_capacity" || name == "l2_capacity" || name == "control_capacity" {
			x["ph"] = "C"
			x["args"] = e.Counters
			if name == "l2_capacity" || name == "control_capacity" {
				x["tid"] = e.Destination
			}
			delete(x, "s")
		}
		x["tid"] = laneID(x["tid"].(string))
		chrome = append(chrome, x)
	}
	if err = f.Close(); err != nil {
		return err
	}
	if err = writeJSON(filepath.Join(*output, "timeline.json"), map[string]any{"traceEvents": chrome, "displayTimeUnit": "ms"}); err != nil {
		return err
	}
	f, err = os.Create(filepath.Join(*output, "requests.csv"))
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	w.Write([]string{"id", "instance", "arrival_us", "ttft_ms", "e2e_ms", "input_tokens", "output_tokens"})
	for _, q := range r.Requests {
		w.Write([]string{q.ID, q.HandledBy, strconv.FormatFloat(q.ArrivedAt*1e6, 'f', 3, 64), strconv.FormatFloat(q.TTFT, 'f', 6, 64), strconv.FormatFloat(q.E2E, 'f', 6, 64), strconv.Itoa(q.NumPrefillTokens), strconv.Itoa(q.NumDecodeTokens)})
	}
	w.Flush()
	err = w.Error()
	f.Close()
	if err != nil {
		return err
	}
	fmt.Printf("completed=%d events=%d block_bytes=%d results=%s\n", len(r.Requests), len(r.Events), r.BlockBytes, *output)
	return runErr
}
