package policylab

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"runtime/pprof"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim/kv"
)

type overheadOutcome struct {
	FirstTokenUS map[string]int64 `json:"first_token_us"`
	FinishedUS   map[string]int64 `json:"finished_us"`
	ArrivalUS    map[string]int64 `json:"arrival_us"`
	CacheMetrics CacheMetrics     `json:"cache_metrics"`
}

type overheadReport struct {
	Mode            string                              `json:"mode"`
	OutcomeSHA256   string                              `json:"outcome_sha256"`
	Requests        int                                 `json:"requests"`
	Parity          bool                                `json:"parity"`
	WallNS          int64                               `json:"run_wall_ns"`
	ProcessCPUNS    int64                               `json:"run_process_cpu_ns"`
	AllocatedBytes  uint64                              `json:"run_allocated_bytes"`
	Allocations     uint64                              `json:"run_allocations"`
	GCs             uint32                              `json:"run_gc_count"`
	GCPauseNS       uint64                              `json:"run_gc_pause_ns"`
	HeapRetained    uint64                              `json:"heap_with_metadata_bytes"`
	HeapReleased    uint64                              `json:"heap_without_metadata_bytes"`
	ObjectsRetained uint64                              `json:"objects_with_metadata"`
	ObjectsReleased uint64                              `json:"objects_without_metadata"`
	MetadataBytes   int64                               `json:"terminal_metadata_bytes"`
	MetadataObjects int64                               `json:"terminal_metadata_objects"`
	Counts          []map[string]int64                  `json:"metadata_counts,omitempty"`
	Methods         map[string]*kv.HotPrefixDiagnostics `json:"methods,omitempty"`
	EventCounts     map[string]int64                    `json:"event_counts"`
}

func TestHotPrefixObservationPreservesExecution(t *testing.T) {
	c := hotPrefixLabConfig()
	a, err := Run(c)
	if err != nil {
		t.Fatal(err)
	}
	called := false
	b, err := runWithObservation(c, PolicyFactories{}, &runObservation{discardEvents: true, afterRun: func(stores []*kv.PeerCache) {
		called = true
		for _, s := range stores {
			r, err := s.CaptureHotPrefixMemoryRoots()
			if err != nil || r.Metadata == nil || r.Counts["request_records"] != int64(len(c.Requests)) {
				t.Fatalf("bad capture: %+v %v", r.Counts, err)
			}
		}
	}})
	if err != nil || !called || len(a.Events) == 0 || len(b.Events) != 0 {
		t.Fatalf("observation did not change retention only: %v", err)
	}
	if !reflect.DeepEqual(a.Counts, b.Counts) || !reflect.DeepEqual(a.FirstTokenUS, b.FirstTokenUS) ||
		!reflect.DeepEqual(a.FinishedUS, b.FinishedUS) || !reflect.DeepEqual(a.ArrivalUS, b.ArrivalUS) ||
		!reflect.DeepEqual(a.CacheMetrics, b.CacheMetrics) || !reflect.DeepEqual(a.HBM, b.HBM) || !reflect.DeepEqual(a.Pools, b.Pools) {
		t.Fatal("observation altered execution")
	}
}

// This entry point is opt-in and compiled into a separate test executable for
// each frozen profiling experiment. Every invocation runs the full workload.
func TestHotPrefixOverheadMeasurement(t *testing.T) {
	path := os.Getenv("KV_PROFILE_CONFIG")
	if path == "" {
		t.Skip("serial experiment entry point; KV_PROFILE_CONFIG is unset")
	}
	if runtime.GOMAXPROCS(0) != 1 || os.Getenv("GOGC") != "100" {
		t.Fatal("requires GOMAXPROCS=1 and GOGC=100")
	}
	mode, output := os.Getenv("KV_PROFILE_MODE"), os.Getenv("KV_PROFILE_OUTPUT")
	if (mode != "cpu" && mode != "memory") || output == "" {
		t.Fatal("requires cpu/memory mode and output path")
	}
	report, metadata, shared := overheadRun(t, path, mode, output)
	if mode == "memory" {
		runtime.GC()
		runtime.GC()
		var held, released runtime.MemStats
		runtime.ReadMemStats(&held)
		runtime.KeepAlive(metadata)
		metadata = nil
		runtime.GC()
		runtime.GC()
		runtime.ReadMemStats(&released)
		runtime.KeepAlive(shared)
		report.HeapRetained, report.HeapReleased = held.HeapAlloc, released.HeapAlloc
		report.ObjectsRetained, report.ObjectsReleased = held.HeapObjects, released.HeapObjects
		report.MetadataBytes = int64(held.HeapAlloc) - int64(released.HeapAlloc)
		report.MetadataObjects = int64(held.HeapObjects) - int64(released.HeapObjects)
	}
	f, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	err = json.NewEncoder(f).Encode(report)
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("write report: %v %v", err, closeErr)
	}
}

// Keep the execution stack out of the retained-heap comparison. Neither Config
// nor Result (and therefore no physical KV or request text) escapes this call.
//
//go:noinline
func overheadRun(t *testing.T, path, mode, output string) (overheadReport, []any, []any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	if c.HotPrefix == nil || c.TraceBatchShapes || c.DecisionPolicy != nil && c.DecisionPolicy.TraceEvents ||
		mode == "memory" && c.HotPrefix.Diagnostics != nil ||
		mode == "cpu" && (c.HotPrefix.Diagnostics == nil || c.HotPrefix.Diagnostics.TraceCandidates || !c.HotPrefix.Diagnostics.MeasureCPU) {
		t.Fatal("profiling configuration does not match retention protocol")
	}
	var metadata, shared []any
	report := overheadReport{Mode: mode}
	obs := &runObservation{discardEvents: true}
	if mode == "memory" {
		obs.afterRun = func(stores []*kv.PeerCache) {
			for _, s := range stores {
				r, err := s.CaptureHotPrefixMemoryRoots()
				if err != nil {
					t.Fatal(err)
				}
				metadata = append(metadata, r.Metadata)
				shared = append(shared, r.SharedModel)
				report.Counts = append(report.Counts, r.Counts)
			}
		}
	}
	runtime.GC()
	var profile *os.File
	if mode == "cpu" {
		profile, err = os.OpenFile(output+".cpu.pprof", os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if err = pprof.StartCPUProfile(profile); err != nil {
			t.Fatal(err)
		}
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	cpuStart, err := overheadProcessCPUNS()
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	out, err := runWithObservation(c, PolicyFactories{}, obs)
	report.WallNS = time.Since(start).Nanoseconds()
	cpuEnd, cpuErr := overheadProcessCPUNS()
	runtime.ReadMemStats(&after)
	if profile != nil {
		pprof.StopCPUProfile()
		if err := profile.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if err != nil || cpuErr != nil {
		t.Fatalf("run: %v CPU: %v", err, cpuErr)
	}
	report.ProcessCPUNS = cpuEnd - cpuStart
	report.AllocatedBytes, report.Allocations = after.TotalAlloc-before.TotalAlloc, after.Mallocs-before.Mallocs
	report.GCs, report.GCPauseNS = after.NumGC-before.NumGC, after.PauseTotalNs-before.PauseTotalNs
	actual := overheadOutcome{out.FirstTokenUS, out.FinishedUS, out.ArrivalUS, out.CacheMetrics}
	ref, err := os.ReadFile(os.Getenv("KV_PROFILE_REFERENCE"))
	if err != nil {
		t.Fatal(err)
	}
	var expected overheadOutcome
	if err := json.Unmarshal(ref, &expected); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatal("frozen first_token/finished/arrival/cache metrics parity failed")
	}
	if len(out.Events) != 0 || len(out.FinishedUS) != len(c.Requests) {
		t.Fatal("unexpected retained events or incomplete workload")
	}
	encoded, err := json.Marshal(actual)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(encoded)
	report.OutcomeSHA256, report.Parity, report.Requests = hex.EncodeToString(digest[:]), true, len(c.Requests)
	report.Methods, report.EventCounts = out.HotPrefixDiagnostics, out.Counts
	for _, d := range report.Methods {
		if len(d.Records) != 0 {
			t.Fatal(fmt.Errorf("unexpected candidate trace retention"))
		}
	}
	return report, metadata, shared
}
