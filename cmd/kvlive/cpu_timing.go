package main

import (
	"encoding/json"
	"os"
	"runtime"
)

// CPU attribution is an opt-in diagnostic. It locks the calling goroutine to
// its current OS thread only while measuring Poll. Sampling and this constraint
// can perturb timing; paired plain runs remain the service-time reference.
type cpuSample struct {
	WallNS, ProcessNS, ThreadNS int64
	Voluntary, Involuntary      int64
	TID                         int
}

type cpuPollTiming struct {
	Sequence            int   `json:"sequence"`
	StartNS             int64 `json:"start_ns"`
	EndNS               int64 `json:"end_ns"`
	ProcessCPUNS        int64 `json:"process_cpu_ns"`
	ThreadCPUNS         int64 `json:"thread_cpu_ns"`
	VoluntarySwitches   int64 `json:"voluntary_switches"`
	InvoluntarySwitches int64 `json:"involuntary_switches"`
	TID                 int   `json:"tid"`
}

func cpuDifference(sequence int, before, after cpuSample) cpuPollTiming {
	return cpuPollTiming{Sequence: sequence, StartNS: before.WallNS, EndNS: after.WallNS,
		ProcessCPUNS: after.ProcessNS - before.ProcessNS, ThreadCPUNS: after.ThreadNS - before.ThreadNS,
		VoluntarySwitches: after.Voluntary - before.Voluntary, InvoluntarySwitches: after.Involuntary - before.Involuntary, TID: before.TID}
}

func saveCPUTimings(path string, baseline, rows []cpuPollTiming) error {
	data, err := json.MarshalIndent(map[string]any{
		"go_version": runtime.Version(), "gomaxprocs": runtime.GOMAXPROCS(0), "num_cpu": runtime.NumCPU(),
		"clock":    "CLOCK_MONOTONIC/CLOCK_PROCESS_CPUTIME_ID/CLOCK_THREAD_CPUTIME_ID",
		"scope":    "Opt-in Linux Poll attribution with LockOSThread and thread rusage; clocks and rusage add measured overhead. Process CPU includes concurrent Go runtime threads, so wall minus process CPU is not off-CPU wait. Thread CPU and wall endpoints have bounded sampling skew; do not clamp negative differences or use as uninstrumented calibration. Baseline is back-to-back samples without policy work; paired plain runs measure perturbation.",
		"baseline": baseline, "rows": rows,
	}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
