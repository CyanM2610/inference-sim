//go:build linux

package main

import (
	"fmt"
	"runtime"

	"golang.org/x/sys/unix"
)

func sampleCPU() (cpuSample, error) {
	var process, thread unix.Timespec
	var usage unix.Rusage
	if err := unix.Getrusage(unix.RUSAGE_THREAD, &usage); err != nil {
		return cpuSample{}, err
	}
	if err := unix.ClockGettime(unix.CLOCK_PROCESS_CPUTIME_ID, &process); err != nil {
		return cpuSample{}, err
	}
	if err := unix.ClockGettime(unix.CLOCK_THREAD_CPUTIME_ID, &thread); err != nil {
		return cpuSample{}, err
	}
	return cpuSample{WallNS: clockNS(), ProcessNS: process.Nano(), ThreadNS: thread.Nano(),
		Voluntary: usage.Nvcsw, Involuntary: usage.Nivcsw, TID: unix.Gettid()}, nil
}

func beginCPUPoll() (cpuSample, error) {
	runtime.LockOSThread()
	sample, err := sampleCPU()
	if err != nil {
		runtime.UnlockOSThread()
	}
	return sample, err
}

func endCPUPoll(sequence int, before cpuSample) (cpuPollTiming, error) {
	defer runtime.UnlockOSThread()
	after, err := sampleCPU()
	if err != nil {
		return cpuPollTiming{}, err
	}
	if before.TID != after.TID || after.ThreadNS < before.ThreadNS || after.ProcessNS < before.ProcessNS || after.WallNS < before.WallNS {
		return cpuPollTiming{}, fmt.Errorf("invalid CPU attribution clock/thread transition")
	}
	return cpuDifference(sequence, before, after), nil
}
