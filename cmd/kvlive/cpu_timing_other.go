//go:build !linux

package main

import "fmt"

func beginCPUPoll() (cpuSample, error) {
	return cpuSample{}, fmt.Errorf("CPU attribution requires Linux process/thread clocks")
}

func endCPUPoll(_ int, _ cpuSample) (cpuPollTiming, error) {
	return cpuPollTiming{}, fmt.Errorf("CPU attribution requires Linux process/thread clocks")
}
