// kvlive is a local JSON-lines policy service. Hardware execution is owned by
// the native Python driver; this process executes the actual BLIS controller.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/policylab"
	"github.com/sirupsen/logrus"
)

func run() error {
	timingPath := flag.String("timing-log", "", "optional protocol boundary timings, written after final reply")
	hashPath := flag.String("hash-work-log", "", "optional deterministic hash counts; separate from service timing")
	cpuPath := flag.String("cpu-time-log", "", "optional Linux CPU attribution; intrusive diagnostic, pair with plain timing")
	flag.Parse()
	if *hashPath != "" && (*timingPath != "" || *cpuPath != "") {
		return fmt.Errorf("hash-work collection must be separated from service timing")
	}
	logrus.SetLevel(logrus.ErrorLevel)
	decoder := json.NewDecoder(os.Stdin)
	encoder := json.NewEncoder(os.Stdout)
	var reader *timedReader
	var writer *timedWriter
	var timings []protocolTiming
	var cpuBaseline, cpuTimings []cpuPollTiming
	if *cpuPath != "" {
		for i := 0; i < 100; i++ {
			before, err := beginCPUPoll()
			if err != nil {
				return err
			}
			row, err := endCPUPoll(i, before)
			if err != nil {
				return err
			}
			cpuBaseline = append(cpuBaseline, row)
		}
	}
	if *timingPath != "" {
		reader = &timedReader{reader: os.Stdin}
		writer = &timedWriter{writer: os.Stdout}
		decoder = json.NewDecoder(reader)
		encoder = json.NewEncoder(writer)
	}
	var config policylab.Config
	if err := decoder.Decode(&config); err != nil {
		return err
	}
	controller, err := policylab.NewLiveController(config)
	if err != nil {
		return err
	}
	var hashWork []kv.HashWork
	if *hashPath != "" {
		controller.EnableHashWorkAccounting()
	}
	if err = encoder.Encode(map[string]bool{"ready": true}); err != nil {
		return err
	}
	for {
		var timing protocolTiming
		if reader != nil {
			reader.calls = nil
			writer.calls = nil
			timing.Sequence = len(timings)
			timing.DecodeStartNS = clockNS()
		}
		var poll policylab.LivePoll
		if err = decoder.Decode(&poll); err != nil {
			return err
		}
		if reader != nil {
			timing.DecodeEndNS = clockNS()
			timing.PollStartNS = clockNS()
		}
		var cpuBefore cpuSample
		if *cpuPath != "" {
			cpuBefore, err = beginCPUPoll()
			if err != nil {
				return err
			}
		}
		reply, e := controller.Poll(poll)
		if *cpuPath != "" {
			row, cpuErr := endCPUPoll(len(cpuTimings), cpuBefore)
			if cpuErr != nil {
				return cpuErr
			}
			cpuTimings = append(cpuTimings, row)
		}
		if e != nil {
			return e
		}
		if *hashPath != "" {
			hashWork = append(hashWork, controller.TakeHashWork())
		}
		if reader != nil {
			timing.PollEndNS = clockNS()
			timing.EncodeStartNS = clockNS()
		}
		if err = encoder.Encode(reply); err != nil {
			return err
		}
		if reader != nil {
			timing.EncodeEndNS = clockNS()
			timing.Reads = reader.calls
			timing.Writes = writer.calls
			timing.Events = reply.EventsProcessed
			timing.Records = len(reply.Records)
			timing.CompletedRequests = len(reply.Requests)
			timing.CopyCommands = len(reply.Copies)
			timing.BatchCommand = reply.Batch != nil
			timings = append(timings, timing)
		}
		if reply.Done {
			if *cpuPath != "" {
				if err := saveCPUTimings(*cpuPath, cpuBaseline, cpuTimings); err != nil {
					return err
				}
			}
			if *hashPath != "" {
				data, err := json.MarshalIndent(hashWork, "", "  ")
				if err != nil {
					return err
				}
				return os.WriteFile(*hashPath, data, 0644)
			}
			if reader != nil {
				return saveProtocolTimings(*timingPath, timings)
			}
			return nil
		}
	}
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
