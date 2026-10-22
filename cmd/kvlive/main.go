// kvlive is a local JSON-lines policy service. Hardware execution is owned by
// the native Python driver; this process executes the actual BLIS controller.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/inference-sim/inference-sim/sim/policylab"
	"github.com/sirupsen/logrus"
)

func run() error {
	timingPath := flag.String("timing-log", "", "optional protocol boundary timings, written after final reply")
	flag.Parse()
	logrus.SetLevel(logrus.ErrorLevel)
	decoder := json.NewDecoder(os.Stdin)
	encoder := json.NewEncoder(os.Stdout)
	var reader *timedReader
	var writer *timedWriter
	var timings []protocolTiming
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
		reply, e := controller.Poll(poll)
		if e != nil {
			return e
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
