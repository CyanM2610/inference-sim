// kvlive is a local JSON-lines policy service. Hardware execution is owned by
// the native Python driver; this process executes the actual BLIS controller.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/inference-sim/inference-sim/sim/policylab"
	"github.com/sirupsen/logrus"
)

func run() error {
	logrus.SetLevel(logrus.ErrorLevel)
	decoder := json.NewDecoder(os.Stdin)
	encoder := json.NewEncoder(os.Stdout)
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
		var poll policylab.LivePoll
		if err = decoder.Decode(&poll); err != nil {
			return err
		}
		reply, e := controller.Poll(poll)
		if e != nil {
			return e
		}
		if err = encoder.Encode(reply); err != nil {
			return err
		}
		if reply.Done {
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
