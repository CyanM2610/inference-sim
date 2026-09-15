// kvsubmission evaluates detached submission views outside measured execution.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/inference-sim/inference-sim/sim/kv"
)

type input struct {
	Mode string                       `json:"mode"`
	View kv.TransferSubmissionContext `json:"view"`
}

func run(in io.Reader, out io.Writer) error {
	decoder := json.NewDecoder(in)
	decoder.DisallowUnknownFields()
	encoder := json.NewEncoder(out)
	for {
		var row input
		if err := decoder.Decode(&row); err == io.EOF {
			return nil
		} else if err != nil {
			return err
		}
		policy, err := kv.NewTransferSubmissionPolicy(row.Mode)
		if err != nil {
			return err
		}
		if err := encoder.Encode(struct {
			Order []int64 `json:"order"`
		}{policy.Order(row.View)}); err != nil {
			return err
		}
	}
}

func main() {
	if err := run(os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
