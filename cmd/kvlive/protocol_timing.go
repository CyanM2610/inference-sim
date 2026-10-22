package main

import (
	"encoding/json"
	"io"
	"os"
)

type ioTiming struct {
	StartNS int64 `json:"start_ns"`
	EndNS   int64 `json:"end_ns"`
	Bytes   int   `json:"bytes"`
}
type timedReader struct {
	reader io.Reader
	calls  []ioTiming
}

func (r *timedReader) Read(p []byte) (int, error) {
	start := clockNS()
	n, err := r.reader.Read(p)
	end := clockNS()
	r.calls = append(r.calls, ioTiming{start, end, n})
	return n, err
}

type timedWriter struct {
	writer io.Writer
	calls  []ioTiming
}

func (w *timedWriter) Write(p []byte) (int, error) {
	start := clockNS()
	n, err := w.writer.Write(p)
	end := clockNS()
	w.calls = append(w.calls, ioTiming{start, end, n})
	return n, err
}

type protocolTiming struct {
	Sequence          int        `json:"sequence"`
	DecodeStartNS     int64      `json:"decode_start_ns"`
	DecodeEndNS       int64      `json:"decode_end_ns"`
	PollStartNS       int64      `json:"poll_start_ns"`
	PollEndNS         int64      `json:"poll_end_ns"`
	EncodeStartNS     int64      `json:"encode_start_ns"`
	EncodeEndNS       int64      `json:"encode_end_ns"`
	Reads             []ioTiming `json:"reads"`
	Writes            []ioTiming `json:"writes"`
	Events            int        `json:"events"`
	Records           int        `json:"records"`
	CompletedRequests int        `json:"completed_requests"`
	CopyCommands      int        `json:"copy_commands"`
	BatchCommand      bool       `json:"batch_command"`
}

func saveProtocolTimings(path string, rows []protocolTiming) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	encoder := json.NewEncoder(f)
	if err = encoder.Encode(map[string]any{"clock": clockSource, "scope": "Optional diagnostic wrappers around the unchanged JSON decoder/encoder. Read calls include blocking and kernel copy; write calls include kernel/pipe work. Rows buffered in memory and written only after final reply. Not pure CPU cycles."}); err != nil {
		return err
	}
	for _, row := range rows {
		if err = encoder.Encode(row); err != nil {
			return err
		}
	}
	return nil
}
