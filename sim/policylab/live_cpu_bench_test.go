package policylab

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/sirupsen/logrus"
)

// BenchmarkLiveControllerReplay attributes CPU work in a verified native ACK
// fixture. It measures local Go work only, never GPU or cross-host performance.
// Set KV_CPU_REPLAY_DIR to an existing CPU-probe run; no fixture is generated.
func BenchmarkLiveControllerReplay(b *testing.B) {
	dir := os.Getenv("KV_CPU_REPLAY_DIR")
	if dir == "" {
		b.Skip("set KV_CPU_REPLAY_DIR to a captured controller CPU fixture")
	}
	data, err := os.ReadFile(filepath.Join(dir, "initial-input.json"))
	if err != nil {
		b.Fatal(err)
	}
	var config Config
	if err = json.Unmarshal(data, &config); err != nil {
		b.Fatal(err)
	}
	f, err := os.Open(filepath.Join(dir, "exchanges.jsonl"))
	if err != nil {
		b.Fatal(err)
	}
	var polls []LivePoll
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		var row struct {
			Poll LivePoll `json:"poll"`
		}
		if err = json.Unmarshal(scanner.Bytes(), &row); err != nil {
			b.Fatal(err)
		}
		polls = append(polls, row.Poll)
	}
	if err = scanner.Err(); err != nil {
		b.Fatal(err)
	}
	f.Close()
	if len(polls) == 0 {
		b.Fatal("empty CPU fixture")
	}
	old := logrus.GetLevel()
	logrus.SetLevel(logrus.ErrorLevel)
	defer logrus.SetLevel(old)
	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		controller, err := NewLiveController(config)
		if err != nil {
			b.Fatal(err)
		}
		var reply *LiveReply
		for _, p := range polls {
			reply, err = controller.Poll(p)
			if err != nil {
				b.Fatal(err)
			}
		}
		if !reply.Done || len(reply.Requests) != len(config.Requests) {
			b.Fatal("replay did not complete")
		}
	}
}
