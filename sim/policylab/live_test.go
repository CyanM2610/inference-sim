package policylab

import (
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
	"testing"
)

func liveConfig() Config {
	c := labConfig(1, "dram")
	c.Model = sim.ModelConfig{NumLayers: 28, HiddenDim: 3584, NumHeads: 28, NumKVHeads: 4, BytesPerParam: 2}
	c.MaxSequences = 1
	c.PrefillChunk = 128
	c.Instances[0].HBMBlocks = 6
	c.Mechanisms = &kv.PeerMechanisms{ReserveAtDispatch: true, RestoreWindow: 2, QueueAware: true}
	return c
}
func TestLiveBatchCannotFinishWithoutHardwareAcknowledgement(t *testing.T) {
	l, err := NewLiveController(liveConfig())
	if err != nil {
		t.Fatal(err)
	}
	a, err := l.Poll(LivePoll{})
	if err != nil || a.Batch == nil {
		t.Fatalf("initial command: %+v %v", a, err)
	}
	id := a.Batch.ID
	b, err := l.Poll(LivePoll{NowUS: 10000000})
	if err != nil {
		t.Fatal(err)
	}
	if b.Batch != nil || b.Done || len(b.Requests) != 0 || b.HBM["active_or_pinned"] == 0 {
		t.Fatal("elapsed clock fabricated a completion or released active pages")
	}
	token := sim.TokenID(12345)
	b, err = l.Poll(LivePoll{NowUS: 10000001, Batch: id, Token: &token})
	if err != nil {
		t.Fatal(err)
	}
	if b.Batch == nil || b.Batch.Prefix != 65 || b.Batch.Query != 1 {
		t.Fatalf("hardware completion did not produce dependent decode: %+v", b)
	}
	if len(b.Batch.Tokens) != 1 || b.Batch.Tokens[0] != token {
		t.Fatal("next decode did not consume the actual observed hardware token")
	}
	if _, err = l.Poll(LivePoll{NowUS: 10000002, Batch: id}); err == nil {
		t.Fatal("duplicate hardware completion accepted")
	}
}
func TestLiveControllerExecutesRealRestoreAndReclaimPolicies(t *testing.T) {
	l, err := NewLiveController(liveConfig())
	if err != nil {
		t.Fatal(err)
	}
	p := LivePoll{}
	stores, restores, batches := 0, 0, 0
	for i := 0; i < 2000; i++ {
		r, err := l.Poll(p)
		if err != nil {
			t.Fatal(err)
		}
		if r.Done {
			if len(r.Requests) != 6 || r.HBM["active_or_pinned"] != 0 || stores == 0 || restores == 0 || batches < 6 {
				t.Fatalf("incomplete live fixture: %+v stores=%d restores=%d", r, stores, restores)
			}
			return
		}
		p = LivePoll{NowUS: p.NowUS + 1000}
		if r.Batch != nil {
			p.Batch = r.Batch.ID
			if r.Batch.Prefix+r.Batch.Query >= 65 {
				token := sim.TokenID(1000 + r.Batch.ID)
				p.Token = &token
			}
			batches++
		}
		for _, x := range r.Copies {
			p.Copies = append(p.Copies, x.ID)
			if x.Destination == "" || x.Source == "" || x.Bytes != 917504 {
				t.Fatal("unbound physical transfer")
			}
			if x.Reason == "store" {
				stores++
			}
			if x.Reason == "restore" {
				restores++
			}
		}
		if p.Batch == 0 && len(p.Copies) == 0 && r.NextEventUS > p.NowUS {
			p.NowUS = r.NextEventUS
		}
	}
	t.Fatal("live hardware completion loop did not terminate")
}

func TestLiveTTFTTimestampIncludesNonzeroArrival(t *testing.T) {
	c := liveConfig()
	c.Requests = c.Requests[:1]
	c.Requests[0].At = 1000
	l, err := NewLiveController(c)
	if err != nil {
		t.Fatal(err)
	}
	r, err := l.Poll(LivePoll{NowUS: 1000})
	if err != nil || r.Batch == nil {
		t.Fatal("initial batch missing", err)
	}
	token := sim.TokenID(9)
	r, err = l.Poll(LivePoll{NowUS: 1023, Batch: r.Batch.ID, Token: &token})
	if err != nil || r.Batch == nil {
		t.Fatal("decode missing", err)
	}
	token = 10
	r, err = l.Poll(LivePoll{NowUS: 1067, Batch: r.Batch.ID, Token: &token})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Done || len(r.Requests) != 1 || r.Requests[0].FirstTokenUS != 1023 || r.Requests[0].FinishedUS != 1067 {
		t.Fatalf("TTFT duration was confused with an absolute timestamp: %+v", r)
	}
}
