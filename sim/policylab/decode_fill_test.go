package policylab

import (
	"encoding/json"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func decodeFillConfig() Config {
	c := promotionPolicyConfig()
	c.PromotionRetention = "request"
	c.RestoreControl = true
	c.Instances[0].HBMBlocks = 32
	c.MaxBatchTokens = 32
	c.PrefillChunk = 8
	c.DecisionPolicy.PrefillDeferrals = true
	c.DecisionPolicy.BalancedBatch = &BalancedBatchConfig{MaxLoadComputeRatio: 1, DecodeFill: true}
	request := func(id string, at int64, n, group, output int) RequestConfig {
		input := make([]sim.TokenID, n)
		for i := range input {
			input[i] = sim.TokenID(group*1000 + i)
		}
		return RequestConfig{ID: id, At: at, Input: input, Output: make([]sim.TokenID, output), MaxOutputTokens: output}
	}
	c.Requests = []RequestConfig{request("seed", 0, 65, 1, 1), request("evict", 10000, 497, 2, 1),
		request("decode", 30000, 17, 3, 64), request("resident", 30500, 257, 4, 1),
		request("hot", 32000, 65, 1, 1), request("cold", 32000, 17, 5, 1)}
	return c
}

func TestDecodeFillPublicConfigUsesActualLoadFeedbackAndReleasesResources(t *testing.T) {
	c := decodeFillConfig()
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Config
	if err = json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	r, err := Run(decoded)
	if err != nil {
		t.Fatal(err)
	}
	var submitted, held, pendingDecode int
	var adopted int64
	for _, e := range r.Events {
		if e.Name == "transfer_adopted" && e.Reason == "promotion" {
			adopted = e.Time
		}
	}
	for _, d := range r.PolicyDecisions {
		for _, o := range d.Feedback.Promotions {
			submitted += int(o.StartedBlocks)
		}
		if len(d.Plan.PrefillDeferrals) > 0 {
			held++
			if len(d.Plan.Promotions) == 0 {
				pendingDecode++
			}
			for _, g := range d.Feedback.Grants {
				if g.Request != "decode" {
					t.Fatal("prefill computed during retained batch load", d)
				}
			}
		}
		for _, g := range d.Feedback.Grants {
			if g.Request == "hot" && (adopted == 0 || d.View.NowUS < adopted) {
				t.Fatal("hot computed before adoption", d, adopted)
			}
		}
	}
	if submitted != 4 || held == 0 || pendingDecode == 0 || len(r.FinishedUS) != 6 {
		t.Fatal("missing real load/decode/cohort progression", submitted, held, pendingDecode, r.FinishedUS)
	}
	if r.HBM["instance_0"]["active_or_pinned"] != 0 || r.Pools["dram"]["reserved"] != 0 || r.Pools["dram"]["read_pins"] != 0 {
		t.Fatal("resources leaked")
	}
	if r.DecisionAdditionalCostCoverage != "decode_fill_policy_and_control_not_independently_calibrated" {
		t.Fatal("new policy cost coverage missing")
	}
	decoded.DecisionPolicy.BalancedBatch.DecodeFill = false
	baseline, err := Run(decoded)
	if err != nil || len(baseline.FinishedUS) != 6 {
		t.Fatal("ordinary baseline failed", err)
	}
	for _, d := range baseline.PolicyDecisions {
		if len(d.Plan.Promotions) > 0 || len(d.Plan.PrefillDeferrals) > 0 {
			t.Fatal("disabled policy emitted actions")
		}
	}
}

func TestDecodeFillConfigRequiresItsDeclaredRuntimeActions(t *testing.T) {
	for _, missing := range []string{"deferral", "promotion", "retention", "probe"} {
		c := decodeFillConfig()
		switch missing {
		case "deferral":
			c.DecisionPolicy.PrefillDeferrals = false
		case "promotion":
			c.PromotionControl = false
		case "retention":
			c.PromotionRetention = "cache"
		case "probe":
			c.DecisionPolicy.PrefillDeferralProbe = &PrefillDeferralProbeConfig{}
		}
		if c.Validate() == nil {
			t.Fatal("unsupported combination accepted", missing)
		}
	}
}
