package kvruntime

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

type TransferPoint struct {
	NUMA        int              `json:"numa"`
	Layout      string           `json:"layout"`
	HostMemory  string           `json:"host_memory"`
	Blocks      int64            `json:"blocks"`
	Direction   string           `json:"direction"`
	Copies      int              `json:"copies"`
	Cost        TransferCost     `json:"cost"`
	ReferenceNS map[string]int64 `json:"reference_ns"`
	Evidence    string           `json:"evidence"`
}

func (p TransferPoint) Key() string {
	return fmt.Sprintf("n%d/%s/%s/b%d/%s", p.NUMA, p.Layout, p.HostMemory, p.Blocks, p.Direction)
}

type AllocationPoint struct {
	NUMA       int              `json:"numa"`
	Layout     string           `json:"layout"`
	HostMemory string           `json:"host_memory"`
	Blocks     int64            `json:"blocks"`
	StagesNS   map[string]int64 `json:"stages_ns"`
}
type ComputePoint struct {
	Bridge                           *BridgePoint               `json:"bridge,omitempty"`
	ForwardWeights                   [][]float64                `json:"forward_weights"`
	BaselineAllocatedBytes           int64                      `json:"baseline_allocated_bytes"`
	ContextInitNS                    int64                      `json:"context_init_ns"`
	ModelLoadNS                      int64                      `json:"model_load_ns"`
	PoolSetupNS                      int64                      `json:"pool_setup_ns"`
	PrefillReferenceToken            int64                      `json:"prefill_reference_token"`
	TransactionResidualNS            int64                      `json:"transaction_residual_ns"`
	LoopResidualNS                   int64                      `json:"loop_residual_ns"`
	StructuralEvidence               string                     `json:"structural_evidence"`
	Length                           int64                      `json:"length"`
	Mode                             string                     `json:"mode"`
	InputCopyNS                      int64                      `json:"input_copy_ns"`
	PrefillNS                        int64                      `json:"prefill_ns"`
	DecodeNS                         []int64                    `json:"decode_ns"`
	CleanupNS                        int64                      `json:"cleanup_ns"`
	StagesNS                         map[string]json.RawMessage `json:"stages_ns"`
	ReferenceTokens                  []int64                    `json:"reference_tokens"`
	ReferenceTokensAreNotPredictions bool                       `json:"reference_tokens_are_not_predictions"`
}
type BridgeStage struct {
	WallNS    int64 `json:"wall_ns"`
	SubmitNS  int64 `json:"submit_ns"`
	ServiceNS int64 `json:"service_ns"`
	ObserveNS int64 `json:"observe_ns"`
}
type BridgePoint struct {
	Stages              map[string]BridgeStage  `json:"stages"`
	Transfers           map[string]TransferCost `json:"transfers"`
	ResidualNS          int64                   `json:"residual_ns"`
	PrefillStorageBytes int64                   `json:"prefill_storage_bytes"`
	Evidence            string                  `json:"evidence"`
}
type Profile struct {
	ComputeGrid        []ComputeShape    `json:"compute_grid,omitempty"`
	Version            int               `json:"schema_version"`
	Condition          string            `json:"condition"`
	QueueDepth         int               `json:"queue_depth"`
	QueueDepthEvidence string            `json:"queue_depth_evidence"`
	Train              []int             `json:"train_processes"`
	HeldOut            []int             `json:"held_out_processes"`
	SourceSHA256       map[string]string `json:"source_sha256"`
	Transfers          []TransferPoint   `json:"transfers"`
	Allocations        []AllocationPoint `json:"allocations"`
	Compute            []ComputePoint    `json:"compute"`
}
type ComputeShape struct {
	Prefix              int64 `json:"prefix"`
	Query               int64 `json:"query"`
	WallNS              int64 `json:"wall_ns"`
	PeakAdditionalBytes int64 `json:"peak_additional_bytes"`
}

func LoadProfile(path string) (*Profile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p Profile
	if err = json.Unmarshal(b, &p); err != nil {
		return nil, err
	}
	if err = p.Validate(); err != nil {
		return nil, err
	}
	return &p, nil
}
func (p *Profile) Validate() error {
	if p.Version != 1 || p.QueueDepth <= 0 || len(p.SourceSHA256) == 0 {
		return fmt.Errorf("invalid profile version, queue depth or provenance")
	}
	for _, a := range p.Train {
		for _, b := range p.HeldOut {
			if a == b {
				return fmt.Errorf("training/held-out overlap")
			}
		}
	}
	seen := map[string]bool{}
	for _, x := range p.Transfers {
		if seen[x.Key()] || x.NUMA < 0 || x.NUMA > 1 {
			return fmt.Errorf("duplicate/invalid transfer point")
		}
		seen[x.Key()] = true
		if x.Direction != "d2h" && x.Direction != "h2d" {
			return fmt.Errorf("invalid direction")
		}
		if x.HostMemory != "pinned" && x.HostMemory != "pageable" && x.HostMemory != "registered" {
			return fmt.Errorf("invalid host memory")
		}
		spans, _, _, err := NativeSpans(x.Layout, x.Blocks, false)
		if err != nil {
			return err
		}
		if len(spans) != x.Copies {
			return fmt.Errorf("copy count does not match layout")
		}
		if err = x.Cost.Validate(); err != nil {
			return err
		}
	}
	for _, x := range p.Allocations {
		for _, ns := range x.StagesNS {
			if ns < 0 {
				return fmt.Errorf("negative allocation profile")
			}
		}
	}
	shapes := map[[2]int64]bool{}
	for _, x := range p.ComputeGrid {
		key := [2]int64{x.Prefix, x.Query}
		if x.Prefix < 0 || x.Query <= 0 || x.WallNS <= 0 || x.PeakAdditionalBytes < 0 || shapes[key] {
			return fmt.Errorf("invalid/duplicate compute shape")
		}
		shapes[key] = true
	}
	for _, c := range p.Compute {
		if c.InputCopyNS < 0 || c.PrefillNS <= 0 || c.CleanupNS < 0 || len(c.DecodeNS) != 8 || len(c.ReferenceTokens) != 8 {
			return fmt.Errorf("invalid model compute profile")
		}
		for _, ns := range c.DecodeNS {
			if ns <= 0 {
				return fmt.Errorf("nonpositive decode service")
			}
		}
		if len(c.ForwardWeights) > 0 {
			if len(c.ForwardWeights) != 9 {
				return fmt.Errorf("expected nine forward profiles")
			}
			for _, ws := range c.ForwardWeights {
				if len(ws) != 30 {
					return fmt.Errorf("expected 30 forward intervals")
				}
				sum := 0.0
				for _, w := range ws {
					if w < 0 || math.IsNaN(w) || math.IsInf(w, 0) {
						return fmt.Errorf("invalid structural weight")
					}
					sum += w
				}
				if math.Abs(sum-1) > 1e-9 {
					return fmt.Errorf("structural fractions must sum to one")
				}
			}
		}
	}
	return nil
}
func (p *Profile) Find(numa int, layout, host string, blocks int64, direction string) (TransferPoint, error) {
	for _, x := range p.Transfers {
		if x.NUMA == numa && x.Layout == layout && x.HostMemory == host && x.Blocks == blocks && x.Direction == direction {
			return x, nil
		}
	}
	return TransferPoint{}, fmt.Errorf("unmeasured transfer shape n%d %s %s blocks=%d %s; extrapolation is not enabled", numa, layout, host, blocks, direction)
}
func (p *Profile) Allocation(x TransferPoint) (AllocationPoint, error) {
	for _, a := range p.Allocations {
		if a.NUMA == x.NUMA && a.Layout == x.Layout && a.HostMemory == x.HostMemory && a.Blocks == x.Blocks {
			return a, nil
		}
	}
	return AllocationPoint{}, fmt.Errorf("missing allocation costs for %s", x.Key())
}
