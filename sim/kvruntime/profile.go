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
	InputPipeline      *InputPipelineProfile `json:"input_pipeline,omitempty"`
	TorchCopies        *TorchCopyProfile     `json:"torch_copies,omitempty"`
	PagedForward       []PagedForwardPoint   `json:"paged_forward,omitempty"`
	ComputeGrid        []ComputeShape        `json:"compute_grid,omitempty"`
	Version            int                   `json:"schema_version"`
	Condition          string                `json:"condition"`
	QueueDepth         int                   `json:"queue_depth"`
	QueueDepthEvidence string                `json:"queue_depth_evidence"`
	Train              []int                 `json:"train_processes"`
	HeldOut            []int                 `json:"held_out_processes"`
	SourceSHA256       map[string]string     `json:"source_sha256"`
	Transfers          []TransferPoint       `json:"transfers"`
	Allocations        []AllocationPoint     `json:"allocations"`
	Compute            []ComputePoint        `json:"compute"`
}
type TorchCopyProfile struct {
	GPUSlots               int64            `json:"gpu_slots"`
	HostSlots              int64            `json:"host_slots"`
	BaselineAllocatedBytes int64            `json:"baseline_allocated_bytes"`
	ReservedBytes          int64            `json:"reserved_bytes"`
	WarmSetupNS            map[string]int64 `json:"warm_setup_ns"`
	Points                 []TransferPoint  `json:"points"`
	Evidence               string           `json:"evidence"`
}
type PagedForwardPoint struct {
	// Optional measured CPU gaps before named phases and after cleanup
	// ("complete"). When present these replace the lumped ResidualNS placement.
	PhaseGapsNS          map[string]int64       `json:"phase_gaps_ns,omitempty"`
	Output               *OutputCopyPoint       `json:"output_copy,omitempty"`
	Prefix               int64                  `json:"prefix"`
	Query                int64                  `json:"query"`
	Stages               map[string]BridgeStage `json:"stages"`
	ResidualNS           int64                  `json:"residual_ns"`
	BeforeAllocatedBytes int64                  `json:"before_allocated_bytes"`
	ReservedBytes        int64                  `json:"reserved_bytes"`
	PhasePeakDelta       map[string]int64       `json:"phase_peak_delta_bytes"`
	PhaseEndDelta        map[string]int64       `json:"phase_end_delta_bytes"`
	Evidence             string                 `json:"evidence"`
}
type OutputCopyPoint struct {
	Cost              TransferCost `json:"cost"`
	GPUStorageBytes   int64        `json:"gpu_storage_bytes"`
	HostStorageBytes  int64        `json:"host_storage_bytes"`
	HostReservedBytes int64        `json:"host_reserved_bytes"`
	ReferenceToken    uint64       `json:"reference_token"`
	Evidence          string       `json:"evidence"`
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
	if p.InputPipeline != nil {
		if err := p.InputPipeline.Validate(); err != nil {
			return err
		}
	}
	if p.Version != 1 || p.QueueDepth <= 0 || len(p.SourceSHA256) == 0 {
		return fmt.Errorf("invalid profile version, queue depth or provenance")
	}
	if c := p.TorchCopies; c != nil {
		if c.GPUSlots != 40 || c.HostSlots != 64 || c.Evidence == "" || len(c.Points) != 6 || c.BaselineAllocatedBytes <= 40*917504 || c.ReservedBytes < c.BaselineAllocatedBytes || c.ReservedBytes > 40<<30 {
			return fmt.Errorf("invalid in-process Torch transfer profile")
		}
		seen := map[string]bool{}
		for _, x := range c.Points {
			if seen[x.Key()] || x.NUMA != 0 || x.Layout != "paged" || x.HostMemory != "pinned" || (x.Blocks != 1 && x.Blocks != 16 && x.Blocks != 32) || (x.Direction != "d2h" && x.Direction != "h2d") || x.Copies != int(x.Blocks)*56 || x.Evidence == "" {
				return fmt.Errorf("invalid/duplicate in-process transfer point")
			}
			seen[x.Key()] = true
			if err := x.Cost.Validate(); err != nil {
				return err
			}
		}
		for _, ns := range c.WarmSetupNS {
			if ns < 0 {
				return fmt.Errorf("negative in-process setup cost")
			}
		}
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
	pagedShapes := map[[2]int64]bool{}
	for _, x := range p.PagedForward {
		key := [2]int64{x.Prefix, x.Query}
		if pagedShapes[key] {
			return fmt.Errorf("duplicate paged forward shape %v", key)
		}
		pagedShapes[key] = true
		if err := x.Validate(); err != nil {
			return err
		}
	}
	for _, x := range p.ComputeGrid {
		key := [2]int64{x.Prefix, x.Query}
		if x.Prefix < 0 || x.Query <= 0 || x.WallNS <= 0 || x.PeakAdditionalBytes < 0 || shapes[key] {
			return fmt.Errorf("invalid/duplicate compute shape")
		}
		shapes[key] = true
	}
	pagedStructure := false
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
			if c.Length == 256 && c.Mode == "resident" {
				pagedStructure = true
			}
		}
	}
	if len(p.PagedForward) > 0 && !pagedStructure {
		return fmt.Errorf("paged pipeline requires measured resident forward structure")
	}
	return nil
}

// Validate fails before graph construction: missing costs or memory envelopes
// must never silently become zero-cost operations in the paged pipeline.
func (x PagedForwardPoint) Validate() error {
	const arena = int64(40 * 917504)
	if x.Prefix < 0 || x.Prefix >= 640 || x.Query <= 0 || x.Query > 640-x.Prefix || x.ResidualNS < 0 || x.Evidence == "" {
		return fmt.Errorf("invalid paged forward shape or provenance")
	}
	if x.BeforeAllocatedBytes <= arena || x.ReservedBytes < x.BeforeAllocatedBytes || x.ReservedBytes > 40<<30 {
		return fmt.Errorf("invalid paged forward allocation envelope")
	}
	if x.PhaseGapsNS != nil {
		allowed := map[string]bool{"index_bind": true, "gather_prefix": true, "forward": true, "observe_token": true, "next_input_d2d": true, "scatter_delta": true, "hf_release": true, "complete": true}
		var sum int64
		for name, ns := range x.PhaseGapsNS {
			if !allowed[name] || ns < 0 || ns > math.MaxInt64-sum {
				return fmt.Errorf("invalid paged phase gap")
			}
			sum += ns
		}
		if sum != x.ResidualNS {
			return fmt.Errorf("paged phase gaps disagree with residual cost")
		}
	}
	required := map[string]bool{"forward": false, "scatter_delta": true, "hf_release": false}
	if o := x.Output; o != nil {
		if o.GPUStorageBytes < 8 || o.GPUStorageBytes%8 != 0 || o.HostStorageBytes < 8 || o.HostStorageBytes%8 != 0 || o.HostReservedBytes < o.HostStorageBytes || o.Evidence == "" {
			return fmt.Errorf("invalid output storage calibration")
		}
		if err := o.Cost.Validate(); err != nil {
			return err
		}
		if o.Cost.WaitForDMA || o.Cost.BlockingAPI {
			return fmt.Errorf("native output copy uses async submission followed by completion observation")
		}
		required["observe_token"] = false
	}
	if x.Prefix > 0 {
		required["gather_prefix"] = true
	}
	if len(x.Stages) != len(required) || len(x.PhasePeakDelta) != len(required) || len(x.PhaseEndDelta) != len(required) {
		return fmt.Errorf("incomplete/unexpected paged forward phases")
	}
	for name, mapping := range required {
		c, ok := x.Stages[name]
		peak, peakOK := x.PhasePeakDelta[name]
		end, endOK := x.PhaseEndDelta[name]
		if !ok || c.WallNS <= 0 || c.SubmitNS < 0 || c.ServiceNS < 0 || c.ObserveNS < 0 || (mapping && (c.SubmitNS == 0 || c.ServiceNS == 0)) {
			return fmt.Errorf("invalid paged forward phase cost %s", name)
		}
		if !peakOK || !endOK || end < 0 || peak < end || peak > x.ReservedBytes-x.BeforeAllocatedBytes {
			return fmt.Errorf("invalid paged forward phase memory %s", name)
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
