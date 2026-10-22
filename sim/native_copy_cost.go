package sim

import (
	"fmt"
	"math"
)

// NativeCopyProfile contains independently measured CUDA phase predictions.
// It deliberately cannot supply CPU/protocol costs or certify service latency.
type NativeCopyProfile struct {
	Schema       int                        `json:"schema"`
	ObjectBytes  int64                      `json:"object_bytes"`
	ChunkTokens  int64                      `json:"host_chunk_tokens"`
	BlockTokens  int64                      `json:"hbm_block_tokens"`
	SegmentBytes int64                      `json:"registration_segment_bytes"`
	StoreBatch   int                        `json:"store_batch_size"`
	LoadBatch    int                        `json:"load_batch_size"`
	Curves       map[string]NativeCopyCurve `json:"curves"`
	Modeled      NativeModeledMedia         `json:"modeled_media"`
}

type NativeCopyCurve struct {
	Coefficients []float64 `json:"coefficients_us"`
	MinBytes     int64     `json:"min_bytes"`
	MaxBytes     int64     `json:"max_bytes"`
	MinObjects   int       `json:"min_objects"`
	MaxObjects   int       `json:"max_objects"`
	MaxExtra     int       `json:"max_extra_segments"`
}

// GB/s equals bytes/ns. These are model inputs, never hardware observations.
type NativeModeledMedia struct {
	ReadGBPS  float64 `json:"read_gbps"`
	WriteGBPS float64 `json:"write_gbps"`
	ReadNS    int64   `json:"read_latency_ns"`
	WriteNS   int64   `json:"write_latency_ns"`
}

// NativeTransferExecution explicitly supplies the still-separate host model.
// Scope must identify its evidence or assumptions; a primitive profile never
// silently turns allocator, IPC, or reply processing into free operations.
type NativeTransferExecution struct {
	PoolMedia                         map[string]string
	HostResource, ComputeResource     string
	CopyResource, ReadModelResource   string
	WriteModelResource                string
	HostSubmitUS, ModelRPCUS, ReplyUS int64
	Scope                             string
}

func (p NativeCopyProfile) Validate() error {
	if p.Schema != 1 || p.ObjectBytes <= 0 || p.ChunkTokens <= 0 || p.BlockTokens <= 0 || p.ChunkTokens%p.BlockTokens != 0 || p.SegmentBytes <= 0 || p.StoreBatch != 1 || p.LoadBatch <= 0 {
		return fmt.Errorf("invalid native copy geometry")
	}
	if p.ObjectBytes > math.MaxInt64/int64(p.LoadBatch) {
		return fmt.Errorf("native staging allocation overflows")
	}
	for _, direction := range []string{"load", "store"} {
		for _, prefix := range []string{"kernel/gpu/", "staging/dram/", "staging/cxl_sim/"} {
			curve, ok := p.Curves[prefix+direction]
			if !ok || len(curve.Coefficients) != 3 || curve.MinBytes <= 0 || curve.MaxBytes < curve.MinBytes || curve.MinObjects <= 0 || curve.MaxObjects < curve.MinObjects || curve.MaxExtra < 0 {
				return fmt.Errorf("missing/invalid native curve %s%s", prefix, direction)
			}
			for _, value := range curve.Coefficients {
				if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
					return fmt.Errorf("invalid native copy coefficient")
				}
			}
		}
	}
	if p.Modeled.ReadGBPS <= 0 || p.Modeled.WriteGBPS <= 0 || math.IsNaN(p.Modeled.ReadGBPS) || math.IsNaN(p.Modeled.WriteGBPS) || math.IsInf(p.Modeled.ReadGBPS, 0) || math.IsInf(p.Modeled.WriteGBPS, 0) || p.Modeled.ReadNS < 0 || p.Modeled.WriteNS < 0 {
		return fmt.Errorf("invalid modeled media parameters")
	}
	return nil
}

func copyTimeUS(value float64) int64 {
	if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value >= float64(math.MaxInt64) {
		panic("invalid predicted copy duration")
	}
	return int64(math.Ceil(value))
}

func (p NativeCopyProfile) primitive(key string, bytes int64, objects, extra int, gaps *[]string) int64 {
	c := p.Curves[key]
	if bytes < c.MinBytes || bytes > c.MaxBytes || objects < c.MinObjects || objects > c.MaxObjects || extra > c.MaxExtra {
		*gaps = append(*gaps, fmt.Sprintf("%s bytes=%d objects=%d extra_segments=%d outside training geometry", key, bytes, objects, extra))
	}
	return copyTimeUS(c.Coefficients[0] + c.Coefficients[1]*float64(bytes)/(1<<20) + c.Coefficients[2]*float64(extra))
}

// One native staging phase may visit several media in object order. Until a
// mixed-media measurement exists, compose contiguous same-media predictions and
// explicitly reject any interpretation of that estimate as a calibrated cost.
// The scatter/gather batch is never split at these media boundaries.
func (p NativeCopyProfile) stagingCost(objects []NativeTransferObject, direction string, media map[string]string, gaps *[]string) int64 {
	var duration int64
	runs := 0
	for start := 0; start < len(objects); {
		end := start + 1
		kind := media[objects[start].Pool]
		for end < len(objects) && media[objects[end].Pool] == kind {
			end++
		}
		var bytes int64
		extra := 0
		for _, object := range objects[start:end] {
			bytes += object.StagingBytes
			extra += len(object.Segments) - 1
		}
		part := p.primitive("staging/"+kind+"/"+direction, bytes, end-start, extra, gaps)
		if duration > math.MaxInt64-part {
			panic("native staging composition overflows")
		}
		duration += part
		runs++
		start = end
	}
	if runs > 1 {
		*gaps = append(*gaps, fmt.Sprintf("mixed-media %s staging batch: uncalibrated sum of %d contiguous media predictions; may repeat fitted intercepts", direction, runs))
	}
	return duration
}

// TransferPlan expands native batch slots into ordered gather/staging/scatter
// stages. The modeled-media branch queues serialization only; its latency tail
// can overlap subsequent objects and joins CUDA before reply/publication.
func (p NativeCopyProfile) TransferPlan(shape NativeTransferShape, x NativeTransferExecution) NativeTransferPlan {
	if err := p.Validate(); err != nil {
		panic(err)
	}
	if len(shape.Objects) == 0 || shape.Direction != "store" && shape.Direction != "load" || x.HostSubmitUS <= 0 || x.ModelRPCUS <= 0 || x.ReplyUS <= 0 || x.Scope == "" || x.HostResource == "" || x.ComputeResource == "" || x.CopyResource == "" || x.ReadModelResource == "" || x.WriteModelResource == "" {
		panic("native transfer requires geometry and explicit host service assumptions")
	}
	plan := NativeTransferPlan{CostScope: "Instrumented CUDA primitive curves with explicit extrapolation/composition gaps; CXLMemSim modeled media; host: " + x.Scope, StagingSlots: p.LoadBatch, StagingBufferBytes: p.ObjectBytes * int64(p.LoadBatch)}
	add := func(s NativeStage) { plan.Stages = append(plan.Stages, s) }
	add(NativeStage{Name: "host_submit", Resource: x.HostResource, Stream: "mp_host", DurationUS: x.HostSubmitUS})
	lastHost := "host_submit"
	var modelEnds []string
	seen := map[string]bool{}
	for i, o := range shape.Objects {
		media := x.PoolMedia[o.Pool]
		if seen[o.Key] || o.Key == "" || media != "dram" && media != "cxl_sim" || o.StagingBytes != p.ObjectBytes || o.KernelBytes <= 0 || o.KernelBytes > o.StagingBytes || o.SkipTokens < 0 || o.SkipTokens >= p.ChunkTokens || o.SkipTokens%p.BlockTokens != 0 || o.KernelBytes != p.ObjectBytes*(p.ChunkTokens-o.SkipTokens)/p.ChunkTokens || o.HostOffset < 0 || len(o.HBMBlocks) != int((p.ChunkTokens-o.SkipTokens)/p.BlockTokens) {
			panic("unsupported or inconsistent native transfer object geometry")
		}
		seen[o.Key] = true
		if i > 0 && o.SkipTokens != 0 {
			panic("native LOAD can only skip a contiguous prefix in its first retained object")
		}
		if shape.Direction == "store" && o.SkipTokens != 0 {
			panic("native STORE may not skip the chunk prefix")
		}
		offset := o.HostOffset
		for _, s := range o.Segments {
			if s.HostOffset != offset || s.Bytes <= 0 || s.Bytes > p.SegmentBytes-offset%p.SegmentBytes {
				panic("native registration segment gap or boundary crossing")
			}
			offset += s.Bytes
		}
		if offset-o.HostOffset != o.StagingBytes {
			panic("native staging segments do not cover the whole object")
		}
		if media == "cxl_sim" {
			rpc, serial, tail := fmt.Sprintf("model_begin_%d", i), fmt.Sprintf("model_serial_%d", i), fmt.Sprintf("model_tail_%d", i)
			add(NativeStage{Name: rpc, Resource: x.HostResource, Stream: "mp_host", DurationUS: x.ModelRPCUS, After: []string{lastHost}})
			bandwidth, latency, resource := p.Modeled.ReadGBPS, p.Modeled.ReadNS, x.ReadModelResource
			if shape.Direction == "store" {
				bandwidth, latency, resource = p.Modeled.WriteGBPS, p.Modeled.WriteNS, x.WriteModelResource
			}
			// Exact native cacheline footprint followed by microsecond clock rounding.
			lines := (o.StagingBytes + o.HostOffset%64 + 63) / 64
			serialNS := math.Max(1, math.Ceil(float64(lines*64)/bandwidth))
			add(NativeStage{Name: serial, Resource: resource, Stream: resource, DurationUS: copyTimeUS(serialNS / 1000), After: []string{rpc}})
			add(NativeStage{Name: tail, Delay: true, DurationUS: copyTimeUS(float64(latency) / 1000), After: []string{serial}})
			modelEnds = append(modelEnds, tail)
			lastHost = rpc
		}
	}
	if len(modelEnds) == 0 {
		modelEnds = []string{lastHost}
	}
	add(NativeStage{Name: "model_complete", After: modelEnds})
	plan.ModelComplete = "model_complete"
	batchSize := p.LoadBatch
	if shape.Direction == "store" {
		batchSize = p.StoreBatch
	}
	lastGPU := ""
	for i := 0; i < len(shape.Objects); i += batchSize {
		objects := shape.Objects[i:min(i+batchSize, len(shape.Objects))]
		kernelBytes := int64(0)
		for _, o := range objects {
			kernelBytes += o.KernelBytes
		}
		kernel := NativeStage{Name: fmt.Sprintf("kernel_%d", i/batchSize), Resource: x.ComputeResource, Stream: "native-kv-stream", DurationUS: p.primitive("kernel/gpu/"+shape.Direction, kernelBytes, len(objects), 0, &plan.CoverageGaps)}
		staging := NativeStage{Name: fmt.Sprintf("staging_%d", i/batchSize), Resource: x.CopyResource, Stream: "native-kv-stream", DurationUS: p.stagingCost(objects, shape.Direction, x.PoolMedia, &plan.CoverageGaps)}
		first, second := kernel, staging
		if shape.Direction == "load" {
			first, second = staging, kernel
		}
		first.After = []string{lastHost}
		if lastGPU != "" {
			first.After = append(first.After, lastGPU)
		}
		second.After = []string{first.Name}
		add(first)
		add(second)
		if plan.ProducerWait == "" {
			plan.ProducerWait = first.Name
		}
		lastGPU = second.Name
	}
	plan.GPUComplete = lastGPU
	add(NativeStage{Name: "reply", Resource: x.HostResource, Stream: "mp_reply", DurationUS: x.ReplyUS, After: []string{lastGPU, plan.ModelComplete}})
	plan.Published = "reply"
	return plan
}
