package sim

import "fmt"

// NativeControlRPCShape is protocol geometry available before execution. It
// contains no future request or observed timing. Operation is lookup or end.
type NativeControlRPCShape struct {
	Operation string `json:"operation"`
	Tokens    int64  `json:"tokens"`
	Objects   int64  `json:"objects"`
}

// Applied is when the host directory mutation occurs; Replied is when its
// result becomes visible. Transport delay and exclusive CPU service differ.
type NativeControlRPCPlan struct {
	Stages       []NativeStage `json:"stages"`
	Applied      string        `json:"applied"`
	Replied      string        `json:"replied"`
	CostScope    string        `json:"cost_scope"`
	CoverageGaps []string      `json:"coverage_gaps,omitempty"`
}

type NativeControlCost interface {
	ControlRPCPlan(NativeControlRPCShape) NativeControlRPCPlan
}

func (e *NativeExecutor) HasControlModel() bool { _, ok := e.n.cost.(NativeControlCost); return ok }

// SubmitControlRPC executes a source lookup or background END on the same
// resource graph as ordinary work. An END can overtake a pending lookup; the
// server-side ownership/tombstone callbacks determine the resulting state.
func (e *NativeExecutor) SubmitControlRPC(now int64, id string, shape NativeControlRPCShape, applied, replied func(int64)) *NativeFence {
	model, ok := e.n.cost.(NativeControlCost)
	if !ok || id == "" || shape.Operation != "lookup" && shape.Operation != "end" || shape.Tokens <= 0 || shape.Objects <= 0 {
		panic("native control geometry/model missing")
	}
	p := model.ControlRPCPlan(shape)
	ancestors, err := validateNativeStages(p.Stages, e.busy)
	if err != nil || ancestors[p.Applied] == nil || ancestors[p.Replied] == nil || !ancestors[p.Replied][p.Applied] || p.CostScope == "" {
		panic("native control requires an ordered directory/reply plan with declared cost scope")
	}
	service := false
	for _, stage := range p.Stages {
		if !ancestors[p.Replied][stage.Name] {
			panic("native control reply leaves unjoined work")
		}
		service = service || stage.Resource != "" && stage.DurationUS > 0
	}
	if !service {
		panic("native control cannot omit all host service")
	}
	fences := map[string]*NativeFence{}
	for _, stage := range p.Stages {
		fences[stage.Name] = &NativeFence{}
	}
	for _, stage := range p.Stages {
		j := &nativeStageRun{spec: stage, fence: fences[stage.Name]}
		j.spec.Name = fmt.Sprintf("control_%s/%s/%s", id, shape.Operation, stage.Name)
		for _, name := range stage.After {
			j.dependencies = append(j.dependencies, fences[name])
		}
		if stage.Name == p.Applied {
			j.end = applied
		}
		if stage.Name == p.Replied {
			prior := j.end
			j.end = func(at int64) {
				if prior != nil {
					prior(at)
				}
				replied(at)
			}
		}
		if stage.Stream != "" {
			if prior := e.streams[stage.Stream]; prior != nil {
				j.dependencies = append(j.dependencies, prior)
			}
			e.streams[stage.Stream] = j.fence
		}
		e.jobs = append(e.jobs, j)
	}
	record := e.n.event(now, "native_control_rpc_plan", nil, nil)
	record.ControlID, record.ControlShape, record.Control = id, &shape, &p
	e.pump(now)
	return fences[p.Replied]
}
