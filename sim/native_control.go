package sim

import "fmt"

// NativeControlRPCShape is protocol geometry available before execution. It
// contains no future request or observed timing. Operation is lookup/end for
// Promotion and lookup/status/end for the foreground request protocol. For
// requests, only LOOKUP carries the full token payload and complete-chunk count;
// STATUS and END carry identities, so Tokens and Objects are zero. Promotion
// retains its historical prefix-geometry cost contract.
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

// NativeRequestRPCCost is explicit because asynchronous foreground lookup is
// not equivalent to the older synchronous directory diagnostic. This models
// the single hybrid L1 server; tiered L2 prefetch needs an additional state.
type NativeRequestRPCCost interface {
	RequestRPCPlan(NativeControlRPCShape) NativeControlRPCPlan
}

func (e *NativeExecutor) HasRequestRPCModel() bool {
	_, ok := e.n.cost.(NativeRequestRPCCost)
	return ok
}

func (e *NativeExecutor) SubmitRequestRPC(now int64, id, request string, shape NativeControlRPCShape, applied, replied func(int64)) *NativeFence {
	model, ok := e.n.cost.(NativeRequestRPCCost)
	if !ok || id == "" || request == "" || shape.Operation != "lookup" && shape.Operation != "status" && shape.Operation != "end" ||
		shape.Operation == "lookup" && (shape.Tokens <= 0 || shape.Objects < 0) || shape.Operation != "lookup" && (shape.Tokens != 0 || shape.Objects != 0) {
		panic("native request RPC geometry/model missing")
	}
	return e.submitRPC(now, id, request, shape, model.RequestRPCPlan(shape), applied, replied)
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
	return e.submitRPC(now, id, "", shape, model.ControlRPCPlan(shape), applied, replied)
}

func (e *NativeExecutor) submitRPC(now int64, id, request string, shape NativeControlRPCShape, p NativeControlRPCPlan, applied, replied func(int64)) *NativeFence {
	ancestors, err := validateNativeStages(p.Stages, e.busy)
	if err != nil || ancestors[p.Applied] == nil || ancestors[p.Replied] == nil || !ancestors[p.Replied][p.Applied] || p.CostScope == "" {
		panic("native control requires an ordered directory/reply plan with declared cost scope")
	}
	service := false
	for _, stage := range p.Stages {
		if !ancestors[p.Replied][stage.Name] {
			panic("native control reply leaves unjoined work")
		}
		service = service || stage.Resource != "" && !stage.Delay && stage.DurationUS > 0
	}
	if !service {
		panic("native control cannot omit all host service")
	}
	fences := map[string]*NativeFence{}
	prefix, event := "control_", "native_control_rpc_plan"
	if request != "" {
		prefix, event = "request_", "native_request_rpc_plan"
	}
	for _, stage := range p.Stages {
		fences[stage.Name] = &NativeFence{}
	}
	for _, stage := range p.Stages {
		j := &nativeStageRun{spec: stage, fence: fences[stage.Name]}
		j.spec.Name = fmt.Sprintf("%s%s/%s/%s", prefix, id, shape.Operation, stage.Name)
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
	record := e.n.event(now, event, nil, nil)
	record.Request = request
	record.ControlID, record.ControlShape, record.Control = id, &shape, &p
	e.pump(now)
	return fences[p.Replied]
}
