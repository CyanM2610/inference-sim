package sim

// cloneNativeTransferShape isolates retained execution evidence from model
// callbacks and caller-owned slices, including nested physical page geometry.
func cloneNativeTransferShape(shape NativeTransferShape) NativeTransferShape {
	shape.Objects = append([]NativeTransferObject(nil), shape.Objects...)
	for i := range shape.Objects {
		shape.Objects[i].HBMBlocks = append([]int64(nil), shape.Objects[i].HBMBlocks...)
		shape.Objects[i].Segments = append([]NativeCopySegment(nil), shape.Objects[i].Segments...)
	}
	return shape
}

func (e *NativeExecutor) HasTransferFailureModel() bool {
	_, ok := e.n.cost.(NativeTransferFailureCost)
	return ok
}

// SubmitFailedTransfer pays only the submitted prefix, retaining both branches
// until a terminal failed reply. Cost models see the acquired geometry as well
// so source acquisition and exception handling are not silently free. Unknown
// CUDA terminal/transport failures are not represented by this drained failure.
func (e *NativeExecutor) SubmitFailedTransfer(now, transaction int64, failure NativeTransferFailure, producer *NativeFence, begin, gpuDone, failed, done func(int64)) *NativeFence {
	model, ok := e.n.cost.(NativeTransferFailureCost)
	if !ok || producer == nil || failure.ReservedShape.Direction != "load" || failure.SubmittedObjects <= 0 || failure.SubmittedObjects >= len(failure.ReservedShape.Objects) {
		panic("partial LOAD failure needs explicit costs and a proper submitted object prefix")
	}
	failure.ReservedShape = cloneNativeTransferShape(failure.ReservedShape)
	input := failure
	input.ReservedShape = cloneNativeTransferShape(input.ReservedShape)
	p := model.FailedTransferPlan(input)
	shape := cloneNativeTransferShape(failure.ReservedShape)
	shape.Objects = shape.Objects[:failure.SubmittedObjects]
	return e.submitTransferPlan(now, transaction, shape, p, &failure, producer, begin, gpuDone, failed, done)
}
