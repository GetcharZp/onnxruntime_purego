package ort

import (
	"fmt"
	"unsafe"
)

type TensorInfo struct {
	Name     string
	DataType TensorElementDataType
	Shape    []int64
}

// ElementCount 返回 Shape 各维度的乘积，便于按形状直接准备输入缓冲区。
//
// # Returns:
//
//	int64: 元素总数
//	bool: 是否成功计算出元素总数
func (t TensorInfo) ElementCount() (int64, bool) {
	return shapeElementCount(t.Shape)
}

type Session struct {
	handle  SessionHandle
	engine  *Engine
	Inputs  []TensorInfo
	Outputs []TensorInfo

	// Deprecated: 改用 Inputs[i].Name
	InputNames []string

	// Deprecated: 改用 Outputs[i].Name
	OutputNames []string
}

type SessionOptions struct {
	handle SessionOptionsHandle
	engine *Engine
}

func (e *Engine) NewSessionOptions() (*SessionOptions, error) {
	var h SessionOptionsHandle
	status := e.funcs.createSessionOptions(&h)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}
	return &SessionOptions{handle: h, engine: e}, nil
}

// SetIntraOpNumThreads 设置线程数
func (o *SessionOptions) SetIntraOpNumThreads(num int32) error {
	return o.engine.checkStatus(o.engine.funcs.setIntraOpNumThreads(o.handle, num))
}

// SetCpuMemArena 设置内存池策略
//
//	false: 禁用内存池，推理速度稍慢，但 Destroy 后立即归还内存给 OS ，解决内存滞留问题
//	true: 启用内存池，推理速度最快，但 Destroy 后内存会被缓存以供复用（默认）
func (o *SessionOptions) SetCpuMemArena(useArena bool) error {
	if useArena {
		return o.engine.checkStatus(o.engine.funcs.enableCpuMemArena(o.handle))
	}
	return o.engine.checkStatus(o.engine.funcs.disableCpuMemArena(o.handle))
}

// EnableCUDA 启用 CUDA
func (o *SessionOptions) EnableCUDA() error {
	var cudaOpts CUDAProviderOptionsV2Handle
	status := o.engine.funcs.createCUDAProviderOptions(&cudaOpts)
	if err := o.engine.checkStatus(status); err != nil {
		return fmt.Errorf("failed to create CUDA provider options: %w", err)
	}
	defer o.engine.funcs.releaseCUDAProviderOptions(cudaOpts)

	status = o.engine.funcs.appendExecutionProvider_CUDA_V2(o.handle, cudaOpts)
	return o.engine.checkStatus(status)
}

func (o *SessionOptions) Destroy() {
	if o.handle != 0 {
		o.engine.funcs.releaseSessionOptions(o.handle)
		o.handle = 0
	}
}

// NewSession 创建会话
//
// # Params:
//
//	modelPath: 模型路径
//	opts: Session 配置项
func (e *Engine) NewSession(modelPath string, opts *SessionOptions) (*Session, error) {
	var optHandle SessionOptionsHandle
	if opts != nil {
		optHandle = opts.handle
	}

	pathPtr, err := stringToPathPtr(modelPath)
	if err != nil {
		return nil, err
	}

	var h SessionHandle
	status := e.funcs.createSession(e.envHandle, pathPtr, optHandle, &h)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}

	s := &Session{
		handle: h,
		engine: e,
	}

	if err := s.initMetadata(); err != nil {
		s.Destroy()
		return nil, err
	}

	return s, nil
}

func (s *Session) initMetadata() error {
	var err error
	if s.Inputs, err = s.collectMetadata(true); err != nil {
		return err
	}
	if s.Outputs, err = s.collectMetadata(false); err != nil {
		return err
	}

	// 兼容已弃用字段，不额外产生 onnxruntime 调用
	s.InputNames, s.OutputNames = tensorNames(s.Inputs), tensorNames(s.Outputs)
	return nil
}

// tensorNames 提取元信息中的名字
func tensorNames(infos []TensorInfo) []string {
	names := make([]string, len(infos))
	for i, info := range infos {
		names[i] = info.Name
	}
	return names
}

// collectMetadata 收集模型数据
//
// # Params:
//
//	isInput: true 表示收集输入元信息，false 表示收集输出元信息
func (s *Session) collectMetadata(isInput bool) ([]TensorInfo, error) {
	f := s.engine.funcs
	count, name, typeInfo := f.sessionGetInputCount, f.sessionGetInputName, f.sessionGetInputTypeInfo
	if !isInput {
		count, name, typeInfo = f.sessionGetOutputCount, f.sessionGetOutputName, f.sessionGetOutputTypeInfo
	}

	var n uintptr
	if err := s.engine.checkStatus(count(s.handle, &n)); err != nil {
		return nil, err
	}

	// 所有条目共用默认分配器，取一次即可
	var allocator AllocatorHandle
	if err := s.engine.checkStatus(f.getAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, err
	}

	infos := make([]TensorInfo, n)
	for i := range infos {
		var namePtr *byte
		if err := s.engine.checkStatus(name(s.handle, uintptr(i), allocator, &namePtr)); err != nil {
			return nil, err
		}
		infos[i].Name = cStringToString(namePtr)
		f.allocatorFree(allocator, unsafe.Pointer(namePtr))

		var ti TypeInfoHandle
		if err := s.engine.checkStatus(typeInfo(s.handle, uintptr(i), &ti)); err != nil {
			return nil, err
		}
		dataType, shape, err := s.readTensorType(ti)
		f.releaseTypeInfo(ti)
		if err != nil {
			return nil, err
		}
		infos[i].DataType, infos[i].Shape = dataType, shape
	}

	return infos, nil
}

// readTensorType 从 OrtTypeInfo 解析出元素类型与形状
func (s *Session) readTensorType(ti TypeInfoHandle) (TensorElementDataType, []int64, error) {
	f := s.engine.funcs

	var info TensorTypeAndShapeInfoHandle
	if err := s.engine.checkStatus(f.castTypeInfoToTensorInfo(ti, &info)); err != nil {
		return TensorElementDataTypeUndefined, nil, err
	}

	if info == 0 {
		return TensorElementDataTypeUndefined, nil, nil
	}

	var dataType TensorElementDataType
	if err := s.engine.checkStatus(f.getTensorElementType(info, &dataType)); err != nil {
		return TensorElementDataTypeUndefined, nil, err
	}

	var dimCount uintptr
	if err := s.engine.checkStatus(f.getDimensionsCount(info, &dimCount)); err != nil {
		return TensorElementDataTypeUndefined, nil, err
	}

	shape := make([]int64, dimCount)
	if dimCount > 0 {
		if err := s.engine.checkStatus(f.getDimensions(info, &shape[0], dimCount)); err != nil {
			return TensorElementDataTypeUndefined, nil, err
		}
	}

	return dataType, shape, nil
}

func (s *Session) Destroy() {
	if s.handle != 0 {
		s.engine.funcs.releaseSession(s.handle)
		s.handle = 0
	}
}

// Run 执行推理
//
// inputs 的 key 为输入名（见 Session.Inputs），返回的 Value 由调用方负责 Destroy。
func (s *Session) Run(inputs map[string]*Value) (map[string]*Value, error) {
	inputCount := len(inputs)
	outputCount := len(s.Outputs)

	if inputCount == 0 {
		return nil, fmt.Errorf("session.Run: no inputs provided, expect %v", tensorNames(s.Inputs))
	}

	// input
	inputNamePtrs := make([]unsafe.Pointer, inputCount)
	inputHandles := make([]ValueHandle, inputCount)
	i := 0
	for name, val := range inputs {
		if val == nil {
			return nil, fmt.Errorf("session.Run: input %q is nil", name)
		}
		cName, err := stringToCString(name)
		if err != nil {
			return nil, err
		}
		inputNamePtrs[i] = unsafe.Pointer(cName)
		inputHandles[i] = val.handle
		i++
	}

	// output
	outputNamePtrs := make([]unsafe.Pointer, outputCount)
	outputHandles := make([]ValueHandle, outputCount)
	for i, out := range s.Outputs {
		cName, err := stringToCString(out.Name)
		if err != nil {
			return nil, err
		}
		outputNamePtrs[i] = unsafe.Pointer(cName)
	}

	// 调用底层执行推理
	status := s.engine.funcs.run(
		s.handle,
		0,
		slicePtr(inputNamePtrs),
		slicePtr(inputHandles),
		uintptr(inputCount),
		slicePtr(outputNamePtrs),
		uintptr(outputCount),
		slicePtr(outputHandles),
	)

	if err := s.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to run session: %w", err)
	}

	results := make(map[string]*Value, outputCount)
	for i := 0; i < outputCount; i++ {
		results[s.Outputs[i].Name] = &Value{
			handle: outputHandles[i],
			engine: s.engine,
		}
	}

	return results, nil
}
