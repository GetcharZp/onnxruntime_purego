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

	// modelData 持有 NewSessionFromBytes 传入的模型字节，保证 Session 存活期间缓冲区不被回收
	modelData []byte
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

// providerOptions 描述一个 V2 形式的 Execution Provider：创建配置 -> 写入键值 -> 挂到 SessionOptions
type providerOptions[H any] struct {
	create  func(*H) StatusHandle
	update  func(H, **byte, **byte, uintptr) StatusHandle
	release func(H)
	append  func(SessionOptionsHandle, H) StatusHandle
}

func (f *apiFuncs) cudaOptions() providerOptions[CUDAProviderOptionsV2Handle] {
	return providerOptions[CUDAProviderOptionsV2Handle]{
		create:  f.createCUDAProviderOptions,
		update:  f.updateCUDAProviderOptions,
		release: f.releaseCUDAProviderOptions,
		append:  f.appendExecutionProvider_CUDA_V2,
	}
}

func (f *apiFuncs) tensorRTOptions() providerOptions[TensorRTProviderOptionsV2Handle] {
	return providerOptions[TensorRTProviderOptionsV2Handle]{
		create:  f.createTensorRTProviderOptions,
		update:  f.updateTensorRTProviderOptions,
		release: f.releaseTensorRTProviderOptions,
		append:  f.appendExecutionProvider_TensorRT_V2,
	}
}

// EnableCUDA 启用 CUDA，options 为 CUDA Provider 的键值配置，可为 nil
//
// 常用键：
//
//	device_id                    = "0"                指定显卡，默认 0
//	cudnn_conv_algo_search       = "HEURISTIC"        卷积算法搜索，默认 EXHAUSTIVE
//	cudnn_conv_use_max_workspace = "1"                卷积使用最大 workspace，默认 1
//	enable_cuda_graph            = "1"                输入 shape 固定时启用 CUDA Graph，默认 0
//	gpu_mem_limit                = "4294967296"       显存上限（字节），默认不限制
//	arena_extend_strategy        = "kSameAsRequested" 显存池扩展策略，默认 kNextPowerOfTwo
//	prefer_nhwc                  = "1"                优先 NHWC 布局，需构建支持，默认 0
func (o *SessionOptions) EnableCUDA(options map[string]string) error {
	return enableProvider(o, o.engine.funcs.cudaOptions(), options)
}

// EnableTensorRT 启用 TensorRT，options 为 TensorRT Provider 的键值配置，可为 nil
//
// 与 CUDA 一样用 device_id 指定显卡，从 0 开始计数，默认 0：
//
//	opts.EnableTensorRT(map[string]string{"device_id": "1"})
//
// 常用键：
//
//	trt_fp16_enable         = "1"     启用 FP16，CV 推理加速明显
//	trt_int8_enable         = "1"     启用 INT8
//	trt_engine_cache_enable = "1"     缓存引擎，避免每次构建耗时
//	trt_engine_cache_path   = "./trt" 引擎缓存目录
//
// Execution Provider 按挂载顺序优先生效，需同时使用 CUDA 兜底时先调用本方法，
// 且两者的 device_id 应指向同一张卡：
//
//	opts.EnableTensorRT(map[string]string{"device_id": "1", "trt_fp16_enable": "1"})
//	opts.EnableCUDA(map[string]string{"device_id": "1"})
func (o *SessionOptions) EnableTensorRT(options map[string]string) error {
	return enableProvider(o, o.engine.funcs.tensorRTOptions(), options)
}

// enableProvider 向 SessionOptions 追加一个 Execution Provider 并写入其键值配置
func enableProvider[H any](o *SessionOptions, p providerOptions[H], options map[string]string) error {
	var cfg H
	if err := o.engine.checkStatus(p.create(&cfg)); err != nil {
		return fmt.Errorf("failed to create provider options: %w", err)
	}
	defer p.release(cfg)

	if len(options) > 0 {
		keys := make([]*byte, 0, len(options))
		values := make([]*byte, 0, len(options))
		for k, v := range options {
			key, _ := stringToCString(k)
			value, _ := stringToCString(v)
			keys, values = append(keys, key), append(values, value)
		}
		status := p.update(cfg, slicePtr(keys), slicePtr(values), uintptr(len(keys)))
		if err := o.engine.checkStatus(status); err != nil {
			return fmt.Errorf("failed to update provider options: %w", err)
		}
	}

	return o.engine.checkStatus(p.append(o.handle, cfg))
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
	pathPtr, err := stringToPathPtr(modelPath)
	if err != nil {
		return nil, err
	}
	return e.newSession(opts, func(optHandle SessionOptionsHandle, h *SessionHandle) StatusHandle {
		return e.funcs.createSession(e.envHandle, pathPtr, optHandle, h)
	})
}

// NewSessionFromBytes 从内存中的模型字节创建会话，适用于 go:embed 等无文件场景
//
// # Params:
//
//	data: 模型字节
//	opts: Session 配置项
func (e *Engine) NewSessionFromBytes(data []byte, opts *SessionOptions) (*Session, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("NewSessionFromBytes: model data is empty")
	}
	s, err := e.newSession(opts, func(optHandle SessionOptionsHandle, h *SessionHandle) StatusHandle {
		return e.funcs.createSessionFromArray(e.envHandle, unsafe.Pointer(slicePtr(data)), uintptr(len(data)), optHandle, h)
	})
	if err != nil {
		return nil, err
	}
	s.modelData = data
	return s, nil
}

// newSession 调用底层创建接口并完成元信息初始化，create 由具体来源（路径 / 内存）提供
func (e *Engine) newSession(opts *SessionOptions, create func(SessionOptionsHandle, *SessionHandle) StatusHandle) (*Session, error) {
	var optHandle SessionOptionsHandle
	if opts != nil {
		optHandle = opts.handle
	}

	var h SessionHandle
	if err := e.checkStatus(create(optHandle, &h)); err != nil {
		return nil, err
	}

	s := &Session{handle: h, engine: e}
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
