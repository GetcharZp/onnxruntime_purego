package ort

import (
	"fmt"
	"unsafe"
)

// TensorInfo 描述模型的输入 / 输出
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
//	bool: 是否成功计算出元素总数，形状含动态维度(-1)或乘积溢出时为 false
func (t TensorInfo) ElementCount() (int64, bool) {
	return shapeElementCount(t.Shape)
}

// Session 表示一个加载好的模型会话
type Session struct {
	handle SessionHandle
	engine *Engine

	// Inputs / Outputs 为模型声明的输入 / 输出元信息，顺序与模型定义一致
	Inputs  []TensorInfo
	Outputs []TensorInfo

	// 输入 / 输出名在创建时转换为 C 字符串并缓存，Run 无需逐次分配；
	// 同时保证 onnxruntime 读取期间这些内存始终可达。
	inputNames     map[string]unsafe.Pointer
	outputNamePtrs []unsafe.Pointer

	// modelData 持有 NewSessionFromBytes 传入的模型字节，保证 Session 存活期间缓冲区不被回收
	modelData []byte
}

// SessionOptions 是创建 Session 时使用的配置，用完后需调用 Close
type SessionOptions struct {
	handle SessionOptionsHandle
	engine *Engine
}

// NewSessionOptions 创建 Session 配置项
func (e *Engine) NewSessionOptions() (*SessionOptions, error) {
	var h SessionOptionsHandle
	status := e.funcs.createSessionOptions(&h)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}
	return &SessionOptions{handle: h, engine: e}, nil
}

// SetIntraOpNumThreads 设置算子内并行线程数
func (o *SessionOptions) SetIntraOpNumThreads(num int) error {
	return o.engine.checkStatus(o.engine.funcs.setIntraOpNumThreads(o.handle, int32(num)))
}

// SetCPUMemArena 设置 CPU 内存池策略
//
//	false: 禁用内存池，推理速度稍慢，但 Close 后立即归还内存给 OS ，解决内存滞留问题
//	true: 启用内存池，推理速度最快，但 Close 后内存会被缓存以供复用（默认）
func (o *SessionOptions) SetCPUMemArena(enabled bool) error {
	arena := o.engine.funcs.enableCpuMemArena
	if !enabled {
		arena = o.engine.funcs.disableCpuMemArena
	}
	return o.engine.checkStatus(arena(o.handle))
}

// AddFreeDimensionOverrideByName 按维度名固化模型的动态维度
//
// 动态模型（如 YOLO）的输入维度常以 -1 出现，导致 TensorInfo.ElementCount 无法计算，
// 也会让 NewTensor 无从下手。在创建 Session 前按维度名覆盖即可消除动态维：
//
//	opts, _ := engine.NewSessionOptions()
//	opts.AddFreeDimensionOverrideByName("batch", 1)
//	session, _ := engine.NewSession("yolo11n.onnx", opts) // session.Inputs[0].Shape 中该维固定为 1
//
// # Params:
//
//	name: 维度名，取自模型定义，例如 batch / height / width
//	dim: 固定的维度值
//
// # Returns:
//
//	error: 失败时的错误信息
func (o *SessionOptions) AddFreeDimensionOverrideByName(name string, dim int64) error {
	return o.setFreeDimensionOverride(name, dim, o.engine.funcs.addFreeDimensionOverrideByName)
}

// AddFreeDimensionOverride 按维度符号（denotation）固化动态维度
//
// 与 AddFreeDimensionOverrideByName 的区别在于匹配的是维度上的 denotation 标记而非维度名，
// 适用于维度名不固定、但带有语义符号（如 DATA_BATCH）的模型。
//
// # Params:
//
//	denotation: 维度符号，例如 DATA_BATCH
//	dim: 固定的维度值
//
// # Returns:
//
//	error: 失败时的错误信息
func (o *SessionOptions) AddFreeDimensionOverride(denotation string, dim int64) error {
	return o.setFreeDimensionOverride(denotation, dim, o.engine.funcs.addFreeDimensionOverride)
}

// setFreeDimensionOverride 按 key 固化动态维度，fn 为对应的 onnxruntime 接口
func (o *SessionOptions) setFreeDimensionOverride(key string, dim int64, fn func(SessionOptionsHandle, *byte, int64) StatusHandle) error {
	return o.engine.checkStatus(fn(o.handle, cString(key), dim))
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
			keys, values = append(keys, cString(k)), append(values, cString(v))
		}
		status := p.update(cfg, slicePtr(keys), slicePtr(values), uintptr(len(keys)))
		if err := o.engine.checkStatus(status); err != nil {
			return fmt.Errorf("failed to update provider options: %w", err)
		}
	}

	return o.engine.checkStatus(p.append(o.handle, cfg))
}

// Close 释放 SessionOptions
func (o *SessionOptions) Close() {
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
//	opts: Session 配置项，可为 nil
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
//	data: 模型字节，Session 存活期间不应修改
//	opts: Session 配置项，可为 nil
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
		s.Close()
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

	s.inputNames = make(map[string]unsafe.Pointer, len(s.Inputs))
	for _, in := range s.Inputs {
		s.inputNames[in.Name] = unsafe.Pointer(cString(in.Name))
	}
	s.outputNamePtrs = make([]unsafe.Pointer, len(s.Outputs))
	for i, out := range s.Outputs {
		s.outputNamePtrs[i] = unsafe.Pointer(cString(out.Name))
	}
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
		infos[i].Name = goString(namePtr)
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

// Close 释放 Session
//
// Session 由 Engine 创建并引用 Engine，必须先于 Engine.Close 调用。
func (s *Session) Close() {
	if s.handle != 0 {
		s.engine.funcs.releaseSession(s.handle)
		s.handle = 0
	}
}

// Run 执行推理
//
// inputs 的 key 为输入名（见 Session.Inputs），返回的 Value 由调用方负责 Close。
func (s *Session) Run(inputs map[string]*Value) (map[string]*Value, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("session.Run: no inputs provided, expect %v", tensorNames(s.Inputs))
	}

	inputNamePtrs := make([]unsafe.Pointer, 0, len(inputs))
	inputHandles := make([]ValueHandle, 0, len(inputs))
	for name, val := range inputs {
		namePtr, ok := s.inputNames[name]
		if !ok {
			return nil, fmt.Errorf("session.Run: unknown input %q, expect %v", name, tensorNames(s.Inputs))
		}
		if val == nil {
			return nil, fmt.Errorf("session.Run: input %q is nil", name)
		}
		inputNamePtrs = append(inputNamePtrs, namePtr)
		inputHandles = append(inputHandles, val.handle)
	}

	outputHandles := make([]ValueHandle, len(s.Outputs))
	status := s.engine.funcs.run(
		s.handle,
		0,
		slicePtr(inputNamePtrs),
		slicePtr(inputHandles),
		uintptr(len(inputHandles)),
		slicePtr(s.outputNamePtrs),
		uintptr(len(s.outputNamePtrs)),
		slicePtr(outputHandles),
	)
	if err := s.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to run session: %w", err)
	}

	results := make(map[string]*Value, len(s.Outputs))
	for i, out := range s.Outputs {
		results[out.Name] = &Value{handle: outputHandles[i], engine: s.engine}
	}
	return results, nil
}
