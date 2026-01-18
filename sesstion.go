package ort

import (
	"fmt"
	"unsafe"
)

type Session struct {
	handle      SessionHandle
	engine      *Engine
	InputNames  []string
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
	// input
	inputCount, err := s.getInputCount()
	if err != nil {
		return err
	}
	s.InputNames = make([]string, inputCount)
	for i := 0; i < inputCount; i++ {
		name, err := s.getInputName(i)
		if err != nil {
			return err
		}
		s.InputNames[i] = name
	}

	// output
	outputCount, err := s.getOutputCount()
	if err != nil {
		return err
	}
	s.OutputNames = make([]string, outputCount)
	for i := 0; i < outputCount; i++ {
		name, err := s.getOutputName(i)
		if err != nil {
			return err
		}
		s.OutputNames[i] = name
	}

	return nil
}

func (s *Session) getInputCount() (int, error) {
	var count uintptr
	status := s.engine.funcs.sessionGetInputCount(s.handle, &count)
	return int(count), s.engine.checkStatus(status)
}

func (s *Session) getOutputCount() (int, error) {
	var count uintptr
	status := s.engine.funcs.sessionGetOutputCount(s.handle, &count)
	return int(count), s.engine.checkStatus(status)
}

func (s *Session) getInputName(index int) (string, error) {
	var allocator AllocatorHandle
	status := s.engine.funcs.getAllocatorWithDefaultOptions(&allocator)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	var namePtr *byte
	status = s.engine.funcs.sessionGetInputName(s.handle, uintptr(index), allocator, &namePtr)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	name := cStringToString(namePtr)
	// 释放内存
	s.engine.funcs.allocatorFree(allocator, unsafe.Pointer(namePtr))

	return name, nil
}

func (s *Session) getOutputName(index int) (string, error) {
	var allocator AllocatorHandle
	status := s.engine.funcs.getAllocatorWithDefaultOptions(&allocator)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	var namePtr *byte
	status = s.engine.funcs.sessionGetOutputName(s.handle, uintptr(index), allocator, &namePtr)
	if err := s.engine.checkStatus(status); err != nil {
		return "", err
	}

	name := cStringToString(namePtr)
	s.engine.funcs.allocatorFree(allocator, unsafe.Pointer(namePtr))

	return name, nil
}

func (s *Session) Destroy() {
	if s.handle != 0 {
		s.engine.funcs.releaseSession(s.handle)
		s.handle = 0
	}
}

// Run 执行推理
func (s *Session) Run(inputs map[string]*Value) (map[string]*Value, error) {
	inputCount := len(inputs)
	outputCount := len(s.OutputNames)

	// input
	inputNamePtrs := make([]unsafe.Pointer, inputCount)
	inputHandles := make([]ValueHandle, inputCount)
	i := 0
	for name, val := range inputs {
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
	for i, name := range s.OutputNames {
		cName, err := stringToCString(name)
		if err != nil {
			return nil, err
		}
		outputNamePtrs[i] = unsafe.Pointer(cName)
	}

	// 调用底层执行推理
	status := s.engine.funcs.run(
		s.handle,
		0,
		&inputNamePtrs[0],
		&inputHandles[0],
		uintptr(inputCount),
		&outputNamePtrs[0],
		uintptr(outputCount),
		&outputHandles[0],
	)

	if err := s.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to run session: %w", err)
	}

	results := make(map[string]*Value, outputCount)
	for i := 0; i < outputCount; i++ {
		results[s.OutputNames[i]] = &Value{
			handle: outputHandles[i],
			engine: s.engine,
		}
	}

	return results, nil
}
