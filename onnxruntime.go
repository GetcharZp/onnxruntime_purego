package ort

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/getcharzp/onnxruntime_purego/internal/sys"
)

type ApiVersion uint32

const (
	ApiVersion23 ApiVersion = 23

	LogVerbose LoggingLevel = 0
	LogInfo    LoggingLevel = 1
	LogWarning LoggingLevel = 2
	LogError   LoggingLevel = 3
	LogFatal   LoggingLevel = 4

	DefaultEnvName = "GETCHARZP"
)

// defaultEngine 由 NewEngine 设置，供包级 NewTensor 使用。
var (
	defaultEngineMu sync.RWMutex
	defaultEngine   *Engine
)

// getDefaultEngine 获取全局默认引擎
func getDefaultEngine() *Engine {
	defaultEngineMu.RLock()
	defer defaultEngineMu.RUnlock()
	return defaultEngine
}

func setDefaultEngine(e *Engine) {
	defaultEngineMu.Lock()
	defer defaultEngineMu.Unlock()
	defaultEngine = e
}

// Engine 推理引擎上下文
type Engine struct {
	handle  uintptr
	version ApiVersion
	api     *ortApi
	funcs   *apiFuncs

	envHandle EnvHandle
	memInfo   MemoryInfoHandle
}

// NewEngine 初始化引擎
func NewEngine(libPath string) (*Engine, error) {
	handle, err := sys.LoadLibrary(libPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load library: %w", err)
	}

	e := &Engine{
		handle:  handle,
		version: ApiVersion23,
		funcs:   &apiFuncs{},
	}

	if err := e.initApi(); err != nil {
		sys.FreeLibrary(handle)
		return nil, err
	}
	if err := e.initEnv(DefaultEnvName); err != nil {
		e.cleanup()
		sys.FreeLibrary(handle)
		return nil, err
	}
	if err := e.initMemInfo(); err != nil {
		e.cleanup()
		sys.FreeLibrary(handle)
		return nil, err
	}

	setDefaultEngine(e)

	return e, nil
}

func (e *Engine) initApi() error {
	// 获取 OrtGetApiBase 导出函数
	var ortGetApiBase func() *ortApiBase
	purego.RegisterLibFunc(&ortGetApiBase, e.handle, "OrtGetApiBase")

	apiBase := ortGetApiBase()
	if apiBase == nil {
		return fmt.Errorf("OrtGetApiBase returned nil")
	}
	purego.RegisterFunc(&e.funcs.getVersionString, apiBase.GetVersionString)

	// 绑定 GetAPI 函数
	var getApi func(ApiVersion) *ortApi
	purego.RegisterFunc(&getApi, apiBase.GetAPI)
	e.api = getApi(e.version)
	if e.api == nil {
		return fmt.Errorf("failed to get OrtApi for version: %d", e.version)
	}

	// status, error
	purego.RegisterFunc(&e.funcs.createStatus, e.api.CreateStatus)
	purego.RegisterFunc(&e.funcs.getErrorCode, e.api.GetErrorCode)
	purego.RegisterFunc(&e.funcs.getErrorMessage, e.api.GetErrorMessage)
	purego.RegisterFunc(&e.funcs.releaseStatus, e.api.ReleaseStatus)

	// env
	purego.RegisterFunc(&e.funcs.createEnv, e.api.CreateEnv)
	purego.RegisterFunc(&e.funcs.releaseEnv, e.api.ReleaseEnv)

	// allocator
	purego.RegisterFunc(&e.funcs.getAllocatorWithDefaultOptions, e.api.GetAllocatorWithDefaultOptions)
	purego.RegisterFunc(&e.funcs.allocatorFree, e.api.AllocatorFree)

	// memory info
	purego.RegisterFunc(&e.funcs.createCpuMemoryInfo, e.api.CreateCpuMemoryInfo)
	purego.RegisterFunc(&e.funcs.releaseMemoryInfo, e.api.ReleaseMemoryInfo)

	// CUDA
	purego.RegisterFunc(&e.funcs.createCUDAProviderOptions, e.api.CreateCUDAProviderOptions)
	purego.RegisterFunc(&e.funcs.releaseCUDAProviderOptions, e.api.ReleaseCUDAProviderOptions)
	purego.RegisterFunc(&e.funcs.updateCUDAProviderOptions, e.api.UpdateCUDAProviderOptions)
	purego.RegisterFunc(&e.funcs.appendExecutionProvider_CUDA_V2, e.api.SessionOptionsAppendExecutionProvider_CUDA_V2)

	// TensorRT
	purego.RegisterFunc(&e.funcs.createTensorRTProviderOptions, e.api.CreateTensorRTProviderOptions)
	purego.RegisterFunc(&e.funcs.releaseTensorRTProviderOptions, e.api.ReleaseTensorRTProviderOptions)
	purego.RegisterFunc(&e.funcs.updateTensorRTProviderOptions, e.api.UpdateTensorRTProviderOptions)
	purego.RegisterFunc(&e.funcs.appendExecutionProvider_TensorRT_V2, e.api.SessionOptionsAppendExecutionProvider_TensorRT_V2)

	// session options
	purego.RegisterFunc(&e.funcs.createSessionOptions, e.api.CreateSessionOptions)
	purego.RegisterFunc(&e.funcs.setIntraOpNumThreads, e.api.SetIntraOpNumThreads)
	purego.RegisterFunc(&e.funcs.sessionOptionsAppendExecutionProvider, e.api.SessionOptionsAppendExecutionProvider)
	purego.RegisterFunc(&e.funcs.releaseSessionOptions, e.api.ReleaseSessionOptions)
	purego.RegisterFunc(&e.funcs.enableCpuMemArena, e.api.EnableCpuMemArena)
	purego.RegisterFunc(&e.funcs.disableCpuMemArena, e.api.DisableCpuMemArena)

	// session
	purego.RegisterFunc(&e.funcs.createSession, e.api.CreateSession)
	purego.RegisterFunc(&e.funcs.createSessionFromArray, e.api.CreateSessionFromArray)
	purego.RegisterFunc(&e.funcs.sessionGetInputCount, e.api.SessionGetInputCount)
	purego.RegisterFunc(&e.funcs.sessionGetOutputCount, e.api.SessionGetOutputCount)
	purego.RegisterFunc(&e.funcs.sessionGetInputName, e.api.SessionGetInputName)
	purego.RegisterFunc(&e.funcs.sessionGetOutputName, e.api.SessionGetOutputName)
	purego.RegisterFunc(&e.funcs.sessionGetInputTypeInfo, e.api.SessionGetInputTypeInfo)
	purego.RegisterFunc(&e.funcs.sessionGetOutputTypeInfo, e.api.SessionGetOutputTypeInfo)
	purego.RegisterFunc(&e.funcs.castTypeInfoToTensorInfo, e.api.CastTypeInfoToTensorInfo)
	purego.RegisterFunc(&e.funcs.releaseTypeInfo, e.api.ReleaseTypeInfo)
	purego.RegisterFunc(&e.funcs.run, e.api.Run)
	purego.RegisterFunc(&e.funcs.releaseSession, e.api.ReleaseSession)

	// tensor, value
	purego.RegisterFunc(&e.funcs.createTensorWithDataAsOrtValue, e.api.CreateTensorWithDataAsOrtValue)
	purego.RegisterFunc(&e.funcs.getValueType, e.api.GetValueType)
	purego.RegisterFunc(&e.funcs.getTensorMutableData, e.api.GetTensorMutableData)
	purego.RegisterFunc(&e.funcs.getTensorTypeAndShape, e.api.GetTensorTypeAndShape)
	purego.RegisterFunc(&e.funcs.getTensorElementType, e.api.GetTensorElementType)
	purego.RegisterFunc(&e.funcs.getDimensionsCount, e.api.GetDimensionsCount)
	purego.RegisterFunc(&e.funcs.getDimensions, e.api.GetDimensions)
	purego.RegisterFunc(&e.funcs.getTensorShapeElementCount, e.api.GetTensorShapeElementCount)
	purego.RegisterFunc(&e.funcs.releaseValue, e.api.ReleaseValue)
	purego.RegisterFunc(&e.funcs.releaseTensorTypeAndShapeInfo, e.api.ReleaseTensorTypeAndShapeInfo)

	// provider
	purego.RegisterFunc(&e.funcs.getAvailableProviders, e.api.GetAvailableProviders)
	purego.RegisterFunc(&e.funcs.releaseAvailableProviders, e.api.ReleaseAvailableProviders)

	return nil
}

func (e *Engine) initEnv(name string) error {
	namePtr, err := stringToCString(name)
	if err != nil {
		return err
	}
	status := e.funcs.createEnv(LogError, namePtr, &e.envHandle)
	if err := e.checkStatus(status); err != nil {
		return fmt.Errorf("failed to create env: %w", err)
	}
	return nil
}

func (e *Engine) initMemInfo() error {
	var memInfo MemoryInfoHandle
	status := e.funcs.createCpuMemoryInfo(DeviceAllocator, DefaultMemType, &memInfo)
	if err := e.checkStatus(status); err != nil {
		return fmt.Errorf("failed to create cpu memory info: %v", err)
	}
	e.memInfo = memInfo
	return nil
}

// GetVersion 获取版本字符串，例如：1.23.2
func (e *Engine) GetVersion() string {
	if e.funcs.getVersionString == nil {
		return "unknown"
	}
	return e.funcs.getVersionString()
}

// GetAvailableProviders 获取当前 onnxruntime 构建支持的 Execution Provider 名称列表，
// 可在调用 SessionOptions.EnableCUDA / EnableTensorRT 前判断环境是否具备对应 EP：
//
// # Example:
//
//	opts, _ := engine.NewSessionOptions()
//	for _, p := range engine.GetAvailableProviders() {
//		if p == "CUDAExecutionProvider" {
//			opts.EnableCUDA(nil)
//		}
//	}
//
// # Returns:
//
//	[]string: 名称列表；查询失败时为 nil。目前 onnxruntime 支持的 EP 有：
//	  CPUExecutionProvider         CPU，内置，始终存在
//	  AzureExecutionProvider       Azure
//	  CUDAExecutionProvider        NVIDIA CUDA
//	  CUDANHWCExecutionProvider    NVIDIA CUDA NHWC
//	  TensorrtExecutionProvider    NVIDIA TensorRT
//	  ROCMExecutionProvider        AMD ROCm
//	  MIGraphXExecutionProvider    AMD MIGraphX
//	  DmlExecutionProvider         DirectML（Windows）
//	  OpenVINOExecutionProvider    Intel OpenVINO
//	  DnnlExecutionProvider        Intel oneDNN
//	  QNNExecutionProvider         Qualcomm QNN
//	  SNPEExecutionProvider        Qualcomm SNPE
//	  CoreMLExecutionProvider      Apple CoreML
//	  ArmNNExecutionProvider       Arm NN
//	  ACLExecutionProvider         Arm Compute Library
//	  NnapiExecutionProvider       Android NNAPI
//	  XnnpackExecutionProvider     XNNPACK
//	  VitisAIExecutionProvider     AMD Vitis AI
//	  CANNExecutionProvider        Huawei CANN
//	  VSINPUExecutionProvider      VeriSilicon NPU
//	  RknpuExecutionProvider       Rockchip NPU
//	  WebNNExecutionProvider       WebNN
//	  JsExecutionProvider          JavaScript
//	  TvmExecutionProvider         Apache TVM
//
// 具体返回范围取决于编译时启用的 EP，不同平台结果可能不同。
func (e *Engine) GetAvailableProviders() []string {
	var providers **byte
	var count int32
	if err := e.checkStatus(e.funcs.getAvailableProviders(&providers, &count)); err != nil {
		return nil
	}
	defer e.funcs.releaseAvailableProviders(providers, count)

	names := make([]string, 0, count)
	for _, p := range unsafe.Slice(providers, count) {
		names = append(names, cStringToString(p))
	}
	return names
}

// cleanup 释放 env / memory info，可重复调用
func (e *Engine) cleanup() {
	if e.memInfo != 0 {
		e.funcs.releaseMemoryInfo(e.memInfo)
		e.memInfo = 0
	}
	if e.envHandle != 0 {
		e.funcs.releaseEnv(e.envHandle)
		e.envHandle = 0
	}
}

// Destroy 释放资源
//
// 调用顺序：session 由 env 创建并引用 env，必须先释放所有 Session（及由它们产出的 Value），
// 再调用本方法。例如：
//
//	engine, _ := ort.NewEngine(ort.DefaultLibraryPath())
//	defer engine.Destroy() // 最先 defer，最后执行
//	session, _ := engine.NewSession("model.onnx", nil)
//	defer session.Destroy()
//
// Destroy 之后包级 NewTensor 会返回 "engine not initialized"，不会再访问已释放的句柄。
func (e *Engine) Destroy() {
	e.cleanup()

	// 避免 defaultEngine 悬垂：否则后续 NewTensor 会把已释放的 MemoryInfoHandle 交给 ORT
	defaultEngineMu.Lock()
	if defaultEngine == e {
		defaultEngine = nil
	}
	defaultEngineMu.Unlock()

	e.handle = 0
}

// checkStatus 检查状态
func (e *Engine) checkStatus(status StatusHandle) error {
	if status == 0 {
		return nil
	}
	defer e.funcs.releaseStatus(status)

	code := e.funcs.getErrorCode(status)
	msgPtr := e.funcs.getErrorMessage(status)
	msg := cStringToString((*byte)(msgPtr))

	return fmt.Errorf("onnxruntime error [code %d]: %s", code, msg)
}
