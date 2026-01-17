package ort

import (
	"fmt"
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

var defaultEngine *Engine

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
		return nil, err
	}
	if err := e.initEnv(DefaultEnvName); err != nil {
		return nil, err
	}
	if err := e.initMemInfo(); err != nil {
		return nil, err
	}

	defaultEngine = e

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

	// session options
	purego.RegisterFunc(&e.funcs.createSessionOptions, e.api.CreateSessionOptions)
	purego.RegisterFunc(&e.funcs.setIntraOpNumThreads, e.api.SetIntraOpNumThreads)
	purego.RegisterFunc(&e.funcs.sessionOptionsAppendExecutionProvider, e.api.SessionOptionsAppendExecutionProvider)
	purego.RegisterFunc(&e.funcs.releaseSessionOptions, e.api.ReleaseSessionOptions)

	// session
	purego.RegisterFunc(&e.funcs.createSession, e.api.CreateSession)
	purego.RegisterFunc(&e.funcs.createSessionFromArray, e.api.CreateSessionFromArray)
	purego.RegisterFunc(&e.funcs.sessionGetInputCount, e.api.SessionGetInputCount)
	purego.RegisterFunc(&e.funcs.sessionGetOutputCount, e.api.SessionGetOutputCount)
	purego.RegisterFunc(&e.funcs.sessionGetInputName, e.api.SessionGetInputName)
	purego.RegisterFunc(&e.funcs.sessionGetOutputName, e.api.SessionGetOutputName)
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
	status := e.funcs.createEnv(LogWarning, namePtr, &e.envHandle)
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

// Destroy 释放资源
func (e *Engine) Destroy() {
	if e.memInfo != 0 {
		e.funcs.releaseMemoryInfo(e.memInfo)
		e.memInfo = 0
	}
	if e.envHandle != 0 {
		e.funcs.releaseEnv(e.envHandle)
		e.envHandle = 0
	}
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
