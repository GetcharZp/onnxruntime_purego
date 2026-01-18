package ort

import "unsafe"

type ortApiBase struct {
	GetAPI           uintptr
	GetVersionString uintptr
}

type ortApi struct {
	CreateStatus                                        uintptr // 0
	GetErrorCode                                        uintptr // 1
	GetErrorMessage                                     uintptr // 2
	CreateEnv                                           uintptr // 3
	CreateEnvWithCustomLogger                           uintptr // 4
	EnableTelemetryEvents                               uintptr // 5
	DisableTelemetryEvents                              uintptr // 6
	CreateSession                                       uintptr // 7
	CreateSessionFromArray                              uintptr // 8
	Run                                                 uintptr // 9
	CreateSessionOptions                                uintptr // 10
	SetOptimizedModelFilePath                           uintptr // 11
	CloneSessionOptions                                 uintptr // 12
	SetSessionExecutionMode                             uintptr // 13
	EnableProfiling                                     uintptr // 14
	DisableProfiling                                    uintptr // 15
	EnableMemPattern                                    uintptr // 16
	DisableMemPattern                                   uintptr // 17
	EnableCpuMemArena                                   uintptr // 18
	DisableCpuMemArena                                  uintptr // 19
	SetSessionLogId                                     uintptr // 20
	SetSessionLogVerbosityLevel                         uintptr // 21
	SetSessionLogSeverityLevel                          uintptr // 22
	SetSessionGraphOptimizationLevel                    uintptr // 23
	SetIntraOpNumThreads                                uintptr // 24
	SetInterOpNumThreads                                uintptr // 25
	CreateCustomOpDomain                                uintptr // 26
	CustomOpDomain_Add                                  uintptr // 27
	AddCustomOpDomain                                   uintptr // 28
	RegisterCustomOpsLibrary                            uintptr // 29
	SessionGetInputCount                                uintptr // 30
	SessionGetOutputCount                               uintptr // 31
	SessionGetOverridableInitializerCount               uintptr // 32
	SessionGetInputTypeInfo                             uintptr // 33
	SessionGetOutputTypeInfo                            uintptr // 34
	SessionGetOverridableInitializerTypeInfo            uintptr // 35
	SessionGetInputName                                 uintptr // 36
	SessionGetOutputName                                uintptr // 37
	SessionGetOverridableInitializerName                uintptr // 38
	CreateRunOptions                                    uintptr // 39
	RunOptionsSetRunLogVerbosityLevel                   uintptr // 40
	RunOptionsSetRunLogSeverityLevel                    uintptr // 41
	RunOptionsSetRunTag                                 uintptr // 42
	RunOptionsGetRunLogVerbosityLevel                   uintptr // 43
	RunOptionsGetRunLogSeverityLevel                    uintptr // 44
	RunOptionsGetRunTag                                 uintptr // 45
	RunOptionsSetTerminate                              uintptr // 46
	RunOptionsUnsetTerminate                            uintptr // 47
	CreateTensorAsOrtValue                              uintptr // 48
	CreateTensorWithDataAsOrtValue                      uintptr // 49
	IsTensor                                            uintptr // 50
	GetTensorMutableData                                uintptr // 51
	FillStringTensor                                    uintptr // 52
	GetStringTensorDataLength                           uintptr // 53
	GetStringTensorContent                              uintptr // 54
	CastTypeInfoToTensorInfo                            uintptr // 55
	GetOnnxTypeFromTypeInfo                             uintptr // 56
	CreateTensorTypeAndShapeInfo                        uintptr // 57
	SetTensorElementType                                uintptr // 58
	SetDimensions                                       uintptr // 59
	GetTensorElementType                                uintptr // 60
	GetDimensionsCount                                  uintptr // 61
	GetDimensions                                       uintptr // 62
	GetSymbolicDimensions                               uintptr // 63
	GetTensorShapeElementCount                          uintptr // 64
	GetTensorTypeAndShape                               uintptr // 65
	GetTypeInfo                                         uintptr // 66
	GetValueType                                        uintptr // 67
	CreateMemoryInfo                                    uintptr // 68
	CreateCpuMemoryInfo                                 uintptr // 69
	CompareMemoryInfo                                   uintptr // 70
	MemoryInfoGetName                                   uintptr // 71
	MemoryInfoGetId                                     uintptr // 72
	MemoryInfoGetMemType                                uintptr // 73
	MemoryInfoGetType                                   uintptr // 74
	AllocatorAlloc                                      uintptr // 75
	AllocatorFree                                       uintptr // 76
	AllocatorGetInfo                                    uintptr // 77
	GetAllocatorWithDefaultOptions                      uintptr // 78
	AddFreeDimensionOverride                            uintptr // 79
	GetValue                                            uintptr // 80
	GetValueCount                                       uintptr // 81
	CreateValue                                         uintptr // 82
	CreateOpaqueValue                                   uintptr // 83
	GetOpaqueValue                                      uintptr // 84
	KernelInfoGetAttribute_float                        uintptr // 85
	KernelInfoGetAttribute_int64                        uintptr // 86
	KernelInfoGetAttribute_string                       uintptr // 87
	KernelContext_GetInputCount                         uintptr // 88
	KernelContext_GetOutputCount                        uintptr // 89
	KernelContext_GetInput                              uintptr // 90
	KernelContext_GetOutput                             uintptr // 91
	ReleaseEnv                                          uintptr // 92
	ReleaseStatus                                       uintptr // 93
	ReleaseMemoryInfo                                   uintptr // 94
	ReleaseSession                                      uintptr // 95
	ReleaseValue                                        uintptr // 96
	ReleaseRunOptions                                   uintptr // 97
	ReleaseTypeInfo                                     uintptr // 98
	ReleaseTensorTypeAndShapeInfo                       uintptr // 99
	ReleaseSessionOptions                               uintptr // 100
	ReleaseCustomOpDomain                               uintptr // 101
	GetDenotationFromTypeInfo                           uintptr // 102
	CastTypeInfoToMapTypeInfo                           uintptr // 103
	CastTypeInfoToSequenceTypeInfo                      uintptr // 104
	GetMapKeyType                                       uintptr // 105
	GetMapValueType                                     uintptr // 106
	GetSequenceElementType                              uintptr // 107
	ReleaseMapTypeInfo                                  uintptr // 108
	ReleaseSequenceTypeInfo                             uintptr // 109
	SessionEndProfiling                                 uintptr // 110
	SessionGetModelMetadata                             uintptr // 111
	ModelMetadataGetProducerName                        uintptr // 112
	ModelMetadataGetGraphName                           uintptr // 113
	ModelMetadataGetDomain                              uintptr // 114
	ModelMetadataGetDescription                         uintptr // 115
	ModelMetadataLookupCustomMetadataMap                uintptr // 116
	ModelMetadataGetVersion                             uintptr // 117
	ReleaseModelMetadata                                uintptr // 118
	CreateEnvWithGlobalThreadPools                      uintptr // 119
	DisablePerSessionThreads                            uintptr // 120
	CreateThreadingOptions                              uintptr // 121
	ReleaseThreadingOptions                             uintptr // 122
	ModelMetadataGetCustomMetadataMapKeys               uintptr // 123
	AddFreeDimensionOverrideByName                      uintptr // 124
	GetAvailableProviders                               uintptr // 125
	ReleaseAvailableProviders                           uintptr // 126
	GetStringTensorElementLength                        uintptr // 127
	GetStringTensorElement                              uintptr // 128
	FillStringTensorElement                             uintptr // 129
	AddSessionConfigEntry                               uintptr // 130
	CreateAllocator                                     uintptr // 131
	ReleaseAllocator                                    uintptr // 132
	RunWithBinding                                      uintptr // 133
	CreateIoBinding                                     uintptr // 134
	ReleaseIoBinding                                    uintptr // 135
	BindInput                                           uintptr // 136
	BindOutput                                          uintptr // 137
	BindOutputToDevice                                  uintptr // 138
	GetBoundOutputNames                                 uintptr // 139
	GetBoundOutputValues                                uintptr // 140
	ClearBoundInputs                                    uintptr // 141
	ClearBoundOutputs                                   uintptr // 142
	TensorAt                                            uintptr // 143
	CreateAndRegisterAllocator                          uintptr // 144
	SetLanguageProjection                               uintptr // 145
	SessionGetProfilingStartTimeNs                      uintptr // 146
	SetGlobalIntraOpNumThreads                          uintptr // 147
	SetGlobalInterOpNumThreads                          uintptr // 148
	SetGlobalSpinControl                                uintptr // 149
	AddInitializer                                      uintptr // 150
	CreateEnvWithCustomLoggerAndGlobalThreadPools       uintptr // 151
	SessionOptionsAppendExecutionProvider_CUDA          uintptr // 152
	SessionOptionsAppendExecutionProvider_ROCM          uintptr // 153
	SessionOptionsAppendExecutionProvider_OpenVINO      uintptr // 154
	SetGlobalDenormalAsZero                             uintptr // 155
	CreateArenaCfg                                      uintptr // 156
	ReleaseArenaCfg                                     uintptr // 157
	ModelMetadataGetGraphDescription                    uintptr // 158
	SessionOptionsAppendExecutionProvider_TensorRT      uintptr // 159
	SetCurrentGpuDeviceId                               uintptr // 160
	GetCurrentGpuDeviceId                               uintptr // 161
	KernelInfoGetAttributeArray_float                   uintptr // 162
	KernelInfoGetAttributeArray_int64                   uintptr // 163
	CreateArenaCfgV2                                    uintptr // 164
	AddRunConfigEntry                                   uintptr // 165
	CreatePrepackedWeightsContainer                     uintptr // 166
	ReleasePrepackedWeightsContainer                    uintptr // 167
	CreateSessionWithPrepackedWeightsContainer          uintptr // 168
	CreateSessionFromArrayWithPrepackedWeightsContainer uintptr // 169
	SessionOptionsAppendExecutionProvider_TensorRT_V2   uintptr // 170
	CreateTensorRTProviderOptions                       uintptr // 171
	UpdateTensorRTProviderOptions                       uintptr // 172
	GetTensorRTProviderOptionsAsString                  uintptr // 173
	ReleaseTensorRTProviderOptions                      uintptr // 174
	EnableOrtCustomOps                                  uintptr // 175
	RegisterAllocator                                   uintptr // 176
	UnregisterAllocator                                 uintptr // 177
	IsSparseTensor                                      uintptr // 178
	CreateSparseTensorAsOrtValue                        uintptr // 179
	FillSparseTensorCoo                                 uintptr // 180
	FillSparseTensorCsr                                 uintptr // 181
	FillSparseTensorBlockSparse                         uintptr // 182
	CreateSparseTensorWithValuesAsOrtValue              uintptr // 183
	UseCooIndices                                       uintptr // 184
	UseCsrIndices                                       uintptr // 185
	UseBlockSparseIndices                               uintptr // 186
	GetSparseTensorFormat                               uintptr // 187
	GetSparseTensorValuesTypeAndShape                   uintptr // 188
	GetSparseTensorValues                               uintptr // 189
	GetSparseTensorIndicesTypeShape                     uintptr // 190
	GetSparseTensorIndices                              uintptr // 191
	HasValue                                            uintptr // 192
	KernelContext_GetGPUComputeStream                   uintptr // 193
	GetTensorMemoryInfo                                 uintptr // 194
	GetExecutionProviderApi                             uintptr // 195
	SessionOptionsSetCustomCreateThreadFn               uintptr // 196
	SessionOptionsSetCustomThreadCreationOptions        uintptr // 197
	SessionOptionsSetCustomJoinThreadFn                 uintptr // 198
	SetGlobalCustomCreateThreadFn                       uintptr // 199
	SetGlobalCustomThreadCreationOptions                uintptr // 200
	SetGlobalCustomJoinThreadFn                         uintptr // 201
	SynchronizeBoundInputs                              uintptr // 202
	SynchronizeBoundOutputs                             uintptr // 203
	SessionOptionsAppendExecutionProvider_CUDA_V2       uintptr // 204
	CreateCUDAProviderOptions                           uintptr // 205
	UpdateCUDAProviderOptions                           uintptr // 206
	GetCUDAProviderOptionsAsString                      uintptr // 207
	ReleaseCUDAProviderOptions                          uintptr // 208
	SessionOptionsAppendExecutionProvider_MIGraphX      uintptr // 209
	AddExternalInitializers                             uintptr // 210
	CreateOpAttr                                        uintptr // 211
	ReleaseOpAttr                                       uintptr // 212
	CreateOp                                            uintptr // 213
	InvokeOp                                            uintptr // 214
	ReleaseOp                                           uintptr // 215
	SessionOptionsAppendExecutionProvider               uintptr // 216
	CopyKernelInfo                                      uintptr // 217
	ReleaseKernelInfo                                   uintptr // 218
	GetTrainingApi                                      uintptr // 219
	SessionOptionsAppendExecutionProvider_CANN          uintptr // 220
	CreateCANNProviderOptions                           uintptr // 221
	UpdateCANNProviderOptions                           uintptr // 222
	GetCANNProviderOptionsAsString                      uintptr // 223
	ReleaseCANNProviderOptions                          uintptr // 224
	MemoryInfoGetDeviceType                             uintptr // 225
	UpdateEnvWithCustomLogLevel                         uintptr // 226
	SetGlobalIntraOpThreadAffinity                      uintptr // 227
	RegisterCustomOpsLibrary_V2                         uintptr // 228
	RegisterCustomOpsUsingFunction                      uintptr // 229
	KernelInfo_GetInputCount                            uintptr // 230
	KernelInfo_GetOutputCount                           uintptr // 231
	KernelInfo_GetInputName                             uintptr // 232
	KernelInfo_GetOutputName                            uintptr // 233
	KernelInfo_GetInputTypeInfo                         uintptr // 234
	KernelInfo_GetOutputTypeInfo                        uintptr // 235
	KernelInfoGetAttribute_tensor                       uintptr // 236
	HasSessionConfigEntry                               uintptr // 237
	GetSessionConfigEntry                               uintptr // 238
	SessionOptionsAppendExecutionProvider_Dnnl          uintptr // 239
	CreateDnnlProviderOptions                           uintptr // 240
	UpdateDnnlProviderOptions                           uintptr // 241
	GetDnnlProviderOptionsAsString                      uintptr // 242
	ReleaseDnnlProviderOptions                          uintptr // 243
	KernelInfo_GetNodeName                              uintptr // 244
	KernelInfo_GetLogger                                uintptr // 245
	KernelContext_GetLogger                             uintptr // 246
	Logger_LogMessage                                   uintptr // 247
	Logger_GetLoggingSeverityLevel                      uintptr // 248
	KernelInfoGetConstantInput_tensor                   uintptr // 249
	CastTypeInfoToOptionalTypeInfo                      uintptr // 250
	GetOptionalContainedTypeInfo                        uintptr // 251
	GetResizedStringTensorElementBuffer                 uintptr // 252
	KernelContext_GetAllocator                          uintptr // 253
	GetBuildInfoString                                  uintptr // 254
	CreateROCMProviderOptions                           uintptr // 255
	UpdateROCMProviderOptions                           uintptr // 256
	GetROCMProviderOptionsAsString                      uintptr // 257
	ReleaseROCMProviderOptions                          uintptr // 258
	CreateAndRegisterAllocatorV2                        uintptr // 259
	RunAsync                                            uintptr // 260
	UpdateTensorRTProviderOptionsWithValue              uintptr // 261
	GetTensorRTProviderOptionsByName                    uintptr // 262
	UpdateCUDAProviderOptionsWithValue                  uintptr // 263
	GetCUDAProviderOptionsByName                        uintptr // 264
	KernelContext_GetResource                           uintptr // 265
	SetUserLoggingFunction                              uintptr // 266
	ShapeInferContext_GetInputCount                     uintptr // 267
	ShapeInferContext_GetInputTypeShape                 uintptr // 268
	ShapeInferContext_GetAttribute                      uintptr // 269
	ShapeInferContext_SetOutputTypeShape                uintptr // 270
	SetSymbolicDimensions                               uintptr // 271
	ReadOpAttr                                          uintptr // 272
	SetDeterministicCompute                             uintptr // 273
	KernelContext_ParallelFor                           uintptr // 274
	SessionOptionsAppendExecutionProvider_OpenVINO_V2   uintptr // 275
	SessionOptionsAppendExecutionProvider_VitisAI       uintptr // 276
	KernelContext_GetScratchBuffer                      uintptr // 277
	KernelInfoGetAllocator                              uintptr // 278
	AddExternalInitializersFromFilesInMemory            uintptr // 279
	CreateLoraAdapter                                   uintptr // 280
	CreateLoraAdapterFromArray                          uintptr // 281
	ReleaseLoraAdapter                                  uintptr // 282
	RunOptionsAddActiveLoraAdapter                      uintptr // 283
	SetEpDynamicOptions                                 uintptr // 284
	ReleaseValueInfo                                    uintptr // 285
	ReleaseNode                                         uintptr // 286
	ReleaseGraph                                        uintptr // 287
	ReleaseModel                                        uintptr // 288
	GetValueInfoName                                    uintptr // 289
	GetValueInfoTypeInfo                                uintptr // 290
	GetModelEditorApi                                   uintptr // 291
	CreateTensorWithDataAndDeleterAsOrtValue            uintptr // 292
	SessionOptionsSetLoadCancellationFlag               uintptr // 293
	GetCompileApi                                       uintptr // 294
	CreateKeyValuePairs                                 uintptr // 295
	AddKeyValuePair                                     uintptr // 296
	GetKeyValue                                         uintptr // 297
	GetKeyValuePairs                                    uintptr // 298
	RemoveKeyValuePair                                  uintptr // 299
	ReleaseKeyValuePairs                                uintptr // 300
	RegisterExecutionProviderLibrary                    uintptr // 301
	UnregisterExecutionProviderLibrary                  uintptr // 302
	GetEpDevices                                        uintptr // 303
	SessionOptionsAppendExecutionProvider_V2            uintptr // 304
	SessionOptionsSetEpSelectionPolicy                  uintptr // 305
	SessionOptionsSetEpSelectionPolicyDelegate          uintptr // 306
	HardwareDevice_Type                                 uintptr // 307
	HardwareDevice_VendorId                             uintptr // 308
	HardwareDevice_Vendor                               uintptr // 309
	HardwareDevice_DeviceId                             uintptr // 310
	HardwareDevice_Metadata                             uintptr // 311
	EpDevice_EpName                                     uintptr // 312
	EpDevice_EpVendor                                   uintptr // 313
	EpDevice_EpMetadata                                 uintptr // 314
	EpDevice_EpOptions                                  uintptr // 315
	EpDevice_Device                                     uintptr // 316
	GetEpApi                                            uintptr // 317
	GetTensorSizeInBytes                                uintptr // 318
	AllocatorGetStats                                   uintptr // 319
	CreateMemoryInfo_V2                                 uintptr // 320
	ValueInfo_GetValueProducer                          uintptr // 321
	ValueInfo_GetValueNumConsumers                      uintptr // 322
	ValueInfo_GetValueConsumers                         uintptr // 323
	ValueInfo_GetInitializerValue                       uintptr // 324
	ValueInfo_GetExternalInitializerInfo                uintptr // 325
	ValueInfo_IsRequiredGraphInput                      uintptr // 326
	ValueInfo_IsOptionalGraphInput                      uintptr // 327
	ValueInfo_IsGraphOutput                             uintptr // 328
	ValueInfo_IsConstantInitializer                     uintptr // 329
	ValueInfo_IsFromOuterScope                          uintptr // 330
	Graph_GetName                                       uintptr // 331
	Graph_GetModelPath                                  uintptr // 332
	Graph_GetOnnxIRVersion                              uintptr // 333
	Graph_GetNumOperatorSets                            uintptr // 334
	Graph_GetOperatorSets                               uintptr // 335
	Graph_GetNumInputs                                  uintptr // 336
	Graph_GetInputs                                     uintptr // 337
	Graph_GetNumOutputs                                 uintptr // 338
	Graph_GetOutputs                                    uintptr // 339
	Graph_GetNumInitializers                            uintptr // 340
	Graph_GetInitializers                               uintptr // 341
	Graph_GetNumNodes                                   uintptr // 342
	Graph_GetNodes                                      uintptr // 343
	Graph_GetParentNode                                 uintptr // 344
	Graph_GetGraphView                                  uintptr // 345
	Node_GetId                                          uintptr // 346
	Node_GetName                                        uintptr // 347
	Node_GetOperatorType                                uintptr // 348
	Node_GetDomain                                      uintptr // 349
	Node_GetSinceVersion                                uintptr // 350
	Node_GetNumInputs                                   uintptr // 351
	Node_GetInputs                                      uintptr // 352
	Node_GetNumOutputs                                  uintptr // 353
	Node_GetOutputs                                     uintptr // 354
	Node_GetNumImplicitInputs                           uintptr // 355
	Node_GetImplicitInputs                              uintptr // 356
	Node_GetNumAttributes                               uintptr // 357
	Node_GetAttributes                                  uintptr // 358
	Node_GetAttributeByName                             uintptr // 359
	OpAttr_GetTensorAttributeAsOrtValue                 uintptr // 360
	OpAttr_GetType                                      uintptr // 361
	OpAttr_GetName                                      uintptr // 362
	Node_GetNumSubgraphs                                uintptr // 363
	Node_GetSubgraphs                                   uintptr // 364
	Node_GetGraph                                       uintptr // 365
	Node_GetEpName                                      uintptr // 366
	ReleaseExternalInitializerInfo                      uintptr // 367
	CreateSharedAllocator                               uintptr // 368
	GetSharedAllocator                                  uintptr // 369
	ReleaseSharedAllocator                              uintptr // 370
	GetTensorData                                       uintptr // 371
	GetSessionOptionsConfigEntries                      uintptr // 372
	SessionGetMemoryInfoForInputs                       uintptr // 373
	SessionGetMemoryInfoForOutputs                      uintptr // 374
	SessionGetEpDeviceForInputs                         uintptr // 375
	CreateSyncStreamForEpDevice                         uintptr // 376
	ReleaseSyncStream                                   uintptr // 377
	CopyTensors                                         uintptr // 378
	Graph_GetModelMetadata                              uintptr // 379
	GetModelCompatibilityForEpDevices                   uintptr // 380
	CreateExternalInitializerInfo                       uintptr // 381
}

// OrtStatus is an opaque pointer to an ONNX Runtime status object.
type (
	StatusHandle                 uintptr
	EnvHandle                    uintptr
	SessionHandle                uintptr
	SessionOptionsHandle         uintptr
	ValueHandle                  uintptr
	AllocatorHandle              uintptr
	MemoryInfoHandle             uintptr
	TensorTypeAndShapeInfoHandle uintptr
	RunOptionsHandle             uintptr
	TypeInfoHandle               uintptr
	CUDAProviderOptionsV2Handle  uintptr
	ErrorCode                    int32
	LoggingLevel                 int32
	OnnxType                     int32
	TensorElementDataType        int32
	AllocatorType                int32
	MemType                      int32
)

const (
	TensorElementDataTypeUndefined      TensorElementDataType = 0
	TensorElementDataTypeFloat          TensorElementDataType = 1
	TensorElementDataTypeUint8          TensorElementDataType = 2
	TensorElementDataTypeInt8           TensorElementDataType = 3
	TensorElementDataTypeUint16         TensorElementDataType = 4
	TensorElementDataTypeInt16          TensorElementDataType = 5
	TensorElementDataTypeInt32          TensorElementDataType = 6
	TensorElementDataTypeInt64          TensorElementDataType = 7
	TensorElementDataTypeString         TensorElementDataType = 8
	TensorElementDataTypeBool           TensorElementDataType = 9
	TensorElementDataTypeFloat16        TensorElementDataType = 10
	TensorElementDataTypeDouble         TensorElementDataType = 11
	TensorElementDataTypeUint32         TensorElementDataType = 12
	TensorElementDataTypeUint64         TensorElementDataType = 13
	TensorElementDataTypeComplex64      TensorElementDataType = 14
	TensorElementDataTypeComplex128     TensorElementDataType = 15
	TensorElementDataTypeBFloat16       TensorElementDataType = 16
	TensorElementDataTypeFloat8E4M3FN   TensorElementDataType = 17
	TensorElementDataTypeFloat8E4M3FNUZ TensorElementDataType = 18
	TensorElementDataTypeFloat8E5M2     TensorElementDataType = 19
	TensorElementDataTypeFloat8E5M2FNUZ TensorElementDataType = 20
	TensorElementDataTypeUint4          TensorElementDataType = 21
	TensorElementDataTypeInt4           TensorElementDataType = 22
	TensorElementDataTypeFloat4E2M1     TensorElementDataType = 23
)

const (
	DeviceAllocator AllocatorType = 0
	ArenaAllocator  AllocatorType = 1

	DefaultMemType MemType = 0
)

// apiFuncs purego 绑定的 Go 函数
type apiFuncs struct {
	getVersionString func() string

	// status, error
	createStatus    func(ErrorCode, *byte) StatusHandle
	getErrorCode    func(StatusHandle) ErrorCode
	getErrorMessage func(StatusHandle) unsafe.Pointer
	releaseStatus   func(StatusHandle)

	// env
	createEnv  func(LoggingLevel, *byte, *EnvHandle) StatusHandle
	releaseEnv func(handle EnvHandle)

	// allocator
	getAllocatorWithDefaultOptions func(*AllocatorHandle) StatusHandle
	allocatorFree                  func(AllocatorHandle, unsafe.Pointer)

	// memory info
	createCpuMemoryInfo func(AllocatorType, MemType, *MemoryInfoHandle) StatusHandle
	releaseMemoryInfo   func(MemoryInfoHandle)

	// CUDA
	createCUDAProviderOptions       func(*CUDAProviderOptionsV2Handle) StatusHandle
	updateCUDAProviderOptions       func(CUDAProviderOptionsV2Handle, **byte, **byte, uintptr) StatusHandle
	releaseCUDAProviderOptions      func(CUDAProviderOptionsV2Handle)
	appendExecutionProvider_CUDA_V2 func(SessionOptionsHandle, CUDAProviderOptionsV2Handle) StatusHandle

	// session options
	createSessionOptions                  func(*SessionOptionsHandle) StatusHandle
	setIntraOpNumThreads                  func(SessionOptionsHandle, int32) StatusHandle
	sessionOptionsAppendExecutionProvider func(SessionOptionsHandle, *byte, **byte, **byte, uintptr) StatusHandle
	releaseSessionOptions                 func(SessionOptionsHandle)
	enableCpuMemArena                     func(SessionOptionsHandle) StatusHandle
	disableCpuMemArena                    func(SessionOptionsHandle) StatusHandle

	// session
	createSession          func(EnvHandle, unsafe.Pointer, SessionOptionsHandle, *SessionHandle) StatusHandle
	createSessionFromArray func(EnvHandle, unsafe.Pointer, uintptr, SessionOptionsHandle, *SessionHandle) StatusHandle
	sessionGetInputCount   func(SessionHandle, *uintptr) StatusHandle
	sessionGetOutputCount  func(SessionHandle, *uintptr) StatusHandle
	sessionGetInputName    func(SessionHandle, uintptr, AllocatorHandle, **byte) StatusHandle
	sessionGetOutputName   func(SessionHandle, uintptr, AllocatorHandle, **byte) StatusHandle
	run                    func(SessionHandle, uintptr, *unsafe.Pointer, *ValueHandle, uintptr, *unsafe.Pointer, uintptr, *ValueHandle) StatusHandle
	//run                    func(SessionHandle, uintptr, **byte, *ValueHandle, uintptr, **byte, uintptr, *ValueHandle) StatusHandle
	releaseSession func(SessionHandle)

	// tensor, value
	createTensorWithDataAsOrtValue func(MemoryInfoHandle, unsafe.Pointer, uintptr, *int64, uintptr, TensorElementDataType, *ValueHandle) StatusHandle
	getValueType                   func(ValueHandle, *OnnxType) StatusHandle
	getTensorMutableData           func(ValueHandle, *unsafe.Pointer) StatusHandle
	getTensorTypeAndShape          func(ValueHandle, *TensorTypeAndShapeInfoHandle) StatusHandle
	getTensorElementType           func(TensorTypeAndShapeInfoHandle, *TensorElementDataType) StatusHandle
	getDimensionsCount             func(TensorTypeAndShapeInfoHandle, *uintptr) StatusHandle
	getDimensions                  func(TensorTypeAndShapeInfoHandle, *int64, uintptr) StatusHandle
	getTensorShapeElementCount     func(TensorTypeAndShapeInfoHandle, *uintptr) StatusHandle
	releaseValue                   func(ValueHandle)
	releaseTensorTypeAndShapeInfo  func(TensorTypeAndShapeInfoHandle)

	// provider
	getAvailableProviders     func(***byte, *int32) StatusHandle
	releaseAvailableProviders func(**byte, int32) StatusHandle
}
