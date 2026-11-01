package ort

import (
	"fmt"
	"math"
	"unsafe"
)

// TensorElement 是 NewTensor / GetTensorData 支持的元素类型。
//
// 仅接受下列内置类型本身，命名类型（如 type Score float32）需转换后再传入。
type TensorElement interface {
	float32 | float64 | int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64 | bool
}

// Value 表示一个输入 / 输出张量，用完后需调用 Close
type Value struct {
	handle ValueHandle
	engine *Engine

	// data 持有传给 CreateTensorWithDataAsOrtValue 的底层数据引用。
	//
	// onnxruntime 只保存指针、不拷贝数据，若不在此保留引用，NewTensor 返回后
	// 该缓冲区即不可达，GC 会回收并复用，而 ORT 仍按原指针读取。
	data any

	dataType TensorElementDataType // 首次调用 info 时填充
	shape    []int64               // 同上；非 nil 表示已加载
}

// NewTensor 创建输入 Tensor
//
// # Params:
//
//	shape: 形状
//	data: 数据，支持 []float32 / []float64 / []int64 / []int32 / []int16 / []int8 /
//	      []uint64 / []uint32 / []uint16 / []uint8 / []bool
//
// onnxruntime 不会拷贝 data，只持有其指针，数据有效性由返回的 Value 托管。
// 因此复用同一个切片会同步影响已创建的 Tensor；调用 Value.Close() 后，
// 由 GetTensorData 取到的切片随之失效。
func NewTensor[T TensorElement](shape []int64, data []T) (*Value, error) {
	engine := getDefaultEngine()
	if engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}

	dataType, elemSize := tensorType[T]()

	// 元素个数由 shape 决定，而 onnxruntime 并不校验缓冲区长度，数据不足会越界读，
	// 这里提前拦截。shape 含动态维度(-1)或乘积溢出时交由 onnxruntime 报错。
	if want, ok := shapeElementCount(shape); ok && int64(len(data)) < want {
		return nil, fmt.Errorf("data length %d is less than the %d elements required by shape %v", len(data), want, shape)
	}

	// 创建 Value 句柄
	var handle ValueHandle
	status := engine.funcs.createTensorWithDataAsOrtValue(
		engine.memInfo,
		unsafe.Pointer(slicePtr(data)),
		uintptr(len(data))*elemSize,
		slicePtr(shape),
		uintptr(len(shape)),
		dataType,
		&handle,
	)
	if err := engine.checkStatus(status); err != nil {
		return nil, err
	}

	return &Value{handle: handle, engine: engine, data: data}, nil
}

// tensorType 返回 T 对应的 onnxruntime 元素类型与单元素字节数
func tensorType[T TensorElement]() (TensorElementDataType, uintptr) {
	var zero T
	switch any(zero).(type) {
	case float32:
		return TensorElementDataTypeFloat, 4
	case float64:
		return TensorElementDataTypeDouble, 8
	case int8:
		return TensorElementDataTypeInt8, 1
	case int16:
		return TensorElementDataTypeInt16, 2
	case int32:
		return TensorElementDataTypeInt32, 4
	case int64:
		return TensorElementDataTypeInt64, 8
	case uint8:
		return TensorElementDataTypeUint8, 1
	case uint16:
		return TensorElementDataTypeUint16, 2
	case uint32:
		return TensorElementDataTypeUint32, 4
	case uint64:
		return TensorElementDataTypeUint64, 8
	case bool:
		return TensorElementDataTypeBool, 1
	default:
		// TensorElement 已穷举所有类型，本分支不可达
		panic(fmt.Sprintf("ort: unsupported tensor element type %T", zero))
	}
}

// shapeElementCount 计算 shape 的元素总数。
//
// 含负数（动态维度）或乘积溢出时返回 ok=false —— 此时不自行拦截，交由 onnxruntime 校验。
func shapeElementCount(shape []int64) (int64, bool) {
	total := int64(1)
	for _, dim := range shape {
		if dim < 0 || (dim != 0 && total > math.MaxInt64/dim) {
			return 0, false
		}
		total *= dim
	}
	return total, true
}

// GetShape 获取 Tensor 的维度信息
//
// 返回的是内部缓存切片本身，调用方不应修改其内容。
func (v *Value) GetShape() ([]int64, error) {
	_, shape, err := v.info()
	return shape, err
}

// GetElementCount 获取 Tensor 中的元素总数
func (v *Value) GetElementCount() (int64, error) {
	_, shape, err := v.info()
	if err != nil {
		return 0, err
	}
	count, ok := shapeElementCount(shape)
	if !ok {
		return 0, fmt.Errorf("tensor has a dynamic shape %v", shape)
	}
	return count, nil
}

// GetTensorData 获取 Tensor 数据
//
// 返回的切片直接指向 onnxruntime 管理的内存，不发生拷贝，Value.Close() 后失效。
func GetTensorData[T TensorElement](v *Value) ([]T, error) {
	dataType, shape, err := v.info()
	if err != nil {
		return nil, err
	}

	if want, _ := tensorType[T](); dataType != want {
		var zero T
		return nil, fmt.Errorf("tensor element type mismatch: onnxruntime type %d does not match requested %T", dataType, zero)
	}

	count, ok := shapeElementCount(shape)
	if !ok {
		return nil, fmt.Errorf("tensor has a dynamic shape %v", shape)
	}

	var ptr unsafe.Pointer
	if err := v.engine.checkStatus(v.engine.funcs.getTensorMutableData(v.handle, &ptr)); err != nil {
		return nil, fmt.Errorf("failed to get tensor data: %w", err)
	}

	return unsafe.Slice((*T)(ptr), int(count)), nil
}

// info 获取并缓存 Tensor 的元素类型与形状，供 GetShape / GetElementCount / GetTensorData 复用，
// 保证每个 Value 最多只调用一次 GetTensorTypeAndShape。
func (v *Value) info() (TensorElementDataType, []int64, error) {
	if v.shape != nil {
		return v.dataType, v.shape, nil
	}

	f := v.engine.funcs
	var info TensorTypeAndShapeInfoHandle
	if err := v.engine.checkStatus(f.getTensorTypeAndShape(v.handle, &info)); err != nil {
		return 0, nil, fmt.Errorf("failed to get tensor type and shape: %w", err)
	}
	defer f.releaseTensorTypeAndShapeInfo(info)

	var dataType TensorElementDataType
	if err := v.engine.checkStatus(f.getTensorElementType(info, &dataType)); err != nil {
		return 0, nil, fmt.Errorf("failed to get tensor element type: %w", err)
	}

	var dimCount uintptr
	if err := v.engine.checkStatus(f.getDimensionsCount(info, &dimCount)); err != nil {
		return 0, nil, fmt.Errorf("failed to get dimensions count: %w", err)
	}

	// 标量张量的 dimCount 为 0，此时不能取 &shape[0]；make 得到的是非 nil 空切片
	shape := make([]int64, dimCount)
	if dimCount > 0 {
		if err := v.engine.checkStatus(f.getDimensions(info, &shape[0], dimCount)); err != nil {
			return 0, nil, fmt.Errorf("failed to get dimensions: %w", err)
		}
	}

	v.dataType, v.shape = dataType, shape
	return dataType, shape, nil
}

// Close 释放 Tensor
func (v *Value) Close() {
	if v.handle != 0 {
		v.engine.funcs.releaseValue(v.handle)
		v.handle = 0
	}
}
