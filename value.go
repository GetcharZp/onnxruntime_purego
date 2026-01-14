package ort

import (
	"fmt"
	"github.com/up-zero/gotool"
	"unsafe"
)

type Value struct {
	handle       ValueHandle
	engine       *Engine
	shape        []int64
	elementCount int
}

// NewTensor 初始化 Tensor
//
// # Params:
//
//	shape: 形状
//	data: 数据
func (e *Engine) NewTensor(shape []int64, data any) (*Value, error) {
	dataType, typeSize, dataLen, dataPtr, err := e.parseInputData(data)
	if err != nil {
		return nil, err
	}

	// 创建 Value 句柄
	var valHandle ValueHandle
	var shapePtr *int64
	if len(shape) > 0 {
		shapePtr = &shape[0]
	}

	status := e.funcs.createTensorWithDataAsOrtValue(
		e.memInfo,
		dataPtr,
		uintptr(dataLen)*typeSize,
		shapePtr,
		uintptr(len(shape)),
		dataType,
		&valHandle,
	)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}

	return &Value{handle: valHandle, engine: e}, nil
}

// GetShape 获取 Tensor 的维度信息
func (v *Value) GetShape() ([]int64, error) {
	if len(v.shape) > 0 {
		return v.shape, nil
	}

	info, err := v.getTypeAndShapeInfo()
	if err != nil {
		return nil, err
	}
	defer v.engine.funcs.releaseTensorTypeAndShapeInfo(info)

	var dimCount uintptr
	status := v.engine.funcs.getDimensionsCount(info, &dimCount)
	if err := v.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get dimensions count: %w", err)
	}

	v.shape = make([]int64, dimCount)
	status = v.engine.funcs.getDimensions(info, &v.shape[0], dimCount)
	if err := v.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get dimensions: %w", err)
	}

	return v.shape, nil
}

// GetElementCount 获取 Tensor 中的元素总数
func (v *Value) GetElementCount() (int, error) {
	if v.elementCount > 0 {
		return v.elementCount, nil
	}

	info, err := v.getTypeAndShapeInfo()
	if err != nil {
		return 0, err
	}
	defer v.engine.funcs.releaseTensorTypeAndShapeInfo(info)

	var elementCount uintptr
	status := v.engine.funcs.getTensorShapeElementCount(info, &elementCount)
	if err := v.engine.checkStatus(status); err != nil {
		return 0, fmt.Errorf("failed to get tensor shape element count: %w", err)
	}

	v.elementCount = int(elementCount)
	return v.elementCount, nil
}

// GetTensorData 获取 Tensor 数据
func GetTensorData[T gotool.Number](v *Value) ([]T, error) {
	elementCount, err := v.GetElementCount()
	if err != nil {
		return nil, err
	}

	info, err := v.getTypeAndShapeInfo()
	if err != nil {
		return nil, fmt.Errorf("failed to get tensor type and shape info: %w", err)
	}
	defer v.engine.funcs.releaseTensorTypeAndShapeInfo(info)

	var dataType TensorElementDataType
	status := v.engine.funcs.getTensorElementType(info, &dataType)
	if err := v.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get tensor element type: %w", err)
	}

	var ptr unsafe.Pointer
	status = v.engine.funcs.getTensorMutableData(v.handle, &ptr)
	if err := v.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to get tensor mutable data: %w", err)
	}

	var rawData any
	switch dataType {
	case TensorElementDataTypeFloat:
		rawData = unsafe.Slice((*float32)(ptr), elementCount)
	case TensorElementDataTypeDouble:
		rawData = unsafe.Slice((*float64)(ptr), elementCount)
	case TensorElementDataTypeInt64:
		rawData = unsafe.Slice((*int64)(ptr), elementCount)
	case TensorElementDataTypeInt32:
		rawData = unsafe.Slice((*int32)(ptr), elementCount)
	case TensorElementDataTypeInt16:
		rawData = unsafe.Slice((*int16)(ptr), elementCount)
	case TensorElementDataTypeInt8:
		rawData = unsafe.Slice((*int8)(ptr), elementCount)
	case TensorElementDataTypeUint64:
		rawData = unsafe.Slice((*uint64)(ptr), elementCount)
	case TensorElementDataTypeUint32:
		rawData = unsafe.Slice((*uint32)(ptr), elementCount)
	case TensorElementDataTypeUint16:
		rawData = unsafe.Slice((*uint16)(ptr), elementCount)
	case TensorElementDataTypeUint8:
		rawData = unsafe.Slice((*uint8)(ptr), elementCount)
	case TensorElementDataTypeBool:
		rawData = unsafe.Slice((*bool)(ptr), elementCount)

	default:
		return nil, fmt.Errorf("unsupported tensor element type: %d", dataType)
	}

	// 类型断言
	if data, ok := rawData.([]T); ok {
		return data, nil
	}

	var t T
	return nil, fmt.Errorf("tensor data type mismatch: actual ORT type %d does not match requested Go type %T", dataType, t)
}

func (v *Value) getTypeAndShapeInfo() (TensorTypeAndShapeInfoHandle, error) {
	var info TensorTypeAndShapeInfoHandle
	status := v.engine.funcs.getTensorTypeAndShape(v.handle, &info)
	if err := v.engine.checkStatus(status); err != nil {
		return 0, err
	}
	return info, nil
}

// parseInputData 解析输入数据
//
// # Params:
//
//	data: 输入数据
//
// # Returns:
//
//	dataType: 数据类型
//	uintptr: 单条元素的字节数
//	int: 数据长度
//	unsafe.Pointer: 数据指针
//	error: 错误信息
func (e *Engine) parseInputData(data any) (TensorElementDataType, uintptr, int, unsafe.Pointer, error) {
	switch d := data.(type) {
	case []float32:
		return TensorElementDataTypeFloat, 4, len(d), unsafe.Pointer(&d[0]), nil
	case []float64:
		return TensorElementDataTypeDouble, 8, len(d), unsafe.Pointer(&d[0]), nil
	case []int64:
		return TensorElementDataTypeInt64, 8, len(d), unsafe.Pointer(&d[0]), nil
	case []int32:
		return TensorElementDataTypeInt32, 4, len(d), unsafe.Pointer(&d[0]), nil
	case []int16:
		return TensorElementDataTypeInt16, 2, len(d), unsafe.Pointer(&d[0]), nil
	case []int8:
		return TensorElementDataTypeInt8, 1, len(d), unsafe.Pointer(&d[0]), nil
	case []uint64:
		return TensorElementDataTypeUint64, 8, len(d), unsafe.Pointer(&d[0]), nil
	case []uint32:
		return TensorElementDataTypeUint32, 4, len(d), unsafe.Pointer(&d[0]), nil
	case []uint16:
		return TensorElementDataTypeUint16, 2, len(d), unsafe.Pointer(&d[0]), nil
	case []uint8:
		return TensorElementDataTypeUint8, 1, len(d), unsafe.Pointer(&d[0]), nil
	case []bool:
		return TensorElementDataTypeBool, 1, len(d), unsafe.Pointer(&d[0]), nil
	default:
		return TensorElementDataTypeUndefined, 0, 0, nil, fmt.Errorf("unsupported input type: %T", data)
	}
}

func (v *Value) Destroy() {
	if v.handle != 0 {
		v.engine.funcs.releaseValue(v.handle)
		v.handle = 0
	}
}
