package ort

import (
	"testing"

	"github.com/up-zero/gotool"
	"github.com/up-zero/gotool/testutil"
)

// mustShape 获取维度信息并断言成功
func mustShape(t *testing.T, v *Value) []int64 {
	t.Helper()

	shape, err := v.GetShape()
	testutil.Equal(t, err, nil)

	return shape
}

// mustElementCount 获取元素总数并断言成功
func mustElementCount(t *testing.T, v *Value) int {
	t.Helper()

	count, err := v.GetElementCount()
	testutil.Equal(t, err, nil)

	return count
}

// newTestTensor 创建测试用 Tensor，并在测试结束后自动释放
func newTestTensor(t *testing.T, shape []int64, data any) *Value {
	t.Helper()

	v, err := NewTensor(shape, data)
	testutil.Equal(t, err, nil)
	t.Cleanup(v.Destroy)

	return v
}

// tensorTestCase 一组「输入数据 + 读回数据」的用例
//
// get 使用泛型函数包装，避免用 any 做类型断言掩盖真实的类型不匹配
type tensorTestCase struct {
	data any
	get  func(*Value) (any, error)
	want any
}

// tensorCase 构造 tensorTestCase，want 需独立于 data（否则比较退化为恒等）
func tensorCase[T gotool.Number](data, want []T) tensorTestCase {
	return tensorTestCase{
		data: data,
		get:  func(v *Value) (any, error) { return GetTensorData[T](v) },
		want: want,
	}
}

func TestValue_GetShape(t *testing.T) {
	newTestEngine(t)

	v := newTestTensor(t, []int64{1, 1, 6}, []float32{1, 2, 3, 4, 5, 6})

	testutil.Equal(t, mustShape(t, v), []int64{1, 1, 6})
	// 二次读取命中缓存，结果需保持一致
	testutil.Equal(t, mustShape(t, v), []int64{1, 1, 6})
}

func TestValue_GetShape_Scalar(t *testing.T) {
	newTestEngine(t)

	// 标量张量的维度数为 0，不能退化成 &shape[0]
	v := newTestTensor(t, []int64{}, []float32{42})

	testutil.Equal(t, mustShape(t, v), []int64{})
	testutil.Equal(t, mustElementCount(t, v), 1)
}

func TestValue_GetElementCount(t *testing.T) {
	newTestEngine(t)

	v := newTestTensor(t, []int64{1, 1, 6}, []float32{1, 2, 3, 4, 5, 6})

	testutil.Equal(t, mustElementCount(t, v), 6)
	// 二次读取命中缓存，结果需保持一致
	testutil.Equal(t, mustElementCount(t, v), 6)
	testutil.Equal(t, len(mustShape(t, v)), 3)
	// 元素总数与数据长度一致
	got, err := GetTensorData[float32](v)
	testutil.Equal(t, err, nil)
	testutil.Equal(t, len(got), 6)
}

func TestValue_Destroy(t *testing.T) {
	newTestEngine(t)

	v := newTestTensor(t, []int64{1, 1, 6}, []float32{1, 2, 3, 4, 5, 6})

	v.Destroy()
	testutil.Equal(t, v.handle, ValueHandle(0))

	// 重复释放不应 panic
	v.Destroy()
}

func TestNewTensor_AllTypes(t *testing.T) {
	newTestEngine(t)

	const elementCount = 3
	cases := []struct {
		name string
		tensorTestCase
	}{
		{"float32", tensorCase([]float32{1, -2, 3.5}, []float32{1, -2, 3.5})},
		{"float64", tensorCase([]float64{1, -2, 3.5}, []float64{1, -2, 3.5})},
		{"int64", tensorCase([]int64{1, -2, 3}, []int64{1, -2, 3})},
		{"int32", tensorCase([]int32{1, -2, 3}, []int32{1, -2, 3})},
		{"int16", tensorCase([]int16{1, -2, 3}, []int16{1, -2, 3})},
		{"int8", tensorCase([]int8{1, -2, 3}, []int8{1, -2, 3})},
		{"uint64", tensorCase([]uint64{1, 2, 3}, []uint64{1, 2, 3})},
		{"uint32", tensorCase([]uint32{1, 2, 3}, []uint32{1, 2, 3})},
		{"uint16", tensorCase([]uint16{1, 2, 3}, []uint16{1, 2, 3})},
		{"uint8", tensorCase([]uint8{1, 2, 3}, []uint8{1, 2, 3})},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := newTestTensor(t, []int64{1, elementCount}, tc.data)

			testutil.Equal(t, mustShape(t, v), []int64{1, elementCount})
			testutil.Equal(t, mustElementCount(t, v), elementCount)

			got, err := tc.get(v)
			testutil.Equal(t, err, nil)
			testutil.Equal(t, got, tc.want)
		})
	}
}

// TestNewTensor_Bool NewTensor 支持 []bool，但 GetTensorData 的类型约束为 gotool.Number
// （不含 bool），无法读回，这里只校验创建与元信息
func TestNewTensor_Bool(t *testing.T) {
	newTestEngine(t)

	v := newTestTensor(t, []int64{1, 3}, []bool{true, false, true})

	testutil.Equal(t, mustShape(t, v), []int64{1, 3})
	testutil.Equal(t, mustElementCount(t, v), 3)
}

func TestNewTensor_DoesNotCopyData(t *testing.T) {
	newTestEngine(t)

	data := []float32{1, 2, 3, 4, 5, 6}
	v := newTestTensor(t, []int64{1, 1, 6}, data)

	// onnxruntime 只持有指针、不拷贝数据，修改原切片后 Tensor 内容应同步变化
	data[0] = 100

	got, err := GetTensorData[float32](v)
	testutil.Equal(t, err, nil)
	testutil.Equal(t, got, []float32{100, 2, 3, 4, 5, 6})
}

func TestNewTensor_DataTooShort(t *testing.T) {
	newTestEngine(t)

	// shape 需要 6 个元素，实际只给了 3 个，需在调用 onnxruntime 前拦截
	v, err := NewTensor([]int64{1, 1, 6}, []float32{1, 2, 3})
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, v, (*Value)(nil))
}

func TestNewTensor_NegativeDim(t *testing.T) {
	newTestEngine(t)

	// 动态维度(-1)时由 onnxruntime 而非包内校验报错
	v, err := NewTensor([]int64{-1, 3}, []float32{1, 2, 3})
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, v, (*Value)(nil))
}

func TestNewTensor_UnsupportedType(t *testing.T) {
	newTestEngine(t)

	v, err := NewTensor([]int64{1, 2}, []string{"a", "b"})
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, v, (*Value)(nil))
}

func TestNewTensor_EngineNotInitialized(t *testing.T) {
	// 模拟 NewEngine 之前调用包级 NewTensor
	setDefaultEngine(nil)

	v, err := NewTensor([]int64{1}, []float32{1})
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, v, (*Value)(nil))
}

func TestGetTensorData(t *testing.T) {
	newTestEngine(t)

	v := newTestTensor(t, []int64{1, 1, 6}, []float32{1, 2, 3, 4, 5, 6})

	data, err := GetTensorData[float32](v)
	testutil.Equal(t, err, nil)
	testutil.Equal(t, data, []float32{1, 2, 3, 4, 5, 6})
}

func TestGetTensorData_TypeMismatch(t *testing.T) {
	newTestEngine(t)

	// Tensor 实际为 float32，按 int64 读取必须报错，而不是返回错误的内存解释
	v := newTestTensor(t, []int64{1, 1, 6}, []float32{1, 2, 3, 4, 5, 6})

	data, err := GetTensorData[int64](v)
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, data, ([]int64)(nil))
}
