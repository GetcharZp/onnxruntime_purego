package ort

import (
	"testing"

	"github.com/up-zero/gotool/testutil"
)

// TestSessionOptions_AddFreeDimensionOverrideByName 按维度名固化动态维度
func TestSessionOptions_AddFreeDimensionOverrideByName(t *testing.T) {
	engine := newTestEngine(t)
	model := dynamicONNX()

	// 未覆盖时，动态维以 -1 呈现
	base, err := engine.NewSessionFromBytes(model, nil)
	testutil.Equal(t, err, nil)
	defer base.Close()
	testutil.Equal(t, base.Inputs[0].Shape, []int64{-1, 3})

	opts, err := engine.NewSessionOptions()
	testutil.Equal(t, err, nil)
	defer opts.Close()
	testutil.Equal(t, opts.AddFreeDimensionOverrideByName("batch", 1), nil)

	session, err := engine.NewSessionFromBytes(model, opts)
	testutil.Equal(t, err, nil)
	defer session.Close()

	// 覆盖后动态维固定为 1，ElementCount 随之可计算
	testutil.Equal(t, session.Inputs[0].Shape, []int64{1, 3})
	count, ok := session.Inputs[0].ElementCount()
	testutil.Equal(t, ok, true)
	testutil.Equal(t, count, int64(3))
}

// TestSessionOptions_AddFreeDimensionOverride 按维度符号固化动态维度
func TestSessionOptions_AddFreeDimensionOverride(t *testing.T) {
	engine := newTestEngine(t)

	opts, err := engine.NewSessionOptions()
	testutil.Equal(t, err, nil)
	defer opts.Close()
	testutil.Equal(t, opts.AddFreeDimensionOverride("DATA_BATCH", 5), nil)

	session, err := engine.NewSessionFromBytes(dynamicONNX(), opts)
	testutil.Equal(t, err, nil)
	defer session.Close()
	testutil.Equal(t, session.Inputs[0].Shape, []int64{5, 3})
}

// dynamicONNX 构造一个最小可用的 ONNX 模型，用于验证动态维度覆盖：
// Identity 算子，输入 / 输出均为 [batch, 3]，其中 batch 同时带有 dim_param("batch")
// 与 denotation("DATA_BATCH")，两者分别对应两个覆盖接口。
//
// 这里手写 protobuf 编码，避免为测试引入 onnx / protobuf 依赖。
func dynamicONNX() []byte {
	shape := pbBytes(1, pbDim("batch", "DATA_BATCH")) // 动态维
	shape = append(shape, pbBytes(1, pbInt(1, 3))...) // 固定维 3

	// ValueInfoProto: name + TypeProto(FLOAT tensor, shape)
	value := func(name string) []byte {
		tensor := pbInt(1, 1) // elem_type = FLOAT
		tensor = append(tensor, pbBytes(2, shape)...)
		return append(pbStr(1, name), pbBytes(2, pbBytes(1, tensor))...)
	}

	// NodeProto: Identity(X) -> Y
	node := pbStr(1, "X")
	node = append(node, pbStr(2, "Y")...)
	node = append(node, pbStr(4, "Identity")...)

	// GraphProto
	graph := pbBytes(1, node)
	graph = append(graph, pbStr(2, "g")...)
	graph = append(graph, pbBytes(11, value("X"))...)
	graph = append(graph, pbBytes(12, value("Y"))...)

	// ModelProto: ir_version + graph + opset_import
	model := pbInt(1, 8)
	model = append(model, pbBytes(7, graph)...)
	return append(model, pbBytes(8, pbInt(2, 13))...)
}

// pbTag 编码字段号与 wire type（0=varint, 2=length-delimited）
func pbTag(field, wire int) []byte { return pbVarint(uint64(field)<<3 | uint64(wire)) }

func pbVarint(v uint64) []byte {
	var b []byte
	for v >= 0x80 {
		b = append(b, byte(v)|0x80)
		v >>= 7
	}
	return append(b, byte(v))
}

func pbInt(field int, v uint64) []byte { return append(pbTag(field, 0), pbVarint(v)...) }

func pbStr(field int, s string) []byte { return pbBytes(field, []byte(s)) }

func pbBytes(field int, b []byte) []byte {
	out := pbTag(field, 2)
	out = append(out, pbVarint(uint64(len(b)))...)
	return append(out, b...)
}

// pbDim 编码 TensorShapeProto.Dimension，dim_param 与 denotation 同时存在
func pbDim(param, denotation string) []byte {
	b := pbStr(2, param)
	return append(b, pbStr(3, denotation)...)
}
