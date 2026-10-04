package ort

import (
	"math"
	"testing"

	"github.com/up-zero/gotool/testutil"
)

const testModelPath = "./testdata/yolo11n.onnx"

// yolo11n.onnx 的输入输出规格
const (
	testInputName  = "images"
	testOutputName = "output0"
	testHeight     = 640
	testWidth      = 640
	testChannels   = 3
	testRows       = 84 // 4 (box) + 80 (cls)
	testAnchors    = 8400
)

// newTestSession 创建测试用 Session，并在测试结束后自动释放
func newTestSession(t *testing.T, opts *SessionOptions) *Session {
	t.Helper()

	engine := newTestEngine(t)
	session, err := engine.NewSession(testModelPath, opts)
	testutil.Equal(t, err, nil)
	t.Cleanup(session.Destroy)

	return session
}

func TestEngine_NewSession(t *testing.T) {
	engine := newTestEngine(t)

	option, err := engine.NewSessionOptions()
	testutil.Equal(t, err, nil)
	defer option.Destroy()

	testutil.Equal(t, option.SetIntraOpNumThreads(1), nil)
	testutil.Equal(t, option.SetCpuMemArena(true), nil)

	session, err := engine.NewSession(testModelPath, option)
	testutil.Equal(t, err, nil)
	defer session.Destroy()

	testutil.NotEqual(t, session.handle, SessionHandle(0))
	testutil.Equal(t, session.Inputs[0].Name, testInputName)
	testutil.Equal(t, session.Outputs[0].Name, testOutputName)

	// 弃用字段需保持可用，直至正式移除
	testutil.Equal(t, session.InputNames, []string{testInputName})
	testutil.Equal(t, session.OutputNames, []string{testOutputName})
}

func TestEngine_NewSession_WithoutOptions(t *testing.T) {
	engine := newTestEngine(t)

	session, err := engine.NewSession(testModelPath, nil)
	testutil.Equal(t, err, nil)
	defer session.Destroy()

	testutil.Equal(t, session.Inputs[0].Name, testInputName)
	testutil.Equal(t, session.Outputs[0].Name, testOutputName)
}

// TestEngine_NewSession_TypeInfo 校验从模型读到的输入/输出类型与形状
func TestEngine_NewSession_TypeInfo(t *testing.T) {
	session := newTestSession(t, nil)

	testutil.Equal(t, len(session.Inputs), 1)
	testutil.Equal(t, session.Inputs[0].Name, testInputName)
	testutil.Equal(t, session.Inputs[0].DataType, TensorElementDataTypeFloat)

	testutil.Equal(t, len(session.Outputs), 1)
	testutil.Equal(t, session.Outputs[0].Name, testOutputName)
	testutil.Equal(t, session.Outputs[0].DataType, TensorElementDataTypeFloat)

	// 同一个模型可以导出成静态或动态形状，动态维度 onnxruntime 统一以 -1 表示
	sameShape := func(got, want []int64) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("shape rank mismatch: got %v, want %v", got, want)
		}
		for i, dim := range got {
			if dim != -1 && dim != want[i] {
				t.Fatalf("shape mismatch: got %v, want %v (-1 表示动态维度)", got, want)
			}
		}
	}
	sameShape(session.Inputs[0].Shape, []int64{1, testChannels, testHeight, testWidth})
	sameShape(session.Outputs[0].Shape, []int64{1, testRows, testAnchors})
}

func TestEngine_NewSession_ModelNotFound(t *testing.T) {
	engine := newTestEngine(t)

	session, err := engine.NewSession("./testdata/not_exists.onnx", nil)
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, session, (*Session)(nil))
}

func TestSession_Destroy(t *testing.T) {
	session := newTestSession(t, nil)

	session.Destroy()
	testutil.Equal(t, session.handle, SessionHandle(0))

	// 重复释放不应 panic
	session.Destroy()
}

func TestSessionOptions(t *testing.T) {
	engine := newTestEngine(t)

	opts, err := engine.NewSessionOptions()
	testutil.Equal(t, err, nil)

	testutil.Equal(t, opts.SetIntraOpNumThreads(2), nil)
	// 内存池的开启 / 关闭两条分支都应调用成功
	testutil.Equal(t, opts.SetCpuMemArena(true), nil)
	testutil.Equal(t, opts.SetCpuMemArena(false), nil)

	opts.Destroy()
	testutil.Equal(t, opts.handle, SessionOptionsHandle(0))

	// 重复释放不应 panic
	opts.Destroy()
}

func TestSessionOptions_EnableCUDA(t *testing.T) {
	engine := newTestEngine(t)

	opts, err := engine.NewSessionOptions()
	testutil.Equal(t, err, nil)
	defer opts.Destroy()

	// 无 CUDA 环境的机器上 onnxruntime 会返回错误，此处只验证调用链不 panic。
	// 生产环境可结合 availableProvider 判断是否具备 CUDA。
	if err := opts.EnableCUDA(map[string]string{"device_id": "0"}); err != nil {
		t.Logf("CUDA provider unavailable: %v", err)
	}
}

func TestSessionOptions_EnableTensorRT(t *testing.T) {
	engine := newTestEngine(t)

	opts, err := engine.NewSessionOptions()
	testutil.Equal(t, err, nil)
	defer opts.Destroy()

	if err := opts.EnableTensorRT(map[string]string{"trt_fp16_enable": "1"}); err != nil {
		t.Logf("TensorRT provider unavailable: %v", err)
		t.Skip("skipping TensorRT session creation")
	}

	// 启用成功后应能正常构建 Session
	session, err := engine.NewSession(testModelPath, opts)
	testutil.Equal(t, err, nil)
	defer session.Destroy()

	testutil.Equal(t, session.Inputs[0].Name, testInputName)
}

func TestSession_Run(t *testing.T) {
	session := newTestSession(t, nil)

	inputData := make([]float32, testChannels*testHeight*testWidth)
	for i := range inputData {
		inputData[i] = 0.5
	}

	inputValue, err := NewTensor([]int64{1, testChannels, testHeight, testWidth}, inputData)
	testutil.Equal(t, err, nil)
	defer inputValue.Destroy()

	outputs, err := session.Run(map[string]*Value{testInputName: inputValue})
	testutil.Equal(t, err, nil)
	defer func() {
		for _, output := range outputs {
			output.Destroy()
		}
	}()

	testutil.Equal(t, len(outputs), 1)
	output, ok := outputs[testOutputName]
	testutil.Equal(t, ok, true)

	// 输出形状固定为 [1, 84, 8400]
	testutil.Equal(t, mustShape(t, output), []int64{1, testRows, testAnchors})
	testutil.Equal(t, mustElementCount(t, output), testRows*testAnchors)

	data, err := GetTensorData[float32](output)
	testutil.Equal(t, err, nil)
	testutil.Equal(t, len(data), testRows*testAnchors)

	// 推理结果不应出现 NaN / Inf
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output contains invalid value at index %d: %v", i, v)
		}
	}
}

func TestSession_Run_Repeated(t *testing.T) {
	session := newTestSession(t, nil)

	inputValue, err := NewTensor([]int64{1, testChannels, testHeight, testWidth}, make([]float32, testChannels*testHeight*testWidth))
	testutil.Equal(t, err, nil)
	defer inputValue.Destroy()

	// 同一 Session 多次推理，输出 shape 应保持一致
	for range 2 {
		outputs, err := session.Run(map[string]*Value{testInputName: inputValue})
		testutil.Equal(t, err, nil)

		testutil.Equal(t, mustShape(t, outputs[testOutputName]), []int64{1, testRows, testAnchors})
		for _, output := range outputs {
			output.Destroy()
		}
	}
}

func TestSession_Run_NoInputs(t *testing.T) {
	session := newTestSession(t, nil)

	outputs, err := session.Run(map[string]*Value{})
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, outputs, (map[string]*Value)(nil))
}

func TestSession_Run_NilValue(t *testing.T) {
	session := newTestSession(t, nil)

	outputs, err := session.Run(map[string]*Value{testInputName: nil})
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, outputs, (map[string]*Value)(nil))
}

func TestSession_Run_UnknownInputName(t *testing.T) {
	session := newTestSession(t, nil)

	inputValue, err := NewTensor([]int64{1, testChannels, testHeight, testWidth}, make([]float32, testChannels*testHeight*testWidth))
	testutil.Equal(t, err, nil)
	defer inputValue.Destroy()

	outputs, err := session.Run(map[string]*Value{"not_a_input": inputValue})
	testutil.NotEqual(t, err, nil)
	testutil.Equal(t, outputs, (map[string]*Value)(nil))
}
