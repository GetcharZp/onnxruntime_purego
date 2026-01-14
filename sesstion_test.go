package ort

import (
	"testing"
)

const testModelPath = "./testdata/yolo11n.onnx"

func TestEngine_NewSession(t *testing.T) {
	engine, _ := NewEngine(DefaultLibraryPath())
	defer engine.Destroy()
	session, err := engine.NewSession(testModelPath, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Destroy()

	t.Logf("input names: %+v", session.InputNames)
	t.Logf("output names: %+v", session.OutputNames)
}

func TestSession_Run(t *testing.T) {
	engine, _ := NewEngine(DefaultLibraryPath())
	defer engine.Destroy()
	session, err := engine.NewSession(testModelPath, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Destroy()

	inputData := make([]float32, 3*640*640)
	inputValue, err := engine.NewTensor([]int64{1, 3, 640, 640}, inputData)
	if err != nil {
		t.Fatal(err)
	}
	defer inputValue.Destroy()

	inputs := map[string]*Value{
		"images": inputValue,
	}

	outputs, err := session.Run(inputs)
	if err != nil {
		t.Fatal(err)
	}

	for name, output := range outputs {
		outputData, err := GetTensorData[float32](output)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%v: %+v", name, outputData[:min(len(outputData), 20)])
	}
}
