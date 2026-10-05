<div align="center" style="text-align: center;">
  <img src="../assets/logo.png" alt="logo" width="200" style="display: block; margin: 0 auto;" />
</div>

<p align="center">
   <a href="https://github.com/getcharzp/onnxruntime_purego/fork" target="blank">
      <img src="https://img.shields.io/github/forks/getcharzp/onnxruntime_purego?style=for-the-badge" alt="onnxruntime_purego forks"/>
   </a>
   <a href="https://github.com/getcharzp/onnxruntime_purego/stargazers" target="blank">
      <img src="https://img.shields.io/github/stars/getcharzp/onnxruntime_purego?style=for-the-badge" alt="onnxruntime_purego stars"/>
   </a>
   <a href="https://github.com/getcharzp/onnxruntime_purego/pulls" target="blank">
      <img src="https://img.shields.io/github/issues-pr/getcharzp/onnxruntime_purego?style=for-the-badge" alt="onnxruntime_purego pull-requests"/>
   </a>
</p>

<p align="center">
  <a href="../README.md">English</a> | 简体中文
</p>

基于 `purego` 实现的无 CGO 纯 Go 项目，通过 `purego` 直接绑定并调用 onnxruntime 原生库接口，无需依赖 CGO 编译环境，
即可实现 ONNX 模型的加载与推理计算，基于 `onnxruntime` 1.26.0 的头文件实现。

## 安装

下载 [onnxruntime 1.26](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0) 动态链接库，安装对应 onnxruntime 1.26 的 `onnxruntime_purego` 库：

| onnxruntime | onnxruntime_purego |
|-------------|--------------------|
| 1.26        | v1.26.0            |

历史版本请参考[版本映射表](./version-mapping.md)。

```shell
# 下载最新版本
go get -u github.com/getcharzp/onnxruntime_purego

# 针对 onnxruntime 1.26 下载特定版本 purego
go get -u github.com/getcharzp/onnxruntime_purego@v1.26.0
```

## 快速开始

```go
package main

import (
	ort "github.com/getcharzp/onnxruntime_purego"
	"log"
)

const testModelPath = "./testdata/yolo11n.onnx"

func main() {
	engine, _ := ort.NewEngine(ort.DefaultLibraryPath())
	defer engine.Destroy()
	session, err := engine.NewSession(testModelPath, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Destroy()

	// 按模型自身的声明准备数据
	input := session.Inputs[0]
	count, ok := input.ElementCount()
	if !ok {
		log.Fatalf("输入 %q 的形状 %v 含动态维度", input.Name, input.Shape)
	}

	inputValue, err := ort.NewTensor(input.Shape, make([]float32, count))
	if err != nil {
		log.Fatal(err)
	}
	defer inputValue.Destroy()

	outputs, err := session.Run(map[string]*ort.Value{input.Name: inputValue})
	if err != nil {
		log.Fatal(err)
	}

	for name, output := range outputs {
		defer output.Destroy()

		outputData, err := ort.GetTensorData[float32](output)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("%v: %+v", name, outputData[:min(len(outputData), 20)])
	}
}
```

## 案例

### YOLOv11 目标检测

| 原图                                                  | Mask图                                                      |
|-----------------------------------------------------|------------------------------------------------------------|
| <img width="100%" src="../testdata/test.png" alt=""> | <img width="100%" src="../testdata/yolov11_det.png" alt=""> |
