package testdata

import (
	"fmt"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/imageutil"
	"image"
	"image/color"
	"image/draw"
	"log"
	"sort"
	"testing"
)

func TestYolo11Detect(t *testing.T) {
	engine, _ := ort.NewEngine("." + ort.DefaultLibraryPath())
	defer engine.Destroy()
	session, err := engine.NewSession("yolo11n.onnx", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Destroy()

	img, err := imageutil.Open("test.png")
	if err != nil {
		t.Fatal(err)
	}

	inputTensor, params, err := preprocess(engine, img, 640)
	if err != nil {
		t.Fatal(err)
	}
	defer inputTensor.Destroy()
	inputs := map[string]*ort.Value{
		"images": inputTensor,
	}

	outputs, err := session.Run(inputs)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, output := range outputs {
			output.Destroy()
		}
	}()

	outputData, err := ort.GetTensorData[float32](outputs["output0"])
	if err != nil {
		t.Fatal(err)
	}

	results, err := postprocess(outputData, []int64{1, 84, 8400}, params)
	if err != nil {
		t.Fatal(err)
	}

	targetImg := image.NewRGBA(img.Bounds())
	draw.Draw(targetImg, img.Bounds(), img, img.Bounds().Min, draw.Src)
	fmt.Printf("检测到目标: %d 个\n", len(results))
	for _, res := range results {
		fmt.Printf("Class: %d, Score: %.2f, Box: %v\n", res.ClassID, res.Score, res.Box)
		imageutil.DrawThickRectOutline(targetImg, res.Box, color.RGBA{R: 255, G: 0, B: 0, A: 255}, 3)
	}
	imageutil.Save("yolov11_det.png", targetImg, 50)
}

// imageParams 图片尺寸信息
type imageParams struct {
	origW, origH int
	scale        float32
}

// 候选结果
type candidate struct {
	box          [4]float32      // 模型输出的 cx, cy, w, h
	origBox      image.Rectangle // 原始图片的检测框
	score        float32
	classID      int
	maskCoeffs   []float32 // Mask 系数
	rawKeyPoints []float32
	angle        float32 // 旋转角度
}

// DetResult 目标检测结果
type DetResult struct {
	// 分类ID，例如：
	//	0: person
	//  1: bicycle
	//  2: car
	// - 详细映射参考：
	//	https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
	ClassID int
	Score   float32
	Box     image.Rectangle // 检测框
}

// postprocess 后处理
func postprocess(data []float32, shape []int64, params imageParams) ([]DetResult, error) {
	numChannels := int(shape[1]) // 4 (box) + 80 (cls) = 84
	numAnchors := int(shape[2])  // 8400

	// 解析候选框
	candidates := parseCandidates(data, numChannels, numAnchors, params)
	// NMS
	keptIndices := nms(candidates, 0.5)

	results := make([]DetResult, 0, len(keptIndices))
	for _, idx := range keptIndices {
		cand := candidates[idx]
		results = append(results, DetResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Box:     cand.origBox,
		})
	}

	return results, nil
}

// parseCandidates 解析候选框
func parseCandidates(data []float32, channels, anchors int, params imageParams) []candidate {
	var cands []candidate

	// 检查通道数
	expectedChannels := 4 + 80
	if channels != expectedChannels {
		log.Printf("警告：传入的通道数(%d)与预期(%d)不匹配", channels, expectedChannels)
		return cands
	}

	for i := 0; i < anchors; i++ {
		// 找最大类别分数
		maxScore := float32(0.0)
		classID := -1
		for c := 0; c < 80; c++ {
			score := data[(4+c)*anchors+i]
			if score > maxScore {
				maxScore = score
				classID = c
			}
		}
		if maxScore < 0.45 {
			continue
		}

		// 提取坐标
		cx := data[0*anchors+i]
		cy := data[1*anchors+i]
		w := data[2*anchors+i]
		h := data[3*anchors+i]

		// 转换回原图矩形坐标
		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2
		origX1 := max(0, int(x1/params.scale))
		origY1 := max(0, int(y1/params.scale))
		origX2 := min(params.origW, int(x2/params.scale))
		origY2 := min(params.origH, int(y2/params.scale))

		cands = append(cands, candidate{
			box:     [4]float32{x1, y1, x2, y2},
			origBox: image.Rect(origX1, origY1, origX2, origY2),
			score:   maxScore,
			classID: classID,
		})
	}
	return cands
}

// preprocess 预处理
func preprocess(engine *ort.Engine, img image.Image, inputSize int) (*ort.Value, imageParams, error) {
	bounds := img.Bounds()
	params := imageParams{
		origW: bounds.Dx(),
		origH: bounds.Dy(),
	}

	scale := float32(inputSize) / float32(max(params.origW, params.origH))
	params.scale = scale

	newW := int(float32(params.origW) * scale)
	newH := int(float32(params.origH) * scale)

	resized := imageutil.Resize(img, newW, newH)

	// 准备 Tensor 数据 (CHW + Normalize 0-1)
	data := make([]float32, 3*inputSize*inputSize)
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()

			idx := y*inputSize + x
			data[idx] = float32(r) / 65535.0                       // R
			data[inputSize*inputSize+idx] = float32(g) / 65535.0   // G
			data[2*inputSize*inputSize+idx] = float32(b) / 65535.0 // B
		}
	}

	tensor, err := engine.NewTensor([]int64{1, 3, 640, 640}, data)
	if err != nil {
		return nil, params, fmt.Errorf("创建 Tensor 失败: %w", err)
	}

	return tensor, params, err
}

func nms(cands []candidate, iouThresh float32) []int {
	sort.Slice(cands, func(i, j int) bool {
		return cands[i].score > cands[j].score
	})

	keep := make([]int, 0)
	suppressed := make([]bool, len(cands))

	for i := 0; i < len(cands); i++ {
		if suppressed[i] {
			continue
		}
		keep = append(keep, i)

		for j := i + 1; j < len(cands); j++ {
			if suppressed[j] {
				continue
			}
			if computeIOU(cands[i].origBox, cands[j].origBox) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func computeIOU(r1, r2 image.Rectangle) float32 {
	intersect := r1.Intersect(r2)
	if intersect.Empty() {
		return 0.0
	}

	interArea := intersect.Dx() * intersect.Dy()
	area1 := r1.Dx() * r1.Dy()
	area2 := r2.Dx() * r2.Dy()

	return float32(interArea) / float32(area1+area2-interArea)
}
