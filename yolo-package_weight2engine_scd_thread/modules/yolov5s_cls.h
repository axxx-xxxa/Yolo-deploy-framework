

//int export_yolov5s_cls(const char* onnx_filename);


#ifndef CLASS_YOLOV5s_cls_H_
#define CLASS_YOLOV5s_cls_H_
#include "yolo.h"
//#include <iostream>
//#include "logging.h"
//#include "NvOnnxParser.h"
//#include "NvInfer.h"
//#include <fstream>

//using namespace nvinfer1;
//using namespace nvonnxparser;
class YoloV5s_cls :public Yolo
{
public:
	YoloV5s_cls(
		const NetworkInfo& network_info_,
		const InferParams& infer_params_);
private:
	std::vector<BBoxInfo> decodeTensor(const int imageIdx,
		const int imageH,
		const int imageW,
		const TensorInfo& tensor) override;
};

#endif