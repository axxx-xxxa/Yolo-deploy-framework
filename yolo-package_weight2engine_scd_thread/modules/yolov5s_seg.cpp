
#include "yolov5s_seg.h"

///
/// \brief YoloV4::YoloV4
/// \param network_info_
/// \param infer_params_
///
YoloV5s_seg::YoloV5s_seg(const NetworkInfo& network_info_, const InferParams& infer_params_)
	: Yolo(network_info_, infer_params_)
{
}
std::vector<BBoxInfo> YoloV5s_seg::decodeTensor(const int imageIdx, const int imageH, const int imageW, const TensorInfo& tensor)
{
	std::vector<BBoxInfo> temp;
	return temp;
}

