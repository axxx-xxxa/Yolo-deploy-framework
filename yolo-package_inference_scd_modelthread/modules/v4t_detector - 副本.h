

#ifndef V4T_DETECTOR_H_
#define V4T_DETECTOR_H_

#include "yaml-cpp/yaml.h"
#include "json-cpp/json.h"
#include <opencv2/opencv.hpp>
#include "ds_image.h"
#include "trt_utils.h"
#include "yolo.h"
#include "yolov4.h"
#include "class_timer.hpp"
#include "API.h"

#include <fstream>
#include <string>
#include <chrono>
#include <stdio.h>  /* defines FILENAME_MAX */


namespace tensor_rt
{
	///
	/// \brief The Result struct
	///
	/// 
	struct yolores
	{
		int	  id = -1;
		float prob = 0.f;
		float x1;
		float y1;
		float x2;
		float y2;

		yolores(int id_, float prob_, float x1_, float y1_, float x2_, float y2_)
			: id(id_), prob(prob_), x1(x1_), y1(y1_), x2(x2_), y2(y2_)
		{
		}
	};
	/// 
	struct Result
	{
		int		 id = -1;
		float	 prob = 0.f;
		cv::Rect rect;

		Result(int id_, float prob_, cv::Rect r)
			: id(id_), prob(prob_), rect(r)
		{
		}
	};

	using BatchResult = std::vector<Result>;
	using BatchResult_yolo = std::vector<yolores>;

	///
	/// \brief The ModelType enum
	///
	enum ModelType
	{
		YOLOV3,
		YOLOV4,
		YOLOV4_TINY,
		YOLOV5,
		YOLOV6,
		YOLOV7
	};

	///
	/// \brief The Precision enum
	///
	enum Precision
	{
		INT8 = 0,
		FP16,
		FP32
	};

	struct Modelinfo
	{
		std::string enginepath = "";
		std::string configpath = "";
		cv::Size inputshape;
		int classes;

	};

	///
	/// \brief The Config struct
	///
	struct Config
	{
		std::string file_model_cfg = "yolov4.cfg";

		std::string file_model_weights = "yolov4.weights";

		float detect_thresh = 0.5f;

		ModelType net_type = YOLOV4;

		Precision inference_precision = FP32;

		int	gpu_id = 0;

		uint32_t batch_size = 1;

		std::string calibration_image_list_file_txt = "configs/calibration_images.txt";
	};

}

using namespace tensor_rt;


class API YoloDectector
{
public:
	YoloDectector() = default;
	~YoloDectector() = default;

	void init(const std::string yaml_path);

	void detect(const std::vector<cv::Mat>& vec_image,
		std::vector<tensor_rt::BatchResult>& vec_batch_result);

	void detect_yolo(const std::vector<cv::Mat>& vec_image,
		std::vector<tensor_rt::BatchResult_yolo>& vec_batch_result);

	void getInfo(std::string& Info);

	void get_cl_n(std::vector<std::string>& cl_n);

	void release();

private:

	void set_gpu_id(const int id = 0);

	void parse_yaml();

	void build_net();

private:
	// my 
	std::string _yaml_path;
	Modelinfo _model_info;
	// ori
	tensor_rt::Config _config;
	NetworkInfo _yolo_info;
	InferParams _infer_param;
	std::vector<std::string> _vec_net_type{ "yolov3", "yolov4", "yolov4-tiny", "yolov5" };
	std::vector<std::string> _vec_precision{ "kINT8","kHALF","kFLOAT" };
	std::unique_ptr<Yolo> _p_net = nullptr;
	Timer _m_timer;
	cv::Mat m_blob;
};


class API Detectors
{
public:

	bool init(const std::string manager_yaml_path);

	bool serial_infer(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result);

	bool serial_infer(const std::vector<cv::Mat>& mat_image, std::string& imgInfo, std::string& result);

	std::string get_info();

	bool release();

	std::vector<std::shared_ptr<YoloDectector>> Dets;

private:


};

#endif
