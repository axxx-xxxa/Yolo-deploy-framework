#include "class_timer.hpp"
#include "class_detector.h"
#include "yaml-cpp/yaml.h"


using namespace tensor_rt;


int TypeID(std::string ModelType) {
	if (ModelType == "yolov4_tiny") return YOLOV4_TINY;
	else if (ModelType == "yolov5s_cls") return YOLOV5s_cls;
	else if (ModelType == "yolov5s_seg") return YOLOV5s_seg;
	std::cout << "Input ModelType is not support! " << std::endl;
	exit(1);
}

Precision PrecID(std::string PrecType) {
	if (PrecType == "FP16") return FP16;
	else if (PrecType == "FP32") return FP32;
	else {
		std::cout << "Input PrecType is not support! " << std::endl;
		exit(1);
	}
	
}



int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cout << "Input export yaml path!" << std::endl;
		exit(1);
	}

	YAML::Node node = YAML::LoadFile(argv[1]);

	if (TypeID(node["type"].as<std::string>()) == YOLOV4_TINY) {
		std::cout << " export type is yolov4-tiny" << std::endl;
		Config config_v4_tiny;
		config_v4_tiny.net_type = YOLOV4_TINY;
		config_v4_tiny.detect_thresh = node["detect_thresh"].as<float>();
		config_v4_tiny.calibration_image_list_file_txt = "../configs/calibration_images.txt";

		config_v4_tiny.batch_size = node["batch_size"].as<int>();
		config_v4_tiny.inference_precision = PrecID(node["precision"].as<std::string>());
		config_v4_tiny.file_model_cfg = node["cfg"].as<std::string>();
		config_v4_tiny.file_model_weights = node["weight"].as<std::string>();
		cv::Mat image0 = cv::imread("test.png");

		std::unique_ptr<Detector> detector(new Detector());
		detector->Init(config_v4_tiny);
		std::vector<BatchResult> batch_res;
		Timer timer;
		for (int j = 0; j < 10; j++)
		{
			//prepare batch data
			std::vector<cv::Mat> batch_img;
			cv::Mat temp0 = image0.clone();
			for (int s = 0; s < config_v4_tiny.batch_size; s++) {
				batch_img.push_back(temp0);
			}

			//detect
			timer.reset();
			detector->Detect(batch_img, batch_res);
			timer.out("Detect");

			//disp
			for (int i = 0; i < batch_img.size(); ++i)
			{
				for (const auto& r : batch_res[i])
				{
					std::cout << "batch " << i << " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
					cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
					std::stringstream stream;
					stream << std::fixed << std::setprecision(2) << "  score:" << r.prob;

					cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 0.5);
				}
				bool status = cv::imwrite("result.jpg", batch_img[i]);
			}
		}
	}
	
	if (TypeID(node["type"].as<std::string>()) == YOLOV5s_cls) {
		std::cout << " input is YOLOV5s_cls" << std::endl;


		Config config_v5s_cls;
		config_v5s_cls.net_type = YOLOV5s_cls;
		config_v5s_cls.batch_size = node["batch_size"].as<int>();
		config_v5s_cls.inference_precision = PrecID(node["precision"].as<std::string>());
		config_v5s_cls.onnx_model = node["onnx"].as<std::string>();
		std::unique_ptr<Detector> detector(new Detector());
		detector->Init(config_v5s_cls);

	}

	if (TypeID(node["type"].as<std::string>()) == YOLOV5s_seg) {
		std::cout << " input is yolov5s_seg" << std::endl;

		Config config_v5s_seg;
		config_v5s_seg.net_type = YOLOV5s_seg;
		config_v5s_seg.batch_size = node["batch_size"].as<int>();
		config_v5s_seg.inference_precision = PrecID(node["precision"].as<std::string>());
		config_v5s_seg.onnx_model = node["onnx"].as<std::string>();
		std::unique_ptr<Detector> detector(new Detector());
		detector->Init(config_v5s_seg);

	}

	
}

