


#include "v4t_detector.h"



using namespace tensor_rt;

void YoloDectector::get_cl_n(std::vector<std::string>& cl_n)
{
	cl_n = _yolo_info.cl_n;
}

void YoloDectector::init(const std::string yaml_path)
{
	_yaml_path = yaml_path;

	this->parse_yaml();

	this->build_net();

}

void YoloDectector::detect(const std::vector<cv::Mat>& vec_image,
	std::vector<tensor_rt::BatchResult>& vec_batch_result)
{
	std::vector<DsImage> vec_ds_images;
	vec_batch_result.clear();
	if (vec_batch_result.capacity() < vec_image.size())
		vec_batch_result.reserve(vec_image.size());
			
	for (const auto& img : vec_image)
	{
		vec_ds_images.emplace_back(img, _vec_net_type[_config.net_type], _p_net->getInputH(), _p_net->getInputW());
	}
	Timer timer;
	timer.reset();

	blobFromDsImages(vec_ds_images, m_blob, _p_net->getInputH(), _p_net->getInputW());
	//timer.out("preprocess");
	timer.reset();
	_p_net->doInference(m_blob.data, static_cast<uint32_t>(vec_ds_images.size()));
	//timer.out("inference");

	for (size_t i = 0; i < vec_ds_images.size(); ++i)
	{
		timer.reset();
		auto curImage = vec_ds_images.at(i);
		//outputensor getshape6
		auto binfo = _p_net->decodeDetections(static_cast<int>(i), curImage.getImageHeight(), curImage.getImageWidth());
		auto remaining = (_p_net->getNMSThresh() > 0) ? nmsAllClasses(_p_net->getNMSThresh(), binfo, _p_net->getNumClasses(), _vec_net_type[_config.net_type]) : binfo;
		//timer.out("postprocess");

		std::vector<tensor_rt::Result> vec_result;
		if (!remaining.empty())
		{
			vec_result.reserve(remaining.size());
			for (const auto& b : remaining)
			{
				const int x = cvRound(b.box.x1);
				const int y = cvRound(b.box.y1);
				const int w = cvRound(b.box.x2 - b.box.x1);
				const int h = cvRound(b.box.y2 - b.box.y1);
				vec_result.emplace_back(b.label, b.prob, cv::Rect(x, y, w, h));
			}
		}
		vec_batch_result.emplace_back(vec_result);
	}
}

void YoloDectector::detect_yolo(const std::vector<cv::Mat>& vec_image,
	std::vector<tensor_rt::BatchResult_yolo>& vec_batch_result)
{
	std::vector<DsImage> vec_ds_images;
	vec_batch_result.clear();
	if (vec_batch_result.capacity() < vec_image.size())
		vec_batch_result.reserve(vec_image.size());

	for (const auto& img : vec_image)
	{
		vec_ds_images.emplace_back(img, _vec_net_type[_config.net_type], _p_net->getInputH(), _p_net->getInputW());
	}
	Timer timer;
	timer.reset();

	blobFromDsImages(vec_ds_images, m_blob, _p_net->getInputH(), _p_net->getInputW());
	//timer.out("preprocess");
	timer.reset();
	_p_net->doInference(m_blob.data, static_cast<uint32_t>(vec_ds_images.size()));
	//timer.out("inference");

	for (size_t i = 0; i < vec_ds_images.size(); ++i)
	{
		timer.reset();
		auto curImage = vec_ds_images.at(i);
		/*std::cout << vec_ds_images[i].getImageHeight() << std::endl;
		std::cout << vec_ds_images[i].getImageWidth() << std::endl;;*/
		//outputensor getshape6
		auto binfo = _p_net->decodeDetections(static_cast<int>(i), curImage.getImageHeight(), curImage.getImageWidth());
		auto remaining = (_p_net->getNMSThresh() > 0) ? nmsAllClasses(_p_net->getNMSThresh(), binfo, _p_net->getNumClasses(), _vec_net_type[_config.net_type]) : binfo;
		//timer.out("postprocess");

		std::vector<tensor_rt::yolores> vec_result;
		if (!remaining.empty())
		{
			vec_result.reserve(remaining.size());
			for (const auto& b : remaining)
			{
				float x1 = b.box.x1 / vec_ds_images[i].getImageWidth();
				float y1 = b.box.y1 / vec_ds_images[i].getImageHeight();
				float x2 = b.box.x2 / vec_ds_images[i].getImageWidth();
				float y2 = b.box.y2 / vec_ds_images[i].getImageHeight();
				vec_result.emplace_back(b.label, b.prob, x1,y1,x2,y2);
			}
		}
		vec_batch_result.emplace_back(vec_result);
	}
}

void YoloDectector::getInfo(std::string& Info)
{
	_model_info.enginepath 	= _yolo_info.enginePath;
	_model_info.configpath 	= _yolo_info.configFilePath;
	_model_info.inputshape = cv::Size(_p_net->getInputH(), _p_net->getInputW());
	_model_info.classes		= _yolo_info.classes;

	Info += "[ModelInfo] enginepath: ";
	Info += _model_info.enginepath;
	Info += " configpath: ";
	Info += _model_info.configpath;
	Info += " inputshape: ";
	Info += std::to_string(_model_info.inputshape.height);
	Info += " ";
	Info += std::to_string(_model_info.inputshape.width);
	Info += " classes: ";
	Info += std::to_string(_model_info.classes);
}

void YoloDectector::release()
{
	_p_net->release();
}



void YoloDectector::set_gpu_id(const int id)
{
	cudaError_t status = cudaSetDevice(id);
	if (status != cudaSuccess)
	{
		std::cout << "gpu id :" + std::to_string(id) + " not exist !" << std::endl;
		assert(0);
	}
}

void YoloDectector::parse_yaml()
{

	YAML::Node node = YAML::LoadFile(_yaml_path);

	_yolo_info.precision = _vec_precision[node["precision"].as<int>()];
	_yolo_info.networkType = _vec_net_type[node["net_type"].as<int>()];
	_yolo_info.enginePath = node["enginePath"].as<std::string>();
	_yolo_info.configFilePath = node["configFilePath"].as<std::string>();
	_yolo_info.classes = node["classes"].as<int>();
	_yolo_info.deviceType = "kGPU";
	_yolo_info.cl_n = node["cl_n"].as<std::vector<std::string>>();
	_yolo_info.calibrationTablePath = _yolo_info.data_path + "-calibration.table";
	_yolo_info.inputBlobName = "data";

	_infer_param.printPerfInfo = false;
	_infer_param.printPredictionInfo = false;
	_infer_param.calibImagesPath = "";
	_infer_param.probThresh = node["detect_thresh"].as<float>();
	_infer_param.nmsThresh = node["nmsThresh"].as<float>();
	_infer_param.batchSize = node["batch_size"].as<int>();
	this->set_gpu_id(node["gpu_id"].as<int>());

}

void YoloDectector::build_net()
{
	_p_net = std::unique_ptr<Yolo>{ new YoloV4(_yolo_info,_infer_param) };
}


bool Detectors::init(const std::string manager_yaml_path)
{
	YAML::Node node = YAML::LoadFile(manager_yaml_path);

	for (int i = 0; i < node["WorkLocation"].size(); i++) {
		for (auto yaml_path : node["WorkLocation"][i]["models"]) {
			//unique_ptrֻ���ƶ����ܸ���
			std::shared_ptr<YoloDectector> detector(new YoloDectector());
			detector->init(yaml_path.as<std::string>());
			Dets.push_back(detector);
		}
	}
	return true;
}

bool Detectors::serial_infer(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result)
{
	for (int num = 0; num < mat_image.size(); num++) vec_batch_result.push_back({});

	for (auto Detector : Dets) {

		std::vector<BatchResult> batch_model_res;
		Detector->detect(mat_image, batch_model_res);
		for (int i = 0; i < batch_model_res.size(); i++) {
			for (int j = 0; j < batch_model_res[i].size(); j++) vec_batch_result[i].push_back(batch_model_res[i][j]);
		}
	}
	return true;
}


bool Detectors::serial_infer(const std::vector<cv::Mat>& mat_image, std::string& imgInfo,std::string& result)
{

	std::string model_result = "";
	int model_id = 0;
	for (auto Detector : Dets) {

		model_id++;
		std::vector<BatchResult_yolo> batch_model_res;

		Detector->detect_yolo(mat_image, batch_model_res);

		Json::Value model_res;
		for (int i = 0; i < batch_model_res.size(); i++)  //每个model的[ ([[],[]]) , ([[],[]]) ]
		{
			Json::Value array = model_res["bboxes_batch"];
			if (batch_model_res[i].size() < 1) {
				for (int j = 0; j < batch_model_res[i].size(); j++) //每个model的[[ ([]),([])] , [([]),([])]]
				{
					array[i][j] = "null";
				}
			}
			else {
				for (int j = 0; j < batch_model_res[i].size(); j++) //每个model的[[ ([]),([])] , [([]),([])]]
				{
					//[1，1，1，1，1，0.99]
					array[i][j][0] = batch_model_res[i][j].id;
					array[i][j][1] = batch_model_res[i][j].x1;
					array[i][j][2] = batch_model_res[i][j].y1;
					array[i][j][3] = batch_model_res[i][j].x2;
					array[i][j][4] = batch_model_res[i][j].y2;
					array[i][j][5] = batch_model_res[i][j].prob;
				}
			}
			
			model_res["bboxes_batch"] = array;
			std::vector<std::string> cl_n;
			Detector->get_cl_n(cl_n);

			Json::Value array_cl_n = model_res["classes"];

			for (int j = 0; j < cl_n.size(); j++) //每个model的[[ ([]),([])] , [([]),([])]]
			{
				array_cl_n[j] = cl_n[j];
			}
			model_res["bboxes_batch"] = array;
			Json::Value arrayclasses = model_res["classes"];
			model_res["classes"] = array_cl_n;
			model_res["model"] = "model" + std::to_string(model_id);
			//std::cout << model_res.toStyledString() << std::endl;
			result += model_res.toStyledString();
			result += ";";
		}
	}
	//std::cout << result << std::endl;
	return true;
}

std::string Detectors::get_info()
{
	std::string Info = "";
	for (auto Detector : Dets) {
		std::string I;
		Detector->getInfo(I);
		Info += I;
	}
	return Info;
}

bool Detectors::release()
{
	for (auto Detector : Dets) {
		Detector->release();
	}
	return true;
}

