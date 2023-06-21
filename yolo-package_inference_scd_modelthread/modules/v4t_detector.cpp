


#include "v4t_detector.h"



using namespace tensor_rt;

void YoloDectector::get_cl_n(std::vector<std::string>& cl_n)
{
	cl_n = _yolo_info.cl_n;
}

void YoloDectector::get_m_n(std::string& model_name)
{
	model_name = _yolo_info.model_name;
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
		std::cout << vec_ds_images[i].getImageHeight() << std::endl;
		std::cout << vec_ds_images[i].getImageWidth() << std::endl;;
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

void YoloDectector::detect_seg(const std::vector<cv::Mat>& vec_image,
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
	int inputsize = static_cast<int>(vec_ds_images.size()) * 3 * _yolo_info.input_h * _yolo_info.input_w;
	int outputsize = static_cast<int>(vec_ds_images.size()) * _yolo_info.classes;

	float* data = new float[inputsize];
	float* prob = new float[outputsize];


	blobFromDsImages(vec_ds_images, m_blob, _p_net->getInputH(), _p_net->getInputW(), _p_net->getNetworkType(), data);

	_p_net->doInference_v5_seg(data, prob, static_cast<int>(vec_ds_images.size()));


	vec_ds_images.clear();
	delete[] data;
	delete[] prob;
}


void YoloDectector::detect_cls(const std::vector<cv::Mat>& vec_image,
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
	int inputsize = static_cast<int>(vec_ds_images.size()) * 3 * _yolo_info.input_h * _yolo_info.input_w;
	int outputsize = static_cast<int>(vec_ds_images.size()) * _yolo_info.classes;

	float* data = new float[inputsize];
	float* prob = new float[outputsize];

	
	blobFromDsImages(vec_ds_images, m_blob, _p_net->getInputH(), _p_net->getInputW(), _p_net->getNetworkType(), data);

	_p_net->doInference_v5_cls(data, prob, static_cast<int>(vec_ds_images.size()));

	std::vector<double> ress;



	for (int i = 0; i < outputsize; i+=4)
	{
		std::vector<double> res;
		for (int j = 0; j < _yolo_info.classes; j++) {
			res.push_back(prob[i+j]);
		}
		ress = _p_net->softmax(res);
		std::vector<double>::iterator itMax = max_element(ress.begin(), ress.end());
		std::vector<tensor_rt::yolores> vec_result;
		vec_result.reserve(1);
		vec_result.emplace_back(std::distance(ress.begin(), itMax), *itMax, 0, 0, 0, 0);
		vec_batch_result.push_back(vec_result);
	}
	
	vec_ds_images.clear();
	delete[] data;
	delete[] prob;
}


void YoloDectector::detect_yolo(const std::vector<cv::Mat>& vec_image,
	std::vector<tensor_rt::BatchResult_yolo>& vec_batch_result)
{
	Timer timer;
	std::vector<DsImage> vec_ds_images;
	vec_batch_result.clear();
	if (vec_batch_result.capacity() < vec_image.size())
		vec_batch_result.reserve(vec_image.size());

	for (const auto& img : vec_image)
	{
		vec_ds_images.emplace_back(img, _vec_net_type[_config.net_type], _p_net->getInputH(), _p_net->getInputW());
	}
	/*Timer timer;
	timer.reset();*/

	blobFromDsImages(vec_ds_images, m_blob, _p_net->getInputH(), _p_net->getInputW());
	_p_net->doInference(m_blob.data, static_cast<uint32_t>(vec_ds_images.size()));

	for (size_t i = 0; i < vec_ds_images.size(); ++i)
	{

		//timer.reset();
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
				vec_result.emplace_back(b.label, b.prob, x1, y1, x2, y2);
			}
		}
		vec_batch_result.emplace_back(vec_result);
	}
}

void YoloDectector::getInfo(std::string& Info)
{
	_model_info.enginepath = _yolo_info.enginePath;
	_model_info.configpath = _yolo_info.configFilePath;
	_model_info.inputshape = cv::Size(_p_net->getInputH(), _p_net->getInputW());
	_model_info.classes = _yolo_info.classes;

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


	_yolo_info.precision			= _vec_precision[node["precision"].as<int>()];
	_yolo_info.networkType			= _vec_net_type[node["net_type"].as<int>()];

	if (_yolo_info.networkType == "yolov4-tiny")
	{
		std::cout << "yolov4-tiny is start load engine" << std::endl;

		_config.net_type = tensor_rt::YOLOV4_TINY;
		_yolo_info.enginePath = node["enginePath"].as<std::string>();
		_yolo_info.configFilePath = node["configFilePath"].as<std::string>();
		_yolo_info.classes = node["classes"].as<int>();
		_yolo_info.model_name = node["model_name"].as<std::string>();
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

	}
	else if (_yolo_info.networkType == "yolov5s_cls") {
		std::cout << "yolov5s_cls is start load engine" << std::endl;

		_config.net_type = tensor_rt::YOLOV5s_cls;
		_yolo_info.enginePath = node["enginePath"].as<std::string>();
		_yolo_info.classes = node["classes"].as<int>();
		_yolo_info.model_name = node["model_name"].as<std::string>();
		_yolo_info.deviceType = "kGPU";
		_yolo_info.cl_n = node["cl_n"].as<std::vector<std::string>>();
		_yolo_info.calibrationTablePath = _yolo_info.data_path + "-calibration.table";
		_yolo_info.inputBlobName = "images";
		_yolo_info.input_h = node["input_h"].as<int>();
		_yolo_info.input_w = node["input_w"].as<int>();
		_yolo_info.classes = node["classes"].as<int>();
		_yolo_info.batch_size = node["batch_size"].as<int>();

		_infer_param.printPerfInfo = false;
		_infer_param.printPredictionInfo = false;
		_infer_param.calibImagesPath = "";
		_infer_param.probThresh = node["detect_thresh"].as<float>();
		_infer_param.batchSize = node["batch_size"].as<int>();

	}
	else {
		std::cout << "model is not support now " << std::endl;
	}
		


	this->set_gpu_id(node["gpu_id"].as<int>());

}

void YoloDectector::build_net()
{
	if (_config.net_type == tensor_rt::YOLOV4 || _config.net_type == tensor_rt::YOLOV4_TINY)
	{
		std::cout << "YOLOV4_tiny start build net" << std::endl;

		_p_net = std::unique_ptr<Yolo>{ new YoloV4(_yolo_info,_infer_param) };
	}
	else if (_config.net_type == tensor_rt::YOLOV5s_cls) {
		std::cout << "YOLOV5s_cls start build net" << std::endl;
		_p_net = std::unique_ptr<Yolo>{ new YoloV5s_cls(_yolo_info,_infer_param) };
	}
	else
	{
		assert(false && "Unrecognised network_type.");
	}
}

bool Detectors::thread_infer(int loc_id, std::vector<cv::Mat>& imgs)
{

	//TQmap thread - queue LTmap
	std::map<int, std::vector<int>>::iterator it;
	for (it = LTmap.begin(); it != LTmap.end(); ++it)
	{
		if(it->first == loc_id)
		{
			for (auto t : it->second)
			{
				producer_queues[t].push(imgs);
				printf("[thread %d][push] produce.size is %d\n", t, producer_queues[t].size());
			}
		}
	}
}

bool Detectors::thread_init(const std::string manager_yaml_path, ConcurrenceQueue<std::string>& Result_queue)
{
	YAML::Node node = YAML::LoadFile(manager_yaml_path);

	int thread_id = 0;
	int thread_num = 0;

	for (int i = 0; i < node["WorkLocation"].size(); i++) {
		std::vector<int> thread_ids;
		//std::cout << "-----------workloc " << i << std::endl;
		for (auto yaml_path : node["WorkLocation"][i]["models"]) {
			std::vector<std::string> yamls;
			thread_num++;
			//std::cout << "[thread]num: " << thread_num << std::endl;

			for (int m = 0; m < yaml_path["thread"].size(); m++)
			{
				std::string yaml = yaml_path["thread"][m].as<std::string>();
				yamls.push_back(yaml);
			}

			
			threads.push_back(std::thread(inthread_infer, std::ref(yamls), std::ref(Result_queue), std::ref(producer_queues[thread_id]),std::ref(thread_id)));
			//std::cout << "[thread]num: " << thread_num << "start!" << std::endl;
			//std::cout << "[thread]producer_queues: " << producer_queues.capacity() << std::endl;
			thread_ids.push_back(thread_id);
			thread_id++;

			Sleep(1000);

		}
		LTmap.insert(std::map<int, std::vector<int>>::value_type(i, thread_ids));
	}

	std::cout <<"threads.size() " << threads.size() << std::endl;
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::detach));
	Timer timer;
	//while (1) {
	//	//Sleep(1000);
	//	std::cout << "[thread alive] size = "<< Result_queue.size() << std::endl;
	//	if(Result_queue.size()>30)
	//	{
	//		std::cout << "-----------------------------------------------------------------------------";
	//		timer.out("Cqueue30 time ");
	//		timer.reset();

	//		while (Result_queue.size() > 1)
	//		{
	//			Result_queue.pop();
	//			//std::cout << "[thread alive] pop size = " << Result_queue.size() << std::endl;
	//			std::string temp = *(Result_queue.pop());
	//			std::cout << temp << std::endl;
	//		}
	//		//timer.out("pop time");
	//		std::cout << "[thread alive] size = " << Result_queue.size() << std::endl;
	//		std::cout << "-----------------------------------------------------------------------------";

	//		timer.reset();
	//	}
	//	else {
	//		Sleep(10);
	//	}
	//}
	return true;
}

//  TODO1 差一个生产队列，差一个Grun做release
void Detectors::inthread_infer(std::vector<std::string> yamls, ConcurrenceQueue<std::string>& Cqueue, ConcurrenceQueue<std::vector<cv::Mat>>& produce_queue, int thread_id)
{
	
	int cycle_id;
	std::vector<std::string> img_names;
	Json::Reader reader;
	Json::Value value;

	std::vector<std::shared_ptr<YoloDectector>> Dets;

	for (auto yaml : yamls)
	{
		std::shared_ptr<YoloDectector> Detector(new YoloDectector());
		Detector->init(yaml);
		Dets.push_back(Detector);
	}
	
	/*if (reader.parse(imgInfo, value)) {

		Json::Value cycle_id_ = value["cycle_id"];
		cycle_id = cycle_id_.asInt();
		Json::Value imgarray_ = value["img_names"];
		for (unsigned int i = 0; i < imgarray_.size(); i++)
		{
			img_names.push_back(imgarray_[i].asString());
		}
	}*/
	Sleep(100);
	while(1)
	{
		Sleep(10);
		if (produce_queue.size() > 0)
		{
			std::vector<cv::Mat> mat_image = *produce_queue.pop();
			for (auto Detector : Dets)
			{
				std::vector<BatchResult_yolo> batch_model_res;

				Timer timer;
				if (Detector->getModelType() == "yolov4-tiny") {
					timer.reset();
					Detector->detect_yolo(mat_image, batch_model_res);
					//timer.out("det  ");

				}
				if (Detector->getModelType() == "yolov5s_cls") {
					timer.reset();
					Detector->detect_cls(mat_image, batch_model_res);
					//timer.out("cls  ");

				}
				if (Detector->getModelType() == "yolov5s_seg") {
					timer.reset();
					Detector->detect_seg(mat_image, batch_model_res);
					//timer.out("seg  ");

				}

				Json::Value model_res;
				Json::Value array = model_res["bboxes_batch"];
				Cqueue.push(Detector->getModelType());
				for (int i = 0; i < batch_model_res.size(); i++)  //每个model的[ ([[],[]]) , ([[],[]]) ]
				{

					if (batch_model_res[i].size() < 1) {
						array[i][0][0] = "null";
					}
					else {
						//array[i][0][0] = "null";
						for (int j = 0; j < batch_model_res[i].size(); j++) //每个model的[[ ([]),([])] , [([]),([])]]
						{
							//[1，1，1，1，1，0.99]
							array[i][j][0] = batch_model_res[i][j].id;
							array[i][j][1] = batch_model_res[i][j].x1;
							array[i][j][2] = batch_model_res[i][j].y1;
							array[i][j][3] = batch_model_res[i][j].x2;
							array[i][j][4] = batch_model_res[i][j].y2;
							array[i][j][5] = batch_model_res[i][j].prob;


							//std::cout << "id " << batch_model_res[i][j].id << "prob " << batch_model_res[i][j].prob << std::endl;
						}
					}
				}
				model_res["bboxes_batch"] = array;

				Json::Value array1 = model_res["img_names"];

				for (int j = 0; j < img_names.size(); j++)
				{
					array1[j] = img_names[j];
				}
				model_res["img_names"] = array1;
				model_res["cycle_id"] = 1;



				std::vector<std::string> cl_n;
				Detector->get_cl_n(cl_n);
				Json::Value array_cl_n = model_res["classes"];
				for (int j = 0; j < cl_n.size(); j++) //每个model的[[ ([]),([])] , [([]),([])]]
				{
					array_cl_n[j] = cl_n[j];
				}

				Json::Value arrayclasses = model_res["classes"];
				model_res["classes"] = array_cl_n;

				std::string m_n;
				Detector->get_m_n(m_n);

				model_res["model"] = m_n;
				//std::cout << model_res.toStyledString() << std::endl;


				//Cqueue.push(model_res.toStyledString());
				////Cqueue += ";";
			}
		}
	}
}

bool Detectors::init(const std::string manager_yaml_path)
{

	YAML::Node node = YAML::LoadFile(manager_yaml_path);
	// 减少printf
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


bool Detectors::serial_infer(const std::vector<cv::Mat>& mat_image, std::string& imgInfo, std::string& result)
{

	std::string model_result = "";
	int model_id = 0;
	for (auto Detector : Dets) {
		model_id++;
		int cycle_id;
		std::vector<std::string> img_names;
		Json::Reader reader;
		Json::Value value;
		if (reader.parse(imgInfo, value)) {

			Json::Value cycle_id_ = value["cycle_id"];
			cycle_id = cycle_id_.asInt();
			Json::Value imgarray_ = value["img_names"];
			for (unsigned int i = 0; i < imgarray_.size(); i++)
			{
				img_names.push_back(imgarray_[i].asString());
			}
		}

		std::vector<BatchResult_yolo> batch_model_res;

		Timer timer;
		if (Detector->getModelType() == "yolov4-tiny") {
			timer.reset();
			Detector->detect_yolo(mat_image, batch_model_res);
			timer.out("det  ");

		}
		if (Detector->getModelType() == "yolov5s_cls") {
			timer.reset();
			Detector->detect_cls(mat_image, batch_model_res);
			timer.out("cls  ");

		}
		if (Detector->getModelType() == "yolov5s_seg") {
			timer.reset();
			Detector->detect_seg(mat_image, batch_model_res);
			timer.out("seg  ");

		}


		Json::Value model_res;

		// batch_model_res shape(imgnum,bboxnum,(6))

		Json::Value array = model_res["bboxes_batch"];

		for (int i = 0; i < batch_model_res.size(); i++)  //每个model的[ ([[],[]]) , ([[],[]]) ]
		{

			if (batch_model_res[i].size() < 1) {
				array[i][0][0] = "null";
			}
			else {
				//array[i][0][0] = "null";
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
		}
		model_res["bboxes_batch"] = array;

		Json::Value array1 = model_res["img_names"];

		for (int j = 0; j < img_names.size(); j++)
		{
			array1[j] = img_names[j];
		}
		model_res["img_names"] = array1;
		model_res["cycle_id"] = cycle_id;



		std::vector<std::string> cl_n;
		Detector->get_cl_n(cl_n);
		Json::Value array_cl_n = model_res["classes"];
		for (int j = 0; j < cl_n.size(); j++) //每个model的[[ ([]),([])] , [([]),([])]]
		{
			array_cl_n[j] = cl_n[j];
		}

		Json::Value arrayclasses = model_res["classes"];
		model_res["classes"] = array_cl_n;

		// TODO 1 get model name
		std::string m_n;
		Detector->get_m_n(m_n);

		model_res["model"] = m_n;
		//std::cout << model_res.toStyledString() << std::endl;


		result += model_res.toStyledString();
		result += ";";
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

