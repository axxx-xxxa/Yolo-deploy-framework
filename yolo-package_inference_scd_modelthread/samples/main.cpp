
#include <iostream>
#include <Windows.h>
#include "v4t_detector.h"
#include "class_timer.hpp"
#include <fstream>
using namespace tensor_rt;




void Stringsplit(const std::string& str, const char split, std::vector<std::string>& res)
{
	if (str == "")		return;
	std::string strs = str + split;
	size_t pos = strs.find(split);

	while (pos != strs.npos)
	{
		std::string temp = strs.substr(0, pos);
		res.push_back(temp);
		strs = strs.substr(pos + 1, strs.size());
		pos = strs.find(split);
	}
}

std::string parse_json(std::string& result)
{
	Json::Value res;
	Json::Reader reader;
	std::vector<std::string> res_list;
	Stringsplit(result, ';', res_list);
	std::string _;
	Json::FastWriter writer;
	for (int i = 0; i < res_list.size() - 1; i++) {
		if (reader.parse(res_list[i], res))
		{
			_ += writer.write(res);
			_ += ";";
		}
	}
	return _;
}

bool Exist() {
	const char* name = "test.jpg";
	std::ifstream f(name);
	return f.good();
}

void producer(ConcurrenceQueue<std::vector<cv::Mat>>& Producer_queue, int bs)
{
	cv::Mat t = cv::imread("test.jpg");
	cv::Mat temp = t.clone();
	std::vector<cv::Mat> temp_v;
	while (1)
	{
		Sleep(300);
		for (int i = 0; i < bs; i++)
		{
			temp_v.push_back(temp);
		}
		Producer_queue.push(temp_v);        
	}
}

int main(int argc, char** argv)
{
	Sleep(1000);
	if (argc < 2) {
		std::cout << "Input 1.model yaml path" << std::endl;
		exit(1);
	}
	if (!Exist()) {
		std::cout << "test.jpg is not exist!" << std::endl;
		exit(1);
	}
	Timer timer;

	//// -----------thread----------- //
	//const std::string yaml_path = argv[1];
	//ConcurrenceQueue<std::string> Result_queue;
	//ConcurrenceQueue<std::vector<cv::Mat>> Producer_queue;
	//std::shared_ptr<Detectors> model_group(new Detectors());
	//model_group->thread_init(yaml_path, Result_queue);

	//std::vector<cv::Mat> imgs0;
	//std::vector<cv::Mat> imgs1;
	//std::vector<cv::Mat> imgs2;
	//cv::Mat t = cv::imread("test.jpg");
	//cv::Mat temp = t.clone();
	//for (int i = 0; i < 2; i++)
	//{
	//	imgs0.push_back(temp);
	//}
	//for (int i = 0; i < 2; i++)
	//{
	//	imgs1.push_back(temp);
	//}
	//for (int i = 0; i < 2; i++)
	//{
	//	imgs2.push_back(temp);
	//}
	//while (1) {
	//	Sleep(2000);
	//	model_group->thread_infer(0, imgs0);
	//	model_group->thread_infer(1, imgs1);
	//	model_group->thread_infer(2, imgs2);
	//	std::cout << "Main Sleep" << std::endl;
	//	
	//}
	// -------------------------------------


	
	std::shared_ptr<Detectors> model_group(new Detectors());
	std::vector<cv::Mat> batch_input;
	std::vector<BatchResult> batch_res;
	std::string imgInfo;
	std::string result;
	const std::string yaml_path = argv[1];
	const int bs = int(argv[2]);
	cv::Mat t = cv::imread("test.jpg");
	cv::Mat temp = t.clone();
	for (int i = 0; i < 18; i++) 
	{
		batch_input.push_back(temp);
	} 
	timer.reset();
	model_group->init(yaml_path);
	timer.out("[Init]");
	
	timer.reset();
	std::string Info = model_group->get_info();
	timer.out("[GetInfo]");

	timer.reset();
	model_group->serial_infer(batch_input, imgInfo, result);
	timer.out("[FirstInfer]");
	std::cout << result << std::endl;


	for (int i = 0; i < 20; i++) {
		timer.reset();
		model_group->serial_infer(batch_input, imgInfo, result);
		timer.out("[Infer]");
	}

	timer.reset();
	model_group->release();
	timer.out("[Release]");

}

