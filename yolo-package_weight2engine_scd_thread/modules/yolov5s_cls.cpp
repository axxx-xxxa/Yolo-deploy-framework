
#include "yolov5s_cls.h"

///
/// \brief YoloV4::YoloV4
/// \param network_info_
/// \param infer_params_
///
YoloV5s_cls::YoloV5s_cls(const NetworkInfo& network_info_, const InferParams& infer_params_)
	: Yolo(network_info_, infer_params_)
{
}
std::vector<BBoxInfo> YoloV5s_cls::decodeTensor(const int imageIdx, const int imageH, const int imageW, const TensorInfo& tensor)
{
	std::vector<BBoxInfo> temp;
	return temp;
}


//int export_yolov5s_cls(const char* onnx_filename) {
//
//	Logger gLogger;
//
//	IBuilder* builder = createInferBuilder(gLogger);
//	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
//
//	auto parser = nvonnxparser::createParser(*network, gLogger);
//
//	parser->parseFromFile(onnx_filename, static_cast<int>(Logger::Severity::kWARNING));
//	for (int i = 0; i < parser->getNbErrors(); ++i)
//	{
//		std::cout << parser->getError(i)->desc() << std::endl;
//	}
//	std::cout << "successfully load the onnx model" << std::endl;
//
//	// 2��build the engine
//	unsigned int maxBatchSize = 1;
//	builder->setMaxBatchSize(maxBatchSize);
//	IBuilderConfig* config = builder->createBuilderConfig();
//	//config->setMaxWorkspaceSize(1 << 20);
//	config->setMaxWorkspaceSize(128 * (1 << 20));  // 16MB
//	config->setFlag(BuilderFlag::kFP16);
//	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
//
//	// 3��serialize Model
//	IHostMemory* gieModelStream = engine->serialize();
//	std::ofstream p("test.engine", std::ios::binary);
//	if (!p)
//	{
//		std::cerr << "could not open plan output file" << std::endl;
//		return -1;
//	}
//	p.write(reinterpret_cast<const char*>(gieModelStream->data()), gieModelStream->size());
//	gieModelStream->destroy();
//
//
//	std::cout << "successfully generate the trt engine model" << std::endl;
//	return 0;
//}