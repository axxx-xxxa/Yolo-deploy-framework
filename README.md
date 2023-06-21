# Yolo-deploy-framework
windows 
tensorrt 7/8 for yolov4-tiny(det) yolov4(det) yolov5(det/cls/seg) and frame for multi model threads


* 只是用来记录这个框架的设计和多线程的设计 其中太多接口设计的比较定制 复用性不高
* based on (enazoe/Yolov5 Yolov4 Yolov3 TensorRT Implementation)[https://github.com/enazoe/yolo-tensorrt]


# Tensorrt模型优化方案（基于单个工位实验）

## 目的

加速原串行方案的推理效率，通过资源换速度的方式并行推理，在推理效果稳定的基础上减少推理时间。



## 实验设计

| 方案         | 原理                         | 逻辑（基于单工位三模型）                                     | 调用方式  |
| ------------ | ---------------------------- | ------------------------------------------------------------ | --------- |
| 串行推理方案 | -                            | 工位线程实例化三个模型，顺序推理，顺序组合结果，输出结果     | 接口      |
| 方案一       | tensorrt生成模型内部自行优化 | 工位线程实例化一个模型(由三个模型组合)，推理，拆分模型结果并组合，输出结果 | 接口      |
| 方案二       | 基于GPU的multi-stream        | 工位线程实例化三个模型，并行推理，队列组合结果，输出结果     | 接口      |
| 方案三       | 基于CPU多线程                | 工位线程开启三个线程，每个线程初始化对应模型，通过线程通信并行推理，队列组合结果，输出结果 | 接口+队列 |



## 方案一

### 实现思路

1. 通过两个yolov4-tiny模型组合到一个engine模型中，对比推理时间及稳定性
2. 通过两个mobilenetV2模型组合到一个engine模型中，对比推理时间及稳定性

### 对比

| 模型               | 组合个数 | batch_size | GPU利用率 | 速度 | 加速效果 | 备注                         |
| ------------------ | -------- | ---------- | --------- | ---- | -------- | ---------------------------- |
| yolov4-tiny        | 1        | 1          | <10%      | 13   | -        |                              |
|                    | 1        | 10         | <10%      | 17   | -        |                              |
|                    | 1        | 18         | 14%       | 20   | -        |                              |
| yolov4-tiny-double | 2        | 1          | 18%       | 28   | - 7%     | 与串行速度*组合个数对比      |
|                    | 2        | 10         | 58%       | 75   | - 120%   |                              |
|                    | 2        | 18         | 70%       | 120  | - 250%   |                              |
| mobilenet          | 1        | 100        | <10%      | 40   | -        | mobilenet太小,batchsize设100 |
| mobilenet-2        | 2        | 100        | <10%      | 67   | 16%      |                              |
| mobilenet-4        | 4        | 100        | <10%      | 120  | 25%      |                              |
| mobilenet-6        | 6        | 100        | <10%      | 172  | 28%      |                              |
| mobilenet-9        | 9        | 100        | <10%      | 265  | 26%      |                              |

### 表格结论

GPU利用率低时，模型组合有20%左右加速效率

### 综合结论

1. 模型简单且参数量较少时，该方案可达到加速效果
2. 该方案还需考虑组合模型的成本（可以从算子支持性/复杂度方面考虑）



## 方案二

### 实现思路

1. 通过多线程推理mobilenet，对比推理时间及稳定性
2. 通过多线程推理yolov4-tiny，对比推理时间及稳定性

### 对比

| 模型          | batchsize | 运行方式 | 速度 |
| ------------- | --------- | -------- | ---- |
| yolov4-tiny*5 | 18        | 单线程   | 100  |
| yolov4-tiny*5 | 18        | 多线程   | 200  |
| mobilenet*1   | 1         | 单线程   | 1.5  |
| mobilenet*5   | 1         | 多线程   | 7.5  |
| mobilenet*1   | 18        | 单线程   | 6    |
| mobilenet*5   | 18        | 多线程   | 30   |



### 表格结论

无加速效果，对于大模型还有降速效果

### 综合结论

1. 每次开启线程需重启模型，额外开销
2. 线程频繁开启关闭，硬件内存交互逻辑复杂且无法管理

## 方案三

### 实现思路

1. 通过多持续线程推理yolov4-tiny，对比推理时间及稳定性

### 对比

* 结果队列长度大于N时，记录推理时间，该推理时间/队列长度*组合个数即为周期推理时间
* 加速效果 = (对比时间-周期推理时间)/对比时间

| 模型                        | 组合个数 | batchsize | 周期推理时间 | 对比时间 | 加速效果 | 显存 | CPU利用率 | GPU利用率 |
| --------------------------- | -------- | --------- | ------------ | -------- | -------- | ---- | --------- | --------- |
| yolov4-tiny                 | 1        | 1         | 3ms          | -        | -        | 1.2G | 26%       | 11%       |
| (基准)                      | 1        | 18        | 25ms         | -        | -        | 1.2G | 26%       | 16%       |
| yolov5s-cls                 | 1        | 1         | 1.5ms        | -        | -        | 0.9G | 25%       | 25%       |
| (基准)                      | 1        | 18        | 13ms         | -        | -        | 0.9G | 25%       | 10%       |
| yolov4-tiny×2+yolov5s-cls   | 3        | 1         | 6ms          | 7.5ms    | 20%      | 1.7G | 55%       | 20%       |
|                             | 3        | 18        | 45ms         | 63ms     | 29%      | 1.7G | 50%       | 22%       |
| yolov4-tiny×5               | 5        | 1         | 10ms         | 15ms     | 33%      | 2.7G | 82%       | 25%       |
|                             | 5        | 18        | 60ms         | 125ms    | 52%      | 2.7G | 75%       | 24%       |
| yolov5s-cls×5               | 5        | 1         | 5ms          | 7.5ms    | 33%      | 1.4G | 67%       | 24%       |
|                             | 5        | 18        | 21ms         | 65ms     | 67%      | 1.4G | 55%       | 17%       |
| yolov4-tiny×6+yolov5s-cls×3 | 9        | 1         | 17ms         | 22.5ms   | 24%      | 3.4G | 82%       | 33%       |
|                             | 9        | 18        | 140ms        | 189ms    | 26%      | 3.4G | 60%       | 25%       |



### 表格结论

加速效果明显，CPU开销大，加速效果随batch_size增大而更加明显

### 综合结论

1. 稳定
2. 加速明显
3. CPU需要合理分配工作
4. 可以合并工位，既然是开N个线程，在代码层面无需根据工位来分配任务，可以直接根据模型 / 线程来分配任务
5. 易保护，根据线程解耦任务
6. 需改接口并且主要考虑CPU压力问题



## 实验结论

| 方案         | 结论                        |
| ------------ | --------------------------- |
| 串行推理方案 | -                           |
| 方案一       | 不适用                      |
| 方案二       | 不适用                      |
| 方案三       | 牺牲CPU资源可以达到加速效果 |

* 可在后续设计中，综合考量CPU性能，一定量组合模型到一个线程，减少线程数量



# 相关探索

* [IS that one GPU can only support one tensorRT ENGINE inference?](https://github.com/NVIDIA/TensorRT/issues/2394)

  方案二基于Multi-stream完成，也许可以解释方案二为什么无法加速

  在性能较差的显卡上，Multi-stream有可能无法加速

  基于显卡各个集群中的 SM个数（流处理器） 

  |        | 显存 | FP32峰值性能 | CUDA cores | CUDA SM |
  | ------ | ---- | ------------ | ---------- | ------- |
  | 2080TI | 11G  | 11 TFLOPS    | 4352       | 8       |
  | 3090   | 24G  | 35.6 TFLOPS  | 10496      | 128     |
  | A100   | 80G  | 19.5 TFLOPS  | 6912       | 64      |

  if the previous kernel is about to finish and already frees up some SMs, it is possible that the next cuda kernel on another stream can start early.

  Note that multi-stream can only happen if a CUDA kernel does not use up all the SMs on the GPU. On smaller GPUs like GTX 1070/1080, the SMs are almost always used up by single kernel, so multi-stream may be relatively rarer. Multi-stream helps more for larger GPUs like A100 or H100.

* [multiple streams parallel inference engine](https://github.com/NVIDIA/TensorRT/issues/2353)

  1.当申请host内存占用时间较长时，可以多线程处理

  2.仅enqueue/enqueueV2 异步接口支持多线程

  Another thing I would try is to launch the three inferences on different threads. This is in case your engine contains required synchronizations in enqueueV2() call.

* [Multi-threaded model loading performance](https://github.com/NVIDIA/TensorRT/issues/1405)

  build engine时做好的做法是无其他程序占用cpu和gpu，且不允许多线程build

  TRT's builder is not thread safe, so I don't think we support building two engines in parallel in the same process.

* [How to run inference in multithread( only Allocate host and device buffers once for all execution contexts](https://github.com/NVIDIA/TensorRT/issues/1367)

  allocate一次buffer，运行在多个context上，但多个context不允许并行，会存在竞争

  基于python的，用在C++不太靠谱

  you can allocate only one set of input/output buffers, and use that set of buffers (called "bindings") to call different contexts' enqueueV2() function. However, you will need to manage the synchronization on your own to make sure that no two contexts are running inference at the same time, creating race conditions.

* [enqueueV2(Asynchronously execute inference) will block calling host thread, at this time , will it take up CPU resources?](https://github.com/NVIDIA/TensorRT/issues/999)

  不会

  For normal networks like CNNs, `enqueueV2()` should return very quickly while the GPUs continue to run. If the network graph contains loops, `enqueueV2()` may be synchronous since CPUs need to check stopping conditions.

  `enqueueV2()` doesn't wait until the GPUs result is ready. It is an asynchronous function. It is up to your application to decide when to call cudaStreamSynchronize(). `enqueueV2()` does not call `cudaStreamSynchronize()`. Is your question about whether cudaStreamSynchronize() takes CPU resources while waiting? If so, I think that's a question for CUDA.
