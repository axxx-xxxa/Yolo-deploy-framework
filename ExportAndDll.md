# Export

## 运行

### 运行环境

系统环境：cuda11.2，cudnn8.2，TensorRT8.2.1.8

依赖环境：yaml

### 运行方式

#### 工程文件

工程路径：yolo-package_weight2engine_scd_thread

执行文件路径：./sln/Export.exe

依赖文件：Model.dll，yaml-cpp.dll

配置文件：sample.yaml

测试图片：test.png

#### 配置文件

![image-20230103135057913](./ExportAndDll.assets\image-20230103135057913.png)

| 参数          | 类型   | 内容     | 注释                                          |
| ------------- | ------ | -------- | --------------------------------------------- |
| type          | string | 模型类型 | support {yolov5s_cls,yolov4_tiny,yolov5s_seg} |
| detect_thresh | float  | 检测阈值 | 可忽略                                        |
| batch_size    | int    | -        | -                                             |
| precision     | string | 转换精度 | support {FP32,FP16}                           |
| cfg           | string | -        | yolov4_tiny                                   |
| weight        | string | -        | yolov4_tiny                                   |
| onnx          | string | -        | yolov5s_cls \|\| yolov5s_seg                  |



#### 运行示例

sln路径下进入cmd

Export.exe sample.yaml

#### 运行结果

在weight或onnx模型路径下生成engine模型

*.weight/ *.onnx -> *-{kFLOAT/kHALF}-batch{batch_size}.engine

## 编译

### 编译软件

VisualStudio2019

### 编译方式

1.设置Release64编译模式

2.props里各个属性表添加到属性管理器中

![image-20230103144131802](./ExportAndDll.assets\image-20230103144131802.png)

3.先生成Model解决方案，会在sln目录下生成Model.dll

4.生成Export解决方案，会在sln目录下生成Export.exe

## 代码

### 参考

[enazoe](https://github.com/enazoe/yolo-tensorrt)

[tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov4)

### 主要代码

![image-20230103145734823](./ExportAndDll.assets\image-20230103145734823.png)

# DLL

## 运行

### 运行环境

系统环境：cuda11.2，cudnn8.2，TensorRT8.2.1.8

依赖环境：yaml，json，opencv

### 运行方式

#### 工程文件

工程路径：yolo-package_inference_scd_modelthread

执行文件路径：./sln/yolo_sample/yolo_sample.exe

依赖文件：yolo_dll.dll，yaml-cpp.dll，jsoncpp.dll

配置文件：loc3_thread.yaml

测试图片：test.jpg

#### 配置文件（单线程）

![image-20230104101254540](./ExportAndDll.assets\image-20230104101254540.png)

#### 配置文件（多线程）

![image-20230104101123829](./ExportAndDll.assets\image-20230104101123829.png)

#### 运行示例

sln路径下进入cmd

可C++试运行：yolo_sample.exe loc3_thread.yaml 1

本任务只需生成yolo_dll.dll即可

## 编译

### 编译软件

VisualStudio2019

### 编译方式

1.设置Release64编译模式

2.props里各个属性表添加到属性管理器中

![image-20230104100116980](./ExportAndDll.assets\image-20230104100116980.png)

3.先生成yolo_dll解决方案，会在sln目录下生成yolo_dll.dll

4.生成yolo_sample解决方案，会在sln目录下生成yolo_sample.exe

## 代码

### 参考

[enazoe](https://github.com/enazoe/yolo-tensorrt)

[tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov4)

### 主要代码

![image-20230104100023496](./ExportAndDll.assets\image-20230104100023496.png)
