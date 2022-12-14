# RegNetY-1.6GF Onnx模型端到端推理指导
-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
	-   [2.1 深度学习框架](#21-深度学习框架)
	-   [2.2 python第三方库](#22-python第三方库)
-   [3 模型转换](#3-模型转换)
	-   [3.1 pth转onnx模型](#31-pth转onnx模型)
	-   [3.2 onnx转om模型](#32-onnx转om模型)
-   [4 数据集预处理](#4-数据集预处理)
	-   [4.1 数据集获取](#41-数据集获取)
	-   [4.2 数据集预处理](#42-数据集预处理)
	-   [4.3 生成数据集信息文件](#43-生成数据集信息文件)
-   [5 离线推理](#5-离线推理)
	-   [5.1 benchmark工具概述](#51-benchmark工具概述)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理TopN精度统计](#61-离线推理TopN精度统计)
	-   [6.2 开源TopN精度](#62-开源TopN精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[RegNetY-1.6GF论文](https://arxiv.org/abs/2003.13678)  

### 1.2 代码地址
[RegNetY-1.6GF代码](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py)  
branch:master commit_id:742c2d524726d426ea2745055a5b217c020ccc72

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.1

torch == 1.8.1
torchvision == 0.9.1
onnx == 1.9.0
```

### 2.2 python第三方库

```
numpy == 1.20.1
Pillow == 8.2.0
opencv-python == 4.5.2.52
timm == 0.4.9
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.下载pth权重文件  
[RegNetY-1.6GF预训练pth权重文件](wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_016-54367f74.pth  )  
```
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_016-54367f74.pth  
```
文件的MD5sum值是：fb4fc8ffd7cbf1a209d66a5ca621b451

2.RegNetY-1.6GF模型代码在timm里，安装timm，arm下需源码安装，参考https://github.com/rwightman/pytorch-image-models
，若安装过程报错请百度解决
```
git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models
python3.7 setup.py install
cd ..
```
3.编写pth2onnx脚本RegNetY_onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 RegNetY_onnx.py regnety_016-54367f74.pth RegNetY-1.6GF.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./RegNetY-1.6GF.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=RegNetY-1.6GF_bs1 --log=debug --soc_version=Ascend310

```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 imagenet_torch_preprocess.py /root/datasets/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 get_info.py bin ./prep_dataset ./RegNetY-1.6GF_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=RegNetY-1.6GF_bs1.om -input_text_path=./RegNetY-1.6GF_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ root/datasets/imagenet/val_label.txt ./ result_bs1.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
batch1的精度
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "77.85%"}, {"key": "Top2 accuracy", "value": "87.5%"}, {"key": "Top3 accuracy", "value": "90.77%"}, {"key": "Top4 accuracy", "value": "92.58%"}, {"key": "Top5 accuracy", "value": "93.72%"}]}
```
batch16的精度
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "77.83%"}, {"key": "Top2 accuracy", "value": "87.5%"}, {"key": "Top3 accuracy", "value": "90.74%"}, {"key": "Top4 accuracy", "value": "92.57%"}, {"key": "Top5 accuracy", "value": "93.75%"}]}
```

### 6.2 开源TopN精度
[timm官网精度](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv)
```
model	    top1	top1_err	top5	top5_err	param_count	img_size	cropt_pct	interpolation
regnety_016	77.862	22.138	    93.720	6.280	    11.20	    224	        0.875	     bicubic
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 185.966, latency: 268866
[data read] throughputRate: 197.421, moduleLatency: 5.06532
[preprocess] throughputRate: 196.443, moduleLatency: 5.09054
[infer] throughputRate: 186.579, Interface throughputRate: 263.872, moduleLatency: 4.71504
[post] throughputRate: 186.579, moduleLatency: 5.35966

```
Interface throughputRate: 263.872，263.872x4=1055.488既是batch1 310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  
```
[e2e] throughputRate: 255.635, latency: 195592
[data read] throughputRate: 271.167, moduleLatency: 3.68776
[preprocess] throughputRate: 270.679, moduleLatency: 3.69441
[infer] throughputRate: 256.689, Interface throughputRate: 714.514, moduleLatency: 2.86055
[post] throughputRate: 16.0426, moduleLatency: 62.3341


```
Interface throughputRate: 714.514，714.514x4=2858.056既是batch16 310单卡吞吐率  
batch4性能：
```
[e2e] throughputRate: 235.164, latency: 212617
[data read] throughputRate: 248.782, moduleLatency: 4.01958
[preprocess] throughputRate: 248.379, moduleLatency: 4.02611
[infer] throughputRate: 236.39, Interface throughputRate: 580.005, moduleLatency: 3.24631
[post] throughputRate: 59.0971, moduleLatency: 16.9213

```
batch4 310单卡吞吐率：580.005x4=2320.02fps  
batch8性能：
```
[e2e] throughputRate: 265.063, latency: 188634
[data read] throughputRate: 282.314, moduleLatency: 3.54216
[preprocess] throughputRate: 281.758, moduleLatency: 3.54915
[infer] throughputRate: 266.417, Interface throughputRate: 673.207, moduleLatency: 2.94789
[post] throughputRate: 33.3015, moduleLatency: 30.0287


```
batch8 310单卡吞吐率：673.207x4=2692.828fps  
batch32性能：
```
[e2e] throughputRate: 169.69, latency: 294654
[data read] throughputRate: 171.811, moduleLatency: 5.82035
[preprocess] throughputRate: 171.55, moduleLatency: 5.82919
[infer] throughputRate: 170.469, Interface throughputRate: 683.088, moduleLatency: 2.91471
[post] throughputRate: 5.32876, moduleLatency: 187.661

```
batch32 310单卡吞吐率：683.088x4=2732.352fps  

 **性能优化：**  
>没有遇到性能不达标的问题，故不需要进行性能优化
