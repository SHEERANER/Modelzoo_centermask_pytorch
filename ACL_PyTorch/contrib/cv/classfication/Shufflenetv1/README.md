 # Shufflenetv1 Onnx模型端到端推理指导
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
 [shufflenetv1论文](https://arxiv.org/pdf/1707.01083.pdf)  

 ### 1.2 代码地址
 [shufflenetv1代码](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1 )  
  branch:master  
  commit_id: d69403d4b5fb3043c7c0da3c2a15df8c5e520d89  

 ## 2 环境说明

 -   **[深度学习框架](#21-深度学习框架)**  

 -   **[python第三方库](#22-python第三方库)**  

 ### 2.1 深度学习框架
 ```
 CANN 5.0.2.alpha003
 pytorch == 1.8.0
 torchvision == 0.9.0
 onnx == 1.9.0
 ```

 ### 2.2 python第三方库

 ```
 numpy == 1.18.5
 Pillow == 7.2.0
 opencv-python == 4.5.1.48
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
 [Shufflenetv1预训练pth权重文件](https://pan.baidu.com/s/1EUQVoFPb74yZm0JWHKjFOw#list/path=%2F)（提取码：mc24）  
  文件md5sum: bc2b8686fe73bf0709af2cbfcff2b895  
 ```
 https://pan.baidu.com/s/1EUQVoFPb74yZm0JWHKjFOw#list/path=%2F
 ``` 
2.shufflenetv1模型代码在代码仓中

 ```
 github上Shufflenetv1没有安装脚本，在pth2onnx脚本中引用代码仓定义的ShuffleNetv1：
 
 git clone https://github.com/megvii-model/ShuffleNet-Series.git

 ```
  3.编写pth2onnx脚本shufflenetv1_pth2onnx.py

  **说明：**  
 >注意目前ATC支持的onnx算子版本为11

 4.执行pth2onnx脚本，生成onnx模型文件
 ```
 python3.7 shufflenetv1_pth2onnx_bs1.py 1.0x.pth.tar shufflenetv1_bs1.onnx
 ```

  **模型转换要点：**  
 >动态batch的onnx转om失败并且测的性能数据也不对，每个batch的om都需要对应batch的onnx来转换，每个batch的性能数据也需要对应batch的onnx来测
 ### 3.2 onnx转om模型

 1.设置环境变量
 ```
 source env.sh
 ```
 2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.2 开发辅助工具指南 (推理) 01
 ```
 atc --framework=5 --model=./shufflenetv1_bs1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=shufflenetv1_bs1 --log=debug --soc_version=Ascend310
 ```

 ## 4 数据集预处理

 -   **[数据集获取](#41-数据集获取)**  

 -   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  
 ### 4.1 数据集获取
 该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt。

### 4.2 数据集预处理
 1.预处理脚本shufflenetv1_torch_preprocess.py

 2.执行预处理脚本，生成数据集预处理后的bin文件
 ```
 python3.7 shufflenetv1_torch_preprocess.py  /opt/npu/imagenet/val ./prep_dataset
 ```
 ### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

 2.执行生成数据集信息脚本，生成数据集信息文件
 ```
 python3.7 get_info.py bin ./prep_dataset ./shufflenetv1_prep_bin.info 224 224
 ```
 第一个参数为生成的bin文件路径，第二个为输出的info文件，后面为宽高信息
 ## 5 离线推理

 -   **[benchmark工具概述](#51-benchmark工具概述)**  

 -   **[离线推理](#52-离线推理)**  

 ### 5.1 benchmark工具概述

 benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.2  推理benchmark工具用户指南 01
 ### 5.2 离线推理
 1.设置环境变量
 ```
 source env.sh
 ```
 2.执行离线推理
 ```
 ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=shufflenetv1_bs1.om -input_text_path=./shufflenetv1_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
 ```
 输出结果默认保存在当前目录result/dumpOutput_devicex，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

 ## 6 精度对比
 -   **[离线推理TopN精度](#61-离线推理TopN精度)**  
 -   **[开源TopN精度](#62-开源TopN精度)**  
 -   **[精度对比](#63-精度对比)**  

 ### 6.1 离线推理TopN精度统计

 后处理统计TopN精度

 调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
 ```
 python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /opt/npu/imagenet/val_label.txt ./ result.json
 ```
 第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
 查看输出结果：
 ```
 {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value
": "67.66%"}, {"key": "Top2 accuracy", "value": "78.61%"}, {"key": "Top3 accuracy", "value": "83.29%"}, {"key": "Top4 accuracy", "value": "85.83%"}, {"key": "Top5 accuracy", "value": "87.61%"}]}
 ```
 经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

 ### 6.2 开源TopN精度
 [开源代码仓精度](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1 )

 ```
 | model                       | top1 | top5 |
 | --------------------------- | ---- | ---- |
 | ShuffleNetV1 1.0x (group=3) | 67.8 | 87.7 |	
 ```
 ### 6.3 精度对比
 将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
  **精度调试：**  
 >遇到精度不对，首先考虑预处理是不是没有和开源代码仓一致。

## 7 性能对比

 -   **[npu性能数据](#71-npu性能数据)**  

 ### 7.1 npu性能数据
 benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
 1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 225.491, latency: 221739
[data read] throughputRate: 238.623, moduleLatency: 4.19071
[preprocess] throughputRate: 238.364, moduleLatency: 4.19526
[infer] throughputRate: 226.585, Interface throughputRate: 392.738, moduleLatency: 3.62481
[post] throughputRate: 226.585, moduleLatency: 4.41336
 ```
 Interface throughputRate: 392.738,392.738乘以4即是310单卡吞吐

 batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  

 ```
[e2e] throughputRate: 179.484, latency: 278577
[data read] throughputRate: 185.65, moduleLatency: 5.38649
[preprocess] throughputRate: 185.36, moduleLatency: 5.39492
[infer] throughputRate: 180.299, Interface throughputRate: 1335.44, moduleLatency: 2.25621
[post] throughputRate: 11.2682, moduleLatency: 88.7455

 ```
 Interface throughputRate: 1335.44，1335.44x4既是batch16 310单卡吞吐率  
 batch4性能：  
./benchmark.x86_64 -round=20 -om_path=shufflenetv1_bs4.om -device_id=3 -batch_size=4
 ```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_shufflenetv1_bs4_in_device_3.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 986.58samples/s, ave_latency: 1.03751ms
----------------------------------------------------------------

 ```
 batch4 310单卡吞吐率：986.58x4=3946.32fps  
 batch8性能：

 ```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_shufflenetv1_bs8_in_device_3.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 1235.73samples/s, ave_latency: 0.821675ms
----------------------------------------------------------------
 ```
 batch8 310单卡吞吐率：1235.73x4=4942.92fps  
 batch32性能：

 ```

[INFO] PureInfer result saved in ./result/PureInfer_perf_of_shufflenetv1_bs32_in_device_3.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 1258.14samples/s, ave_latency: 0.795141ms
----------------------------------------------------------------
 ```
 batch32 310单卡吞吐率： 1258.14x4=5032.56fps  

 **性能优化：**  
 1.   CANN 5.0.2.alpha003版本将PadV3D与AvgPoolV2融合提高了性能，所以基于此版本测。  
  2.  从profiling数据的op_statistic_0_1.csv看出影响性能的是Conv2D算子，TransData，Transpose算子，Conv2D算子不存在问题，由于格式转换om模型Transpose前后需要有TransData算子，从op_summary_0_1.csv可以看出单个TransData或Transpose算子aicore耗时，确定是否可以优化。  
  3.  dynamic/transpose.py：68中已经通过shape白名单优化掉了TransposeD  
>
     five_2_four.py：9928  
      修改如下：  
 >   elif dst_format.lower() == "nchw" and dst_shape in [[2560, 512, 4, 26], [2560, 512, 1, 26], [2560, 256, 8, 25],  
                                                    [16, 240, 7, 7], [16, 120, 14, 14],  
						    [1, 240, 7, 7], [1, 120, 14, 14],  
                                                    [4, 240, 7, 7], [4, 120, 14, 14],  
						    [8, 240, 7, 7], [8, 120, 14, 14],  
                                                    [32, 240, 7, 7], [32, 120, 14, 14]]:  
>  
     four_2_five.py：1219  
     修改如下：  
 >    if src_format.upper() == "NCHW" and shape_input in [[16, 240, 7, 7], [16, 120, 14, 14],  
                                                    [1, 240, 7, 7], [1, 120, 14, 14],  
                                                    [4, 240, 7, 7], [4, 120, 14, 14],  
						    [8, 240, 7, 7], [8, 120, 14, 14],  
                                                    [32, 240, 7, 7], [32, 120, 14, 14]] and dtype_input == "float16":  
 > 