# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
from DistributedResnet50.image_classification import resnet
import torch.onnx

from collections import OrderedDict


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(pth_file_path, onnx_file_path):
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = resnet.build_resnet("resnet50", "classic")
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}} 
    dummy_input = torch.randn(64, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_file_path, dynamic_axes = dynamic_axes, input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    src_file_path = "./resnet50_pytorch_1.4.pth.tar"
    dst_file_path = "resnet50_npu_dynamic.onnx"
    convert(src_file_path, dst_file_path)

