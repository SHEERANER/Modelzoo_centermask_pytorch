# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import torch

sys.path.append('./AdvancedEAST-PyTorch')

from model_VGG import advancedEAST


def pth2onnx(input_file, output_file):
    model = advancedEAST()
    state_dict = {k.replace('module.', ''): v for k, v in torch.load(
        input_file, map_location='cpu').items()}
    model.load_state_dict(state_dict)

    model.eval()
    input_names = ["input_1"]
    output_names = ["output_1"]
    dynamic_axes = {'input_1': {0: '-1'}, 'output_1': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 736, 736)

    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes = dynamic_axes,
        verbose=True,
        opset_version=11)


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    pth2onnx(input_file, output_file)
