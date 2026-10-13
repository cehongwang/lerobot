#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Export a trained pi0.5 policy as AOTInductor packages that run on TensorRT.

Run it on the device the policy will run on, because a TensorRT engine only runs on the GPU
model that built it:

    python examples/export/pi05_torch_tensorrt.py \
        --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model \
        --task="pick up the block and place it in the cup"

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`, with the same task.
Running it needs PyTorch and Torch-TensorRT.
"""

from pathlib import Path

import torch
import torch_tensorrt
from pi05_recipe import TENSORRT_OPTIONS, export_parts, parse_args
from torch import nn


def compile_part(
    module: nn.Module, inputs: tuple, input_names: list[str], output_names: list[str], path: Path
) -> None:
    with torch.no_grad():
        program = torch.export.export(module, inputs)
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=inputs, **TENSORRT_OPTIONS)
    torch_tensorrt.save(engine, str(path), output_format="aot_inductor", arg_inputs=inputs)


if __name__ == "__main__":
    export_parts(
        parse_args(__doc__, "torch_tensorrt"), __file__, "{name}.pt2", compile_part, device_resident=True
    )
