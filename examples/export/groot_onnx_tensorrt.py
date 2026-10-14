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

"""Export a trained GR00T N1.7 policy to ONNX, then build TensorRT engines from it.

Run it on the device the policy will run on, because a TensorRT engine only runs on the GPU
model that built it:

    python examples/export/groot_onnx_tensorrt.py \
        --policy.path=outputs/train/groot_so101/checkpoints/last/pretrained_model \
        --dataset.repo_id=<user>/so101_dataset

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`, with the same task.
"""

from pathlib import Path

import torch
from build_engine import build_engine
from groot_recipe import TENSORRT_OPTIONS, export_parts, free_memory, parse_args
from torch import nn


def compile_part(
    module: nn.Module, inputs: tuple, input_names: list[str], output_names: list[str], path: Path
) -> None:
    onnx_path = path.with_suffix(".onnx")
    with torch.no_grad():
        torch.onnx.export(
            module, inputs, onnx_path, dynamo=True, input_names=input_names, output_names=output_names
        )
    # As offload_module_to_cpu does on the other routes: CPU memory can swap during the build; GPU memory cannot.
    module.cpu()
    free_memory()
    build_engine(
        onnx_path,
        path,
        workspace_gib=TENSORRT_OPTIONS["workspace_size"] / (1 << 30),
        optimization_level=TENSORRT_OPTIONS["optimization_level"],
    )


if __name__ == "__main__":
    # CUDA graphs stay off: TensorRT cannot record this chain of engines in one (the capture is invalidated).
    export_parts(parse_args(__doc__, "onnx_tensorrt"), __file__, "{name}.engine", compile_part)
