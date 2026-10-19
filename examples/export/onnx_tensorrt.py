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

"""Export a trained policy to ONNX, and build TensorRT engines from it, with no PyTorch at run time.

Run it on the board that will run the policy, because a TensorRT engine only runs on the GPU model that
built it:

    python examples/export/onnx_tensorrt.py \
        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model

Then run the folder it writes with `lerobot-rollout --policy.path=<folder>`. To run the same export on
another board, copy the folder there and rebuild its engines with build_engine.py.
"""

from pathlib import Path

import torch
from build_engine import build_engine
from export import TENSORRT_OPTIONS, Part, export, free_memory, parse_args


def compile_part(part: Part, path: Path) -> None:
    onnx_path = path.with_suffix(".onnx")
    with torch.no_grad():
        torch.onnx.export(
            part.module,
            part.inputs,
            onnx_path,
            dynamo=True,
            input_names=part.input_names,
            output_names=part.output_names,
        )
    # As offload_module_to_cpu does on the other routes: CPU memory can swap during the build; GPU memory cannot.
    part.module.cpu()
    free_memory()
    build_engine(onnx_path, path, TENSORRT_OPTIONS["workspace_size"], TENSORRT_OPTIONS["optimization_level"])


if __name__ == "__main__":
    args, recipe = parse_args(__doc__, "onnx_tensorrt")
    export(__file__, args, recipe, compile_part, ".engine")
