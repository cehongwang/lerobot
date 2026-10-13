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

"""Export a trained GR00T N1.7 policy as a chain of ExecuTorch programs that run on TensorRT.

Run it on the device the policy will run on, because a TensorRT engine only runs on the GPU
model that built it:

    python examples/export/groot_executorch_tensorrt.py \
        --policy.path=outputs/train/groot_so101/checkpoints/last/pretrained_model \
        --dataset.repo_id=<user>/so101_dataset

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`, with the same task.
"""

from pathlib import Path

import torch
import torch_tensorrt
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
from groot_recipe import TENSORRT_OPTIONS, export_parts, parse_args
from torch import nn


def compile_part(
    module: nn.Module, inputs: tuple, input_names: list[str], output_names: list[str], path: Path
) -> None:
    with torch.no_grad():
        program = torch.export.export(module, inputs)
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=inputs, **TENSORRT_OPTIONS)
    torch_tensorrt.save(
        engine,
        str(path),
        output_format="executorch",
        arg_inputs=inputs,
        retrace=False,
        use_cuda_graphs=True,
        backend_config=ExecutorchBackendConfig(
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
            # Each program reads and writes CUDA tensors the rollout owns, so the chain never copies to the host.
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
        ),
    )


if __name__ == "__main__":
    export_parts(
        parse_args(__doc__, "executorch_tensorrt"), __file__, "{name}.pte", compile_part, device_resident=True
    )
