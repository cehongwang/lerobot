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

"""Export a trained pi0.5 policy as ExecuTorch programs that run on ExecuTorch's CUDA backend.

The CUDA backend compiles each program with AOTInductor into CUDA and Triton kernels, without
TensorRT, and saves its weights in a `.ptd` file beside it. Run it on the device the policy will
run on, because the kernels are tuned for the GPU that compiles them:

    python examples/export/pi05_executorch_cuda.py \
        --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model \
        --task="pick up the block and place it in the cup"

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`, with the same task.
"""

from pathlib import Path

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
from pi05_recipe import export_parts, parse_args
from torch import nn


def compile_part(
    module: nn.Module, inputs: tuple, input_names: list[str], output_names: list[str], path: Path
) -> None:
    with torch.no_grad():
        program = torch.export.export(module, inputs)
    # Compile the whole program into one CUDA delegate, as ExecuTorch's CUDA example does.
    lowered = to_edge_transform_and_lower(
        program,
        partitioner=[CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    executorch_program = lowered.to_executorch(
        config=ExecutorchBackendConfig(
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
            # Each program reads and writes CUDA tensors the rollout owns, so the chain never copies to the host.
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
        )
    )
    # Every program's weights file has the same name, so each program gets its own folder.
    path.parent.mkdir()
    executorch_program.save(str(path))
    executorch_program.write_tensor_data_to_file(str(path.parent))


if __name__ == "__main__":
    export_parts(
        parse_args(__doc__, "executorch_cuda"),
        __file__,
        "{name}/{name}.pte",
        compile_part,
        device_resident=True,
    )
