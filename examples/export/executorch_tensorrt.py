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

"""Export a trained policy as ExecuTorch programs that run on TensorRT, with no PyTorch at run time.

Run it on the board that will run the policy, because a TensorRT engine only runs on the GPU model that
built it:

    python examples/export/executorch_tensorrt.py \
        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model

Then run the folder it writes with `lerobot-rollout --policy.path=<folder>`.
"""

from pathlib import Path

import torch
import torch_tensorrt
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
from export import TENSORRT_OPTIONS, Part, export, parse_args


def compile_part(part: Part, path: Path) -> None:
    with torch.no_grad():
        program = torch.export.export(part.module, part.inputs)
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=part.inputs, **TENSORRT_OPTIONS)
    del program
    torch_tensorrt.save(
        engine,
        str(path),
        output_format="executorch",
        arg_inputs=part.inputs,
        retrace=False,
        use_cuda_graphs=True,
        backend_config=ExecutorchBackendConfig(
            # The rollout hands each program its inputs, and takes its outputs, on the GPU.
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
        ),
    )


if __name__ == "__main__":
    args, recipe = parse_args(__doc__, "executorch_tensorrt")
    export(__file__, args, recipe, compile_part, ".pte")
