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

"""Export a trained ACT policy as an ExecuTorch program that runs on TensorRT.

Run it on the device the policy will run on, because a TensorRT engine only runs on the GPU
model that built it:

    python examples/export/act_executorch_tensorrt.py \
        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`.
"""

import torch
import torch_tensorrt
from act_recipe import ACTExport, parse_args, save_for_build_engine
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass


def main() -> None:
    args = parse_args(__doc__, "executorch_tensorrt")
    export = ACTExport(args)
    pte_path = export.output_dir / "model.pte"

    with torch.no_grad():
        program = torch.export.export(export.module, export.inputs)
    if args.export_only:
        save_for_build_engine(export, program, pte_path)
        return
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=export.inputs, min_block_size=1)
    del program
    export.release_policy()
    torch_tensorrt.save(
        engine,
        str(pte_path),
        output_format="executorch",
        retrace=False,
        use_cuda_graphs=True,
        backend_config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
        ),
    )
    export.write(pte_path)


if __name__ == "__main__":
    main()
