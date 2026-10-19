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

"""Build the TensorRT engine of a folder exported with `--export_only`, for this device's GPU.

A TensorRT engine only runs on the GPU model that built it. So export once with `--export_only`, copy
the folder to each device, and build its engine there:

    python examples/export/build_engine.py outputs/export/act_onnx_tensorrt

Then run the folder with `lerobot-rollout --policy.path=<folder>`.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from act_recipe import EXPORTED_PROGRAM

from lerobot.inference.export import ExportInferenceEngine

GIB = 1 << 30


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gib: float | None = None,
    tactic_gib: float | None = None,
    optimization_level: int | None = None,
) -> None:
    """Build a TensorRT engine that keeps the ONNX file's own dtypes, as the ExecuTorch route does."""
    import tensorrt as trt

    class FileWriter(trt.IStreamWriter):
        """Writes the engine to a file as TensorRT serializes it, without another copy in memory."""

        def __init__(self, file):
            trt.IStreamWriter.__init__(self)
            self._file = file

        def write(self, data: bytes) -> int:
            return self._file.write(data)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if workspace_gib is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * GIB))
    if tactic_gib is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM, int(tactic_gib * GIB))
    if optimization_level is not None:
        config.builder_optimization_level = optimization_level
    start = time.perf_counter()
    with engine_path.open("wb") as file:
        if not builder.build_serialized_network_to_stream(network, config, FileWriter(file)):
            raise SystemExit("TensorRT could not build the engine.")
    print(f"Built {engine_path} in {time.perf_counter() - start:.0f} s")


def build_executorch(folder: Path, program_file: str, args: argparse.Namespace) -> None:
    import torch_tensorrt
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    program = torch.export.load(folder / EXPORTED_PROGRAM)  # nosec B614: a folder the user exported
    inputs = program.example_inputs[0]
    options = {}
    if args.workspace_gib is not None:
        options["workspace_size"] = int(args.workspace_gib * GIB)
    if args.optimization_level is not None:
        options["optimization_level"] = args.optimization_level
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=inputs, min_block_size=1, **options)
    del program
    gc.collect()
    torch.cuda.empty_cache()
    torch_tensorrt.save(
        engine,
        str(folder / program_file),
        output_format="executorch",
        retrace=False,
        backend_config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
        ),
    )
    # The startup check loads the engine again; a compiled GraphModule is only freed by the cycle collector.
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path, help="A folder an export script wrote with --export_only.")
    parser.add_argument("--workspace_gib", type=float, help="Most scratch memory a TensorRT layer may use.")
    parser.add_argument(
        "--tactic_gib", type=float, help="Most memory TensorRT may use to time its kernels (ONNX only)."
    )
    parser.add_argument(
        "--optimization_level", type=int, choices=range(6), help="Lower builds faster, with less memory."
    )
    args = parser.parse_args()
    info = json.loads((args.folder / "export.json").read_text())

    if info["backend"] == "onnx_tensorrt":
        build_engine(
            args.folder / "model.onnx",
            args.folder / info["file"],
            args.workspace_gib,
            args.tactic_gib,
            args.optimization_level,
        )
    else:
        start = time.perf_counter()
        build_executorch(args.folder, info["file"], args)
        print(f"Built {args.folder / info['file']} in {time.perf_counter() - start:.0f} s")

    # The rollout's own startup check: the engine must reproduce the PyTorch test case.
    ExportInferenceEngine(args.folder, task=info.get("task", ""), robot_type="")
    print(f"{args.folder} reproduces its test case")


if __name__ == "__main__":
    main()
