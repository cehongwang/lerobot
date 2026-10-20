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

"""Rebuild the TensorRT engines of an ONNX-TensorRT folder for this board's GPU.

A TensorRT engine only runs on the GPU model that built it. A folder `onnx_trt.py` wrote keeps its
ONNX files, so copy it to another board and rebuild its engines there:

    python examples/export/build_engine.py outputs/export/act_onnx_trt

Then run the folder with `lerobot-rollout --policy.path=<folder>`.
"""

import argparse
import json
import time
from pathlib import Path

from lerobot.inference.export import ExportInferenceEngine


def build_engine(onnx_path: Path, engine_path: Path, workspace_bytes: int, optimization_level: int) -> None:
    """Build a TensorRT engine that keeps the ONNX file's own dtypes, as the other routes do."""
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.builder_optimization_level = optimization_level
    start = time.perf_counter()
    with engine_path.open("wb") as file:
        if not builder.build_serialized_network_to_stream(network, config, FileWriter(file)):
            raise SystemExit(f"TensorRT could not build {engine_path.name}.")
    print(f"Built {engine_path.name} in {time.perf_counter() - start:.0f} s")


def main() -> None:
    from export import TENSORRT_OPTIONS

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path, help="A folder onnx_trt.py wrote.")
    args = parser.parse_args()
    info = json.loads((args.folder / "export.json").read_text())
    if info["backend"] != "onnx_trt":
        raise SystemExit("Only an ONNX-TensorRT folder keeps the ONNX files to rebuild its engines from.")
    for program in info["programs"].values():
        engine = args.folder / program["file"]
        build_engine(
            engine.with_suffix(".onnx"),
            engine,
            TENSORRT_OPTIONS["workspace_size"],
            TENSORRT_OPTIONS["optimization_level"],
        )
    # The rollout's own startup check: the engines must reproduce the test case.
    ExportInferenceEngine(args.folder, task=info.get("task", ""), robot_type="")
    print(f"{args.folder} reproduces its test case")


if __name__ == "__main__":
    main()
