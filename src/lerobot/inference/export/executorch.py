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

"""Run an exported policy's ExecuTorch programs on TensorRT, without PyTorch."""

import os
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.import_utils import require_package

# ExecuTorch returns its own results rather than torch tensors, even where torch is installed, so running
# a program never imports torch. The bindings read this once, when they first load, and the code below
# only reads their own results, so a shell setting of "torch" is overridden.
os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"

# DLPack's device type for CUDA memory.
DLPACK_CUDA = 2


class ExecuTorchProgram:
    """Run a `.pte` program whose inputs and outputs live on the GPU.

    `run_device` keeps its outputs on the GPU, so the next program in a chain reads them where they are.
    Calling the program takes and returns host arrays.
    """

    def __init__(self, path: Path):
        try:
            import torch_tensorrt_executorch_runtime  # noqa: F401  # registers the TensorRT backend

            # ExecuTorchResult arrived with the API this module uses, so an older ExecuTorch fails here.
            from executorch.runtime import ExecuTorchResult, Runtime  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "An ExecuTorch-TensorRT export needs ExecuTorch 1.6.0.dev20261020 or newer and "
                f"torch-tensorrt-executorch-runtime built for it; see examples/export/README.md. ({error})"
            ) from error

        runtime = Runtime.get()
        # The programs of a chain run one after another, so they can share one activation scratch.
        runtime.backend_registry.set_option("TensorRTBackend", {"use_shared_activation_scratch": True})
        self._method = runtime.load_program(path).load_method("forward")

    def run_device(self, *inputs: np.ndarray | Any) -> tuple[Any, ...]:
        return tuple(self._method.execute([on_device(value) for value in inputs]))

    def __call__(self, *inputs: np.ndarray | Any) -> np.ndarray | tuple[np.ndarray, ...]:
        outputs = [to_host(value) for value in self.run_device(*inputs)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def upload(self, value: np.ndarray) -> Any:
        return on_device(value)


def on_device(value: np.ndarray | Any) -> Any:
    """`value` in GPU memory: a result already there passes straight through, anything else is copied.

    The bindings hand a CUDA program whatever pointer an input has, so a host input would be read as
    GPU memory.
    """
    from executorch.runtime import ExecuTorchResult

    if not isinstance(value, ExecuTorchResult):
        return ExecuTorchResult.from_host(np.ascontiguousarray(value))
    return value if value.__dlpack_device__()[0] == DLPACK_CUDA else ExecuTorchResult.from_host(value)


def to_host(result: Any) -> np.ndarray:
    """A host copy of a program's result, in the result's own layout, read without copying it again.

    NumPy has no bfloat16, so the bindings export a bfloat16 result's raw bytes in storage order, and
    ml_dtypes reads them with the result's strides.
    """
    host = np.asarray(result.to_host())
    if result.dtype_name == "bfloat16":
        require_package("ml-dtypes", extra="export", import_name="ml_dtypes")
        from ml_dtypes import bfloat16

        return np.lib.stride_tricks.as_strided(host.view(bfloat16), result.shape, result.strides)
    return host
