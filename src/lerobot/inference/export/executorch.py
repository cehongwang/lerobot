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

"""Run an exported policy's ExecuTorch program, on its TensorRT or CUDA backend."""

from pathlib import Path
from typing import Any

import numpy as np


class ExecuTorchProgram:
    """Run a `.pte` program with host arrays and read its outputs through DLPack."""

    def __init__(self, path: Path, tensorrt: bool = True, shared_scratch: bool = False):
        if tensorrt:
            import torch_tensorrt_executorch_runtime  # noqa: F401
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        # Engines read this process-wide option when they load; older runtimes keep one scratch per engine.
        if tensorrt and hasattr(runtime.backend_registry, "set_option"):
            runtime.backend_registry.set_option(
                "TensorRTBackend", {"use_shared_activation_scratch": shared_scratch}
            )
        data_path = next(path.parent.glob("*.ptd"), None)
        self._method = runtime.load_program(path, data_path=data_path).load_method("forward")
        self._torch_inputs = False

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self._method.metadata.input_tensor_meta(0).sizes())

    def initialize(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Select the input type while running the saved test case at load time."""
        try:
            return self(*inputs)
        except (TypeError, RuntimeError) as error:
            if "Unsupported python type <class 'numpy.ndarray'>" not in str(error):
                raise
        self._torch_inputs = True
        return self(*inputs)

    def __call__(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        arrays = [np.ascontiguousarray(value) for value in inputs]
        if self._torch_inputs:
            arrays = [to_torch(value) for value in arrays]
        outputs = self._method.execute(arrays)
        result = []
        for output in outputs:
            if output.__dlpack_device__()[0] == 1:
                try:
                    result.append(np.from_dlpack(output).copy())
                    continue
                except (BufferError, RuntimeError):
                    # NumPy cannot consume every DLPack dtype, including bfloat16.
                    pass
            import torch

            result.append(to_numpy(torch.from_dlpack(output)))
        return result[0] if len(result) == 1 else tuple(result)


class DeviceResidentProgram(ExecuTorchProgram):
    """Run a `.pte` program exported to take and return CUDA tensors, with no copies at its boundary.

    `run_device` returns the outputs as CUDA torch tensors, which the next program in a chain reads as they are.
    """

    def run_device(self, *inputs: np.ndarray | Any) -> tuple[Any, ...]:
        import torch

        tensors = [
            value if isinstance(value, torch.Tensor) else to_torch(np.ascontiguousarray(value)).cuda()
            for value in inputs
        ]
        # The delegate runs on its own CUDA stream, so the PyTorch work that wrote the inputs must finish first.
        torch.cuda.current_stream().synchronize()
        # ExecuTorch can return DLPack objects instead of tensors; from_dlpack shares their GPU memory.
        return tuple(
            output if isinstance(output, torch.Tensor) else torch.from_dlpack(output)
            for output in self._method.execute(tensors)
        )

    def __call__(self, *inputs: np.ndarray | Any) -> np.ndarray | tuple[np.ndarray, ...]:
        outputs = [to_numpy(value) for value in self.run_device(*inputs)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def to_torch(array: np.ndarray):
    """A torch tensor sharing the array's memory; PyTorch cannot read NumPy's bfloat16 extension dtype."""
    import torch

    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def to_numpy(tensor) -> np.ndarray:
    """A host copy of the tensor; NumPy reads bfloat16 through the ml_dtypes extension dtype."""
    import torch

    tensor = tensor.cpu()
    if tensor.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return tensor.view(torch.uint16).numpy().view(bfloat16).copy()
    return tensor.numpy().copy()
