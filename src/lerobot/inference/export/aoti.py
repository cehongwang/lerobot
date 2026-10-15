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

"""Run an exported policy's AOTInductor package, with its TensorRT engine inside."""

from pathlib import Path

import numpy as np
import torch


class AOTInductorPackage:
    """Runs a `.pt2` package on GPU tensors, with PyTorch and Torch-TensorRT installed.

    `run_device` keeps the outputs on the GPU, so the next package in a chain reads them without a copy.
    """

    def __init__(self, path: Path):
        import torch_tensorrt  # noqa: F401  # registers the TensorRT engine op the package calls

        self._model = torch._inductor.aoti_load_package(str(path))

    def run_device(self, *inputs: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            tensors = [torch.from_numpy(v).to("cuda") if isinstance(v, np.ndarray) else v for v in inputs]
            outputs = self._model(*tensors)
        return tuple(outputs) if isinstance(outputs, list | tuple) else (outputs,)

    def __call__(self, *inputs: np.ndarray | torch.Tensor) -> np.ndarray | tuple[np.ndarray, ...]:
        outputs = [value.cpu().numpy() for value in self.run_device(*inputs)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
