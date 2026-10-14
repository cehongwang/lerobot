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

"""The inference engine for an exported policy.

An exported folder holds the compiled program, the policy's `config.json`, and an `export.json` naming the
backend, the program's inputs and a test case. The engine feeds the program the robot's frame (and, for a
flow-matching policy, fresh noise each chunk) and returns actions in robot units. A folder compiled for one
task (`task_fixed`) refuses any other task. A policy too large for one program is exported as a chain that
`ProgramChain` runs in order.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from ..base import InferenceEngine

logger = logging.getLogger(__name__)

EXPORT_INFO = "export.json"
NOISE = "noise"

Program = Callable[..., np.ndarray]


def export_dir(path: str | Path | None) -> Path | None:
    """The folder an export script wrote, or None when `path` holds a regular policy checkpoint."""
    if path is not None and (Path(path) / EXPORT_INFO).is_file():
        return Path(path)
    return None


def load_program(folder: Path, info: dict) -> Program:
    """Load the folder's program, or its chain of programs, with the backend `export.json` names."""
    backend = info["backend"]
    if backend not in ("executorch_tensorrt", "executorch_cuda", "onnx_tensorrt", "torch_tensorrt"):
        raise ValueError(f"Unknown export backend {backend!r} in {folder / EXPORT_INFO}")
    cuda_graphs = info.get("cuda_graphs", False)
    if "programs" in info:
        if backend == "onnx_tensorrt":
            from .tensorrt import load_engines

            return ProgramChain(load_engines(folder, info["programs"], cuda_graphs), info)
        files = {name: folder / program["file"] for name, program in info["programs"].items()}
        device_resident = info.get("device_resident", False)
        return ProgramChain(
            {
                name: load_one(path, backend, device_resident, shared_scratch=True)
                for name, path in files.items()
            },
            info,
        )
    if backend == "onnx_tensorrt":
        from .tensorrt import TensorRTEngine

        return TensorRTEngine(
            folder / info["file"], info["inputs"], [info["output"]], cuda_graphs=cuda_graphs
        )
    return load_one(folder / info["file"], backend)


def load_one(
    path: Path, backend: str, device_resident: bool = False, shared_scratch: bool = False
) -> Program:
    """Load one ExecuTorch program or AOTInductor package.

    The ExecuTorch-TensorRT programs of a chain share one activation scratch, since they run one after another.
    """
    if backend == "torch_tensorrt":
        from .aoti import AOTInductorPackage

        return AOTInductorPackage(path)
    from .executorch import DeviceResidentProgram, ExecuTorchProgram

    program = DeviceResidentProgram if device_resident else ExecuTorchProgram
    return program(path, tensorrt=backend == "executorch_tensorrt", shared_scratch=shared_scratch)


class ProgramChain:
    """Run a chunk exported as a chain of programs, in the order `export.json` lists them.

    Each program reads the frame, the noise, or the outputs of a program before it, by name. The `step`
    program (or `step_0`, `step_1`, ...) runs once per denoising step, from `timesteps` with step size `dt`.
    A program with `run_device` hands its outputs to the next one on the GPU.
    """

    def __init__(self, programs: Mapping[str, Any], info: dict):
        if info["num_steps"] <= 0:
            raise ValueError("num_steps must be positive")
        self._programs = programs
        self._device_resident = info.get("device_resident", False)
        self._names: list[str] = info["inputs"]
        self._inputs = {name: program["inputs"] for name, program in info["programs"].items()}
        self._outputs = {name: program["outputs"] for name, program in info["programs"].items()}
        self._steps = [name for name in programs if name == "step" or name.startswith("step_")]
        self._dt: float = info.get("dt", -1.0 / info["num_steps"])
        self._timesteps: list[float] = info.get(
            "timesteps", [1.0 + step * self._dt for step in range(info["num_steps"])]
        )

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        return getattr(next(iter(self._programs.values())), "input_shape", None)

    def initialize(self, *inputs: np.ndarray) -> np.ndarray:
        return self._run(inputs, initialize=True)

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        return self._run(inputs)

    def _run(self, inputs: tuple[np.ndarray, ...], initialize: bool = False) -> np.ndarray:
        values = dict(zip(self._names, inputs, strict=True))
        *names, last = self._programs
        for name in names:
            if name not in self._steps:
                self._run_program(name, values, initialize)
            elif name == self._steps[0]:
                self._denoise(values, initialize)
        return _call(self._programs[last], initialize)(*(values[n] for n in self._inputs[last]))

    def _denoise(self, values: dict, initialize: bool) -> None:
        *inner, velocity = self._steps
        sample_name, time_name = self._inputs[self._steps[0]][-2:]
        sample = values[NOISE]
        timesteps = np.asarray(self._timesteps, dtype=np.float32)
        if self._device_resident:
            import torch

            sample, timesteps = (
                torch.from_numpy(np.ascontiguousarray(value)).cuda() for value in (sample, timesteps)
            )
        for step, timestep in enumerate(timesteps):
            values[sample_name], values[time_name] = sample, timestep[None]
            for name in inner:
                self._run_program(name, values, initialize and step == 0)
            program = self._programs[velocity]
            run = program.run_device if self._device_resident else _call(program, initialize and step == 0)
            result = run(*(values[n] for n in self._inputs[velocity]))
            if self._device_resident:
                (result,) = result
            sample = sample + self._dt * result
        values[sample_name] = sample

    def _run_program(self, name: str, values: dict, initialize: bool) -> None:
        program = self._programs[name]
        run = getattr(program, "run_device", None) or _call(program, initialize)
        outputs = run(*(values[n] for n in self._inputs[name]))
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        values.update(zip(self._outputs[name], outputs, strict=True))


def _call(program: Program, initialize: bool) -> Program:
    """The program's `initialize`, which runs the startup check, or the program itself."""
    return getattr(program, "initialize", program) if initialize else program


class ExportInferenceEngine(InferenceEngine):
    """Inline inference with an exported program: one program call per action chunk.

    Plays the chunk one action per call, as `SyncInferenceEngine` does with the PyTorch policy's
    action queue. Before running, it replays the folder's test case and refuses to start if the
    program's actions differ from what `lerobot-rollout` played with PyTorch at export time.
    """

    def __init__(
        self, folder: Path, task: str, robot_type: str, rename_map: dict[str, str] | None = None
    ) -> None:
        super().__init__(task=task)
        info = json.loads((folder / EXPORT_INFO).read_text())
        self._input_names: list[str] = info["inputs"]
        self._robot_type = robot_type
        self._rename = rename_map or {}
        self._noise_shape: tuple[int, ...] | None = (
            tuple(info["noise_shape"]) if "noise_shape" in info else None
        )
        self._text_steps = None
        if "text_steps" in info:
            from lerobot.processor import PolicyProcessorPipeline

            self._text_steps = PolicyProcessorPipeline.from_pretrained(
                folder, config_filename=info["text_steps"]
            )
        self._program = load_program(folder, info)
        self._raw_frame: bool | None = info.get("raw_frame")
        self._image_resize = info.get("image_resize")
        self._actions: deque[np.ndarray] = deque()
        self._fixed_task: str | None = info["task"] if info.get("task_fixed") else None
        self._check_task(task)
        self._check(folder / info["test_case"], info.get("task", task), info["tolerance"])
        logger.info("Exported policy loaded from %s (%s)", folder, info["backend"])

    def _run_chunk(
        self,
        frame: dict[str, np.ndarray],
        task: str,
        noise: np.ndarray | None = None,
        initialize: bool = False,
    ) -> np.ndarray:
        if self._image_resize is not None:
            from .images import resize_images

            frame = resize_images(frame, self._image_resize)
        observation = {}
        names = [name for name in self._input_names if name != NOISE] if self._raw_frame else frame
        for name in names:
            value = frame[name]
            if not self._raw_frame:
                if "image" in name:
                    if value.dtype == np.uint8:
                        value = value.astype(np.float32) / 255
                    value = value.transpose(2, 0, 1)
                value = value[None]
            observation[name] = np.ascontiguousarray(value)
        if self._text_steps is not None:
            import torch

            tensors = {name: torch.from_numpy(value) for name, value in observation.items()}
            tensors.update(task=task or "", robot_type=self._robot_type or "")
            with torch.inference_mode():
                observation = self._text_steps(tensors)
            observation = {
                name: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                for name, value in observation.items()
            }
        if self._noise_shape is not None:
            observation[NOISE] = (
                noise
                if noise is not None
                else np.random.standard_normal(self._noise_shape).astype(np.float32)
            )
        actions = _call(self._program, initialize)(*(observation[name] for name in self._input_names))
        return (actions if self._raw_frame else actions[0]).copy()

    def _check_task(self, task: str) -> None:
        if self._fixed_task is not None and task != self._fixed_task:
            raise ValueError(
                f"This exported policy only runs the task it was exported for, {self._fixed_task!r}, "
                f"not {task!r}. Export it again with the new task."
            )

    def _check(self, path: Path, task: str, tolerance: float) -> None:
        case = load_file(path)
        expected = case.pop("expected_actions")
        input_shape = getattr(self._program, "input_shape", None)
        if self._raw_frame is None:
            self._raw_frame = (
                self._text_steps is None
                and input_shape is not None
                and input_shape == case[self._input_names[0]].shape
            )
        actual = self._run_chunk(case, task, case.pop(NOISE, None), initialize=True)
        error = float(np.abs(actual - expected).max()) if actual.shape == expected.shape else np.inf
        logger.info("Exported policy test case: largest difference %.2e (tolerance %g)", error, tolerance)
        if not error <= tolerance:
            raise RuntimeError(
                f"The exported policy does not reproduce its test case ({error:.2e} > {tolerance}). "
                "Export it again on this device."
            )

    @property
    def control_thread_owns_policy(self) -> bool:
        return True

    def start(self) -> None:
        """No background resources to start."""

    def stop(self) -> None:
        """No background resources to stop."""

    def reset(self) -> None:
        self._actions.clear()
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> np.ndarray | None:
        if obs_frame is None:
            return None
        task, task_changed = self._take_task()
        if task_changed:
            self._check_task(task)
            self._actions.clear()
        if not self._actions:
            start = time.perf_counter()
            frame = {self._rename.get(name, name): value for name, value in obs_frame.items()}
            self._actions.extend(self._run_chunk(frame, task))
            self.inference_seconds.append(time.perf_counter() - start)
        self._set_dispatched_task(task)
        return self._actions.popleft()
