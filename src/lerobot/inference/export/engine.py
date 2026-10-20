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

An exported folder holds the compiled programs, the policy's `config.json`, and an `export.json` that
lists the programs with their inputs and outputs, and a test case. A whole policy is one program; a
policy too large to compile at once is a chain. The engine feeds the chain the robot's frame (and, for a
flow-matching policy, fresh noise each chunk) and returns actions in robot units. A folder compiled for
one task (`task_fixed`) refuses any other task.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from safetensors.numpy import load_file

from ..base import InferenceEngine

logger = logging.getLogger(__name__)

EXPORT_INFO = "export.json"
NOISE = "noise"
BACKENDS = ("executorch_trt", "onnx_trt", "torch_trt")


class Program(Protocol):
    def run_device(self, *inputs: Any) -> tuple[Any, ...]:
        """The outputs where the program left them, for the next program to read without a copy."""

    def __call__(self, *inputs: Any) -> np.ndarray | tuple[np.ndarray, ...]:
        """The outputs as host arrays."""


def export_dir(path: str | Path | None) -> Path | None:
    """The folder an export script wrote, or None when `path` holds a regular policy checkpoint."""
    if path is not None and (Path(path) / EXPORT_INFO).is_file():
        return Path(path)
    return None


def load_programs(folder: Path, info: dict) -> Mapping[str, Program]:
    """Load the folder's programs with the backend `export.json` names."""
    backend = info["backend"]
    programs = info["programs"]
    if backend == "onnx_trt":
        from .tensorrt import load_engines

        return load_engines(folder, programs, info.get("cuda_graphs", False))
    if backend == "executorch_trt":
        from .executorch import ExecuTorchProgram

        # The programs run one after another, so they can share one activation scratch.
        return {
            name: ExecuTorchProgram(folder / p["file"], shared_scratch=True) for name, p in programs.items()
        }
    if backend == "torch_trt":
        from .aoti import AOTInductorPackage

        return {name: AOTInductorPackage(folder / p["file"]) for name, p in programs.items()}
    raise ValueError(
        f"Unknown export backend {backend!r} in {folder / EXPORT_INFO}; expected one of {BACKENDS}."
    )


class ProgramChain:
    """Run the programs `export.json` lists, in order, each reading earlier values by name.

    The frame and the noise are the first values. The programs named in `steps` run once per denoising
    step, from the noisy actions `x_t` and the `timestep`. The last returns the next noisy actions, `x_next`,
    which become `x_t`, so no arithmetic runs between programs.
    Values stay where the programs left them until the last program returns the actions to the host.
    """

    def __init__(self, programs: Mapping[str, Program], info: dict):
        self._programs = programs
        self._names: list[str] = info["inputs"]
        self._inputs = {name: program["inputs"] for name, program in info["programs"].items()}
        self._outputs = {name: program["outputs"] for name, program in info["programs"].items()}
        self._steps: list[str] = info.get("steps", [])
        self._timesteps = [np.array([t], dtype=np.float32) for t in info.get("timesteps", [])]

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        values: dict[str, Any] = dict(zip(self._names, inputs, strict=True))
        *names, last = self._programs
        for name in names:
            if name not in self._steps:
                self._run(name, values)
            elif name == self._steps[0]:
                values["x_t"] = values[NOISE]
                for timestep in self._timesteps:
                    values["timestep"] = timestep
                    for step in self._steps:
                        self._run(step, values)
                    values["x_t"] = values["x_next"]
        return self._programs[last](*(values[n] for n in self._inputs[last]))

    def _run(self, name: str, values: dict[str, Any]) -> None:
        outputs = self._programs[name].run_device(*(values[n] for n in self._inputs[name]))
        values.update(zip(self._outputs[name], outputs, strict=True))


class ExportInferenceEngine(InferenceEngine):
    """Inline inference with an exported policy: one run of its programs per action chunk.

    Plays the chunk one action per call, as `SyncInferenceEngine` does with the PyTorch policy's action
    queue. Before running, it replays the folder's test case and refuses to start if the actions differ by
    more than the folder's tolerance from the test case's expected actions, which the export computed in
    PyTorch.
    """

    def __init__(
        self, folder: Path, task: str, robot_type: str, rename_map: dict[str, str] | None = None
    ) -> None:
        super().__init__(task=task)
        info = json.loads((folder / EXPORT_INFO).read_text())
        self._frame_names = [name for name in info["inputs"] if name != NOISE]
        self._noise_shape = tuple(info["noise_shape"]) if "noise_shape" in info else None
        self._image_resize = info.get("image_resize")
        self._rename = rename_map or {}
        self._chain = ProgramChain(load_programs(folder, info), info)
        self._actions: deque[np.ndarray] = deque()
        self._fixed_task: str | None = info["task"] if info.get("task_fixed") else None
        self._check_task(task)
        self._check(folder / info["test_case"], info["tolerance"])
        logger.info("Exported policy loaded from %s (%s)", folder, info["backend"])

    def _run_chunk(self, frame: dict[str, np.ndarray], noise: np.ndarray | None = None) -> np.ndarray:
        if self._image_resize is not None:
            from .images import resize_images

            frame = resize_images(frame, self._image_resize)
        inputs = [np.ascontiguousarray(frame[name]) for name in self._frame_names]
        if self._noise_shape is not None:
            inputs.append(
                noise
                if noise is not None
                else np.random.standard_normal(self._noise_shape).astype(np.float32)
            )
        return self._chain(*inputs)

    def _task_error(self, task: str) -> str | None:
        if self._fixed_task is not None and task != self._fixed_task:
            return (
                f"This exported policy only runs the task it was exported for, {self._fixed_task!r}, "
                f"not {task!r}. Export it again with the new task."
            )
        return None

    def _check_task(self, task: str) -> None:
        if error := self._task_error(task):
            raise ValueError(error)

    def _check(self, path: Path, tolerance: float) -> None:
        case = load_file(path)
        expected = case.pop("expected_actions")
        actual = self._run_chunk(case, case.pop(NOISE, None))
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

    def text_input_error(self, text: str, *, instruction: bool = False) -> str | None:
        if instruction and (error := self._task_error(text)):
            return error
        return super().text_input_error(text, instruction=instruction)

    def dispatch_allowed(self) -> bool:
        # Inline inference has no worker to pause, so the planner hold is applied here.
        return not self.hold_for_planner()

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
            self._actions.clear()
        if not self._actions:
            start = time.perf_counter()
            frame = {self._rename.get(name, name): value for name, value in obs_frame.items()}
            self._actions.extend(self._run_chunk(frame))
            end = time.perf_counter()
            self.inference_times.append((end, end - start))
        self._set_dispatched_task(task)
        return self._actions.popleft()
