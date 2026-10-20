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
"""Modules that must import without torch.

Each module is imported in a fresh interpreter where torch is installed but refuses to load, so a
failure's traceback points at the import that pulled torch in.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.utils.import_utils import _datasets_available

# The spec is still found, so code that only checks whether torch is installed keeps working. Each
# load attempt is recorded, so an import wrapped in try/except still fails the test.
BLOCK_TORCH = """
import importlib
import importlib.abc
import importlib.machinery
import sys
import traceback

attempts = []


class RefuseLoader(importlib.abc.Loader):
    def exec_module(self, module):
        attempts.append("".join(traceback.format_stack()))
        raise RuntimeError("torch must not be imported here")


class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        spec.loader = RefuseLoader()
        return spec


sys.meta_path.insert(0, BlockTorch())
"""

IMPORT_WITHOUT_TORCH = (
    BLOCK_TORCH
    + """
importlib.import_module(sys.argv[1])
if attempts:
    sys.exit(attempts[0])
"""
)


@pytest.mark.parametrize(
    "module",
    [
        "lerobot.scripts.lerobot_calibrate",
        "lerobot.scripts.lerobot_find_cameras",
        "lerobot.scripts.lerobot_find_joint_limits",
        "lerobot.scripts.lerobot_find_port",
        "lerobot.scripts.lerobot_setup_can",
        "lerobot.scripts.lerobot_setup_motors",
        "lerobot.scripts.lerobot_teleoperate",
        "lerobot.configs",
        "lerobot.policies",
        "lerobot.inference",
        "lerobot.processor",
        "lerobot.utils.action_interpolator",
        "lerobot.inference.export.tensorrt",
        pytest.param(
            "lerobot.scripts.lerobot_rollout",
            marks=pytest.mark.skipif(not _datasets_available, reason="datasets not installed"),
        ),
    ],
)
def test_imports_without_torch(module):
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_WITHOUT_TORCH, module], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


# Stand-ins for the compiled programs and both torch-free runtimes, imported from disk the way the real
# packages are. The chain has the shape of a flow-matching policy: a prefix, a denoising step run once per
# timestep, and the actions, with fresh noise each chunk and, optionally, a camera the engine resizes.
EXPORT_FAKES = """
import ctypes
import json
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

STATE = np.array([1, 2, 3], dtype=np.float32)
TENSORS = {
    "observation.state": ((3,), np.float32),
    "observation.images.camera": ((4, 4, 3), np.uint8),
    "hidden": ((3,), np.float32),
    "x_t": ((2, 3), np.float32),
    "timestep": ((1,), np.float32),
    "x_next": ((2, 3), np.float32),
    "action": ((2, 3), np.float32),
}
# The two denoising steps leave the noise behind: x = state + the last timestep.
ACTIONS = np.stack([STATE + 0.5, 2 * (STATE + 0.5)])


def run(name, values):
    if name == "prefix":
        if "observation.images.camera" in values:
            image = values["observation.images.camera"]
            assert image.shape == (4, 4, 3) and image.dtype == np.uint8, (image.shape, image.dtype)
        return {"hidden": values["observation.state"].copy()}
    if name == "step":
        return {"x_next": values["x_t"] * 0 + values["hidden"] + values["timestep"]}
    return {"action": values["x_t"] * np.array([[1], [2]], np.float32)}


def write_folder(folder, backend, camera):
    frame = {"observation.state": STATE}
    if camera:
        frame["observation.images.camera"] = np.full((6, 8, 3), 255, dtype=np.uint8)
    names = list(frame)
    programs = {
        "prefix": {"file": "prefix.bin", "inputs": names, "outputs": ["hidden"]},
        "step": {"file": "step.bin", "inputs": ["hidden", "x_t", "timestep"], "outputs": ["x_next"]},
        "actions": {"file": "actions.bin", "inputs": ["x_t"], "outputs": ["action"]},
    }
    info = {
        "backend": backend,
        "programs": programs,
        "inputs": [*names, "noise"],
        "noise_shape": [2, 3],
        "steps": ["step"],
        "timesteps": [1.0, 0.5],
        "test_case": "case.safetensors",
        "tolerance": 0.0,
    }
    if camera:
        info["image_resize"] = {
            "cameras": ["observation.images.camera"],
            "target_size": None,
            "resize_edge": None,
            "crop_fraction": None,
            "letterbox": False,
            "image_size": [4, 4],
        }
    (folder / "export.json").write_text(json.dumps(info))
    features = {"observation.state": {"type": "STATE", "shape": [3]}}
    if camera:
        features["observation.images.camera"] = {"type": "VISUAL", "shape": [3, 6, 8]}
    (folder / "config.json").write_text(json.dumps({
        "type": "act",
        "input_features": features,
        "output_features": {"action": {"type": "ACTION", "shape": [3]}},
        "action_feature_names": ["motor_1.pos", "motor_2.pos", "motor_3.pos"],
    }))
    case = {**frame, "noise": np.zeros((2, 3), np.float32), "expected_actions": ACTIONS}
    save_file(case, folder / "case.safetensors")
    return frame


def install_tensorrt():
    # CUDA and TensorRT on the CPU: memory is host memory, and copies and engines run when queued.
    from lerobot.inference.export import tensorrt

    memory = []
    captures = []

    def malloc(size):
        memory.append(ctypes.create_string_buffer(size))
        return 0, ctypes.addressof(memory[-1])

    def memcpy(dst, src, size, kind, stream):
        ctypes.memmove(dst, src, size)
        return (0,)

    def array(name, address):
        shape, dtype = TENSORS[name]
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return np.frombuffer((ctypes.c_char * size).from_address(address), dtype).reshape(shape)

    def launch(graph, stream):
        for work in graph:
            work()
        return (0,)

    class Context:
        def __init__(self, name):
            self.name, self.addresses = name, {}

        def set_tensor_address(self, name, address):
            self.addresses[name] = address
            return True

        def set_device_memory(self, address, size):
            pass

        def execute(self):
            values = {name: array(name, address) for name, address in self.addresses.items()}
            for name, value in run(self.name, values).items():
                array(name, self.addresses[name])[:] = value

        def execute_async_v3(self, stream):
            if captures:
                captures[-1].append(self.execute)
            else:
                self.execute()
            return True

    def load_engine(path):
        return SimpleNamespace(
            create_execution_context=lambda strategy: Context(path.stem),
            get_tensor_shape=lambda name: TENSORS[name][0],
            get_tensor_dtype=lambda name: TENSORS[name][1],
            device_memory_size_v2=0,
        )

    cuda = SimpleNamespace(
        cudaMalloc=malloc,
        cudaFree=lambda address: (0,),
        cudaMemcpyAsync=memcpy,
        cudaMemcpyKind=SimpleNamespace(cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2, cudaMemcpyDeviceToDevice=3),
        cudaStreamCreate=lambda: (0, 1),
        cudaStreamSynchronize=lambda stream: (0,),
        cudaStreamDestroy=lambda stream: (0,),
        cudaStreamCaptureMode=SimpleNamespace(cudaStreamCaptureModeGlobal=0),
        cudaStreamBeginCapture=lambda stream, mode: captures.append([]) or (0,),
        cudaStreamEndCapture=lambda stream: (0, captures.pop()),
        cudaGraphInstantiate=lambda graph, flags: (0, graph),
        cudaGraphDestroy=lambda graph: (0,),
        cudaGraphLaunch=launch,
        cudaGraphExecDestroy=lambda graph: (0,),
    )
    import sys

    sys.modules["tensorrt"] = SimpleNamespace(
        nptype=lambda dtype: dtype,
        bfloat16="bfloat16",
        ExecutionContextAllocationStrategy=SimpleNamespace(STATIC=0, USER_MANAGED=1),
    )
    tensorrt.cuda_runtime = lambda: cuda
    tensorrt.load_engine = load_engine
"""

# ExecuTorch on the CPU. It records the output type the bindings would have chosen when they loaded, and
# returns the actions in bfloat16, which NumPy can only read through ml_dtypes.
FAKE_EXECUTORCH_RUNTIME = """
import json
import os

import numpy as np
from ml_dtypes import bfloat16

from export_fakes import run

OUTPUT_AT_IMPORT = os.environ.get("EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT")


class ExecuTorchResult:
    def __init__(self, array):
        self.array, self.shape, self.strides = array, array.shape, array.strides
        self.dtype_name = array.dtype.name

    @classmethod
    def from_host(cls, array, device_index=0):
        return cls(np.array(array))

    def to_host(self):
        copy = np.array(self.array)
        # The bindings export a bfloat16 result's raw bytes, since the buffer protocol has no bfloat16.
        return copy.view(np.uint8) if self.dtype_name == "bfloat16" else copy

    def __dlpack_device__(self):
        return (2, 0)


class Method:
    def __init__(self, path):
        self.name = path.stem
        program = json.loads((path.parent / "export.json").read_text())["programs"][self.name]
        self.inputs = program["inputs"]

    def execute(self, inputs):
        assert all(isinstance(value, ExecuTorchResult) for value in inputs), inputs
        outputs = run(self.name, {name: value.array for name, value in zip(self.inputs, inputs)})
        if self.name == "actions":
            outputs = {name: value.astype(bfloat16) for name, value in outputs.items()}
        return [ExecuTorchResult(value) for value in outputs.values()]


class Runtime:
    backend_registry = type("Registry", (), {"set_option": lambda self, backend, options: None})()

    @staticmethod
    def get():
        return Runtime()

    def load_program(self, path):
        return type("Program", (), {"load_method": lambda self, name: Method(path)})()
"""


def write_fakes(tmp_path: Path) -> Path:
    fake = tmp_path / "fake"
    (fake / "executorch").mkdir(parents=True)
    (fake / "executorch" / "__init__.py").write_text("")
    (fake / "executorch" / "runtime.py").write_text(FAKE_EXECUTORCH_RUNTIME)
    # Like the real package, the TensorRT backend loads ExecuTorch's bindings when it is imported.
    (fake / "torch_tensorrt_executorch_runtime.py").write_text("import executorch.runtime\n")
    (fake / "export_fakes.py").write_text(EXPORT_FAKES)
    return fake


def run_without_torch(code: str, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(write_fakes(tmp_path)), os.environ.get("PYTHONPATH", "")]),
        # The engine needs ExecuTorch's own results, whatever the shell asks for.
        "EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT": "torch",
    }
    folder = tmp_path / "export"
    folder.mkdir()
    return subprocess.run(
        [sys.executable, "-c", code, str(folder), *args], capture_output=True, text=True, timeout=300, env=env
    )


# Runs the export engine on a chain with noise, denoising steps and a resized camera.
ENGINE_WITHOUT_TORCH = (
    BLOCK_TORCH
    + """
from pathlib import Path

import numpy as np

import export_fakes

folder, backend = Path(sys.argv[1]), sys.argv[2]
frame = export_fakes.write_folder(folder, backend, camera=True)
if backend == "onnx_trt":
    export_fakes.install_tensorrt()

from lerobot.inference.export import ExportInferenceEngine

engine = ExportInferenceEngine(folder, task="")
np.testing.assert_array_equal(engine.get_action(frame), export_fakes.ACTIONS[0])
np.testing.assert_array_equal(engine.get_action(frame), export_fakes.ACTIONS[1])
if attempts:
    sys.exit(attempts[0])
assert "torch" not in sys.modules
"""
)


@pytest.mark.parametrize("backend", ["executorch_trt", "onnx_trt"])
def test_export_engine_runs_a_denoising_chain_without_torch(tmp_path, backend):
    """With torch installed but refusing to load, the engine runs a flow-matching chain with a camera."""
    result = run_without_torch(ENGINE_WITHOUT_TORCH, tmp_path, backend)
    assert result.returncode == 0, result.stderr[-4000:]


# Runs `lerobot-rollout` itself on a mock robot, which has no camera.
ROLLOUT_WITHOUT_TORCH = (
    BLOCK_TORCH
    + """
from pathlib import Path
from types import SimpleNamespace

import export_fakes

folder, backend, planner = Path(sys.argv[1]), sys.argv[2], sys.argv[4] == "planner"
sys.path.insert(0, sys.argv[3])
export_fakes.write_folder(folder, backend, camera=False)
if backend == "onnx_trt":
    export_fakes.install_tensorrt()

from tests.mocks.mock_robot import MockRobot

sent = []
send_action = MockRobot.send_action
MockRobot.send_action = lambda self, action: sent.append(dict(action)) or send_action(self, action)

from lerobot.scripts import lerobot_rollout

sys.argv = [
    "lerobot-rollout", f"--policy.path={folder}", "--robot.type=mock_robot", "--robot.random_values=false",
    "--robot.static_values=[1,2,3]", "--strategy.type=base", "--duration=0.3", "--fps=20",
]
if planner:
    from lerobot.rollout import planner as planner_module

    # No VLM answers here; the rollout only sets the planner up, which reads the checkpoint's vocabulary.
    planner_module.make_vlm_client = lambda config: SimpleNamespace()
    sys.argv.append("--planner.model_id=planner")
lerobot_rollout.main()

if attempts:
    sys.exit(attempts[0])
assert "torch" not in sys.modules
# The first actions the robot gets are the chunk the programs computed from its state, one per tick.
names = ["motor_1.pos", "motor_2.pos", "motor_3.pos"]
assert sent[:2] == [dict(zip(names, row.tolist())) for row in export_fakes.ACTIONS], sent[:2]
if backend == "executorch_trt":
    import executorch.runtime

    assert executorch.runtime.OUTPUT_AT_IMPORT == "executorch", executorch.runtime.OUTPUT_AT_IMPORT
"""
)


@pytest.mark.skipif(not _datasets_available, reason="datasets not installed")
@pytest.mark.parametrize("planner", ["", "planner"], ids=["no_planner", "planner"])
@pytest.mark.parametrize("backend", ["executorch_trt", "onnx_trt"])
def test_rollout_runs_an_export_with_torch_unimportable(tmp_path, backend, planner):
    """`lerobot-rollout` loads an ExecuTorch-TensorRT or ONNX-TensorRT folder and drives a robot without torch.

    With a planner, the rollout also looks for the vocabulary of the dataset the checkpoint was trained on.
    """
    repo = Path(__file__).resolve().parents[1]
    result = run_without_torch(ROLLOUT_WITHOUT_TORCH, tmp_path, backend, str(repo), planner)
    assert result.returncode == 0, result.stderr[-4000:]
