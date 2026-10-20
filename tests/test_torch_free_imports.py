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


def test_executorch_export_runs_without_importing_torch(tmp_path):
    """With torch installed, the rollout reads an ExecuTorch-TensorRT folder and runs it without importing torch."""
    code = r"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

from lerobot.configs import parser
from lerobot.inference.export import ExportInferenceEngine
from lerobot.robots.config import RobotConfig
from lerobot.rollout.configs import RolloutConfig

folder = Path(sys.argv[1])
info = {
    "backend": "executorch_trt",
    "programs": {
        "model": {
            "file": "model.pte",
            "inputs": ["observation.state", "observation.images.camera"],
            "outputs": ["action"],
        }
    },
    "inputs": ["observation.state", "observation.images.camera"],
    "test_case": "case.safetensors",
    "tolerance": 0.0,
}
(folder / "export.json").write_text(json.dumps(info))
(folder / "config.json").write_text(json.dumps({
    "type": "act", "input_features": {
        "observation.images.camera": {"type": "VISUAL", "shape": [3, 2, 2]}
    }, "action_feature_names": ["joint.pos"],
}))
state = np.array([2.0], dtype=np.float32)
image = np.full((2, 2, 3), 255, dtype=np.uint8)
save_file({"observation.state": state, "observation.images.camera": image,
           "expected_actions": np.array([[2.0], [4.0]], dtype=np.float32)},
          folder / "case.safetensors")


class Result:
    # Stands in for ExecuTorch's own result: device memory with a host copy on request.
    dtype_name = "float32"

    def __init__(self, array):
        self.array, self.shape = array, array.shape

    @classmethod
    def from_host(cls, array):
        return cls(np.array(array))

    def to_host(self):
        return self.array

    def __dlpack_device__(self):
        return (2, 0)


class Method:
    def execute(self, inputs):
        assert all(isinstance(value, Result) for value in inputs)
        state, image = (value.array for value in inputs)
        assert image.shape == (2, 2, 3) and image.dtype == np.uint8
        return [Result(np.stack([state, 2 * state]))]


runtime = SimpleNamespace(
    load_program=lambda path: SimpleNamespace(load_method=lambda name: Method()),
    backend_registry=SimpleNamespace(set_option=lambda backend, options: None),
)
sys.modules["executorch.runtime"] = SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime), ExecuTorchResult=Result)
sys.modules["torch_tensorrt_executorch_runtime"] = SimpleNamespace()

@RobotConfig.register_subclass("test_export_robot")
@dataclass
class ExportRobotConfig(RobotConfig):
    pass

@parser.wrap()
def parse_config(cfg: RolloutConfig):
    return cfg

sys.argv = ["lerobot-rollout", f"--policy.path={folder}", "--robot.type=test_export_robot"]
cfg = parse_config()
assert cfg.policy.type == "act" and cfg.policy.action_feature_names == ["joint.pos"]
engine = ExportInferenceEngine(folder, task="", robot_type="test_export_robot")
frame = {"observation.state": state, "observation.images.camera": image}
np.testing.assert_array_equal(engine.get_action(frame), [2.0])
np.testing.assert_array_equal(engine.get_action(frame), [4.0])
import os
assert os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] == "executorch"
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


# A stand-in for ExecuTorch, imported from disk the way the real package is, that runs the program on
# the CPU and records the output type the bindings would have chosen when they loaded.
FAKE_EXECUTORCH_RUNTIME = """
import os

import numpy as np

OUTPUT_AT_IMPORT = os.environ.get("EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT")


class ExecuTorchResult:
    dtype_name = "float32"

    def __init__(self, array):
        self.array, self.shape, self.strides = array, array.shape, array.strides

    @classmethod
    def from_host(cls, array, device_index=0):
        return cls(np.array(array))

    def to_host(self):
        return np.array(self.array)

    def __dlpack_device__(self):
        return (2, 0)


class Method:
    def execute(self, inputs):
        assert all(isinstance(value, ExecuTorchResult) for value in inputs), inputs
        (state,) = (value.array for value in inputs)
        return [ExecuTorchResult(np.stack([state, 2 * state]))]


class Runtime:
    backend_registry = type("Registry", (), {"set_option": lambda self, backend, options: None})()

    @staticmethod
    def get():
        return Runtime()

    def load_program(self, path):
        return type("Program", (), {"load_method": lambda self, name: Method()})()
"""

# Runs `lerobot-rollout` itself on a mock robot, with a CPU stand-in for the compiled program.
ROLLOUT_WITHOUT_TORCH = (
    BLOCK_TORCH
    + """
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

folder, backend = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, sys.argv[3])
state = np.array([1, 2, 3], dtype=np.float32)
save_file({"observation.state": state, "expected_actions": np.stack([state, 2 * state])}, folder / "case.safetensors")
(folder / "export.json").write_text(json.dumps({
    "backend": backend,
    "programs": {"model": {"file": "model.bin", "inputs": ["observation.state"], "outputs": ["action"]}},
    "inputs": ["observation.state"],
    "test_case": "case.safetensors",
    "tolerance": 0.0,
}))
(folder / "config.json").write_text(json.dumps({
    "type": "act",
    "input_features": {"observation.state": {"type": "STATE", "shape": [3]}},
    "output_features": {"action": {"type": "ACTION", "shape": [3]}},
    "action_feature_names": ["motor_1.pos", "motor_2.pos", "motor_3.pos"],
}))

if backend == "onnx_trt":
    from lerobot.inference.export import tensorrt

    memory = []

    def malloc(size):
        memory.append(ctypes.create_string_buffer(size))
        return 0, ctypes.addressof(memory[-1])

    def memcpy(dst, src, size, kind, stream):
        ctypes.memmove(dst, src, size)
        return (0,)

    def floats(address, count):
        return np.ctypeslib.as_array((ctypes.c_float * count).from_address(address))

    class Context:
        addresses = {}

        def set_tensor_address(self, name, address):
            self.addresses[name] = address
            return True

        def set_device_memory(self, address, size):
            pass

        def execute_async_v3(self, stream):
            x = floats(self.addresses["observation.state"], 3)
            floats(self.addresses["action"], 6)[:] = np.concatenate([x, 2 * x])
            return True

    cuda = SimpleNamespace(
        cudaMalloc=malloc,
        cudaFree=lambda address: (0,),
        cudaMemcpyAsync=memcpy,
        cudaMemcpyKind=SimpleNamespace(cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2),
        cudaStreamCreate=lambda: (0, 1),
        cudaStreamSynchronize=lambda stream: (0,),
        cudaStreamDestroy=lambda stream: (0,),
    )
    engine = SimpleNamespace(
        create_execution_context=lambda strategy: Context(),
        get_tensor_shape=lambda name: (3,) if name == "observation.state" else (2, 3),
        get_tensor_dtype=lambda name: "float",
        device_memory_size_v2=0,
    )
    sys.modules["tensorrt"] = SimpleNamespace(
        nptype=lambda dtype: np.float32,
        bfloat16="bfloat16",
        ExecutionContextAllocationStrategy=SimpleNamespace(STATIC=0, USER_MANAGED=1),
    )
    tensorrt.cuda_runtime = lambda: cuda
    tensorrt.load_engine = lambda path: engine

from tests.mocks.mock_robot import MockRobot

sent = []
send_action = MockRobot.send_action
MockRobot.send_action = lambda self, action: sent.append(dict(action)) or send_action(self, action)

from lerobot.scripts import lerobot_rollout

sys.argv = [
    "lerobot-rollout", f"--policy.path={folder}", "--robot.type=mock_robot", "--robot.random_values=false",
    "--robot.static_values=[1,2,3]", "--strategy.type=base", "--duration=0.3", "--fps=20",
]
lerobot_rollout.main()

if attempts:
    sys.exit(attempts[0])
assert "torch" not in sys.modules
# The first actions the robot gets are the chunk the program computed from its state, one per tick.
assert sent[:2] == [
    {"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0},
    {"motor_1.pos": 2.0, "motor_2.pos": 4.0, "motor_3.pos": 6.0},
], sent[:2]
if backend == "executorch_trt":
    import executorch.runtime

    assert executorch.runtime.OUTPUT_AT_IMPORT == "executorch", executorch.runtime.OUTPUT_AT_IMPORT
"""
)


@pytest.mark.skipif(not _datasets_available, reason="datasets not installed")
@pytest.mark.parametrize("backend", ["executorch_trt", "onnx_trt"])
def test_rollout_runs_an_export_with_torch_unimportable(tmp_path, backend):
    """`lerobot-rollout` loads an ExecuTorch-TensorRT or ONNX-TensorRT folder and drives a robot without torch."""
    fake = tmp_path / "fake"
    (fake / "executorch").mkdir(parents=True)
    (fake / "executorch" / "__init__.py").write_text("")
    (fake / "executorch" / "runtime.py").write_text(FAKE_EXECUTORCH_RUNTIME)
    # Like the real package, the TensorRT backend loads ExecuTorch's bindings when it is imported.
    (fake / "torch_tensorrt_executorch_runtime.py").write_text("import executorch.runtime\n")
    folder = tmp_path / "export"
    folder.mkdir()
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(fake), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, "-c", ROLLOUT_WITHOUT_TORCH, str(folder), backend, str(repo)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-4000:]
