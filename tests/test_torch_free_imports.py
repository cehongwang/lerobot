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

import subprocess
import sys

import pytest

from lerobot.utils.import_utils import _datasets_available

# The spec is still found, so code that only checks whether torch is installed keeps working. Each
# load attempt is recorded, so an import wrapped in try/except still fails the test.
IMPORT_WITHOUT_TORCH = """
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
importlib.import_module(sys.argv[1])
if attempts:
    sys.exit(attempts[0])
"""


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
