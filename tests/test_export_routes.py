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

"""The route scripts: each compiles a part with its backend, the same way for every policy."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "export"


@pytest.fixture
def route(monkeypatch):
    """Load a route script with its compiler stubbed out, and return its `compile_part` and the stubs."""
    monkeypatch.syspath_prepend(str(EXAMPLES))

    def load(name):
        stubs = {
            "torch_tensorrt": MagicMock(),
            "executorch.exir": SimpleNamespace(ExecutorchBackendConfig=MagicMock()),
            "executorch.exir.passes.memory_planning_pass": SimpleNamespace(MemoryPlanningPass=MagicMock()),
            "executorch.exir.passes.propagate_device_pass": SimpleNamespace(
                PropagateDeviceConfig=MagicMock()
            ),
            "build_engine": SimpleNamespace(build_engine=MagicMock()),
        }
        for module, stub in stubs.items():
            monkeypatch.setitem(sys.modules, module, stub)
        monkeypatch.setattr(torch.export, "export", MagicMock())
        monkeypatch.setattr(torch.onnx, "export", MagicMock())
        return runpy.run_path(str(EXAMPLES / f"{name}.py"))["compile_part"], stubs

    yield load
    sys.modules.pop("export", None)


def part():
    from export import Part

    return Part("step", torch.nn.Linear(2, 2), (torch.zeros(2),), ["x_t"], ["x_next"])


def test_executorch_route_keeps_inputs_and_outputs_on_the_gpu_and_records_cuda_graphs(route, tmp_path):
    compile_part, stubs = route("executorch_tensorrt")
    compile_part(part(), tmp_path / "step.pte")
    tensorrt = stubs["torch_tensorrt"]
    kwargs = tensorrt.save.call_args.kwargs
    assert tensorrt.save.call_args.args[1] == str(tmp_path / "step.pte")
    assert kwargs["output_format"] == "executorch" and kwargs["use_cuda_graphs"] is True
    stubs["executorch.exir.passes.propagate_device_pass"].PropagateDeviceConfig.assert_called_once_with(
        skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
    )
    stubs["executorch.exir.passes.memory_planning_pass"].MemoryPlanningPass.assert_called_once_with(
        alloc_graph_input=False, alloc_graph_output=False
    )


def test_onnx_route_keeps_the_onnx_file_and_builds_the_engine_beside_it(route, tmp_path):
    compile_part, stubs = route("onnx_tensorrt")
    compile_part(part(), tmp_path / "step.engine")
    onnx_path = torch.onnx.export.call_args.args[2]
    assert onnx_path == tmp_path / "step.onnx"
    assert torch.onnx.export.call_args.kwargs["input_names"] == ["x_t"]
    assert torch.onnx.export.call_args.kwargs["output_names"] == ["x_next"]
    build = stubs["build_engine"].build_engine
    assert build.call_args.args[:2] == (tmp_path / "step.onnx", tmp_path / "step.engine")


def test_torch_tensorrt_route_saves_an_aoti_package(route, tmp_path):
    compile_part, stubs = route("torch_tensorrt")
    compile_part(part(), tmp_path / "step.pt2")
    tensorrt = stubs["torch_tensorrt"]
    assert tensorrt.save.call_args.args[1] == str(tmp_path / "step.pt2")
    assert tensorrt.save.call_args.kwargs["output_format"] == "aot_inductor"


@pytest.mark.parametrize("name", ["executorch_tensorrt", "onnx_tensorrt", "torch_tensorrt"])
def test_routes_compile_with_the_shared_tensorrt_options(route, tmp_path, name):
    from export import TENSORRT_OPTIONS

    compile_part, stubs = route(name)
    compile_part(part(), tmp_path / "step.bin")
    if name == "onnx_tensorrt":
        build = stubs["build_engine"].build_engine
        assert build.call_args.args[2:] == (
            TENSORRT_OPTIONS["workspace_size"],
            TENSORRT_OPTIONS["optimization_level"],
        )
    else:
        options = stubs["torch_tensorrt"].dynamo.compile.call_args.kwargs
        assert {key: options[key] for key in TENSORRT_OPTIONS} == TENSORRT_OPTIONS


def test_route_scripts_name_no_policy():
    for name in ("executorch_tensorrt", "onnx_tensorrt", "torch_tensorrt"):
        code = (EXAMPLES / f"{name}.py").read_text().split('"""', 2)[2]
        assert not any(policy in code for policy in ("act", "smolvla", "pi05", "groot")), name
