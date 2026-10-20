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
        }
        for module, stub in stubs.items():
            monkeypatch.setitem(sys.modules, module, stub)
        import build_engine

        monkeypatch.setattr(build_engine, "build_engine", MagicMock())
        stubs["build_engine"] = build_engine
        monkeypatch.setattr(torch.export, "export", MagicMock())
        monkeypatch.setattr(torch.onnx, "export", MagicMock())
        return runpy.run_path(str(EXAMPLES / f"{name}.py"))["compile_part"], stubs

    yield load
    sys.modules.pop("export", None)
    sys.modules.pop("build_engine", None)


def part():
    from export import Part

    return Part("step", torch.nn.Linear(2, 2), (torch.zeros(2),), ["x_t"], ["x_next"])


def test_executorch_route_keeps_inputs_and_outputs_on_the_gpu_and_records_cuda_graphs(route, tmp_path):
    compile_part, stubs = route("executorch_trt")
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
    compile_part, stubs = route("onnx_trt")
    compile_part(part(), tmp_path / "step.engine")
    onnx_path = torch.onnx.export.call_args.args[2]
    assert onnx_path == tmp_path / "step.onnx"
    assert torch.onnx.export.call_args.kwargs["input_names"] == ["x_t"]
    assert torch.onnx.export.call_args.kwargs["output_names"] == ["x_next"]
    build = stubs["build_engine"].build_engine
    assert build.call_args.args[:2] == (tmp_path / "step.onnx", tmp_path / "step.engine")


def test_torch_tensorrt_route_saves_an_aoti_package(route, tmp_path):
    compile_part, stubs = route("torch_trt")
    compile_part(part(), tmp_path / "step.pt2")
    tensorrt = stubs["torch_tensorrt"]
    assert tensorrt.save.call_args.args[1] == str(tmp_path / "step.pt2")
    assert tensorrt.save.call_args.kwargs["output_format"] == "aot_inductor"


@pytest.mark.parametrize("name", ["executorch_trt", "onnx_trt", "torch_trt"])
def test_routes_compile_with_the_shared_tensorrt_options(route, tmp_path, name):
    compile_part, stubs = route(name)
    from export import TENSORRT_OPTIONS

    compile_part(part(), tmp_path / "step.bin")
    if name == "onnx_trt":
        assert TENSORRT_OPTIONS["workspace_size"] == stubs["build_engine"].WORKSPACE_BYTES
        assert TENSORRT_OPTIONS["optimization_level"] == stubs["build_engine"].OPTIMIZATION_LEVEL
    else:
        options = stubs["torch_tensorrt"].dynamo.compile.call_args.kwargs
        assert {key: options[key] for key in TENSORRT_OPTIONS} == TENSORRT_OPTIONS


@pytest.mark.parametrize("checkpoint", [False, True], ids=["alone", "with_checkpoint"])
def test_help_lists_the_route_and_recipe_arguments(tmp_path, monkeypatch, capsys, checkpoint):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from export import parse_args

    argv = ["executorch_trt.py", "--help"]
    if checkpoint:
        (tmp_path / "config.json").write_text('{"type": "smolvla"}')
        argv.insert(1, f"--policy.path={tmp_path}")
    monkeypatch.setattr(sys, "argv", argv)
    try:
        with pytest.raises(SystemExit) as exited:
            parse_args("Export a policy.", "executorch_trt")
    finally:
        for name in ("export", "smolvla"):
            sys.modules.pop(name, None)
    assert exited.value.code == 0
    text = capsys.readouterr().out
    assert "--policy.path" in text and "--output_dir" in text
    if checkpoint:
        assert "--tolerance" in text and "--dataset.repo_id" in text and "--task" in text


def test_export_refuses_a_part_the_recipe_does_not_build(tmp_path, monkeypatch):
    """A recipe whose part names and parts disagree would otherwise write a folder with nothing compiled."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import export

    sys.modules.pop("export", None)

    class Recipe:
        __name__ = "recipe"

        @staticmethod
        def part_names(args):
            return ["a", "b"]

        class Export:
            def __init__(self, args):
                pass

            def parts(self):
                for name in ("a", "b_misnamed"):
                    yield export.Part(name, torch.nn.Identity(), (torch.zeros(1),), ["x"], ["y"])

            def write(self, programs, args):
                raise AssertionError("nothing was compiled")

    compiled = []
    args = SimpleNamespace(part="b", output_dir=tmp_path, route="onnx_trt")
    with pytest.raises(SystemExit, match="'b'"):
        export.export("unused", args, Recipe, lambda part, path: compiled.append(part.name), ".engine")
    assert not compiled


def test_route_scripts_name_no_policy():
    for name in ("executorch_trt", "onnx_trt", "torch_trt"):
        code = (EXAMPLES / f"{name}.py").read_text().split('"""', 2)[2]
        assert not any(policy in code for policy in ("act", "smolvla", "pi05", "groot")), name


def test_no_export_script_shadows_a_package_it_imports():
    """A script named like a package it imports hides that package, because its own folder is first on
    sys.path when it runs. torch_trt.py importing torch_tensorrt is the case this guards against. The
    check is structural, so it catches the clash even where the package is not installed."""
    import ast

    scripts = sorted(EXAMPLES.glob("*.py"))
    filenames = {s.name for s in scripts}
    # The modules a sibling is meant to import locally: the shared driver, the engine rebuilder, the recipes.
    local = {"export", "build_engine", "act", "smolvla", "pi05", "groot"}
    imported = set()
    for script in scripts:
        for node in ast.walk(ast.parse(script.read_text())):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    shadowed = sorted(name for name in imported - local if f"{name}.py" in filenames)
    assert not shadowed, f"these scripts are named like packages they import and shadow them: {shadowed}"
