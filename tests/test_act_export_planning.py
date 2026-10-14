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

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("deferred,export_only", [(False, False), (True, False), (False, True)])
def test_executorch_export_shares_host_inputs_only(tmp_path, monkeypatch, deferred, export_only):
    torch = MagicMock()
    tensorrt = MagicMock()
    config = MagicMock()
    planning = MagicMock()
    export = MagicMock(output_dir=tmp_path)
    save_for_build_engine = MagicMock()
    args = SimpleNamespace(
        policy_path="policy",
        output_dir=tmp_path,
        job_name="act",
        export_only=export_only,
        tolerance=0.5,
        workspace_gib=None,
        optimization_level=None,
    )
    modules = {
        "torch": torch,
        "torch_tensorrt": tensorrt,
        "executorch.exir": SimpleNamespace(ExecutorchBackendConfig=config),
        "executorch.exir.passes": SimpleNamespace(MemoryPlanningPass=planning),
        "act_recipe": SimpleNamespace(
            ACTExport=MagicMock(return_value=export),
            parse_args=lambda *a: args,
            save_for_build_engine=save_for_build_engine,
            EXPORTED_PROGRAM="model.pt2",
        ),
        "tensorrt": SimpleNamespace(IStreamWriter=object),
        "lerobot.rollout.inference.export": SimpleNamespace(ExportInferenceEngine=MagicMock()),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    scripts = Path(__file__).resolve().parents[1] / "examples" / "export"
    if deferred:
        namespace = runpy.run_path(str(scripts / "build_engine.py"))
        namespace["build_executorch"](tmp_path, "model.pte", args)
    else:
        runpy.run_path(str(scripts / "act_executorch_tensorrt.py"), run_name="__main__")
    if export_only:
        tensorrt.save.assert_not_called()
        save_for_build_engine.assert_called_once_with(
            export, torch.export.export.return_value, tmp_path / "model.pte"
        )
        planning.assert_not_called()
        config.assert_not_called()
        return
    planning.assert_called_once_with(alloc_graph_input=False)
    config.assert_called_once_with(memory_planning_pass=planning.return_value)
    tensorrt.save.assert_called_once_with(
        tensorrt.dynamo.compile.return_value,
        str(tmp_path / "model.pte"),
        output_format="executorch",
        retrace=False,
        backend_config=config.return_value,
        **({} if deferred else {"use_cuda_graphs": True}),
    )


def test_smolvla_direct_export_enables_cuda_graphs(tmp_path, monkeypatch):
    torch = MagicMock()
    tensorrt = MagicMock()
    export = MagicMock(output_dir=tmp_path)
    modules = {
        "torch": torch,
        "torch_tensorrt": tensorrt,
        "act_recipe": SimpleNamespace(save_for_build_engine=MagicMock()),
        "smolvla_recipe": SimpleNamespace(
            SmolVLAExport=MagicMock(return_value=export),
            parse_args=lambda *args: SimpleNamespace(export_only=False),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    script = Path(__file__).resolve().parents[1] / "examples" / "export" / "smolvla_executorch_tensorrt.py"
    runpy.run_path(str(script), run_name="__main__")
    assert tensorrt.save.call_args.kwargs["use_cuda_graphs"] is True


@pytest.mark.parametrize("policy", ["pi05", "groot"])
def test_chain_executorch_export_enables_cuda_graphs(tmp_path, monkeypatch, policy):
    tensorrt = MagicMock()
    export_parts = MagicMock()
    modules = {
        "torch_tensorrt": tensorrt,
        "executorch.exir": SimpleNamespace(ExecutorchBackendConfig=MagicMock()),
        "executorch.exir.passes.memory_planning_pass": SimpleNamespace(MemoryPlanningPass=MagicMock()),
        "executorch.exir.passes.propagate_device_pass": SimpleNamespace(PropagateDeviceConfig=MagicMock()),
        f"{policy}_recipe": SimpleNamespace(
            TENSORRT_OPTIONS={}, export_parts=export_parts, parse_args=lambda *args: SimpleNamespace()
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    script = Path(__file__).resolve().parents[1] / "examples" / "export" / f"{policy}_executorch_tensorrt.py"
    runpy.run_path(str(script), run_name="__main__")
    compile_part = export_parts.call_args.args[3]
    monkeypatch.setattr("torch.export.export", MagicMock())
    compile_part(MagicMock(), (), [], [], tmp_path / "part.pte")
    assert tensorrt.save.call_args.kwargs["use_cuda_graphs"] is True


@pytest.mark.parametrize("policy,cuda_graphs", [("pi05", True), ("groot", False)])
def test_chain_onnx_export_sets_cuda_graphs(monkeypatch, policy, cuda_graphs):
    export_parts = MagicMock()
    recipe = SimpleNamespace(
        TENSORRT_OPTIONS={},
        export_parts=export_parts,
        free_memory=MagicMock(),
        parse_args=lambda *args: SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, f"{policy}_recipe", recipe)
    monkeypatch.setitem(sys.modules, "build_engine", SimpleNamespace(build_engine=MagicMock()))
    script = Path(__file__).resolve().parents[1] / "examples" / "export" / f"{policy}_onnx_tensorrt.py"
    runpy.run_path(str(script), run_name="__main__")
    assert export_parts.call_args.kwargs.get("cuda_graphs", False) is cuda_graphs


@pytest.mark.parametrize("policy", ["pi05", "groot"])
def test_chain_torch_tensorrt_export_keeps_values_on_the_gpu(monkeypatch, policy):
    export_parts = MagicMock()
    recipe = SimpleNamespace(
        TENSORRT_OPTIONS={}, export_parts=export_parts, parse_args=lambda *args: SimpleNamespace()
    )
    monkeypatch.setitem(sys.modules, f"{policy}_recipe", recipe)
    monkeypatch.setitem(sys.modules, "torch_tensorrt", MagicMock())
    script = Path(__file__).resolve().parents[1] / "examples" / "export" / f"{policy}_torch_tensorrt.py"
    runpy.run_path(str(script), run_name="__main__")
    assert export_parts.call_args.kwargs.get("device_resident", False) is True


def test_pi05_executorch_cuda_export_keeps_values_on_the_gpu(tmp_path, monkeypatch):
    export_parts = MagicMock()
    exir = SimpleNamespace(
        EdgeCompileConfig=MagicMock(),
        ExecutorchBackendConfig=MagicMock(),
        to_edge_transform_and_lower=MagicMock(),
    )
    planning = MagicMock()
    propagate = MagicMock()
    modules = {
        "executorch.backends.cuda.cuda_backend": SimpleNamespace(CudaBackend=MagicMock()),
        "executorch.backends.cuda.cuda_partitioner": SimpleNamespace(CudaPartitioner=MagicMock()),
        "executorch.exir": exir,
        "executorch.exir.passes.memory_planning_pass": SimpleNamespace(MemoryPlanningPass=planning),
        "executorch.exir.passes.propagate_device_pass": SimpleNamespace(PropagateDeviceConfig=propagate),
        "pi05_recipe": SimpleNamespace(export_parts=export_parts, parse_args=lambda *args: SimpleNamespace()),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    script = Path(__file__).resolve().parents[1] / "examples" / "export" / "pi05_executorch_cuda.py"
    runpy.run_path(str(script), run_name="__main__")
    assert export_parts.call_args.kwargs.get("device_resident", False) is True
    compile_part = export_parts.call_args.args[3]
    monkeypatch.setattr("torch.export.export", MagicMock())
    compile_part(MagicMock(), (), [], [], tmp_path / "part" / "part.pte")
    propagate.assert_called_once_with(skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True)
    planning.assert_called_once_with(alloc_graph_input=False, alloc_graph_output=False)
    exir.ExecutorchBackendConfig.assert_called_once_with(
        propagate_device_config=propagate.return_value,
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=planning.return_value,
    )
    lowered = exir.to_edge_transform_and_lower.return_value
    lowered.to_executorch.assert_called_once_with(config=exir.ExecutorchBackendConfig.return_value)
