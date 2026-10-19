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

import importlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def recipe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    return importlib.import_module("act_recipe")


@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_act_folder_records_what_the_engine_needs(recipe, tmp_path, cuda_graphs):
    export = recipe.ACTExport.__new__(recipe.ACTExport)
    export.__dict__.update(
        output_dir=tmp_path,
        backend="onnx_tensorrt",
        frame={"observation.state": np.zeros(6, np.float32)},
        expected_actions=np.zeros((2, 6), np.float32),
        config=SimpleNamespace(save_pretrained=lambda path: None),
        input_names=["observation.state"],
        tolerance=0.5,
        start=0.0,
    )
    export.write(tmp_path / "model.engine", cuda_graphs=cuda_graphs)
    info = json.loads((tmp_path / "export.json").read_text())
    assert info["raw_frame"] is True
    assert info.get("cuda_graphs", False) is cuda_graphs
    assert info["file"] == "model.engine" and info["inputs"] == ["observation.state"]


def test_act_with_temporal_ensembling_is_refused(recipe, monkeypatch):
    policy = SimpleNamespace(config=SimpleNamespace(temporal_ensemble_coeff=0.01))
    policy.to = lambda device: policy
    policy.eval = lambda: policy
    monkeypatch.setattr(recipe.ACTPolicy, "from_pretrained", lambda path: policy)
    with pytest.raises(ValueError, match="temporal_ensemble_coeff"):
        recipe.ACTExport(Namespace(policy_path="checkpoint"))
