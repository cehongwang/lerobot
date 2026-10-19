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

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.inference.export.engine import ExportInferenceEngine, load_program


@pytest.mark.parametrize("noise_shape", [(1, 2, 3), (2, 3)])
@pytest.mark.parametrize("torch_inputs", [False, True])
def test_executorch_three_program_loop(tmp_path, monkeypatch, noise_shape, torch_inputs):
    calls = []
    probes = []
    times = []

    def step(mask, key, sample, timestep):
        assert mask.item()
        np.testing.assert_array_equal(key, np.full((1, 3), 2.0))
        assert timestep.shape == (1,) and timestep.dtype == np.float32
        times.append(timestep.item())
        return [sample * timestep]

    programs = {
        "prefix.pte": lambda state: [np.array([[True]]), 2 * state],
        "step.pte": step,
        "actions.pte": lambda sample: [10 * sample],
    }

    def load(path, data_path=None):
        def execute(inputs):
            if torch_inputs:
                if isinstance(inputs[0], np.ndarray):
                    probes.append(path.name)
                    raise RuntimeError(
                        "Unsupported python type <class 'numpy.ndarray'>. "
                        "Ensure that inputs are passed as a flat list of tensors."
                    )
                inputs = [value.numpy() for value in inputs]
            calls.append(path.name)
            assert all(isinstance(value, np.ndarray) for value in inputs)
            return programs[path.name](*inputs)

        return SimpleNamespace(load_method=lambda name: SimpleNamespace(execute=execute))

    runtime = SimpleNamespace(load_program=load, backend_registry=SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime))
    )
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", SimpleNamespace())
    names = {
        "prefix": (["observation.state"], ["prefix_pad_masks", "past_key_0"]),
        "step": (["prefix_pad_masks", "past_key_0", "x_t", "timestep"], ["v_t"]),
        "actions": (["x_t"], ["action"]),
    }
    info = {
        "backend": "executorch_tensorrt",
        "inputs": ["observation.state", "noise"],
        "programs": {
            name: {"file": f"{name}.pte", "inputs": inputs, "outputs": outputs}
            for name, (inputs, outputs) in names.items()
        },
        "num_steps": 4,
    }
    program = load_program(tmp_path, info)
    actions = program.initialize(np.ones((1, 3)), np.ones(noise_shape))
    assert probes == (["prefix.pte", "step.pte", "actions.pte"] if torch_inputs else [])
    assert actions.shape == noise_shape
    assert calls == ["prefix.pte", *(["step.pte"] * 4), "actions.pte"]
    assert times == [1.0, 0.75, 0.5, 0.25]
    expected = 10 * (1 - 0.25) * (1 - 0.1875) * (1 - 0.125) * (1 - 0.0625)
    np.testing.assert_allclose(actions, expected)
    program(np.ones((1, 3)), np.ones(noise_shape))
    assert probes == (["prefix.pte", "step.pte", "actions.pte"] if torch_inputs else [])
    assert len(calls) == 12


def test_executorch_chained_prefix_programs(tmp_path, monkeypatch):
    """A prefix split into programs runs them in order, each reading the values its inputs name."""
    calls = []
    programs = {
        "embed.pte": lambda state: [state + 1, np.array([[True]])],
        "layers_0.pte": lambda hidden, mask: [2 * hidden, 3 * hidden],
        "layers_1.pte": lambda hidden, mask: [hidden, 5 * hidden],
        "step.pte": lambda mask, key_0, key_1, sample, timestep: [sample * 0 + key_0 + key_1],
        "actions.pte": lambda sample: [sample],
    }

    def load(path, data_path=None):
        def execute(inputs):
            calls.append(path.name)
            return programs[path.name](*inputs)

        return SimpleNamespace(load_method=lambda name: SimpleNamespace(execute=execute))

    runtime = SimpleNamespace(load_program=load, backend_registry=SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime))
    )
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", SimpleNamespace())
    names = {
        "embed": (["observation.state"], ["prefix_embs", "prefix_pad_masks"]),
        "layers_0": (["prefix_embs", "prefix_pad_masks"], ["hidden_0", "past_key_0"]),
        "layers_1": (["hidden_0", "prefix_pad_masks"], ["hidden_1", "past_key_1"]),
        "step": (["prefix_pad_masks", "past_key_0", "past_key_1", "x_t", "timestep"], ["v_t"]),
        "actions": (["x_t"], ["action"]),
    }
    info = {
        "backend": "executorch_tensorrt",
        "inputs": ["observation.state", "noise"],
        "programs": {
            name: {"file": f"{name}.pte", "inputs": inputs, "outputs": outputs}
            for name, (inputs, outputs) in names.items()
        },
        "num_steps": 2,
    }

    program = load_program(tmp_path, info)
    actions = program(np.ones((1, 2)), np.zeros((1, 2)))

    assert calls == ["embed.pte", "layers_0.pte", "layers_1.pte", "step.pte", "step.pte", "actions.pte"]
    # state 1 -> embeddings 2 -> past_key_0 = 3 * 2, hidden_0 = 4 -> past_key_1 = 5 * 4; two steps of -0.5 * 26.
    np.testing.assert_allclose(actions, np.full((1, 2), -26.0))


def test_executorch_output_survives_next_execution(tmp_path, monkeypatch):
    buffer = np.zeros((1, 2), dtype=np.float32)

    def execute(inputs):
        buffer[:] = inputs[0]
        return [buffer]

    runtime = SimpleNamespace(
        load_program=lambda *args, **kw: SimpleNamespace(
            load_method=lambda name: SimpleNamespace(execute=execute)
        )
    )
    monkeypatch.setitem(
        sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime))
    )
    program = load_program(tmp_path, {"backend": "executorch_cuda", "file": "model.pte"})
    first = program(np.ones((1, 2), dtype=np.float32))
    program(np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(first, 1.0)


def test_executorch_bfloat16_compatibility_preserves_values_and_ownership(tmp_path, monkeypatch):
    import torch

    ml_dtypes = pytest.importorskip("ml_dtypes")
    buffer = torch.zeros((1, 2), dtype=torch.bfloat16)
    input_types = []

    def execute(inputs):
        input_types.append(type(inputs[0]))
        if not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError(
                "Unsupported python type <class 'numpy.ndarray'>. "
                "Ensure that inputs are passed as a flat list of tensors."
            )
        buffer.copy_(inputs[0])
        return [buffer]

    runtime = SimpleNamespace(
        load_program=lambda *args, **kw: SimpleNamespace(
            load_method=lambda name: SimpleNamespace(execute=execute)
        )
    )
    monkeypatch.setitem(
        sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime))
    )
    program = load_program(tmp_path, {"backend": "executorch_cuda", "file": "model.pte"})
    first = program.initialize(np.array([[1.25, -2.5]], dtype=ml_dtypes.bfloat16))
    second = program(np.zeros((1, 2), dtype=ml_dtypes.bfloat16))
    assert first.dtype == np.dtype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(first.astype(np.float32), [[1.25, -2.5]])
    np.testing.assert_array_equal(second.astype(np.float32), [[0, 0]])
    assert input_types == [np.ndarray, torch.Tensor, torch.Tensor]


@pytest.mark.parametrize("error_type", [TypeError, RuntimeError])
@pytest.mark.parametrize("after_load", [False, True])
def test_executorch_does_not_retry_execution_errors(tmp_path, monkeypatch, error_type, after_load):
    calls = []
    error = error_type("Unsupported python type <class 'numpy.ndarray'>" if after_load else "Invalid shape")

    def execute(inputs):
        calls.append(inputs)
        if after_load and len(calls) == 1:
            return inputs
        raise error

    runtime = SimpleNamespace(
        load_program=lambda *args, **kw: SimpleNamespace(
            load_method=lambda name: SimpleNamespace(execute=execute)
        )
    )
    monkeypatch.setitem(
        sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime))
    )
    program = load_program(tmp_path, {"backend": "executorch_cuda", "file": "model.pte"})
    inputs = np.ones((1, 2), dtype=np.float32)
    if after_load:
        program.initialize(inputs)
    run = program if after_load else program.initialize
    with pytest.raises(error_type) as caught:
        run(inputs)
    assert caught.value is error
    assert len(calls) == (2 if after_load else 1)
    assert all(isinstance(call[0], np.ndarray) for call in calls)


def test_program_chain_invalid_step_count():
    from lerobot.inference.export.engine import ProgramChain

    with pytest.raises(ValueError, match="num_steps"):
        ProgramChain({}, {"num_steps": 0, "inputs": [], "programs": {}})


@pytest.mark.parametrize("backend", ["executorch_tensorrt", "executorch_cuda"])
@pytest.mark.parametrize("layout", ["single", "chain", "device_resident_chain"])
@pytest.mark.parametrize("has_set_option", [False, True])
def test_executorch_shares_activation_scratch_only_within_a_tensorrt_chain(
    tmp_path, monkeypatch, backend, layout, has_set_option
):
    """Each TensorRT program of a chain turns the shared scratch on before it loads; a lone one turns it off.

    A runtime without `set_option` loads the same programs, each engine with its own scratch.
    """
    events = []

    def load(path, data_path=None):
        events.append(("load", path.name))
        return SimpleNamespace(load_method=lambda name: SimpleNamespace())

    registry = SimpleNamespace()
    if has_set_option:
        registry.set_option = lambda name, options: events.append(("set", name, options))
    runtime = SimpleNamespace(load_program=load, backend_registry=registry)
    monkeypatch.setitem(
        sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime))
    )
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", SimpleNamespace())
    names = ["prefix", "step"]
    info = {
        "backend": backend,
        "device_resident": layout == "device_resident_chain",
        "inputs": ["noise"],
        "programs": {name: {"file": f"{name}.pte", "inputs": [], "outputs": []} for name in names},
        "num_steps": 1,
    }
    if layout == "single":
        names = ["model"]
        info = {"backend": backend, "file": "model.pte"}

    load_program(tmp_path, info)

    option = [("set", "TensorRTBackend", {"use_shared_activation_scratch": layout != "single"})]
    if backend == "executorch_cuda" or not has_set_option:
        option = []
    assert events == [event for name in names for event in (*option, ("load", f"{name}.pte"))]


@pytest.mark.parametrize("image_first", [False, True])
@pytest.mark.parametrize("declared_layout", [False, True])
def test_raw_frame_engine_preserves_inputs_noise_and_action_queue(
    tmp_path, monkeypatch, image_first, declared_layout
):
    from safetensors.numpy import save_file

    from lerobot.inference.export import engine as module

    state = np.array([1, 2, 3], dtype=np.float32)
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    noise = np.ones((4, 3), dtype=np.float32)
    expected = np.stack([state, 2 * state])
    names = ["observation.state", "observation.images.camera"]
    if image_first:
        names.reverse()
    frame = {"observation.state": state, "observation.images.camera": image}
    info = {
        "backend": "onnx_tensorrt",
        "file": "model.engine",
        "inputs": [*names, "noise"],
        "output": "action",
        "noise_shape": [4, 3],
        "task": "pick",
        "task_fixed": True,
        "test_case": "case.safetensors",
        "tolerance": 0.0,
    }
    if declared_layout:
        info["raw_frame"] = True
    (tmp_path / "export.json").write_text(json.dumps(info))
    save_file({**frame, "noise": noise, "expected_actions": expected}, tmp_path / "case.safetensors")
    calls = []

    class Program:
        input_shape = frame[names[0]].shape

        def __call__(self, *arrays):
            assert len(arrays) == 3
            for name, value in zip(names, arrays[:-1], strict=True):
                np.testing.assert_array_equal(value, frame[name])
                assert value.dtype == frame[name].dtype and value.flags.c_contiguous
            assert arrays[-1].shape == (4, 3) and arrays[-1].dtype == np.float32
            calls.append(arrays[-1].copy())
            return expected

    if declared_layout:
        del Program.input_shape
    monkeypatch.setattr(module, "load_program", lambda *args: Program())
    engine = ExportInferenceEngine(tmp_path, "pick", "replay", {"camera": "observation.images.camera"})
    np.testing.assert_array_equal(calls[0], noise)
    live = {
        "observation.state": state,
        "camera": np.asfortranarray(image),
        "unused.image": np.zeros(1, dtype=np.uint8),
    }
    np.testing.assert_array_equal(engine.get_action(live), state)
    np.testing.assert_array_equal(engine.get_action(live), 2 * state)
    assert len(calls) == 2
    assert not np.array_equal(calls[1], noise)
    with pytest.raises(ValueError, match="only runs the task"):
        engine._check_task("place")


def test_legacy_text_pipeline_keeps_state_not_listed_as_program_input(tmp_path, monkeypatch):
    from safetensors.numpy import save_file

    from lerobot.inference.export import engine as module

    state = np.array([1, 2], dtype=np.float32)
    (tmp_path / "export.json").write_text(
        json.dumps(
            {
                "backend": "onnx_tensorrt",
                "file": "model.engine",
                "inputs": ["tokens"],
                "output": "action",
                "test_case": "case.safetensors",
                "tolerance": 0.0,
                "text_steps": "text.json",
            }
        )
    )
    save_file({"observation.state": state, "expected_actions": state[None]}, tmp_path / "case.safetensors")

    def pipeline(frame):
        return {"tokens": frame["observation.state"]}

    monkeypatch.setattr(
        "lerobot.processor.PolicyProcessorPipeline.from_pretrained", lambda *a, **kw: pipeline
    )
    monkeypatch.setattr(module, "load_program", lambda *a: lambda tokens: tokens[:, None])
    engine = ExportInferenceEngine(tmp_path, "pick", "replay")
    np.testing.assert_array_equal(engine.get_action({"observation.state": state}), state)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("dlpack_outputs", [False, True])
def test_device_resident_chain_passes_cuda_tensors_between_programs(
    tmp_path, monkeypatch, split, positive, dlpack_outputs
):
    """Only final actions cross to NumPy, including when the velocity uses several programs.

    With `dlpack_outputs`, the programs return DLPack objects that are not torch tensors, and the chain still
    passes torch tensors between programs.
    """
    torch = pytest.importorskip("torch")
    from lerobot.inference.export import executorch

    class DLPackOnly:
        def __init__(self, tensor):
            self._tensor = tensor

        def __dlpack__(self, **kwargs):
            return self._tensor.__dlpack__(**kwargs)

        def __dlpack_device__(self):
            return self._tensor.__dlpack_device__()

    received, downloaded, times = [], [], []
    to_numpy = executorch.to_numpy

    def download(tensor):
        downloaded.append(tensor)
        return to_numpy(tensor)

    def load(path, data_path=None):
        def execute(inputs):
            outputs = run(inputs)
            return [DLPackOnly(output) for output in outputs] if dlpack_outputs else outputs

        def run(inputs):
            received.append((path.name, [type(value) for value in inputs]))
            if path.name == "prefix.pte":
                return [inputs[0] * 2]
            if path.name in ("step.pte", "step_0.pte"):
                assert inputs[-1].shape == (1,) and inputs[-1].dtype == torch.float32
                times.append(inputs[-1].item())
                return [inputs[0] + inputs[1] * 0]
            if path.name == "step_1.pte":
                return [inputs[0]]
            return [inputs[0] * 10]

        return SimpleNamespace(load_method=lambda name: SimpleNamespace(execute=execute))

    runtime = SimpleNamespace(load_program=load, backend_registry=SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime))
    )
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", SimpleNamespace())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(executorch, "to_numpy", download)
    names = {"prefix": (["observation.state"], ["past_key_0"])}
    if split:
        names.update(
            step_0=(["past_key_0", "x_t", "timestep"], ["hidden"]),
            step_1=(["hidden"], ["v_t"]),
        )
    else:
        names["step"] = (["past_key_0", "x_t", "timestep"], ["v_t"])
    names["actions"] = (["x_t"], ["action"])
    info = {
        "backend": "executorch_tensorrt",
        "device_resident": True,
        "inputs": ["observation.state", "noise"],
        "programs": {
            name: {"file": f"{name}.pte", "inputs": inputs, "outputs": outputs}
            for name, (inputs, outputs) in names.items()
        },
        "num_steps": 2,
    }
    if positive:
        info.update(dt=0.5, timesteps=[0, 500])
    program = load_program(tmp_path, info)
    inputs = (np.ones((1, 3), np.float32), np.zeros((1, 3), np.float32))
    actions = program.initialize(*inputs)

    steps = ["step_0.pte", "step_1.pte"] if split else ["step.pte"]
    assert [name for name, _ in received] == ["prefix.pte", *steps, *steps, "actions.pte"]
    assert all(kind is torch.Tensor for _, kinds in received for kind in kinds), received
    assert times == ([0, 500] if positive else [1.0, 0.5])
    assert isinstance(actions, np.ndarray)
    assert len(downloaded) == 1
    np.testing.assert_allclose(actions, np.full((1, 3), 20.0 if positive else -20.0))
    np.testing.assert_array_equal(program(*inputs), actions)
    assert len(downloaded) == 2


@pytest.mark.parametrize("chain", [False, True])
def test_torch_tensorrt_package_takes_host_arrays_and_returns_host_actions(tmp_path, monkeypatch, chain):
    """The AOTInductor runner hands the package tensors and returns NumPy, alone and as a chain."""
    import torch

    loaded, received = [], []

    def load_package(path):
        loaded.append(Path(path).name)

        def run(*inputs):
            received.append((Path(path).name, [type(x) for x in inputs]))
            if Path(path).name == "prefix.pt2":
                return [inputs[0] * 2]
            if Path(path).name == "step.pt2":
                return [inputs[0] + inputs[1] * 0]
            return [inputs[0] * 10]

        return run

    monkeypatch.setitem(sys.modules, "torch_tensorrt", SimpleNamespace())
    monkeypatch.setattr(torch._inductor, "aoti_load_package", load_package)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)
    if chain:
        names = {
            "prefix": (["observation.state"], ["past_key_0"]),
            "step": (["past_key_0", "x_t", "timestep"], ["v_t"]),
            "actions": (["x_t"], ["action"]),
        }
        info = {
            "backend": "torch_tensorrt",
            "inputs": ["observation.state", "noise"],
            "programs": {
                name: {"file": f"{name}.pt2", "inputs": inputs, "outputs": outputs}
                for name, (inputs, outputs) in names.items()
            },
            "num_steps": 1,
        }
        actions = load_program(tmp_path, info)(np.ones((1, 3), np.float32), np.zeros((1, 3), np.float32))
        assert loaded == ["prefix.pt2", "step.pt2", "actions.pt2"]
        np.testing.assert_allclose(actions, np.full((1, 3), -20.0))
    else:
        info = {"backend": "torch_tensorrt", "file": "model.pt2"}
        actions = load_program(tmp_path, info)(np.full((1, 3), 2.0, np.float32))
        assert loaded == ["model.pt2"]
        np.testing.assert_array_equal(actions, np.full((1, 3), 20.0))
    assert isinstance(actions, np.ndarray)
    assert all(kind is torch.Tensor for _, kinds in received for kind in kinds), received
