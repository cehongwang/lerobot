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

"""The export engine's chain and its ExecuTorch and AOTInductor runners, with stand-ins for the runtimes."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.inference.export import engine as export_engine
from lerobot.inference.export.engine import ExportInferenceEngine, ProgramChain, load_programs

# DLPack's device types.
CPU, CUDA = 1, 2


class Result:
    """Stands in for ExecuTorch's own result: device memory with a host copy on request."""

    def __init__(self, array, dtype_name="float32", device=CUDA):
        self.array, self.shape, self.dtype_name = np.asarray(array), np.shape(array), dtype_name
        self.device = device

    @classmethod
    def from_host(cls, source):
        return cls(np.array(source.array if isinstance(source, cls) else source))

    def to_host(self):
        # Like the bindings, a new host copy that later runs do not overwrite.
        return self.array.copy()

    def __dlpack_device__(self):
        return (self.device, 0)


@pytest.fixture
def executorch(monkeypatch):
    """A stand-in ExecuTorch runtime; `programs` maps a file name to its function on host arrays."""
    programs, calls, options = {}, [], []

    def load(path):
        def execute(inputs):
            assert all(isinstance(value, Result) and value.device == CUDA for value in inputs), (
                "programs only see device buffers"
            )
            calls.append(path.name)
            return [Result(out) for out in programs[path.name](*(value.array for value in inputs))]

        return SimpleNamespace(load_method=lambda name: SimpleNamespace(execute=execute))

    registry = SimpleNamespace(set_option=lambda backend, opts: options.append((backend, opts)))
    runtime = SimpleNamespace(load_program=load, backend_registry=registry)
    monkeypatch.setitem(
        sys.modules,
        "executorch.runtime",
        SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime), ExecuTorchResult=Result),
    )
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", SimpleNamespace())
    return SimpleNamespace(programs=programs, calls=calls, options=options)


def chain_info(names, backend="executorch_trt", suffix=".pte", **info):
    return {
        "backend": backend,
        "inputs": info.pop("inputs"),
        "programs": {
            name: {"file": f"{name}{suffix}", "inputs": inputs, "outputs": outputs}
            for name, (inputs, outputs) in names.items()
        },
        **info,
    }


def test_denoising_steps_run_in_order_and_feed_each_other(tmp_path, executorch):
    """Each step program gets the previous one's x_next as x_t, and its timestep; the engine does no math."""
    times = []

    def step(mask, key, x_t, timestep):
        assert mask.item()
        np.testing.assert_array_equal(key, np.full((1, 3), 2.0))
        assert timestep.shape == (1,) and timestep.dtype == np.float32
        times.append(timestep.item())
        return [x_t * timestep]

    executorch.programs.update(
        {
            "prefix.pte": lambda state: [np.array([[True]]), 2 * state],
            "step.pte": step,
            "actions.pte": lambda x_t: [10 * x_t],
        }
    )
    names = {
        "prefix": (["observation.state"], ["prefix_pad_masks", "past_key_0"]),
        "step": (["prefix_pad_masks", "past_key_0", "x_t", "timestep"], ["x_next"]),
        "actions": (["x_t"], ["action"]),
    }
    info = chain_info(
        names, inputs=["observation.state", "noise"], steps=["step"], timesteps=[1.0, 0.75, 0.5]
    )
    chain = ProgramChain(load_programs(tmp_path, info), info)
    actions = chain(np.ones((1, 3), np.float32), np.ones((2, 3), np.float32))
    assert executorch.calls == ["prefix.pte", "step.pte", "step.pte", "step.pte", "actions.pte"]
    assert times == [1.0, 0.75, 0.5]
    np.testing.assert_allclose(actions, np.full((2, 3), 10 * 1.0 * 0.75 * 0.5))
    assert isinstance(actions, np.ndarray)


def test_a_split_prefix_reads_the_values_its_inputs_name(tmp_path, executorch):
    executorch.programs.update(
        {
            "embed.pte": lambda state: [state + 1, np.array([[True]])],
            "layers_0.pte": lambda hidden, mask: [2 * hidden, 3 * hidden],
            "layers_1.pte": lambda hidden, mask: [hidden, 5 * hidden],
            "step.pte": lambda mask, key_0, key_1, x_t, timestep: [x_t * 0 + key_0 + key_1],
            "actions.pte": lambda x_t: [x_t],
        }
    )
    names = {
        "embed": (["observation.state"], ["prefix_embs", "prefix_pad_masks"]),
        "layers_0": (["prefix_embs", "prefix_pad_masks"], ["hidden_0", "past_key_0"]),
        "layers_1": (["hidden_0", "prefix_pad_masks"], ["hidden_1", "past_key_1"]),
        "step": (["prefix_pad_masks", "past_key_0", "past_key_1", "x_t", "timestep"], ["x_next"]),
        "actions": (["x_t"], ["action"]),
    }
    info = chain_info(names, inputs=["observation.state", "noise"], steps=["step"], timesteps=[1.0])
    actions = ProgramChain(load_programs(tmp_path, info), info)(
        np.array([[1.0]], np.float32), np.zeros((1, 1), np.float32)
    )
    # key_0 = 3 * (state + 1) = 6 and key_1 = 5 * 2 * (state + 1) = 20.
    np.testing.assert_allclose(actions, [[26.0]])
    assert executorch.calls == ["embed.pte", "layers_0.pte", "layers_1.pte", "step.pte", "actions.pte"]


def test_a_whole_policy_is_a_chain_of_one(tmp_path, executorch):
    executorch.programs["model.pte"] = lambda state: [2 * state]
    info = chain_info({"model": (["observation.state"], ["action"])}, inputs=["observation.state"])
    actions = ProgramChain(load_programs(tmp_path, info), info)(np.array([[1.5]], np.float32))
    np.testing.assert_array_equal(actions, [[3.0]])


@pytest.mark.parametrize(
    "make_input", [memoryview, lambda array: Result(array, device=CPU)], ids=["buffer", "host_result"]
)
def test_executorch_copies_every_host_input_to_the_gpu(tmp_path, executorch, make_input):
    """The bindings would hand a CUDA program a host pointer without checking, so nothing host-side passes."""
    executorch.programs["model.pte"] = lambda state: [2 * state]
    info = chain_info({"model": (["observation.state"], ["action"])}, inputs=["observation.state"])
    program = load_programs(tmp_path, info)["model"]
    np.testing.assert_array_equal(program(make_input(np.ones(2, np.float32))), [2, 2])


def test_executorch_programs_of_a_chain_share_one_activation_scratch(tmp_path, executorch):
    executorch.programs["model.pte"] = lambda state: [state]
    info = chain_info({"model": (["observation.state"], ["action"])}, inputs=["observation.state"])
    load_programs(tmp_path, info)
    assert executorch.options == [("TensorRTBackend", {"use_shared_activation_scratch": True})]


class HostResult:
    """Stands in for a CPU result of ExecuTorch's bindings, read through the buffer protocol.

    The protocol exports it in its own layout, except bfloat16, which the protocol cannot describe, so
    that is exported as raw bytes in storage order.
    """

    def __init__(self, storage, view):
        self._storage, self._view = storage, view

    def __buffer__(self, flags):
        if self._view.dtype.name == "bfloat16":
            return memoryview(self._storage.reshape(-1).view(np.uint8))
        return memoryview(self._view)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("channels_last", [False, True])
def test_executorch_reads_results_in_their_own_layout(dtype, channels_last):
    from lerobot.inference.export.executorch import to_host

    if dtype == "bfloat16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    expected = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2).astype(dtype)
    # A channels-last result stores each pixel's channels together and keeps the logical shape.
    storage = expected.transpose(0, 2, 3, 1).copy() if channels_last else expected.copy()
    view = storage.transpose(0, 3, 1, 2) if channels_last else storage
    result = SimpleNamespace(
        to_host=lambda: HostResult(storage, view),
        dtype_name=np.dtype(dtype).name,
        shape=view.shape,
        strides=view.strides,
    )
    host = to_host(result)
    # The bindings' to_host already made the copy, so reading it copies nothing more.
    assert np.shares_memory(host, storage)
    assert host.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(host.astype(np.float32), expected.astype(np.float32))


def test_executorch_host_copies_survive_the_next_run(tmp_path, executorch):
    buffer = np.zeros((1, 2), np.float32)

    def run(state):
        buffer[:] = state
        return [buffer]

    executorch.programs["model.pte"] = run
    info = chain_info({"model": (["observation.state"], ["action"])}, inputs=["observation.state"])
    chain = ProgramChain(load_programs(tmp_path, info), info)
    first = chain(np.ones((1, 2), np.float32))
    chain(np.zeros((1, 2), np.float32))
    np.testing.assert_array_equal(first, 1.0)


def test_unknown_backends_are_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown export backend"):
        load_programs(tmp_path, {"backend": "executorch_cuda", "programs": {}})


@pytest.mark.parametrize("image_first", [False, True])
def test_engine_feeds_the_raw_frame_and_fresh_noise_and_plays_the_chunk(tmp_path, monkeypatch, image_first):
    from safetensors.numpy import save_file

    state = np.array([1, 2, 3], dtype=np.float32)
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    noise = np.ones((4, 3), dtype=np.float32)
    expected = np.stack([state, 2 * state])
    names = ["observation.state", "observation.images.camera"]
    if image_first:
        names.reverse()
    frame = {"observation.state": state, "observation.images.camera": image}
    info = chain_info(
        {"model": ([*names, "noise"], ["action"])},
        backend="onnx_trt",
        suffix=".engine",
        inputs=[*names, "noise"],
        noise_shape=[4, 3],
        task="pick",
        task_fixed=True,
        test_case="case.safetensors",
        tolerance=0.0,
    )
    (tmp_path / "export.json").write_text(json.dumps(info))
    save_file({**frame, "noise": noise, "expected_actions": expected}, tmp_path / "case.safetensors")
    calls = []

    def program(*arrays):
        assert len(arrays) == 3
        for name, value in zip(names, arrays[:-1], strict=True):
            np.testing.assert_array_equal(value, frame[name])
            assert value.dtype == frame[name].dtype and value.flags.c_contiguous
        assert arrays[-1].shape == (4, 3) and arrays[-1].dtype == np.float32
        calls.append(arrays[-1].copy())
        return expected

    monkeypatch.setattr(export_engine, "load_programs", lambda *args: {"model": program})
    engine = ExportInferenceEngine(tmp_path, "pick", "replay", {"camera": "observation.images.camera"})
    np.testing.assert_array_equal(calls[0], noise)
    live = {
        "observation.state": state,
        "camera": np.asfortranarray(image),
        "unused.image": np.zeros(1, np.uint8),
    }
    np.testing.assert_array_equal(engine.get_action(live), state)
    np.testing.assert_array_equal(engine.get_action(live), 2 * state)
    assert len(calls) == 2
    assert not np.array_equal(calls[1], noise)
    with pytest.raises(ValueError, match="only runs the task"):
        engine._check_task("place")


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
            "step": (["past_key_0", "x_t", "timestep"], ["x_next"]),
            "actions": (["x_t"], ["action"]),
        }
        info = {
            "backend": "torch_trt",
            "inputs": ["observation.state", "noise"],
            "programs": {
                name: {"file": f"{name}.pt2", "inputs": inputs, "outputs": outputs}
                for name, (inputs, outputs) in names.items()
            },
            "steps": ["step"],
            "timesteps": [1.0],
        }
        actions = ProgramChain(load_programs(tmp_path, info), info)(
            np.ones((1, 3), np.float32), np.zeros((1, 3), np.float32)
        )
        assert loaded == ["prefix.pt2", "step.pt2", "actions.pt2"]
        np.testing.assert_allclose(actions, np.full((1, 3), 20.0))
    else:
        info = chain_info(
            {"model": (["observation.state"], ["action"])},
            "torch_trt",
            ".pt2",
            inputs=["observation.state"],
        )
        actions = ProgramChain(load_programs(tmp_path, info), info)(np.full((1, 3), 2.0, np.float32))
        assert loaded == ["model.pt2"]
        np.testing.assert_array_equal(actions, np.full((1, 3), 20.0))
    assert isinstance(actions, np.ndarray)
    assert all(kind is torch.Tensor for _, kinds in received for kind in kinds), received
