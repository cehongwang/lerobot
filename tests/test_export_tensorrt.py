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

import ctypes
import functools
import gc
import itertools
import sys
import weakref
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.inference.export import tensorrt as backend

HOST_TO_DEVICE, DEVICE_TO_HOST, DEVICE_TO_DEVICE = 1, 2, 3


@pytest.fixture
def fake_cuda(monkeypatch):
    """CUDA on the CPU: each stream queues its copies and engine runs until it is synchronized."""
    allocations = {}
    queues = defaultdict(list)
    copies = []
    handles = itertools.count(100)

    def malloc(size):
        memory = ctypes.create_string_buffer(size)
        pointer = ctypes.addressof(memory)
        allocations[pointer] = memory
        return 0, pointer

    def free(pointer):
        allocations.pop(pointer)
        return (0,)

    def memcpy(dst, src, size, kind, stream):
        copies.append(kind)
        queues[stream].append(lambda: ctypes.memmove(dst, src, size))
        return (0,)

    def synchronize(stream):
        queue = queues[stream]
        while queue:
            queue.pop(0)()
        return (0,)

    # A recorded graph is the work queued during the capture, replayed with the addresses it was queued with.
    captures = {}

    def begin_capture(stream, mode):
        captures[stream] = len(queues[stream])
        return (0,)

    def end_capture(stream):
        start = captures.pop(stream)
        graph = queues[stream][start:]
        del queues[stream][start:]
        return 0, graph

    def launch(graph, stream):
        queues[stream].extend(graph)
        return (0,)

    cuda = SimpleNamespace(
        cudaMalloc=malloc,
        cudaFree=free,
        cudaMemcpyAsync=memcpy,
        cudaMemcpyKind=SimpleNamespace(
            cudaMemcpyHostToDevice=HOST_TO_DEVICE,
            cudaMemcpyDeviceToHost=DEVICE_TO_HOST,
            cudaMemcpyDeviceToDevice=DEVICE_TO_DEVICE,
        ),
        cudaStreamCreate=lambda: (0, next(handles)),
        cudaStreamSynchronize=synchronize,
        cudaStreamDestroy=lambda stream: (0,),
        cudaStreamCaptureMode=SimpleNamespace(cudaStreamCaptureModeGlobal=0),
        cudaStreamBeginCapture=begin_capture,
        cudaStreamEndCapture=end_capture,
        cudaGraphInstantiate=lambda graph, flags: (0, graph),
        cudaGraphDestroy=lambda graph: (0,),
        cudaGraphLaunch=launch,
        cudaGraphExecDestroy=MagicMock(return_value=(0,)),
        allocations=allocations,
        queues=queues,
        copies=copies,
    )
    monkeypatch.setattr(backend, "cuda_runtime", lambda: cuda)
    return cuda


def floats(pointer, shape):
    return np.ctypeslib.as_array((ctypes.c_float * int(np.prod(shape))).from_address(pointer)).reshape(shape)


# Each fake engine doubles its input: `x` to `y` alone, `x_t` to `x_next` as a denoising step, and `x_t` to
# `action` at the end of a chain. They ask for different scratch sizes, as real engines do.
SCRATCH_BYTES = {"prefix": 64, "step": 256, "actions": 128}


@pytest.fixture
def fake_engine(monkeypatch, fake_cuda):
    contexts = []

    class Context:
        def __init__(self, strategy):
            self.strategy = strategy
            self.addresses = {}
            self.enqueued = 0
            contexts.append(self)

        def set_tensor_address(self, name, pointer):
            self.addresses[name] = pointer
            return True

        def execute_async_v3(self, stream):
            # TensorRT reads the bound addresses when the work is queued, not when it runs.
            addresses = dict(self.addresses)
            source = addresses["x"] if "x" in addresses else addresses["x_t"]
            target = next(
                addresses[name] for name in ("y", "x_next", "action", "hidden") if name in addresses
            )

            def execute():
                floats(target, (1, 3))[:] = 2 * floats(source, (1, 3))

            self.enqueued += 1
            fake_cuda.queues[stream].append(execute)
            return True

        def set_device_memory(self, pointer, size):
            self.scratch = (pointer, size)

    def load_engine(path):
        return SimpleNamespace(
            create_execution_context=Context,
            get_tensor_shape=lambda name: (1,) if name == "timestep" else (1, 3),
            get_tensor_dtype=lambda name: "float",
            device_memory_size_v2=SCRATCH_BYTES.get(path.stem, 64),
        )

    trt = SimpleNamespace(
        nptype=lambda dtype: np.float32,
        bfloat16="bfloat16",
        ExecutionContextAllocationStrategy=SimpleNamespace(STATIC=0, USER_MANAGED=1),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(backend, "load_engine", load_engine)
    return contexts


def test_engine_copies_host_inputs_and_owns_returned_actions(tmp_path, fake_cuda, fake_engine):
    engine = backend.TensorRTEngine(tmp_path / "model.engine", ["x"], ["y"])
    values = np.arange(6, dtype=np.float64).reshape(1, 6)[:, ::2]
    first = engine(values)
    np.testing.assert_array_equal(first, [[0, 4, 8]])
    np.testing.assert_array_equal(engine(np.ones((1, 3))), [[2, 2, 2]])
    np.testing.assert_array_equal(first, [[0, 4, 8]])
    assert fake_cuda.copies == [HOST_TO_DEVICE, DEVICE_TO_HOST] * 2
    assert not any(fake_cuda.queues.values())
    del engine
    assert not fake_cuda.allocations


def test_engines_share_cache_on_device_and_retain_scratch(tmp_path, fake_cuda, fake_engine):
    stream = backend.CudaStream()
    prefix = backend.TensorRTEngine(tmp_path / "prefix.engine", ["x"], ["y"], False, stream)
    step = backend.TensorRTEngine(tmp_path / "step.engine", ["x"], ["y"], False, stream)
    scratch = backend.CudaBuffer((SCRATCH_BYTES["step"],), np.uint8)
    prefix.use_scratch(scratch)
    step.use_scratch(scratch)
    pointer = scratch.ptr
    del scratch
    cache = prefix.run_device(np.ones((1, 3), np.float32))
    np.testing.assert_array_equal(step(*cache), [[4, 4, 4]])
    assert fake_engine[1].addresses["x"] == cache[0].ptr
    assert fake_cuda.copies == [HOST_TO_DEVICE, DEVICE_TO_HOST]
    assert fake_engine[0].scratch == (pointer, SCRATCH_BYTES["prefix"])
    assert fake_engine[1].scratch == (pointer, SCRATCH_BYTES["step"])
    assert pointer in fake_cuda.allocations


def test_chained_engines_wait_for_the_gpu_only_to_return_the_actions(tmp_path, fake_cuda, fake_engine):
    """Engines on one stream run in order, so waiting after each one would only stall the CPU."""
    stream = backend.CudaStream()
    first = backend.TensorRTEngine(tmp_path / "first.engine", ["x"], ["y"], False, stream)
    second = backend.TensorRTEngine(tmp_path / "second.engine", ["x"], ["y"], False, stream)
    synchronize = fake_cuda.cudaStreamSynchronize
    waits = []
    fake_cuda.cudaStreamSynchronize = lambda handle: waits.append(handle) or synchronize(handle)
    hidden = first.run_device(np.arange(3, dtype=np.float32).reshape(1, 3))
    assert fake_cuda.queues[stream.handle] and not waits
    np.testing.assert_array_equal(second(*hidden), [[0, 4, 8]])
    assert waits == [stream.handle]


def test_a_chain_of_engines_shares_one_stream_and_scratch_and_feeds_the_step_its_own_output(
    tmp_path, fake_cuda, fake_engine
):
    """pi0.5's shape: one step engine, run once per timestep on the noisy actions it returned last."""
    from lerobot.inference.export.engine import ProgramChain

    programs = {
        "step": {"file": "step.engine", "inputs": ["x_t", "timestep"], "outputs": ["x_next"]},
        "actions": {"file": "actions.engine", "inputs": ["x_t"], "outputs": ["action"]},
    }
    engines = backend.load_engines(tmp_path, programs)
    step, actions = fake_engine
    strategy = sys.modules["tensorrt"].ExecutionContextAllocationStrategy.USER_MANAGED
    assert step.strategy == actions.strategy == strategy
    # One scratch the size of the largest engine's, each engine using its own share of it.
    scratch = step.scratch[0]
    assert step.scratch == (scratch, SCRATCH_BYTES["step"])
    assert actions.scratch == (scratch, SCRATCH_BYTES["actions"])
    info = {"inputs": ["noise"], "programs": programs, "steps": ["step"], "timesteps": [1.0, 0.5, 0.25]}
    noise = np.arange(3, dtype=np.float32).reshape(1, 3)
    np.testing.assert_array_equal(ProgramChain(engines, info)(noise), 16 * noise)
    # The engine never reads the buffer it writes: its output is copied into its input on the GPU.
    assert step.addresses["x_t"] != step.addresses["x_next"]
    assert fake_cuda.copies.count(DEVICE_TO_DEVICE) == 2
    assert not any(fake_cuda.queues.values())


def test_engine_keeps_host_inputs_alive_until_the_stream_finishes(
    tmp_path, fake_cuda, fake_engine, monkeypatch
):
    """A queued copy reads its host array later, so the array must outlive the call that queued it."""
    copies = []
    contiguous = np.ascontiguousarray

    def tracked(value, dtype=None):
        result = contiguous(value, dtype=dtype)
        copies.append(weakref.ref(result))
        return result

    monkeypatch.setattr(np, "ascontiguousarray", tracked)
    stream = backend.CudaStream()
    engine = backend.TensorRTEngine(tmp_path / "model.engine", ["x"], ["y"], stream=stream)
    (output,) = engine.run_device(np.ones((1, 3), np.float64))
    gc.collect()
    assert copies and all(ref() is not None for ref in copies)
    np.testing.assert_array_equal(output.numpy(stream), [[2, 2, 2]])
    gc.collect()
    assert all(ref() is None for ref in copies)


def test_engine_refuses_a_gpu_input_of_another_dtype(tmp_path, fake_cuda, fake_engine):
    engine = backend.load_engines(
        tmp_path, {"model": {"file": "model.engine", "inputs": ["x"], "outputs": ["y"]}}
    )
    with pytest.raises(ValueError, match="the engine expects float32"):
        engine["model"].run_device(backend.CudaBuffer((1, 3), np.float16))


def test_engine_refuses_its_own_output_of_another_dtype(tmp_path, fake_cuda, fake_engine, monkeypatch):
    """Copying a float16 output into a float32 input would read past the end of the output."""
    load_engine = backend.load_engine

    def half_output(path):
        engine = load_engine(path)
        engine.get_tensor_dtype = lambda name: "half" if name == "x_next" else "float"
        return engine

    monkeypatch.setattr(backend, "load_engine", half_output)
    monkeypatch.setattr(
        sys.modules["tensorrt"], "nptype", lambda dtype: np.float16 if dtype == "half" else np.float32
    )
    engine = backend.TensorRTEngine(tmp_path / "step.engine", ["x_t"], ["x_next"])
    (x_next,) = engine.run_device(np.ones((1, 3), np.float32))
    with pytest.raises(ValueError, match="the engine expects float32"):
        engine.run_device(x_next)


def test_engine_synchronizes_copies_after_execution_failure(tmp_path, fake_cuda, fake_engine):
    engine = backend.TensorRTEngine(tmp_path / "model.engine", ["x"], ["y"])
    fake_engine[0].execute_async_v3 = lambda stream: False
    with pytest.raises(RuntimeError, match="could not run"):
        engine(np.ones((1, 3), np.float32))
    assert not any(fake_cuda.queues.values())


def test_engine_records_one_cuda_graph_and_replays_it(tmp_path, fake_cuda, fake_engine, monkeypatch):
    queued_before_capture = []
    begin_capture = fake_cuda.cudaStreamBeginCapture
    monkeypatch.setattr(
        fake_cuda,
        "cudaStreamBeginCapture",
        lambda stream, mode: queued_before_capture.append(fake_engine[0].enqueued)
        or begin_capture(stream, mode),
    )
    engine = backend.TensorRTEngine(tmp_path / "model.engine", ["x"], ["y"], cuda_graphs=True)
    np.testing.assert_array_equal(engine(np.ones((1, 3))), [[2, 2, 2]])
    np.testing.assert_array_equal(engine(np.full((1, 3), 3.0)), [[6, 6, 6]])
    # TensorRT updates its state on the first run, so the graph records the second; later runs replay it.
    assert queued_before_capture == [1]
    assert fake_engine[0].enqueued == 2
    del engine
    fake_cuda.cudaGraphExecDestroy.assert_called_once()


def test_engine_graph_refuses_an_input_buffer_it_did_not_record(tmp_path, fake_cuda, fake_engine):
    engine = backend.TensorRTEngine(tmp_path / "model.engine", ["x"], ["y"], cuda_graphs=True)
    engine.run_device(backend.CudaBuffer((1, 3), np.float32))
    with pytest.raises(ValueError, match="same ones every run"):
        engine.run_device(backend.CudaBuffer((1, 3), np.float32))


def test_a_split_denoising_step_runs_with_cuda_graphs(tmp_path, fake_cuda, fake_engine):
    """GR00T's shape: the step is two engines, so the first reads x_t from the noise, then from the last."""
    from lerobot.inference.export.engine import ProgramChain

    programs = {
        "step_0": {"file": "step_0.engine", "inputs": ["x_t", "timestep"], "outputs": ["hidden"]},
        "step_1": {"file": "step_1.engine", "inputs": ["hidden", "x_t"], "outputs": ["x_next"]},
        "actions": {"file": "actions.engine", "inputs": ["x_t"], "outputs": ["action"]},
    }
    engines = backend.load_engines(tmp_path, programs, cuda_graphs=True)
    info = {
        "inputs": ["noise"],
        "programs": programs,
        "steps": ["step_0", "step_1"],
        "timesteps": [1.0, 0.5, 0.25],
    }
    chain = ProgramChain(engines, info)
    noise = np.arange(3, dtype=np.float32).reshape(1, 3)
    for _ in range(2):
        np.testing.assert_array_equal(chain(noise), 16 * noise)


def test_allocator_uses_plain_malloc_and_handles_failure(monkeypatch):
    cuda = SimpleNamespace(
        cudaMalloc=MagicMock(return_value=(0, 4096)),
        cudaFree=MagicMock(return_value=(0,)),
        cudaStreamSynchronize=MagicMock(return_value=(0,)),
    )
    monkeypatch.setattr(backend, "cuda_runtime", lambda: cuda)
    trt = SimpleNamespace(
        IGpuAllocator=type("Allocator", (), {}), Runtime=lambda logger: SimpleNamespace(), Logger=MagicMock()
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(backend, "runtime", functools.cache(backend.runtime.__wrapped__))
    allocator = backend.runtime().gpu_allocator
    assert allocator.allocate_async(512, 256, 0, 123) == 4096
    cuda.cudaMalloc.assert_called_once_with(512)
    assert allocator.deallocate_async(4096, 123)
    cuda.cudaFree.assert_called_once_with(4096)
    assert cuda.cudaStreamSynchronize.call_count == 2
    cuda.cudaMalloc.return_value = (2, 0)
    assert allocator.allocate(512, 256, 0) == 0
    # TensorRT needs the alignment it asks for; a misaligned allocation is returned and refused.
    cuda.cudaMalloc.return_value = (0, 4097)
    assert allocator.allocate(512, 256, 0) == 0
    cuda.cudaFree.assert_called_with(4097)


def test_bfloat16_bindings_use_extension_dtype(monkeypatch):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    trt = SimpleNamespace(bfloat16="bfloat16", nptype=MagicMock(side_effect=TypeError))
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    dtype = backend.numpy_dtype(trt.bfloat16)
    assert dtype == np.dtype(ml_dtypes.bfloat16)
    assert dtype.itemsize == 2
    values = np.asarray([1.25, -2.5], dtype=dtype)
    np.testing.assert_array_equal(values.astype(np.float32), [1.25, -2.5])
    trt.nptype.assert_not_called()


def test_cuda_error_is_not_ignored():
    with pytest.raises(RuntimeError, match="CUDA runtime call failed"):
        backend.check_cuda((2, 0))


@pytest.mark.parametrize("noise_shape", [(1, 4, 3), (4, 3)])
def test_chained_engines_share_scratch_and_keep_the_cache_on_device(tmp_path, monkeypatch, noise_shape):
    from lerobot.inference.export.engine import ProgramChain

    times = []
    cache = object()
    scratch = []

    class Engine:
        scratch_bytes = 64
        input_shape = (3,)

        def __init__(self, path, *args, **kwargs):
            self.name = path.stem
            self.kwargs = kwargs

        def use_scratch(self, buffer):
            scratch.append(buffer)

        def run_device(self, *inputs):
            if self.name == "prefix":
                return (cache,)
            saved, sample, timestep = inputs
            assert saved is cache
            assert timestep.shape == (1,) and timestep.dtype == np.float32
            times.append(timestep.item())
            return (sample * timestep,)

        def __call__(self, *inputs):
            return inputs[0]

        def upload(self, value):
            return value

    monkeypatch.setattr(backend, "TensorRTEngine", Engine)
    monkeypatch.setattr(backend, "CudaStream", lambda: "stream")
    monkeypatch.setattr(backend, "CudaBuffer", lambda *a: object())
    programs = {
        "prefix": {"file": "prefix.engine", "inputs": ["state"], "outputs": ["past_key_0"]},
        "step": {"file": "step.engine", "inputs": ["past_key_0", "x_t", "timestep"], "outputs": ["x_next"]},
        "actions": {"file": "actions.engine", "inputs": ["x_t"], "outputs": ["action"]},
    }
    engines = backend.load_engines(tmp_path, programs)
    assert all(
        engine.kwargs == {"own_scratch": False, "stream": "stream", "cuda_graphs": False}
        for engine in engines.values()
    )
    assert len(scratch) == 3 and all(buffer is scratch[0] for buffer in scratch)
    info = {
        "inputs": ["state", "noise"],
        "programs": programs,
        "steps": ["step"],
        "timesteps": [1.0, 0.75, 0.5],
    }
    result = ProgramChain(engines, info)(np.ones(3), np.ones(noise_shape, dtype=np.float32))
    assert result.shape == noise_shape
    assert times == [1.0, 0.75, 0.5]
    np.testing.assert_allclose(result, 1.0 * 0.75 * 0.5)
