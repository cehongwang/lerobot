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

"""Run an exported policy's TensorRT engine with CUDA buffers and NumPy inputs."""

import functools
from pathlib import Path

import numpy as np

# Limit the host copy while TensorRT reads a large engine.
READ_CHUNK_BYTES = 64 << 20


@functools.cache
def cuda_runtime():
    """The CUDA runtime bindings, imported on first use so the module loads without them."""
    from cuda.bindings import runtime

    return runtime


def check_cuda(result):
    """The value a CUDA runtime call returns, or an error if its status is not success."""
    status, *values = result
    if status != 0:
        raise RuntimeError(f"CUDA runtime call failed: {status}")
    return values[0] if values else None


class CudaStream:
    def __init__(self):
        self.handle = check_cuda(cuda_runtime().cudaStreamCreate())

    def synchronize(self):
        check_cuda(cuda_runtime().cudaStreamSynchronize(self.handle))

    def __del__(self):
        if getattr(self, "handle", None) is not None:
            cuda_runtime().cudaStreamDestroy(self.handle)


class CudaBuffer:
    """Own a fixed-shape device allocation, including buffers passed between engines."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        if any(size < 0 for size in self.shape):
            raise ValueError(f"Exported TensorRT engines require fixed shapes, got {self.shape}")
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.ptr = check_cuda(cuda_runtime().cudaMalloc(max(self.nbytes, 1)))

    def numpy(self, stream: CudaStream) -> np.ndarray:
        value = np.empty(self.shape, self.dtype)
        cuda = cuda_runtime()
        check_cuda(
            cuda.cudaMemcpyAsync(
                value.ctypes.data,
                self.ptr,
                self.nbytes,
                cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream.handle,
            )
        )
        stream.synchronize()
        return value

    def __del__(self):
        if getattr(self, "ptr", 0):
            cuda_runtime().cudaFree(self.ptr)


@functools.cache
def runtime():
    """Use plain allocations so large engines fit without a stream-ordered memory pool."""
    import tensorrt as trt

    class CudaAllocator(trt.IGpuAllocator):
        def __init__(self):
            trt.IGpuAllocator.__init__(self)

        def allocate(self, size: int, alignment: int, flags: int) -> int:
            status, pointer = cuda_runtime().cudaMalloc(size)
            if status != 0:
                return 0
            if alignment and pointer % alignment:
                cuda_runtime().cudaFree(pointer)
                return 0
            return pointer

        def allocate_async(self, size: int, alignment: int, flags: int, stream: int) -> int:
            if cuda_runtime().cudaStreamSynchronize(stream)[0] != 0:
                return 0
            return self.allocate(size, alignment, flags)

        def deallocate(self, memory: int) -> bool:
            return cuda_runtime().cudaFree(memory)[0] == 0

        def deallocate_async(self, memory: int, stream: int) -> bool:
            if cuda_runtime().cudaStreamSynchronize(stream)[0] != 0:
                return False
            return self.deallocate(memory)

    result = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    result.gpu_allocator = CudaAllocator()
    return result


def load_engine(path: Path):
    """Deserialize an engine without holding another complete copy on the host."""
    import tensorrt as trt

    class FileReader(trt.IStreamReaderV2):
        def __init__(self, file):
            trt.IStreamReaderV2.__init__(self)
            self._file = file

        def read(self, num_bytes: int, stream: int) -> bytes:
            return self._file.read(min(num_bytes, READ_CHUNK_BYTES))

        def seek(self, offset: int, where: trt.SeekPosition) -> bool:
            whence = {trt.SeekPosition.SET: 0, trt.SeekPosition.CUR: 1, trt.SeekPosition.END: 2}[where]
            self._file.seek(offset, whence)
            return True

    with path.open("rb") as file:
        engine = runtime().deserialize_cuda_engine(FileReader(file))
    if engine is None:
        raise RuntimeError(f"TensorRT could not load {path}")
    return engine


def numpy_dtype(dtype):
    """NumPy needs an extension dtype for TensorRT's bfloat16 bindings."""
    import tensorrt as trt

    if dtype == trt.bfloat16:
        from ml_dtypes import bfloat16

        return np.dtype(bfloat16)
    return np.dtype(trt.nptype(dtype))


class TensorRTEngine:
    """Run a fixed-shape engine; related engines can share a stream and scratch allocation.

    `run_device` keeps the outputs in device buffers, which another engine on the same stream can read.
    With `cuda_graphs`, the first run records the engine's work in a CUDA graph and later runs replay it,
    so every run must pass the same input buffers.
    """

    def __init__(
        self,
        path: Path,
        input_names: list[str],
        output_names: list[str],
        own_scratch: bool = True,
        stream: CudaStream | None = None,
        cuda_graphs: bool = False,
    ):
        import tensorrt as trt

        self._engine = load_engine(path)
        strategy = trt.ExecutionContextAllocationStrategy
        self._context = self._engine.create_execution_context(
            strategy.STATIC if own_scratch else strategy.USER_MANAGED
        )
        if self._context is None:
            raise RuntimeError(f"TensorRT could not create an execution context for {path}")
        self._stream = stream if stream is not None else CudaStream()
        self._input_names = input_names
        self._input_shapes = [tuple(self._engine.get_tensor_shape(name)) for name in input_names]
        self._input_dtypes = [numpy_dtype(self._engine.get_tensor_dtype(name)) for name in input_names]
        self._inputs: dict[str, CudaBuffer] = {}
        self._outputs = [
            CudaBuffer(self._engine.get_tensor_shape(name), numpy_dtype(self._engine.get_tensor_dtype(name)))
            for name in output_names
        ]
        for name, output in zip(output_names, self._outputs, strict=True):
            if not self._context.set_tensor_address(name, output.ptr):
                raise RuntimeError(f"TensorRT could not bind output {name}")
        self._cuda_graphs = cuda_graphs
        self._graph = None
        self._graph_inputs: tuple[int, ...] | None = None

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shapes[0]

    @property
    def scratch_bytes(self) -> int:
        return self._engine.device_memory_size_v2

    def use_scratch(self, scratch: CudaBuffer) -> None:
        if scratch.nbytes < self.scratch_bytes:
            raise ValueError("The scratch allocation is too small")
        self._scratch = scratch
        self._context.set_device_memory(scratch.ptr, self.scratch_bytes)

    def run_device(self, *inputs: np.ndarray | CudaBuffer) -> tuple[CudaBuffer, ...]:
        for name, shape, value in zip(self._input_names, self._input_shapes, inputs, strict=True):
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(value.shape)}; the engine was built for {shape}.")
        cuda = cuda_runtime()
        host_inputs = []
        addresses = []
        try:
            for name, dtype, value in zip(self._input_names, self._input_dtypes, inputs, strict=True):
                if isinstance(value, CudaBuffer):
                    if value.dtype != dtype:
                        raise ValueError(f"{name} has dtype {value.dtype}; the engine expects {dtype}")
                    buffer = value
                else:
                    value = np.ascontiguousarray(value, dtype=dtype)
                    host_inputs.append(value)
                    if name not in self._inputs:
                        self._inputs[name] = CudaBuffer(value.shape, dtype)
                    buffer = self._inputs[name]
                    check_cuda(
                        cuda.cudaMemcpyAsync(
                            buffer.ptr,
                            value.ctypes.data,
                            buffer.nbytes,
                            cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
                            self._stream.handle,
                        )
                    )
                if not self._context.set_tensor_address(name, buffer.ptr):
                    raise RuntimeError(f"TensorRT could not bind input {name}")
                addresses.append(buffer.ptr)
            if self._cuda_graphs:
                self._replay(tuple(addresses))
            elif not self._context.execute_async_v3(self._stream.handle):
                raise RuntimeError("TensorRT could not run the engine.")
        finally:
            # Input arrays must remain alive until their queued copies finish, including on errors.
            self._stream.synchronize()
        return tuple(self._outputs)

    def _replay(self, addresses: tuple[int, ...]) -> None:
        """Launch the engine's CUDA graph, recorded on the first run the way NVIDIA documents for TensorRT."""
        cuda = cuda_runtime()
        stream = self._stream.handle
        if self._graph is None:
            # The first enqueue updates TensorRT's internal state, so the graph records the second one.
            if not self._context.execute_async_v3(stream):
                raise RuntimeError("TensorRT could not run the engine.")
            check_cuda(
                cuda.cudaStreamBeginCapture(stream, cuda.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            )
            recorded = self._context.execute_async_v3(stream)
            graph = check_cuda(cuda.cudaStreamEndCapture(stream))
            if not recorded:
                raise RuntimeError("TensorRT could not record the engine in a CUDA graph.")
            self._graph = check_cuda(cuda.cudaGraphInstantiate(graph, 0))
            check_cuda(cuda.cudaGraphDestroy(graph))
            self._graph_inputs = addresses
        elif addresses != self._graph_inputs:
            raise ValueError(
                "The CUDA graph reads the input buffers it recorded; pass the same ones every run."
            )
        check_cuda(cuda.cudaGraphLaunch(self._graph, stream))

    def __call__(self, *inputs: np.ndarray | CudaBuffer) -> np.ndarray | tuple[np.ndarray, ...]:
        outputs = [value.numpy(self._stream) for value in self.run_device(*inputs)]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def __del__(self):
        # Destroy the graph and the context before releasing their bound buffers and shared scratch.
        if getattr(self, "_graph", None) is not None:
            cuda_runtime().cudaGraphExecDestroy(self._graph)
        self._context = None
        self._engine = None


def load_engines(
    folder: Path, programs: dict[str, dict], cuda_graphs: bool = False
) -> dict[str, TensorRTEngine]:
    """Load a chain's engines on one stream, with one scratch allocation the size of the largest.

    The engines run one after another, so none needs its own scratch.
    """
    stream = CudaStream()
    engines = {
        name: TensorRTEngine(
            folder / program["file"],
            program["inputs"],
            program["outputs"],
            own_scratch=False,
            stream=stream,
            cuda_graphs=cuda_graphs,
        )
        for name, program in programs.items()
    }
    scratch = CudaBuffer((max(engine.scratch_bytes for engine in engines.values()),), np.uint8)
    for engine in engines.values():
        engine.use_scratch(scratch)
    return engines
