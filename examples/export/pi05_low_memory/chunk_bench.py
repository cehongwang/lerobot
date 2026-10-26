"""Time and check one exported pi0.5 chunk through ExecuTorch's Python runtime with the TensorRT delegate.

    python chunk_bench.py anth <folder>   # Anthony's chain, run by his own ProgramChain (as lerobot-rollout)
    python chunk_bench.py onnx <folder>   # his ONNX-TensorRT engines, the same way
    python chunk_bench.py ours <folder>   # our prefix / step_table / actions, device-resident like his

Prints phase markers ([bench] ...) so measure.py's timeline separates load from steady state, then the
median, p10 and p90 of RUNS chunks per repeat, and the largest action error against the folder's test case.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file


WARMUP, RUNS, REPEATS = (int(os.environ.get(k, d)) for k, d in (("WARMUP", 10), ("RUNS", 100), ("REPEATS", 3)))
flow, folder = sys.argv[1], Path(sys.argv[2])
t0 = time.perf_counter()
case = load_file(folder / "test_case.safetensors")
expected = case.pop("expected_actions").astype(np.float32)


def mark(what):
    print(f"[bench] {what} at {time.perf_counter() - t0:.2f} s", flush=True)


FIELDS = ("MemAvailable", "AnonPages", "Mapped", "Shmem", "Cached", "Unevictable", "Mlocked", "PageTables",
          "SUnreclaim", "KReclaimable", "VmallocUsed", "HugePages_Free")


def meminfo():
    m = {line.split(":")[0]: int(line.split()[1]) for line in open("/proc/meminfo")}
    m["HugePages_Free"] *= m["Hugepagesize"]
    return m


def own_rss():
    return {line.split(":")[0]: int(line.split()[1]) for line in open("/proc/self/status") if line.startswith("Rss")}


at_start = meminfo()


if flow in ("anth", "onnx"):  # an export.json folder, run by Anthony's ProgramChain
    info = json.loads((folder / "export.json").read_text())
    try:  # lerobot's export engine since 2026-10-20
        from lerobot.inference.export.engine import ProgramChain, load_programs

        chain = ProgramChain(load_programs(folder, info), info)
        initialize = chain
    except ImportError:
        from lerobot.inference.export.engine import load_program

        chain = load_program(folder, info)
        initialize = chain.initialize
    inputs = [case[name] for name in info["inputs"]]
    mark("loaded")

    def chunk():
        return chain(*inputs)

    first = initialize(*inputs)
else:
    import torch_tensorrt_executorch_runtime  # noqa: F401  (registers TensorRTBackend)
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    # The programs run one after another, so their engines can share one activation scratch.
    runtime.backend_registry.set_option("TensorRTBackend", {"use_shared_activation_scratch": True})

    class Program:
        """A .pte's forward method, given and giving torch tensors where they are."""

        def __init__(self, path):
            self._method = runtime.load_program(path).load_method("forward")

        def run(self, *inputs):
            # The delegate runs on its own CUDA stream, so the PyTorch work that wrote the inputs must finish.
            torch.cuda.current_stream().synchronize()
            return tuple(o if isinstance(o, torch.Tensor) else torch.from_dlpack(o)
                         for o in self._method.execute(list(inputs)))

    prefix, step = Program(folder / "prefix.pte"), Program(folder / "step_table.pte")
    # actions.pte is an empty program here (this checkpoint's postprocessor is the identity): its one
    # input, on the host, is its output. So it runs with host I/O, which is the copy Anthony's chain
    # also makes when it hands the actions back as NumPy.
    actions = Program(folder / "actions.pte")
    mark("loaded")
    meta = prefix._method.metadata
    dtypes = {6: torch.float32, 15: torch.bfloat16, 4: torch.int64, 11: torch.bool, 3: torch.int32, 0: torch.uint8}

    def bin_input(path, tensor_meta):
        dtype = dtypes[int(tensor_meta.dtype())]
        raw = np.fromfile(path, dtype=np.uint8)
        return torch.from_numpy(raw).view(dtype).reshape(tuple(tensor_meta.sizes())).cuda()

    prefix_in = [bin_input(folder / "inputs_prefix" / f"in{i}.bin", meta.input_tensor_meta(i))
                 for i in range(meta.num_inputs())]
    smeta = step._method.metadata
    n_cache = smeta.num_inputs() - 2
    x_meta, hot_meta = smeta.input_tensor_meta(n_cache), smeta.input_tensor_meta(n_cache + 1)
    steps = int(np.prod(hot_meta.sizes()))
    noise = bin_input(folder / "inputs_actions" / "in0.bin", actions._method.metadata.input_tensor_meta(0))
    hots = [torch.nn.functional.one_hot(torch.tensor(k), steps).to(dtypes[int(hot_meta.dtype())])
            .reshape(tuple(hot_meta.sizes())).cuda() for k in range(steps)]
    x_dtype = dtypes[int(x_meta.dtype())]

    def chunk():
        # The prefix returns prefix_pad_masks (968 bools) on the host: its program ends in et_copy::_d2h_copy.
        cache = [t.cuda(non_blocking=True) for t in prefix.run(*prefix_in)]
        x = noise.to(x_dtype)
        for hot in hots:
            (v,) = step.run(*cache, x, hot)
            x = (x.float() + (-1.0 / steps) * v.float()).to(x_dtype)  # anth_step_export's Euler update
        (action,) = actions.run(x.cpu())
        return action.float().numpy()

    first = chunk()
mark("first chunk")
error = float(np.abs(np.asarray(first, dtype=np.float32).reshape(expected.shape) - expected).max())
for _ in range(WARMUP):
    chunk()
mark("warm")
medians = []
for r in range(REPEATS):
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        chunk()  # returns host actions, so the GPU work is finished
        times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    medians.append(statistics.median(times))
    print(f"[bench] repeat {r}: median {medians[-1]:.1f} ms, p10 {times[RUNS // 10]:.1f}, "
          f"p90 {times[RUNS * 9 // 10]:.1f}, min {times[0]:.1f}", flush=True)
mark("timed")
now = meminfo()
print("[mem] change since start, MiB: " + ", ".join(f"{k} {(now[k] - at_start[k]) / 1024:+.0f}" for k in FIELDS)
      + "; own " + ", ".join(f"{k} {v / 1024:.0f}" for k, v in own_rss().items()), flush=True)
print(f"[result] {flow}: median of {REPEATS} x {RUNS} chunks {statistics.median(medians):.1f} ms, "
      f"max action error vs test case {error:.4f}", flush=True)
