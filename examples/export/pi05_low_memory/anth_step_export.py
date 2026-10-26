"""Anthony's pi0.5 export with the denoising loop rolled up: three ExecuTorch + TensorRT programs.

His recipe's `PI05Export.denoising_programs()` splits the chunk into prefix (cameras and prompt in, KV
cache out), one denoising step, and actions; only his ONNX script uses it. Here each is exported, compiled
and saved to its own .pte, so no engine holds the 10-step loop and the action expert's weights are built
once. Run it through run_anthony.py like his script:

    python run_anthony.py anth_step_export.py --policy.path=... --task=... --output_dir=...

Checks the three engines, looped with LeRobot's euler_integrate as lerobot-rollout's
TensorRTDenoisingLoop does, against the PyTorch policy's chunk, and times one chunk (median of 50
synchronized calls after 10 warmup). UNROLLED=1 compiles his unrolled chunk instead, for comparison;
NOSAVE=1 skips writing the .pte files. CPU_COMPILE=1 (with UNROLLED=1) moves the policy, its
processors and the example inputs to the CPU before torch.export, so export, lowering and TensorRT
network construction run on CPU weights and only the engine build uses the GPU.
"""

import gc
import os
import time

import numpy as np
import torch
import torch_tensorrt
from act_executorch_tensorrt import GPU_RESIDENT
from pi05_recipe import PI05Export, parse_args

from lerobot.policies.common.flow_matching import euler_integrate


def first(out):
    return out[0] if isinstance(out, (tuple, list)) and len(out) == 1 else out


def latency_ms(fn, warmup=10, runs=50):
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)
    lat.sort()
    return lat[len(lat) // 2], lat[int(len(lat) * 0.9)]


def dump_weights(policy_path: str, task: str, path: str) -> None:
    """DUMP_WEIGHTS (one-time, its own process): the policy's weights as PI05Export leaves them (bfloat16
    storage, the task's compact token embedding), keyed by parameter name, so MMAP_WEIGHTS can map them in
    the dtypes the model uses. The policy loads on the CPU, since nothing here runs it; only the prompt's
    tokenizer steps touch the GPU. The file is written one tensor at a time, so the process holds the
    policy once and no copy of the file."""
    from types import SimpleNamespace

    import pi05_recipe

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep
    from lerobot.utils.constants import OBS_STATE
    from torch_tensorrt.dynamo.tools import save_weights_for_mapping

    config = PreTrainedConfig.from_pretrained(policy_path)
    config.device = "cpu"
    policy = PI05Policy.from_pretrained(policy_path, config=config).eval()
    rss_anon("loading the policy on the CPU")
    # What PI05Export.__init__ does to the policy after its eager reference run.
    pi05_recipe.store_in_bfloat16(policy)
    export = SimpleNamespace(policy=policy, policy_path=policy_path, task=task,
                             frame={OBS_STATE: np.zeros(policy.config.input_features[OBS_STATE].shape)})
    preprocessor, _ = pi05_recipe.PI05Export.processors(export)
    tokenizer = next(i for i, s in enumerate(preprocessor.steps) if isinstance(s, TokenizerProcessorStep))
    export.text_steps = PolicyProcessorPipeline(steps=preprocessor.steps[: tokenizer + 1])
    language_model = policy.model.paligemma_with_expert.paligemma.model.language_model
    language_model.embed_tokens = pi05_recipe.CompactEmbedding(
        language_model.embed_tokens, pi05_recipe.PI05Export.prompt_token_ids(export))
    rss_anon("bfloat16 storage and the compact embedding")
    save_weights_for_mapping(policy, path)
    rss_anon("writing the weights")
    print(f"dumped the weights to {path} ({os.path.getsize(path) / 2**30:.2f} GiB)", flush=True)


def map_weights(export: PI05Export, path: str) -> None:
    """MMAP_WEIGHTS: point the policy's weights at zero-copy views of a safetensors file, so a weight is
    resident only while something reads it (TensorRT, during its partition's build). run_anthony.py's
    FREE_WEIGHTS then drops its pages with madvise once the engine is built. Weights the file lacks, or
    whose dtype or shape differ, are copied to the CPU as before."""
    import builtins
    import ctypes
    import json
    import mmap
    import struct

    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64,
              "I32": torch.int32, "BOOL": torch.bool, "U8": torch.uint8}
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header, start = json.loads(f.read(n)), 8 + n
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)  # private: the file is never written
    mapped = copied = checked = 0
    with torch.no_grad():
        for name, t in [*export.policy.named_parameters(remove_duplicate=False),
                        *export.policy.named_buffers(remove_duplicate=False)]:
            if not t.is_cuda:
                continue  # already moved (a tied weight seen under another name)
            meta = header.get(name)
            if meta and dtypes.get(meta["dtype"]) == t.dtype and list(meta["shape"]) == list(t.shape):
                a, b = meta["data_offsets"]
                view = torch.frombuffer(mm, dtype=torch.uint8, count=b - a, offset=start + a).view(t.dtype).view(t.shape)
                if checked < 3:
                    assert torch.equal(t.cpu(), view), f"{name} differs from {path}"
                    checked += 1
                t.data, mapped = view, mapped + (b - a)
            else:
                t.data, copied = t.data.cpu(), copied + t.nbytes
    builtins.MAPPED_WEIGHTS = {"mm": mm, "base": ctypes.addressof(ctypes.c_char.from_buffer(mm)), "size": len(mm)}
    print(f"mapped {mapped / 2**30:.2f} GiB of weights from {path}, copied {copied / 2**30:.2f} GiB", flush=True)


def meta_load() -> None:
    """META_LOAD=1 (with CPU_COMPILE, MMAP_WEIGHTS, REF_CACHE): build the policy on the meta device instead
    of loading the checkpoint, so the load allocates no weights. cpu_compile() then maps every weight, the
    non-persistent buffers included, from the weights-as-loaded file (DUMP_WEIGHTS). The policy's .to("cuda")
    is skipped while it is still on meta."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    def from_pretrained(cls, path, **kwargs):
        config = PreTrainedConfig.from_pretrained(path)
        with torch.device("meta"):
            return cls(config)

    real_to = PI05Policy.to

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        if device is not None and str(device).startswith("cuda") and next(self.parameters()).is_meta:
            return self
        return real_to(self, *args, **kwargs)

    PI05Policy.from_pretrained = classmethod(from_pretrained)
    PI05Policy.to = to


def cpu_compile(export: PI05Export) -> None:
    """Put the chunk his recipe built on the GPU onto the CPU: policy, processors (rebuilt with
    the CPU device step, the tokenizer padded the way PI05Export.processors pads it) and inputs."""
    from pi05_recipe import PI05Chunk

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep

    cpu = {"device_processor": {"device": "cpu"}}
    pre, post = make_pre_post_processors(export.config, pretrained_path=export.policy_path,
                                         preprocessor_overrides=cpu, postprocessor_overrides=cpu)
    tokenizer = next(i for i, s in enumerate(pre.steps) if isinstance(s, TokenizerProcessorStep))
    pre.steps[tokenizer].padding = "max_length"
    if os.environ.get("MMAP_WEIGHTS") and os.environ.get("MMAP_IMPL"):
        # MMAP_IMPL=<path to torch_tensorrt/dynamo/tools/weight_mapping.py>: the library implementation (Torch-TRT
        # PR "map_weights"), loaded from its file because the measured stack predates it.
        import builtins
        import importlib.util

        spec = importlib.util.spec_from_file_location("_weight_mapping", os.environ["MMAP_IMPL"])
        lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib)
        mapping = lib.map_weights(export.policy, os.environ["MMAP_WEIGHTS"], strict=True)
        print(f"map_weights: {mapping.mapped_bytes / 2**30:.2f} GiB mapped, {len(mapping.missing)} missing", flush=True)
        builtins.RELEASE_MAPPED = lib.release_mapped_storage
    elif os.environ.get("MMAP_WEIGHTS"):
        map_weights(export, os.environ["MMAP_WEIGHTS"])
    export.policy.to("cpu")
    export.module = PI05Chunk(export.policy, PolicyProcessorPipeline(steps=pre.steps[tokenizer + 1:]), post,
                              export.input_names)
    export.inputs = tuple(x.cpu() for x in export.inputs)
    # The vision tower's bfloat16 weights rely on CUDA autocast to cast its activations, and lerobot
    # enables it only for CUDA images (PaliGemmaWithExpertModel._vision_autocast); use CPU autocast.
    paligemma = export.policy.model.paligemma_with_expert.paligemma.model
    get_image_features = paligemma.get_image_features

    def autocast_image_features(*a, **kw):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return get_image_features(*a, **kw)

    paligemma.get_image_features = autocast_image_features
    gc.collect()
    torch.cuda.empty_cache()
    on_gpu = sum(p.is_cuda for p in export.policy.parameters()) + sum(b.is_cuda for b in export.policy.buffers())
    print(f"CPU compilation: policy on CPU ({on_gpu} tensors left on GPU)", flush=True)


def raw_trt_latency(engines: dict, programs: dict, export: PI05Export) -> None:
    """RAW_TRT=1: time the chunk on the TensorRT runtime directly, as a C++ runner would drive it: every
    buffer on the GPU, each engine reading its producer's outputs in place (the step engines read the prefix
    engines' KV cache), the 10 steps queued without a sync; then once more as one CUDA graph. A program may
    be several engines (resource partitioning) but no PyTorch ops."""
    import operator

    import tensorrt as trt
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    to_torch = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16,
                trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool, trt.int8: torch.int8,
                trt.uint8: torch.uint8}
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    keep = []  # engines and buffers stay alive for the whole benchmark

    def bind(gm, args):
        """The program's engines, in graph order, with its arguments and every intermediate bound in place;
        returns the contexts, the outputs in the program's output order, and for each argument the
        (context, binding) pairs that read it."""
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        value = dict(zip(placeholders, args))
        readers = {p: [] for p in placeholders}
        ctxs, outputs = [], None
        for node in gm.graph.nodes:
            if node.op == "call_module" and isinstance(gm.get_submodule(node.target), TorchTensorRTModule):
                mod = gm.get_submodule(node.target)
                engine = runtime.deserialize_cuda_engine(mod.serialized_engine)
                ctx = engine.create_execution_context()
                for binding, arg in zip(mod.input_binding_names, node.args):
                    tensor = value[arg]
                    assert tensor.is_cuda and tensor.is_contiguous(), f"{binding}: not a contiguous GPU tensor"
                    ctx.set_tensor_address(binding, tensor.data_ptr())
                    if arg in readers:
                        readers[arg].append((ctx, binding))
                outs = []
                for binding in mod.output_binding_names:
                    out = torch.empty(tuple(ctx.get_tensor_shape(binding)),
                                      dtype=to_torch[engine.get_tensor_dtype(binding)], device="cuda")
                    ctx.set_tensor_address(binding, out.data_ptr())
                    outs.append(out)
                value[node] = outs if len(outs) > 1 else outs[0]
                ctxs.append(ctx)
                keep.extend([engine, ctx, outs])
            elif node.op == "call_function" and node.target is operator.getitem:
                value[node] = value[node.args[0]][node.args[1]]
            elif node.op == "output":
                flat = node.args[0] if isinstance(node.args[0], (tuple, list)) else (node.args[0],)
                outputs = [value[n] for n in flat]
            elif node.op == "call_module" and str(node.target) == "_guards_fn":
                continue
            elif node.op != "placeholder":
                raise AssertionError(f"{node.format_node()}: not a TensorRT engine")
        print(f"RAW_TRT: {len(ctxs)} engine(s)", flush=True)
        return ctxs, outputs, [readers[p] for p in placeholders]

    def enqueue(ctxs):
        s = torch.cuda.current_stream().cuda_stream
        for ctx in ctxs:
            ctx.execute_async_v3(s)

    stream = torch.cuda.Stream()

    def measure(label, chunk, action):
        """The chunk queued without syncs, then as one CUDA graph: latency and action error of each."""
        with torch.cuda.stream(stream), torch.no_grad():
            error = float(np.abs(chunk()[0].float().cpu().numpy() - export.expected_actions).max())
            median, p90 = latency_ms(chunk)
            print(f"raw TensorRT {label}: median {median:.2f} ms, p90 {p90:.2f} ms, max abs action error "
                  f"{error:.4f}", flush=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                chunk()
            graph.replay()
            torch.cuda.synchronize()
            error = float(np.abs(action[0].float().cpu().numpy() - export.expected_actions).max())
            median, p90 = latency_ms(graph.replay)
            print(f"raw TensorRT {label} as one CUDA graph: median {median:.2f} ms, p90 {p90:.2f} ms, "
                  f"max abs action error {error:.4f}", flush=True)

    def graph_ms(fn):
        with torch.cuda.stream(stream), torch.no_grad():
            piece = torch.cuda.CUDAGraph()
            with torch.cuda.graph(piece, stream=stream):
                fn()
            return latency_ms(piece.replay)[0]

    if "model" in engines:
        ctxs, (action,), _ = bind(engines["model"], tuple(x.cuda() for x in export.inputs))

        def chunk():
            enqueue(ctxs)
            return action

        measure("chunk", chunk, action)
        return

    steps = export.config.num_inference_steps
    dt = -1.0 / steps
    prefix_ctxs, cache, _ = bind(engines["prefix"], programs["prefix"][1])
    x_t = export.noise.cuda().contiguous()
    actions_gm = engines["actions"]
    if any(n.op == "call_module" and str(n.target) != "_guards_fn" for n in actions_gm.graph.nodes):
        actions_ctxs, (action,), _ = bind(actions_gm, (x_t,))
    else:  # the actions program is the identity for this checkpoint (no postprocessing left)
        (out,) = [n for n in actions_gm.graph.nodes if n.op == "output"]
        flat = out.args[0] if isinstance(out.args[0], (tuple, list)) else (out.args[0],)
        assert [n.op for n in flat] == ["placeholder"], f"actions program returns {flat}"
        actions_ctxs, action = [], x_t
    print(f"raw TensorRT prefix ({len(prefix_ctxs)} engines) as a CUDA graph: median "
          f"{graph_ms(lambda: enqueue(prefix_ctxs)):.2f} ms", flush=True)

    # Each step variant: its program, and its last input per step (the time, or a one-hot step index).
    variants = []
    if "step" in engines:
        variants.append(("step", "chunk", torch.tensor([[1.0 + k * dt] for k in range(steps)], device="cuda")))
    if "step_table" in engines:
        onehots = torch.eye(steps, dtype=programs["step_table"][1][-1].dtype, device="cuda")
        variants.append(("step_table", "chunk, time conditioning looked up", onehots.reshape(steps, 1, steps)))
    for program, label, per_step in variants:
        step_ctxs, (v_t,), step_readers = bind(engines[program], (*cache, x_t, per_step[0]))
        assert step_readers[-1], f"the {program} engines do not read its last input"

        def run_steps(step_ctxs=step_ctxs, v_t=v_t, readers=step_readers[-1], per_step=per_step):
            for k in range(steps):
                for ctx, binding in readers:
                    ctx.set_tensor_address(binding, per_step[k].data_ptr())
                enqueue(step_ctxs)
                x_t.add_(v_t, alpha=dt)

        def chunk(run_steps=run_steps):
            x_t.copy_(export.noise)
            enqueue(prefix_ctxs)
            run_steps()
            enqueue(actions_ctxs)
            return action

        measure(f"{label} ({len(prefix_ctxs)} prefix + {len(step_ctxs)} step engines)", chunk, action)
        print(f"raw TensorRT {steps} steps ({program}) as a CUDA graph: median "
              f"{graph_ms(lambda: (x_t.copy_(export.noise), run_steps())):.2f} ms", flush=True)
    if "step_const_t" in engines:
        const_ctxs, (v_const,), _ = bind(engines["step_const_t"], (*cache, x_t))

        def const_steps():
            x_t.copy_(export.noise)
            for _ in range(steps):
                enqueue(const_ctxs)
                x_t.add_(v_const, alpha=dt)

        print(f"raw TensorRT {steps} steps, timestep folded in, as a CUDA graph: median "
              f"{graph_ms(const_steps):.2f} ms", flush=True)


class _TimeTableStep(torch.nn.Module):
    """STEP_TABLE=1: the denoising step with its time conditioning looked up. The Euler timesteps are fixed,
    so each adaptive norm's modulation, dense(time_mlp(t_k)), is the same for every observation: it is
    computed here for each step and the norm's 1024x3072 dense becomes a (steps x 3072) table applied to a
    one-hot step index, which the step program takes instead of the time (the time MLP is then dead)."""

    def __init__(self, step: torch.nn.Module, noise: torch.Tensor, steps: int):
        super().__init__()
        from lerobot.policies.pi_gemma import PiGemmaRMSNorm

        self.step = step
        model = step.model
        expert = model.paligemma_with_expert.gemma_expert
        self.norms = [m for m in expert.modules() if isinstance(m, PiGemmaRMSNorm) and m.dense is not None]
        self.t0 = torch.ones(1, device=noise.device)
        with torch.no_grad():
            conds = [model.embed_suffix(noise, torch.full((1,), 1.0 - k / steps, device=noise.device))[3]
                     for k in range(steps)]
            tables = [torch.cat([norm.dense(c) for c in conds]) for norm in self.norms]  # (steps, 3 * dim)
        self.lookups = torch.nn.ModuleList()
        for table in tables:
            lookup = torch.nn.Linear(steps, table.shape[1], bias=False, dtype=table.dtype, device=table.device)
            lookup.weight.data = table.T.contiguous()
            self.lookups.append(lookup)
        self.onehots = torch.eye(steps, dtype=tables[0].dtype, device=noise.device).reshape(steps, 1, steps)

    def forward(self, *inputs):
        *cache_and_x_t, onehot = inputs
        model, saved = self.step.model, [(n.dense, n.cond_dim) for n in self.norms]
        embed_suffix = model.embed_suffix
        try:
            model.embed_suffix = lambda x_t, timestep: (*embed_suffix(x_t, self.t0)[:3], onehot)
            for norm, lookup in zip(self.norms, self.lookups):
                norm.dense, norm.cond_dim = lookup, lookup.in_features
            return self.step(*cache_and_x_t, self.t0)
        finally:
            del model.embed_suffix
            for norm, (dense, cond_dim) in zip(self.norms, saved):
                norm.dense, norm.cond_dim = dense, cond_dim


class _ConstTimestep(torch.nn.Module):
    """STEP_CONST_T=1 (diagnosis): the denoising step with its timestep a constant, as in the unrolled
    chunk, so TensorRT folds the time conditioning away."""

    def __init__(self, step: torch.nn.Module, timestep: torch.Tensor):
        super().__init__()
        self.step, self.timestep = step, timestep

    def forward(self, *cache_and_x_t):
        return self.step(*cache_and_x_t, self.timestep)


def rss_anon(stage: str) -> None:
    if os.environ.get("RSS_TRACE") == "1":
        anon = next(int(line.split()[1]) for line in open("/proc/self/status") if line.startswith("RssAnon"))
        print(f"RssAnon after {stage}: {anon / 2**20:.2f} GiB", flush=True)


def host_rss() -> float:
    return int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**30


def use_reference_cache(path: str) -> None:
    """REF_CACHE: compute the PyTorch reference chunk in a separate process (REF_SAVE=1) and load it
    here, so the export process never runs the policy eagerly. Both processes seed the RNGs, so the
    random frame and noise match; their checksums are saved and checked."""
    import random

    import pi05_recipe

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if os.environ.get("REF_SAVE") == "1":
        return
    ref = np.load(path)
    pi05_recipe.PI05Export.rollout_actions = lambda self: ref["actions"]
    real_init = pi05_recipe.PI05Export.__init__

    def init(self, *a, **kw):
        real_init(self, *a, **kw)
        assert np.isclose(float(self.noise.sum()), float(ref["noise_sum"])), "noise differs from REF_CACHE"
        assert np.isclose(sum(float(np.asarray(v, np.float64).sum()) for v in self.frame.values()),
                          float(ref["frame_sum"])), "frame differs from REF_CACHE"

    pi05_recipe.PI05Export.__init__ = init


def main() -> None:
    args = parse_args(__doc__, "executorch_tensorrt")
    rss_anon("imports")
    if os.environ.get("REF_CACHE"):
        use_reference_cache(os.environ["REF_CACHE"])
    if os.environ.get("DUMP_WEIGHTS"):
        dump_weights(args.policy_path, args.task, os.environ["DUMP_WEIGHTS"])
        return
    if os.environ.get("META_LOAD") == "1":
        meta_load()
    export = PI05Export(args.policy_path, args.task, args.output_dir, args.job_name)
    rss_anon("PI05Export (load, eager reference, bf16 storage)")
    if os.environ.get("REF_SAVE") == "1":
        np.savez(os.environ["REF_CACHE"], actions=np.asarray(export.expected_actions),
                 noise_sum=float(export.noise.sum()),
                 frame_sum=sum(float(np.asarray(v, np.float64).sum()) for v in export.frame.values()))
        print(f"saved the reference chunk to {os.environ['REF_CACHE']}", flush=True)
        return
    run_inputs = export.inputs
    if os.environ.get("WRITE_BINS") == "1":  # for et_bench: inputs/in<i>.bin raw bytes, expected.bin fp32
        (export.output_dir / "inputs").mkdir(exist_ok=True)
        for i, (name, x) in enumerate(zip(export.input_names, export.inputs)):
            raw = x.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
            (export.output_dir / "inputs" / f"in{i}.bin").write_bytes(raw)
            (export.output_dir / "inputs" / f"{name}.bin").write_bytes(raw)  # trt_bench: by binding name
        (export.output_dir / "expected.bin").write_bytes(np.asarray(export.expected_actions, np.float32).tobytes())
        if os.environ.get("BINS_ONLY") == "1":
            print(f"wrote inputs for {export.input_names} and expected.bin to {export.output_dir}", flush=True)
            return
    if os.environ.get("EXPORT_DIFF") == "1":  # what CPU compilation changes in the traced graph
        import collections

        def histogram(ep):
            out = collections.Counter()
            for n in ep.graph.nodes:
                if n.op == "call_function":
                    val = n.meta.get("val")
                    out[(str(n.target), str(getattr(val, "dtype", "")))] += 1
            return out

        with torch.no_grad():
            on_gpu = histogram(torch.export.export(export.module, export.inputs))
            cpu_compile(export)
            on_cpu = histogram(torch.export.export(export.module, export.inputs))
        diff = {k: (on_gpu[k], on_cpu[k]) for k in on_gpu.keys() | on_cpu.keys() if on_gpu[k] != on_cpu[k]}
        print(f"EXPORT_DIFF: {sum(on_gpu.values())} ops on the GPU, {sum(on_cpu.values())} on the CPU; "
              f"{len(diff)} (op, dtype) counts differ", flush=True)
        for (op, dtype), (g, c) in sorted(diff.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]))[:25]:
            print(f"EXPORT_DIFF {op} {dtype}: gpu {g} cpu {c}", flush=True)
        return
    if os.environ.get("CPU_COMPILE") == "1":
        cpu_compile(export)
        export.noise = export.noise.cpu()  # the step programs' x_t example
        rss_anon("moving the chunk to the CPU")
    if os.environ.get("UNROLLED") == "1":
        programs = {"model": (export.module, export.inputs, export.input_names, ["action"])}
    else:
        if os.environ.get("FAKE_CACHE_EXAMPLE") == "1":
            # denoising_programs runs the prefix eagerly only for the step program's KV-cache examples,
            # reading every prefix weight; their shapes are all that export and compile need.
            import pi05_recipe
            from torch._subclasses.fake_tensor import FakeTensorMode

            real_forward = pi05_recipe.PI05Prefix.forward

            def shapes_only(self, *observation):
                with FakeTensorMode(allow_non_fake_inputs=True) as mode:
                    fake = real_forward(self, *(mode.from_tensor(x) for x in observation))
                return tuple(torch.zeros(t.shape, dtype=t.dtype, device=t.device.type) for t in fake)

            pi05_recipe.PI05Prefix.forward = shapes_only
            programs = export.denoising_programs()
            pi05_recipe.PI05Prefix.forward = real_forward
        else:
            programs = export.denoising_programs()
        step_m, step_in, names, outs = programs["step"]
        if os.environ.get("STEP_CONST_T") == "1":
            programs["step_const_t"] = (_ConstTimestep(step_m, step_in[-1]), step_in[:-1], names[:-1], outs)
        if os.environ.get("CPU_COMPILE") == "1":  # denoising_programs makes its timestep example on CUDA
            step_in = tuple(x.cpu() for x in step_in)
            programs["step"] = (step_m, step_in, names, outs)
        if os.environ.get("STEP_TABLE") in ("1", "only"):
            table_step = _TimeTableStep(step_m, export.noise, export.config.num_inference_steps)
            programs["step_table"] = (table_step, (*step_in[:-1], table_step.onehots[0]), [*names[:-1], "step"],
                                      outs)
            if os.environ.get("STEP_TABLE") == "only":
                del programs["step"]

    start = time.perf_counter()
    with torch.no_grad():
        exported = {name: torch.export.export(m, inputs) for name, (m, inputs, _, _) in programs.items()}
    rss_anon(f"torch.export ({sum(len(ep.graph.nodes) for ep in exported.values())} graph nodes)")
    if os.environ.get("MMAP_IMPL") and "model" not in programs:
        # denoising_programs runs the prefix eagerly for its KV-cache examples, paging every prefix weight
        # in; drop those pages again (TensorRT re-reads each weight while it builds that weight's engine).
        import builtins

        tensors = {id(t): t for t in [*export.policy.parameters(), *export.policy.buffers()]}
        before = host_rss()
        dropped = sum(builtins.RELEASE_MAPPED(t.untyped_storage()) for t in tensors.values())
        gc.collect()
        print(f"dropped the pages of {dropped} mapped weights: host RSS {before:.2f} -> {host_rss():.2f} GiB",
              flush=True)
    if os.environ.get("TRIM") == "1":
        # torch.export traced the processors into the graph; the chunk module, its processors and the
        # tokenizer pipeline are no longer needed (the policy stays: FREE_WEIGHTS finds its weights).
        from types import SimpleNamespace

        from smolvla_recipe import TEXT_STEPS

        before = host_rss()
        # lerobot-rollout tokenizes the prompt with these steps, so write them beside the .pte now;
        # export.write() then finds them already saved.
        export.text_steps.save_pretrained(export.output_dir, config_filename=TEXT_STEPS)
        export.text_steps = SimpleNamespace(save_pretrained=lambda *args, **kwargs: None)
        programs = {name: (None, *rest) for name, (_, *rest) in programs.items()}
        export.module = None
        gc.collect()
        print(f"dropped the chunk module, processors and tokenizer: host RSS {before:.2f} -> {host_rss():.2f} GiB",
              flush=True)
    # truncate_double: with the time an input, the step's sinusoidal time embedding keeps its float64
    # frequencies (vla_utils.create_sinusoidal_pos_embedding); TensorRT computes it in float32.
    engines = {}
    for name in list(exported):
        # CAP_<PROGRAM>=<GiB>: this program's per-engine cap (run_anthony.py reads ENGINE_CAP_GB at compile).
        if os.environ.get(f"CAP_{name.upper()}"):
            os.environ["ENGINE_CAP_GB"] = os.environ[f"CAP_{name.upper()}"]
        engines[name] = torch_tensorrt.dynamo.compile(exported.pop(name), arg_inputs=programs[name][1],
                                                      min_block_size=1, truncate_double=True)
    print(f"compiled {', '.join(engines)} in {time.perf_counter() - start:.0f} s", flush=True)
    if os.environ.get("ENGINE_SIZES") == "1":
        for name, gm in engines.items():
            for sub, m in gm.named_children():
                if "_run_on_acc" in sub:
                    import hashlib

                    sha = hashlib.sha256(m.serialized_engine).hexdigest()[:16]
                    print(f"engine size {name}/{sub}: {len(m.serialized_engine) / 2**20:.0f} MiB sha {sha}",
                          flush=True)

    if os.environ.get("DROP_ENGINES") == "1":
        print("TRT vs PyTorch chunk: skipped, engines were dropped after their build (DROP_ENGINES)", flush=True)
        return
    if "model" in engines:
        def chunk():
            return first(engines["model"](*run_inputs))
    elif "step" in engines:
        def chunk():
            cache = engines["prefix"](*programs["prefix"][1])
            x_0 = euler_integrate(lambda x_t, t: first(engines["step"](*cache, x_t, t)), export.noise.cuda(),
                                  export.config.num_inference_steps)
            return first(engines["actions"](x_0))
    else:  # STEP_TABLE=only: the same Euler loop, the step indexed by a one-hot instead of the time
        steps = export.config.num_inference_steps
        onehots = torch.eye(steps, dtype=programs["step_table"][1][-1].dtype, device="cuda").reshape(steps, 1, steps)

        def chunk():
            cache = engines["prefix"](*programs["prefix"][1])
            x_t = export.noise.cuda()  # CPU_COMPILE leaves the noise on the CPU
            for k in range(steps):
                x_t = x_t + (-1.0 / steps) * first(engines["step_table"](*cache, x_t, onehots[k]))
            return first(engines["actions"](x_t))

    if os.environ.get("NOCHECK") == "1":
        # Save straight from the compiled programs, without loading their engines (with engine_output_dir
        # they stay on disk); the saved .pte files can be checked instead.
        print("TRT vs PyTorch chunk: skipped (NOCHECK)", flush=True)
    else:
        with torch.no_grad():
            actions = chunk()[0].float().cpu().numpy()
            median, p90 = latency_ms(chunk)
        error = float(np.abs(actions - export.expected_actions).max())
        print(f"TRT vs PyTorch chunk: max abs action error {error:.4f} (tolerance {args.tolerance})", flush=True)
        print(f"chunk latency: median {median:.2f} ms, p90 {p90:.2f} ms", flush=True)
    if os.environ.get("RAW_TRT") == "1":
        raw_trt_latency(engines, programs, export)
    if os.environ.get("ENGINE_OUTPUT_DIR"):
        # The check loaded the engines on the GPU; with engine_output_dir the save reads them from their files.
        for gm in engines.values():
            for m in gm.modules():
                if getattr(m, "serialized_engine_path", None):
                    m.engine = None
        gc.collect()
        torch.cuda.empty_cache()
    if os.environ.get("NOSAVE") == "1":
        return

    # Free the PyTorch policy before saving, as his script does (release_policy after compile).
    programs = {name: (None, *rest) for name, (_, *rest) in programs.items()}
    del chunk
    export.release_policy()
    files = {}
    for name, (_, inputs, input_names, output_names) in programs.items():
        path = export.output_dir / f"{name}.pte"
        torch_tensorrt.save(engines.pop(name), str(path), output_format="executorch", retrace=False,
                            arg_inputs=inputs, backend_config=GPU_RESIDENT,
                            **({"use_cuda_graphs": True} if os.environ.get("CUDA_GRAPHS") == "1" else {}))
        files[name] = {"file": path.name, "inputs": input_names, "outputs": output_names,
                       "bytes": path.stat().st_size}
        if os.environ.get("RELEASE_AFTER_SAVE") == "1":  # what Torch-TensorRT #4865 does inside save()
            import ctypes

            from executorch.exir.backend.utils import is_same_node
            from torch._subclasses.fake_tensor import FakeTensorMode

            before = host_rss()
            # Both caches keep the lowered program, and so every engine blob, alive after the save.
            is_same_node.cache_clear()
            FakeTensorMode.cache_clear()
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            print(f"after saving {name}: host RSS {before:.2f} -> {host_rss():.2f} GiB "
                  "(is_same_node + FakeTensorMode caches, gc, malloc_trim)", flush=True)
        if os.environ.get("WRITE_PROGRAM_BINS") == "1":  # et_bench inputs per program: inputs_<name>/in<i>.bin
            (export.output_dir / f"inputs_{name}").mkdir(exist_ok=True)
            for i, x in enumerate(inputs):
                raw = x.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
                (export.output_dir / f"inputs_{name}" / f"in{i}.bin").write_bytes(raw)
        print(f"saved {path} ({path.stat().st_size / 1e9:.2f} GB)", flush=True)
    print(f"Exported {export.output_dir} in {time.perf_counter() - start:.0f} s", flush=True)
    export.write("executorch_tensorrt", files, args.tolerance)


if __name__ == "__main__":
    main()
