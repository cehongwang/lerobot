"""Run one of Anthony's export scripts (pi0.5, GR00T) unchanged, with Torch-TensorRT's INFO logs for phase
markers and host RSS printed right before compile. Switches (environment variables):

Partitioner and build
  BUDGET_GB=<n>            resource partitioner with this absolute RSS budget (+ clean, plan release)
  BUDGET_HEADROOM_GB=<n>   budget = host RSS at compile + n (keeps the per-engine cap across configurations)
  ENGINE_CAP_GB=<n>        pin the per-engine weight cap directly, independent of what is resident
  PATCH_4783=1             pytorch/TensorRT#4783 (getitem with its producer) for builds that predate it
  CLEAN=0 / RELEASE=0      skip the clean before compile / the plan-release monkeypatch
  OPT_LEVEL=<n>            TensorRT builder optimization level; TIMING_CACHE=<file> per run
  OFFLOAD=1, DYN_RES=1     offload_module_to_cpu, dynamically_allocate_resources
Memory prototypes
  FREE_WEIGHTS=1           free never-read weights at partition time and each partition's after its engine
  DROP_ENGINES=1           measurement only: stream each plan to /dev/null and drop the engine (no .pte)
  ENGINE_OUTPUT_DIR=<dir>  Torch-TensorRT's engine_output_dir: each engine written to <dir> and kept there
  TRIM=1                   empty the ExportedProgram graphs after lowering (anth_step_export also trims)
  RSS_BREAKDOWN=1          print resident memory by mapping right before compile, then exit
ExecuTorch
  ET_SHIM=0                ExecuTorch 1.6 (env-2.15): keep his GPU-resident config instead of shimming it
  ET_DEVICE_INTERMEDIATES=1  plan tensors between delegates on the GPU, method inputs/outputs on the host
GR00T (Isaac checkpoints)
  GROOT_EMBODIMENT=<tag>   load with strict=False and this embodiment tag; identity postprocessor
  GROOT_CPU_COMPILE=1      move the chunk to the CPU before export; GROOT_DUMP / GROOT_MMAP=<file>
  GROOT_WRITE_BINS=<dir>   et_bench inputs and the eager reference, then exit
"""
import logging, os, runpy, sys

logging.basicConfig(format="%(name)s: %(message)s")
logging.getLogger("torch_tensorrt").setLevel(logging.INFO)
logging.getLogger().handlers[0].addFilter(
    lambda r: not r.getMessage().startswith(("Converted node", "skip broadcast")))

# Our ExecuTorch (1.3.1 + overlay) predates the GPU-resident I/O config and _flatbuffer_to_program
# his branch uses. Drop those so the .pte keeps host I/O (KB-sized tensors), and skip his
# post-save check that asserts GPU-resident I/O. The compile and save paths are unchanged.
import executorch.exir as exir
import executorch.exir._serialize._program as et_program
import executorch.exir.passes.propagate_device_pass as propagate_device_pass

if os.environ.get("ET_SHIM", "1") == "1":  # ET_SHIM=0 on ExecuTorch 1.6 (env-2.15), which has both
    _Config = exir.ExecutorchBackendConfig
    exir.ExecutorchBackendConfig = lambda propagate_device_config=None, enable_non_cpu_memory_planning=False, **kw: _Config(**kw)
    et_program._flatbuffer_to_program = None
    propagate_device_pass.PropagateDeviceConfig = lambda **kw: None
import act_executorch_tensorrt

act_executorch_tensorrt.check_program = lambda path, backend="TensorRTBackend": print(f"skipped check_program({path})")
if os.environ.get("ET_DEVICE_INTERMEDIATES") == "1":
    # Plan the tensors between delegates (engines) in GPU memory, but keep the method's inputs and
    # outputs on the host: isolates the boundary copies from GPU-resident I/O (et_bench feeds host inputs).
    from executorch.exir.passes import MemoryPlanningPass

    act_executorch_tensorrt.GPU_RESIDENT = exir.ExecutorchBackendConfig(
        propagate_device_config=propagate_device_pass.PropagateDeviceConfig(
            skip_h2d_for_method_inputs=False, skip_d2h_for_method_outputs=False),
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True))
if os.environ.get("PATCH_4783") == "1":
    # pytorch/TensorRT#4783 for Torch-TensorRT builds that predate it: keep a boundary node's getitem
    # users with it, so a size cut never ends a partition on a tuple.
    import operator

    from torch_tensorrt.dynamo.partitioning._resource_partitioner import ResourcePartitioner

    _step = ResourcePartitioner.step_if_break_fusion

    def step_if_break_fusion(self, subgraphs, leaf_nodes, first, second):
        moved = False
        for leaf in leaf_nodes:
            for node in [u for u in leaf.users if u.target is operator.getitem]:
                if node not in first and node in second:
                    moved = True
                    first.add(node)
                    second.remove(node)
                    subgraphs[0].nodes.append(node)
                    subgraphs[1].nodes.remove(node)
        return _step(self, subgraphs, leaf_nodes, first, second) or moved

    ResourcePartitioner.step_if_break_fusion = step_if_break_fusion
if os.environ.get("PATCH_REMAT") == "1":
    # validate_and_correct_subgraphs hoists every node whose inputs are ready into the earliest subgraph,
    # so weightless chains that read only program inputs (the attention mask, rotary cos/sin) all land in
    # the first engine and reach every later one as inputs. On Thor (sm_110, TensorRT 11.0-11.4) an engine
    # with Gemma attention fed both that way fails: the fused gemm_mha kernel needs 0x39174 bytes of
    # shared memory, 0x38c00 max. Recompute such chains in each subgraph that uses them instead.
    import torch
    from torch_tensorrt.dynamo.partitioning._resource_partitioner import ResourcePartitioner

    _break = ResourcePartitioner.break_subgraphs
    REMAT_MAX_CONST_BYTES = 64 << 10  # mask fill values, rotary inv_freq; any real weight is far larger
    REMAT_MAX_FOLDED_BYTES = 4 << 20  # constant folding turns the fixed part of the 968x968 mask into one

    def break_subgraphs(self, subgraphs, subgraph_size_budget):
        subgraphs = _break(self, subgraphs, subgraph_size_budget)
        where = {n: i for i, s in enumerate(subgraphs) for n in s.nodes}
        params = self.module.state_dict(keep_vars=True)
        cheap = {}

        def is_cheap(n):  # computable from program inputs and small constants alone
            if n not in cheap:
                if n.op == "placeholder":
                    cheap[n] = True
                elif n.op == "get_attr":
                    w = params.get(n.target)
                    limit = REMAT_MAX_FOLDED_BYTES if n.target.startswith("_frozen_param") else REMAT_MAX_CONST_BYTES
                    cheap[n] = w is None or w.numel() * w.element_size() <= limit
                else:
                    cheap[n] = (n.op == "call_function"
                                and torch.Tag.nondeterministic_seeded not in getattr(n.target, "tags", ())
                                and all(is_cheap(i) for i in n._input_nodes))
            return cheap[n]

        def reads_input(n):
            return any(i.op == "placeholder" for i in n._input_nodes)

        graph, copied = self.module.graph, 0
        order = {n: k for k, n in enumerate(graph.nodes)}  # subgraph lists aren't in graph order
        for j, sub in enumerate(subgraphs):
            copies = {}

            def copy(n, before):
                # A node reading a program input stays where it is and crosses as a small tensor (the
                # bool padding mask): GPU-resident ExecuTorch inputs (skip_h2d) allow one user each.
                if n.op in ("placeholder", "get_attr") or where.get(n) == j or reads_input(n):
                    return n
                if n not in copies:
                    args = {i: copy(i, before) for i in n._input_nodes}
                    with graph.inserting_before(before):
                        copies[n] = graph.node_copy(n, lambda i: args[i])
                    sub.nodes.insert(sub.nodes.index(before), copies[n])
                return copies[n]

            for user in sorted(sub.nodes, key=order.__getitem__):
                for n in list(user._input_nodes):
                    if where.get(n, j) < j and is_cheap(n) and not reads_input(n):
                        user.replace_input_with(n, copy(n, user))
                        copied += 1
        for sub in subgraphs:  # the hoisted originals nothing reads any more
            for n in reversed(list(sub.nodes)):
                if not n.users and n.op == "call_function" and n in cheap and cheap[n]:
                    sub.nodes.remove(n)
                    graph.erase_node(n)
        print(f"[remat] recomputed {copied} input-only inputs in their own subgraphs", flush=True)
        return [s for s in subgraphs if s.nodes]

    ResourcePartitioner.break_subgraphs = break_subgraphs
if os.environ.get("PATCH_FIX_B") == "1":
    sys.exit("PATCH_FIX_B was dropped on 2026-10-13: the rolled-up export doesn't need it (docs/pi05_export_memory.md).")
if os.environ.get("ENGINE_CAP_GB"):
    from torch_tensorrt.dynamo.partitioning._resource_partitioner import ResourcePartitioner

    # The per-engine weight cap directly, (budget - RSS at partition) / 4 otherwise: a split then
    # depends only on the graph and the cap, not on what else is resident (CPU or GPU weights, mmap).
    _init = ResourcePartitioner.__init__

    def init(self, *args, **kwargs):
        _init(self, *args, **kwargs)
        self.remaining_memory_budget = int(4 * float(os.environ["ENGINE_CAP_GB"]) * 2**30)

    ResourcePartitioner.__init__ = init

# With BUDGET_GB, CLEAN=0 skips the clean (to measure a Torch-TensorRT that cleans itself) and
# RELEASE=0 skips the plan-release monkeypatch (a Torch-TensorRT with #4795 releases natively).
# TIMING_CACHE gives concurrent runs separate timing caches; OPT_LEVEL sets TensorRT's builder
# optimization level (default 3); OFFLOAD=1 sets offload_module_to_cpu; DYN_RES=1 sets
# dynamically_allocate_resources; FREE_WEIGHTS=1 (and DROP_ENGINES=1), see free_weights_after_each_engine.
import torch_tensorrt
from torch_tensorrt.dynamo.utils import release_host_and_device_memory

budget = os.environ.get("BUDGET_GB")
clean = bool(budget) and os.environ.get("CLEAN", "1") == "1"
if budget and os.environ.get("RELEASE", "1") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pi05 import release_engine_plans

    release_engine_plans()
real_compile = torch_tensorrt.dynamo.compile


def trace_save_steps():
    """SAVE_TRACE=1: print anonymous and file-backed RSS around each step of the ExecuTorch save."""
    import functools

    def rss(tag):
        s = {l.split(":")[0]: int(l.split()[1]) / 2**20 for l in open("/proc/self/status") if l.startswith("Rss")}
        print(f"[save] {tag}: anon {s['RssAnon']:.2f} file {s['RssFile']:.2f} shmem {s['RssShmem']:.2f} GiB", flush=True)

    def wrap(owner, name, label):
        real = getattr(owner, name)
        fn = real.__func__ if isinstance(real, staticmethod) else real

        @functools.wraps(fn)
        def traced(*a, **kw):
            rss(f"{label} start")
            out = fn(*a, **kw)
            rss(f"{label} end")
            return out

        setattr(owner, name, staticmethod(traced) if isinstance(owner.__dict__.get(name), staticmethod) else traced)

    import executorch.exir.program._program as et_program_mod
    import torch_tensorrt.executorch as tte
    import torch_tensorrt.executorch.backend as ttb

    wrap(tte, "export", "torch_tensorrt.executorch.export (to_edge + lower)")
    wrap(ttb.TensorRTBackend, "preprocess", "  backend preprocess")
    wrap(et_program_mod.EdgeProgramManager, "to_executorch", "to_executorch")
    wrap(et_program_mod.ExecutorchProgramManager, "write_to_file", "write_to_file")


if os.environ.get("SAVE_TRACE") == "1":
    trace_save_steps()


def rss_gib():
    return int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**30


def rss_breakdown(program):
    """Resident memory right before compile, by mapping, plus the program's CPU weights; then exit."""
    import collections

    import torch

    by_map, path = collections.Counter(), "[anon]"
    for line in open("/proc/self/smaps"):
        head = line.split()
        if len(head) >= 5 and "-" in head[0] and ":" in head[3]:  # mapping header
            path = head[5] if len(head) > 5 else "[anon]"
        elif head[0] == "Rss:":
            by_map[path] += int(head[1]) * 1024
    status = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/self/status") if l.startswith("Rss")}
    seen, weights = set(), 0
    for t in [*program.state_dict.values(), *program.constants.values()]:
        if isinstance(t, torch.Tensor) and not t.is_cuda and t.untyped_storage().data_ptr() not in seen:
            seen.add(t.untyped_storage().data_ptr())
            weights += t.untyped_storage().nbytes()
    gib = lambda b: f"{b / 2**30:.2f} GiB"
    print("RSS breakdown: " + ", ".join(f"{k} {gib(v)}" for k, v in status.items()))
    print(f"  program weights on CPU (unique storages): {gib(weights)}")
    for p, b in by_map.most_common(15):
        print(f"  {gib(b):>10}  {p}")
    release_host_and_device_memory()
    status = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/self/status") if l.startswith("Rss")}
    print("after release_host_and_device_memory (gc, empty_cache, malloc_trim): "
          + ", ".join(f"{k} {gib(v)}" for k, v in status.items()))
    sys.stdout.flush()
    os._exit(0)


_compiled_programs = set()  # ids of the ExportedPrograms compile() was given (TRIM empties only these)


def compile(*args, **kwargs):
    if args:
        _compiled_programs.add(id(args[0]))
    if clean:
        release_host_and_device_memory()
    if os.environ.get("RSS_BREAKDOWN") == "1":
        rss_breakdown(args[0])
    print(f"host RSS before compile{' (cleaned)' if clean else ''}: {rss_gib():.1f} GiB", flush=True)
    kwargs["timing_cache_path"] = os.environ.get(
        "TIMING_CACHE", f"/tmp/torch_tensorrt_engine_cache/timing_cache_budget{budget}.bin")
    cap = budget or (1024 if os.environ.get("ENGINE_CAP_GB") else None)  # ENGINE_CAP_GB overrides it
    if os.environ.get("BUDGET_HEADROOM_GB"):  # the budget caps absolute RSS; keep the cap per engine fixed
        cap = rss_gib() + float(os.environ["BUDGET_HEADROOM_GB"])
        print(f"budget = host RSS + headroom = {cap:.2f} GiB", flush=True)
    if cap:
        kwargs.update(enable_resource_partitioning=True, cpu_memory_budget=int(float(cap) * 2**30))
    if os.environ.get("OPT_LEVEL"):
        kwargs["optimization_level"] = int(os.environ["OPT_LEVEL"])
    if os.environ.get("OFFLOAD") == "1":
        kwargs["offload_module_to_cpu"] = True
    if os.environ.get("DYN_RES") == "1":
        kwargs["dynamically_allocate_resources"] = True
    if os.environ.get("ENGINE_OUTPUT_DIR"):  # Torch-TensorRT engine_output_dir: engines stay on disk
        kwargs["engine_output_dir"] = os.environ["ENGINE_OUTPUT_DIR"]
    return real_compile(*args, **kwargs)


torch_tensorrt.dynamo.compile = compile


def free_weights_after_each_engine():
    """FREE_WEIGHTS=1: once a partition's engine is built, free the CPU storage of every weight that
    no partition still to be built uses. Torch-TensorRT already does this for constant folding's
    _frozen_param tensors (_conversion.py, after interpreter.run()); this extends it to the model's own
    parameters, which the ExportedProgram, the graph modules and the policy all still reference, so
    it frees the storage itself. Weights read by a PyTorch-run partition are never freed."""
    import collections

    import torch
    import torch_tensorrt.dynamo._compiler as compiler

    uses = collections.Counter()
    real_partition, real_convert = compiler.resource_partition, compiler.convert_module

    def release(storage):
        """Drop a weight's memory: madvise its pages away if it is a view of the MMAP_WEIGHTS file
        (anth_step_export.map_weights; re-read from the file if touched again), else resize it to 0."""
        import builtins
        import mmap

        lib_release = getattr(builtins, "RELEASE_MAPPED", None)  # anth_step_export MMAP_IMPL
        if lib_release is not None and lib_release(storage):
            return
        m = getattr(builtins, "MAPPED_WEIGHTS", None)
        ptr, n = storage.data_ptr(), storage.nbytes()
        if m and m["base"] <= ptr < m["base"] + m["size"]:
            lo = (ptr - m["base"]) // mmap.PAGESIZE * mmap.PAGESIZE
            hi = min(m["size"], -(-(ptr - m["base"] + n) // mmap.PAGESIZE) * mmap.PAGESIZE)
            m["mm"].madvise(mmap.MADV_DONTNEED, lo, hi - lo)
        else:
            storage.resize_(0)

    def storages(module):
        found = {}
        for node in module.graph.nodes:
            if node.op == "get_attr":
                t = module
                for part in node.target.split("."):
                    t = getattr(t, part)
                if isinstance(t, torch.Tensor) and not t.is_cuda:
                    found[t.untyped_storage().data_ptr()] = t.untyped_storage()
        return found

    def partition(gm, *args, **kwargs):
        partitioned = real_partition(gm, *args, **kwargs)
        for _, sub in partitioned.named_children():
            if isinstance(sub, torch.fx.GraphModule):
                uses.update(storages(sub).keys())
        free_unused(set(uses) | set(storages(partitioned)))
        if os.environ.get("TRIM") == "1":
            drop_exported_programs()
        return partitioned

    def drop_exported_programs():
        """Lowering is done: compile() passed graph_signature to compile_module and reads the
        ExportedProgram no more, yet it (and this wrapper's args) keep its graph and metadata alive."""
        import gc

        from torch.export import ExportedProgram

        before = rss_gib()
        # Only programs already handed to compile(): with several programs (prefix, step, actions), the
        # ones still to be compiled must keep their graphs.
        eps = [o for o in gc.get_objects() if isinstance(o, ExportedProgram) and id(o) in _compiled_programs]
        for ep in eps:
            ep._graph_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        del eps
        release_host_and_device_memory()
        print(f"dropped the exported programs' graphs: host RSS {before:.2f} -> {rss_gib():.2f} GiB", flush=True)

    def free_unused(keep):
        """Weights the lowered graph never reads (the policy and the ExportedProgram still hold them)."""
        import gc

        unused = {}
        for policy in [o for o in gc.get_objects()
                       if isinstance(o, torch.nn.Module) and type(o).__name__.endswith("Policy")]:
            for name, t in [*policy.named_parameters(remove_duplicate=False),
                            *policy.named_buffers(remove_duplicate=False)]:
                s = t.untyped_storage()
                if not t.is_cuda and s.nbytes() and s.data_ptr() not in keep:
                    unused.setdefault(s.data_ptr(), (name, s))
        top = sorted(unused.values(), key=lambda v: -v[1].nbytes())
        total = sum(s.nbytes() for _, s in top)
        for name, s in top[:8]:
            print(f"  unused weight {name}: {s.nbytes() / 2**20:.0f} MiB", flush=True)
        for _, s in top:
            release(s)
        release_host_and_device_memory()
        print(f"freed {total / 2**30:.2f} GiB of weights the graph never reads ({len(top)} tensors)", flush=True)

    def convert(submodule, *args, **kwargs):
        mine = storages(submodule)
        result = real_convert(submodule, *args, **kwargs)
        freed = 0
        for ptr, storage in mine.items():
            uses[ptr] -= 1
            if uses[ptr] == 0 and storage.nbytes():
                freed += storage.nbytes()
                release(storage)
        if os.environ.get("DROP_ENGINES") == "1":
            # Measurement only: stream the plan out (to /dev/null) and drop the GPU engine, as a save that
            # writes each engine while building would. No .pte, no accuracy check.
            plan = result._plan_buffer()
            with open("/dev/null", "wb") as sink:
                sink.write(memoryview(plan.numpy() if isinstance(plan, torch.Tensor) else plan))
            del plan
            result.engine, result._serialized_engine = None, None
        release_host_and_device_memory()
        print(f"freed {freed / 2**30:.2f} GiB of weights after building {kwargs.get('name')}", flush=True)
        return result

    compiler.resource_partition, compiler.convert_module = partition, convert


if os.environ.get("FREE_WEIGHTS") == "1":
    free_weights_after_each_engine()

if os.environ.get("GROOT_EMBODIMENT"):
    # His GrootExport expects a lerobot-trained GR00T folder. NVIDIA's Isaac-GR00T checkpoints load with
    # strict=False once the embodiment tag names one the checkpoint has statistics for (libero_sim for
    # GR00T-N1.7-LIBERO); lerobot's processors then load from the same folder.
    import groot_recipe
    import torch

    real_from_pretrained = groot_recipe.GrootPolicy.from_pretrained

    def groot_from_pretrained(path, *args, **kwargs):
        policy = real_from_pretrained(pretrained_name_or_path=path, strict=False)
        policy.config.embodiment_tag = os.environ["GROOT_EMBODIMENT"]
        return policy

    groot_recipe.GrootPolicy.from_pretrained = groot_from_pretrained

    # The Isaac folder's postprocessor (processor_groot.py) un-normalizes through numpy, which
    # torch.export can't trace; a lerobot-trained checkpoint brings a traceable one. The program then
    # returns the policy's normalized actions (memory-neutral: a few elementwise ops at the end).
    real_chunk_init = groot_recipe.GrootChunk.__init__

    def groot_chunk_init(self, *args, **kwargs):
        real_chunk_init(self, *args, **kwargs)
        self.postprocessor = torch.nn.Identity()

    groot_recipe.GrootChunk.__init__ = groot_chunk_init

    if os.environ.get("GROOT_WRITE_BINS"):
        # et_bench inputs (in<i>.bin) and the reference output (expected.bin, fp32) from his GrootChunk run
        # eagerly on the same inputs, under its CUDA autocast: the program returns these normalized actions.
        from pathlib import Path

        import numpy as np

        real_bins_init = groot_recipe.GrootExport.__init__

        def groot_bins_init(self, *args, **kwargs):
            real_bins_init(self, *args, **kwargs)
            out = Path(os.environ["GROOT_WRITE_BINS"])
            (out / "inputs").mkdir(parents=True, exist_ok=True)
            for i, x in enumerate(self.inputs):
                (out / "inputs" / f"in{i}.bin").write_bytes(
                    x.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
            with torch.no_grad():
                ref = self.module(*self.inputs)
            (out / "expected.bin").write_bytes(ref.float().cpu().numpy().astype(np.float32).tobytes())
            print(f"wrote {len(self.inputs)} inputs and expected.bin {tuple(ref.shape)} to {out}", flush=True)
            os._exit(0)

        groot_recipe.GrootExport.__init__ = groot_bins_init

    if os.environ.get("GROOT_CPU_COMPILE") == "1":
        # CPU compilation for his GR00T export: after GrootExport is built (on the GPU, as his loader
        # does), move the chunk, the policy and the inputs to the CPU before torch.export, so TensorRT
        # reads the weights in place. GROOT_DUMP=<file> writes the weights as loaded and exits;
        # GROOT_MMAP=<file> maps them first (anth_step_export.dump_weights / map_weights).
        import gc

        real_export_init = groot_recipe.GrootExport.__init__

        def groot_export_init(self, *args, **kwargs):
            real_export_init(self, *args, **kwargs)
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import anth_step_export

            if os.environ.get("GROOT_DUMP"):
                anth_step_export.dump_weights(self, os.environ["GROOT_DUMP"])
                os._exit(0)
            if os.environ.get("GROOT_MMAP"):
                anth_step_export.map_weights(self, os.environ["GROOT_MMAP"])
            self.module.to("cpu")
            self.policy.to("cpu")
            self.inputs = tuple(x.cpu() for x in self.inputs)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"GR00T CPU compilation: chunk on the CPU, host RSS {rss_gib():.2f} GiB", flush=True)

        groot_recipe.GrootExport.__init__ = groot_export_init

        # GrootChunk.forward runs under torch.autocast("cuda"), which does nothing for CPU tensors.
        real_chunk_forward = groot_recipe.GrootChunk.forward

        def groot_chunk_forward(self, pixel_values, state, noise):
            with torch.autocast("cpu", torch.bfloat16, enabled=self.use_bf16 and pixel_values.device.type == "cpu"):
                return real_chunk_forward(self, pixel_values, state, noise)

        groot_recipe.GrootChunk.forward = groot_chunk_forward

script, sys.argv = sys.argv[1], sys.argv[1:]
runpy.run_path(script, run_name="__main__")
