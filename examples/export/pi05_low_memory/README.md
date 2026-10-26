# pi0.5 export in about 6 GB: ExecuTorch + TensorRT, one process

This folder exports `lerobot/pi05_base` as three ExecuTorch programs that run on TensorRT:

- `prefix.pte`: cameras and prompt in, the KV cache out; 8 TensorRT engines.
- `step_table.pte`: one denoising step; the time conditioning of the 10 steps is precomputed and picked
  by a one-hot.
- `actions.pte`.

The export runs in one process.

Measured on a Jetson AGX Thor. Memory is system used memory plus GPU memory above idle, measured by
`measure.py --gpu -1`. Latency is the median of 3 × 100 chunks:

| | Export time | Export peak | Run-time memory | Time per chunk |
|---|---:|---:|---:|---:|
| `pi05_executorch_tensorrt.py` as a chain of 9 processes (commit `128f2660`) | 394 s | 11,752 MiB | 6,124 MiB | 125.9 ms |
| **This folder** | 233 s | **4,983 MiB** (812 MiB GPU) | 5,908 MiB | 117.9 ms |

The largest action error against the PyTorch test case is 0.029; the rollout's tolerance is 5.

Every technique and its measured effect is in `docs/pi05_export_memory.md` section 2.1 of
`cehongwang/lerobot_project`. The weights are memory-mapped from a bf16 dump, and the policy is built on the
meta device and compiled on the CPU. Each weight is released once its engine is built, and each engine goes
to a file as it is built. The save streams those files into the `.pte` without reading them into memory.

## 1. Software

Tested on a Jetson AGX Thor (sm_110, CUDA 13), Python 3.12:

| Package | Version |
|---|---|
| torch, torchvision | `2.15.0.dev20260930+cu132`, `0.30.0.dev20260930+cu132` (`download.pytorch.org/whl/nightly/cu132`) |
| executorch | `1.6.0.dev20261020+cu132` (same index) |
| TensorRT | `tensorrt_cu13_libs` and `tensorrt_cu13_bindings` `11.4.0.106` (`pypi.nvidia.com`) |
| transformers | `5.5.4` |
| Torch-TensorRT | branch `cehongw/pi05-low-memory-repro` of `pytorch/TensorRT`, used from source, Python-only |
| lerobot | this branch (`cehongwang/lerobot:pi05-6gb-repro`), from `src/` |

Install torch, torchvision and executorch with `pip install --no-deps`, then their dependencies with a
constraint file that pins torch (`torch==2.15.0.dev20260930+cu132`). If pip resolves torch's dependencies
itself, it replaces the CUDA build with PyPI's CPU-only one to satisfy torch's triton pin. Triton isn't needed.

Torch-TensorRT needs no build. Put its `py/` folder on `PYTHONPATH` and give it the version file that
`setup.py` would generate:

```bash
git clone -b cehongw/pi05-low-memory-repro https://github.com/pytorch/TensorRT.git
cat > TensorRT/py/torch_tensorrt/_version.py <<'EOF'
__version__ = "2.16.0.dev0+pi05"
__cuda_version__ = "13.2"
__tensorrt_version__ = "11.4.0"
__tensorrt_rtx_version__ = "1.6.1"
__tensorrt_llm_version__ = "0.17.0.post1"
EOF
export TORCH_TENSORRT_PY=$PWD/TensorRT/py
```

The branch is one commit on `main` that combines #4783, #4853, #4854, #4855, #4863, #4864 and #4865. It
also changes the resource partitioner so that every engine rebuilds the attention mask and rotary tables
from the padding mask. On sm_110 TensorRT 11.0–11.4 can't build an attention engine that receives them
as inputs.

## 2. The weights dump (once)

The export maps the policy's weights from one bf16 file, which this step writes once. The file holds
them as the export's model holds them: in bfloat16, with the task's compact token embedding.

The step loads nothing. It builds the policy on the meta device and makes the export's changes to it.
Then Torch-TensorRT's `save_weights_for_mapping(..., source=checkpoint_source(...))` reads the
checkpoint in 64 MiB chunks, casts them and writes them.

Measured: 0.91 GiB of process memory and 0.3 GB of GPU, for the prompt's tokenizer steps, in 25 s. The
pages of the 14.5 GB float32 checkpoint it reads are mapped clean, so the kernel can reclaim them. It runs
on the board itself, an 8 GB Orin Nano included. It needs the Torch-TensorRT branch at `4d77b3ad13` or later.

The policy folder needs `config.json`, `policy_preprocessor.json`, `policy_postprocessor.json` and
`model.safetensors`:

```bash
H=examples/export/pi05_low_memory
PYTHONPATH=src:examples/export:$TORCH_TENSORRT_PY:$H DUMP_WEIGHTS=pi05_dump.safetensors \
  REF_CACHE=$H/pi05_ref.npz UNROLLED=1 \
  python $H/run_anthony.py $H/anth_step_export.py --policy.path=<pi05_base folder> \
  "--task=pick up the red block" --output_dir=/tmp/pi05_dump_run
# prints "dumped 6.74 GiB of weights to pi05_dump.safetensors" (7,241,148,522 bytes)
```

Copy the file to the board's disk, not to tmpfs: a tmpfs copy would hold its 6.8 GB in RAM.

`pi05_ref.npz` is the PyTorch reference chunk for `lerobot/pi05_base` with the task above. The export checks
its frame and noise checksums and compares the actions with it. For another checkpoint or task, write a new
one: add `REF_SAVE=1` to the command above.

## 3. The export

```bash
bash examples/export/pi05_low_memory/export.sh <policy folder> pi05_dump.safetensors <output folder>
```

The policy folder needs only the three JSON files; the weights come from the dump. The output gets
`prefix.pte`, `step_table.pte`, `actions.pte`, the test case and each program's inputs. Memory goes to
`<output folder>.measure/pi05_low_memory.mem.json`: `sys_peak_above_idle_mb` is the export peak, and
`gpu_peak_mb` is its GPU part.

## 4. Run and check the programs

The programs run on ExecuTorch's Python runtime with the Torch-TensorRT delegate. Build the delegate for
the installed ExecuTorch from `TensorRT/py/torch-tensorrt-executorch-runtime/native`.

1. **Prerequisites:** `pip install "cmake>=3.28,<4.2" patchelf`. You also need the TensorRT headers and a
   `lib/` folder holding `libnvinfer.so` and `libnvinfer_plugin.so`, as links to the pip
   `tensorrt_libs/*.so.11`.
2. **Build:**

   ```bash
   SP=$(python -c "import site; print(site.getsitepackages()[0])")
   cmake -S TensorRT/py/torch-tensorrt-executorch-runtime/native -B build -DCMAKE_BUILD_TYPE=Release \
     -DTORCH_TENSORRT_SOURCE_DIR=$PWD/TensorRT -DCMAKE_PREFIX_PATH=$SP/executorch/share/cmake \
     -DTensorRT_ROOT=<headers and lib/> -DTORCH_TENSORRT_PATCHELF=$(which patchelf)
   cmake --build build -j8
   ```

3. **Install:** put `build/libexecutorch_backend_tensorrt.so` in
   `$SP/torch_tensorrt_executorch_runtime/lib/`, next to that package's `__init__.py`, copied from
   `TensorRT/py/torch-tensorrt-executorch-runtime/torch_tensorrt_executorch_runtime/`.

Then time the chunk and check it against the test case:

```bash
python examples/export/pi05_low_memory/chunk_bench.py ours <output folder>
# [result] ours: median of 3 x 100 chunks ... ms, max action error vs test case ...
```

## Notes

- **GPU memory on a Thor with a hugepage pool** comes from that pool, which `MemAvailable` doesn't show.
  `measure.py --gpu -1` adds the hugepages in use. On a Jetson without the pool, that term is zero.
- **tmpfs output:** if you write the output to `/dev/shm`, keep a login session open. logind's `RemoveIPC`
  deletes a user's `/dev/shm` files when the user's last session closes.
- **Checksums:** check a large file copied to the Thor with `sha256sum` before using it. Its disk can take
  a few seconds after `scp` returns to show the complete file.
