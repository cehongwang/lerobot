# Compiled LeRobot policies on an NVIDIA Jetson

Export a trained ACT, SmolVLA, pi0.5 or GR00T N1.7 policy into a compiled folder. Then run that folder with `lerobot-rollout`, the same command you use for a PyTorch checkpoint.

## Hardware

- [SO-101](../../docs/source/so101.mdx) arm with a scene camera and a wrist camera
- [Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), to train, export and run
- [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), to export and run the policies that fit in its memory (see Limits)

## Routes

A route is the way a policy is compiled. All three use [TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's compiler for its GPUs.

| route                 | what it writes                                    | loads PyTorch at runtime                    |
| --------------------- | ------------------------------------------------- | ------------------------------------------- |
| `executorch_tensorrt` | Torch-TensorRT into an ExecuTorch `.pte` program  | yes, through the public ExecuTorch bindings |
| `onnx_tensorrt`       | ONNX, then a TensorRT `.engine`                   | no for raw-frame folders; see Limits        |
| `torch_tensorrt`      | Torch-TensorRT into an AOTInductor `.pt2` package | yes                                         |

## Install

Run these commands on each Jetson (JetPack 7.2.1), with [uv](https://docs.astral.sh/uv/). The nightly packages must include recent export and runtime fixes. The results below were measured with ExecuTorch and Torch-TensorRT built from source with those fixes.

```bash
git clone --branch export https://github.com/shoumikhin/lerobot.git
cd lerobot
uv venv --python 3.12
source .venv/bin/activate
uv pip install \
    -e ".[smolvla,pi,groot,feetech,export]" \
    "torch-tensorrt[executorch]" onnx onnxscript \
    --index-url https://download.pytorch.org/whl/nightly/cu132 \
    --extra-index-url https://pypi.org/simple \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease allow
```

This also installs `torch-tensorrt-executorch-runtime`, which runs TensorRT inside ExecuTorch programs. The `export` extra adds the CUDA bindings that the ONNX-TensorRT route needs to run.

## Export a policy

Export on the same GPU model that will run the policy, because a TensorRT engine only runs on the GPU model that built it. For ACT, pick one of these commands:

```bash
python examples/export/act_executorch_tensorrt.py --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --output_dir=outputs/export/act_executorch_tensorrt
python examples/export/act_onnx_tensorrt.py --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --output_dir=outputs/export/act_onnx_tensorrt
python examples/export/act_torch_tensorrt.py --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --output_dir=outputs/export/act_torch_tensorrt
```

| policy  | `executorch_tensorrt`                    | `onnx_tensorrt`                    | `torch_tensorrt`                    |
| ------- | ---------------------------------------- | ---------------------------------- | ----------------------------------- |
| ACT     | [script](act_executorch_tensorrt.py)     | [script](act_onnx_tensorrt.py)     | [script](act_torch_tensorrt.py)     |
| SmolVLA | [script](smolvla_executorch_tensorrt.py) | [script](smolvla_onnx_tensorrt.py) | [script](smolvla_torch_tensorrt.py) |
| pi0.5   | [script](pi05_executorch_tensorrt.py)    | [script](pi05_onnx_tensorrt.py)    | [script](pi05_torch_tensorrt.py)    |
| GR00T   | [script](groot_executorch_tensorrt.py)   | [script](groot_onnx_tensorrt.py)   | [script](groot_torch_tensorrt.py)   |

Every ExecuTorch-TensorRT and ONNX-TensorRT script turns on CUDA graphs (the GPU records its work once and replays it), except GR00T with ONNX-TensorRT, where recording the chain of engines fails. The Torch-TensorRT scripts leave them off.

SmolVLA and GR00T read the camera names and the task from the dataset they were trained on. Add `--dataset.root=<folder>` if the dataset is on your disk:

```bash
python examples/export/smolvla_executorch_tensorrt.py \
    --policy.path=outputs/train/smolvla_so101/checkpoints/last/pretrained_model \
    --dataset.repo_id="<user>/so101_dataset" \
    --output_dir=outputs/export/smolvla_executorch_tensorrt
```

pi0.5 needs the task it was trained on, and the camera slots your robot fills. The policy fills the slot you leave out by itself. For a scene camera and a wrist camera:

```bash
python examples/export/pi05_executorch_tensorrt.py \
    --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model \
    --task="pick up the block and place it in the cup" \
    --cameras observation.images.base_0_rgb observation.images.left_wrist_0_rgb \
    --output_dir=outputs/export/pi05_executorch_tensorrt
```

### Export once, build on each device

The ACT and SmolVLA `executorch_tensorrt` and `onnx_tensorrt` scripts take `--export_only`, which saves the exported model without building its TensorRT engine. Copy the folder to each device, then build the engine there:

```bash
python examples/export/act_onnx_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act_onnx_tensorrt \
    --export_only
python examples/export/build_engine.py outputs/export/act_onnx_tensorrt
```

`build_engine.py` takes `--workspace_gib`, `--tactic_gib` (ONNX only) and `--optimization_level` to use less memory while it builds, but the engine may still not fit. ExecuTorch-TensorRT folders it builds run with CUDA graphs off.

### Large policies are exported in parts

pi0.5 and GR00T are too large to compile as one program on a small board. So their scripts export a chain of small programs, each built in its own process from only its own weights, and `lerobot-rollout` runs them in order:

- pi0.5: the image and prompt embeddings, the language model in groups of three layers, one denoising step, and the actions.
- GR00T: vision, the language model in groups of six layers, the diffusion model in groups of eight blocks, and the actions.

## Run it

Use the same calibrated robot and camera settings as when you recorded the training data. Change the camera paths, resolution and rate below to match your setup:

```bash
lerobot-rollout \
    --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: '/dev/video-wrist', width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: '/dev/video-scene', width: 640, height: 480, fps: 30} }" \
    --duration=30
```

SmolVLA, pi0.5 and GR00T run only the task they were exported for, so pass that same `--task`. If your camera names differ from the policy's, add `--rename_map`. For the pi0.5 export above:

```bash
--task="pick up the block and place it in the cup" \
--rename_map='{"observation.images.scene": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
```

Each folder keeps a test case: one observation, and the actions the PyTorch policy gave for it. Before the robot moves, the rollout runs it and stops if the actions differ by more than the folder's tolerance.

### Without a robot

[replay_robot](replay_robot) is a LeRobot robot plugin. It plays back the cameras and joint positions of recorded SO-101 episodes, and it accepts every action. Set `--robot.fps` to the recording's rate:

```bash
uv pip install -e examples/export/replay_robot
python examples/export/replay_robot/write_frames.py \
    --dataset.repo_id="<user>/so101_dataset" \
    --episodes 0 1 2 3 \
    --output_dir=outputs/replay_frames
lerobot-rollout \
    --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=replay \
    --robot.frames_dir=outputs/replay_frames \
    --robot.fps=30 \
    --duration=60 \
    --return_to_initial_position=false \
    --play_sounds=false
```

Add `--robot.log_path=actions.npz` to save the actions and their times. The observations do not react to the actions.

## Results

Each policy runs five ways on the same model: the three TensorRT routes, `torch.compile` and plain PyTorch. Bold marks the best value in each row, for each of the two numbers in an accuracy cell. OOM means the route ran out of memory on the 8 GB Orin Nano. Thor runs at MAXN power with locked clocks. The Orin Nano runs with a locked GPU clock.

### The policies

All four policies were trained on the same SO-101 dataset (225 episodes recorded at 30 Hz) with a wrist camera and a scene camera, both 640 by 480. Each run gets one observation and returns a chunk of actions:

|                           |     ACT | SmolVLA |   pi0.5 | GR00T N1.7 |
| ------------------------- | ------: | ------: | ------: | ---------: |
| Parameters                |     52M |    450M |    4.1B |       3.1B |
| Actions per chunk         |     100 |      50 |      50 |         40 |
| Motion per chunk at 30 Hz |  3.33 s |  1.67 s |  1.67 s |     1.33 s |
| Denoising steps per chunk |       0 |      10 |      10 |          4 |
| Image size the model sees | 640x480 | 512x512 | 224x224 |    256x256 |

`lerobot-rollout` plays every action of a chunk, then runs the policy again while the arm holds still. So the latency below is a pause between chunks.

### Accuracy

How far the actions are from PyTorch's, on 50 dataset samples with the same random noise: largest / average, in degrees (gripper in %). Lower is better.

Thor:

|         | ExecuTorch-TensorRT |    ONNX-TensorRT | Torch-TensorRT |      `torch.compile` |
| ------- | ------------------: | ---------------: | -------------: | -------------------: |
| ACT     |        0.11 / 0.005 |     0.12 / 0.005 |   0.11 / 0.005 | **0.06** / **0.001** |
| SmolVLA |        2.23 / 0.070 | **1.46** / 0.061 |   2.23 / 0.070 |     1.83 / **0.057** |
| pi0.5   |        2.44 / 0.121 |     4.72 / 0.140 |   2.44 / 0.121 | **1.60** / **0.099** |
| GR00T   |        2.97 / 0.094 |     1.53 / 0.087 |   2.97 / 0.094 | **1.08** / **0.069** |

Orin Nano:

|         |  ExecuTorch-TensorRT |        ONNX-TensorRT |   Torch-TensorRT |      `torch.compile` |
| ------- | -------------------: | -------------------: | ---------------: | -------------------: |
| ACT     |         0.10 / 0.005 |         0.11 / 0.005 |     0.10 / 0.005 | **0.07** / **0.002** |
| SmolVLA |     1.59 / **0.054** |     **1.46** / 0.058 | 1.59 / **0.054** |         2.05 / 0.059 |
| pi0.5   |     3.80 / **0.138** | **3.16** / **0.138** |              OOM |                  OOM |
| GR00T   | **1.42** / **0.094** |         2.75 / 0.107 | 1.50 / **0.094** |                  OOM |

Every cell passed the 50-sample comparison, and every exported folder passed its own startup check. `torch.compile` has no startup check.

### Latency

Time for one action chunk, in milliseconds: the median of three process medians, each over 100 chunks after 20 warmup chunks. The Thor `torch.compile` pi0.5 cell uses one process. Repeated runs of the larger policies can differ by 1 ms or more, so values are rounded to whole milliseconds. A 1 ms gap between two routes does not show that one is faster. Lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |               **3** |         **3** |              4 |              13 |      16 |
| SmolVLA |              **28** |            30 |             29 |              44 |     167 |
| pi0.5   |                 119 |       **111** |            119 |             130 |     244 |
| GR00T   |              **93** |            95 |         **93** |             206 |     211 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |              **25** |            26 |             26 |              62 |      65 |
| SmolVLA |                 126 |       **121** |            135 |             172 |     762 |
| pi0.5   |                 631 |       **616** |            OOM |             OOM |     OOM |
| GR00T   |                 277 |       **276** |        **276** |             OOM |     OOM |

The TensorRT routes use CUDA graphs wherever their scripts turn them on. On pi0.5 the graphs do not make ExecuTorch-TensorRT faster, because its chain of small programs copies each input and output through private buffers on every replay.

### Memory

Memory the whole board uses while the policy runs, in MiB: the highest steady value across three valid runs, plus any memory the run pushed to swap. On a Jetson the CPU and GPU share memory, so the memory of the process alone misses some GPU use. Lower is better.

The ExecuTorch-TensorRT cells come from an ExecuTorch build that does not load PyTorch. That build is still in progress, so treat these cells as early estimates. The ONNX-TensorRT cells, and the ExecuTorch-TensorRT pi0.5 and GR00T cells, were measured with CUDA graphs off.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |             **538** |           547 |           1510 |            2489 |    2327 |
| SmolVLA |            **1263** |          1295 |           3233 |            4422 |    3940 |
| pi0.5   |                6031 |      **5924** |           7072 |           12215 |   11610 |
| GR00T   |            **5388** |          5409 |           6489 |           16138 |   15722 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |             **207** |           216 |            579 |            1651 |    1140 |
| SmolVLA |             **346** |           427 |           1708 |            2957 |    2034 |
| pi0.5   |                5001 |      **4893** |            OOM |             OOM |     OOM |
| GR00T   |            **4402** |          4422 |           7605 |             OOM |     OOM |

On the Orin Nano, GR00T with Torch-TensorRT pushes about 3 GB to swap.

## Limits

- **Orin Nano memory.** The board has 8 GB, so SmolVLA, pi0.5 and GR00T may need swap space to compile there.
- **GR00T resizes camera images outside the program**, with NumPy, because not every route can compile that resize. Its latency includes the resize.
- **SmolVLA folders from older versions of these scripts load PyTorch** to turn the task into tokens, even with ONNX-TensorRT. New exports store the task's tokens in the program.
- **These checks show that the compiled policy matches PyTorch. They do not show that the robot is safe or that it does the task.**
