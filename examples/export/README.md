# Compiled LeRobot policies on an NVIDIA Jetson

Train a policy as usual, export it into a compiled folder, and run that folder with `lerobot-rollout`, the same command you use for a PyTorch checkpoint. In the results below, a compiled policy runs 2 to 6 times faster than in PyTorch and needs a sixth to a half of its memory, so the larger policies also fit on a small board.

Supported policies: ACT, SmolVLA, pi0.5 and GR00T N1.7.

## Hardware

- [SO-101](../../docs/source/so101.mdx) arm with a scene camera and a wrist camera
- [Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), to train, export and run
- [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), to export and run in 8 GB

## Install

On each Jetson (JetPack 7.2.1), with [uv](https://docs.astral.sh/uv/):

```bash
git clone --branch export https://github.com/shoumikhin/lerobot.git
cd lerobot
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[smolvla,pi,groot,feetech,export]" \
    torch==2.15.0.dev20261005 torch-tensorrt==2.15.0.dev20261006 "executorch>=1.6.0.dev20261020" \
    --index-url https://download.pytorch.org/whl/nightly/cu132 \
    --extra-index-url https://pypi.org/simple \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease allow
```

This brings TensorRT 11.3.0.99 with Torch-TensorRT. ExecuTorch 1.6.0.dev20261020 is the first nightly with the API the rollout uses to keep values on the GPU.

To run ExecuTorch-TensorRT exports, you also need `torch-tensorrt-executorch-runtime`, Torch-TensorRT's backend for ExecuTorch, built for that ExecuTorch. As of October 20, 2026, no published build of it does, so build it from source. In a clone of [Torch-TensorRT](https://github.com/pytorch/TensorRT), at the last commit that uses TensorRT 11.3, and with [Bazel](https://bazel.build) installed:

```bash
git checkout 190a72b006
TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH=1 uv pip install --no-build-isolation --no-deps ./py/torch-tensorrt-executorch-runtime
```

## Export

Export on the kind of board that will run the policy: a compiled folder only runs on the GPU model that built it.

```bash
python examples/export/executorch_trt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act
```

Each script compiles any supported policy one way. All use [TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's compiler for its GPUs. If you are not sure, start with ExecuTorch-TensorRT: it is fast and small, and it runs without PyTorch.

| script              | writes                      | needs PyTorch to run |
| ------------------- | --------------------------- | -------------------- |
| `executorch_trt.py` | ExecuTorch `.pte` programs  | no                   |
| `onnx_trt.py`       | TensorRT `.engine` files    | no                   |
| `torch_trt.py`      | AOTInductor `.pt2` packages | yes                  |

Some policies need more arguments:

- **SmolVLA and GR00T** take their camera names and task from the dataset they were trained on: add `--dataset.repo_id=<user>/so101_dataset`, and `--task="<one of its tasks>"` to pick a task other than its first.
- **pi0.5** takes the task it was trained on, and the camera slots your robot fills: add `--task="pick up the block and place it in the cup" --cameras observation.images.base_0_rgb observation.images.left_wrist_0_rgb`.

pi0.5 and GR00T are too large to compile in one piece on a small board, so the scripts build them as a chain of smaller programs, one at a time.

To run an ONNX-TensorRT export on another board, copy the folder there and rebuild its engines with `python examples/export/build_engine.py <folder>`.

## Run

Use the same calibrated robot and camera settings as when you recorded the training data:

```bash
lerobot-rollout \
    --policy.path=outputs/export/act \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: '/dev/video-wrist', width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: '/dev/video-scene', width: 640, height: 480, fps: 30} }" \
    --duration=30
```

SmolVLA, pi0.5 and GR00T run only the task they were exported for, so pass the same `--task`. If your camera names differ from the policy's, add `--rename_map`, for example `--rename_map='{"observation.images.scene": "observation.images.base_0_rgb"}'`.

Before the robot moves, the rollout checks that the compiled policy reproduces the actions PyTorch computed at export time, and stops if it does not.

### Without a robot

[replay_robot](replay_robot) plays back recorded episodes and accepts every action, so you can try a policy at your desk:

```bash
uv pip install -e examples/export/replay_robot
python examples/export/replay_robot/write_frames.py \
    --dataset.repo_id=<user>/so101_dataset --episodes 0 1 2 3 --output_dir=outputs/replay_frames
lerobot-rollout \
    --policy.path=outputs/export/act \
    --robot.type=replay --robot.frames_dir=outputs/replay_frames \
    --duration=60 --return_to_initial_position=false --play_sounds=false
```

## Results

Measured with PyTorch 2.15.0.dev20261005 and TensorRT 11.3.0.99, with Torch-TensorRT and ExecuTorch built from source in mid-October 2026.

Each policy, on each board, compiled each way, next to `torch.compile` and plain PyTorch. The four policies were trained on the same SO-101 dataset (304 episodes at 30 Hz). Bold marks the best value in a row. OOM means it ran out of memory on the 8 GB Orin Nano. Both boards ran at maximum power with locked GPU clocks.

### Time per chunk

A policy returns a chunk of actions at a time (ACT 100, SmolVLA and pi0.5 50, GR00T 40). `lerobot-rollout` plays the whole chunk, then runs the policy again while the arm holds still, so this time is the pause between chunks. Milliseconds, lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |               **3** |         **3** |              4 |              13 |      16 |
| SmolVLA |              **29** |            30 |             30 |              44 |     164 |
| pi0.5   |                 125 |       **115** |            121 |             124 |     237 |
| GR00T   |              **94** |            96 |         **94** |             206 |     211 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |              **25** |        **25** |             26 |              61 |      64 |
| SmolVLA |                 125 |       **121** |            137 |             170 |     749 |
| pi0.5   |                 630 |       **612** |            OOM |             OOM |     OOM |
| GR00T   |             **266** |           274 |            275 |             OOM |     OOM |

Each value is the median of three runs of 100 chunks. Differences of a few milliseconds are within run-to-run noise.

### Memory

Extra memory the board needs while the policy runs: the drop in available memory plus any growth in swap, in MiB (the CPU and the GPU share it on a Jetson). Lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |             **545** |           551 |           1523 |            2404 |    2039 |
| SmolVLA |            **1247** |          1310 |           2836 |            4706 |    4568 |
| pi0.5   |                6050 |      **5932** |           7600 |           13186 |   11618 |
| GR00T   |            **5405** |          5420 |           6068 |           16361 |   15708 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |             **207** |           222 |            841 |            1677 |    1136 |
| SmolVLA |             **373** |           475 |           1764 |            2493 |    2447 |
| pi0.5   |                4983 |      **4882** |            OOM |             OOM |     OOM |
| GR00T   |            **4402** |          4408 |           7534 |             OOM |     OOM |

### Accuracy

How far the compiled policy's actions are from PyTorch's, on 50 dataset samples with the same random noise: largest / average, in degrees (gripper in %). Lower is better. Every cell passed the export's own check.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT |   Torch-TensorRT |      `torch.compile` |
| ------- | ------------------: | ------------: | ---------------: | -------------------: |
| ACT     |        0.14 / 0.004 |  0.14 / 0.004 |     0.14 / 0.004 | **0.05** / **0.001** |
| SmolVLA |        1.72 / 0.048 |  2.00 / 0.049 |     1.72 / 0.048 | **1.68** / **0.045** |
| pi0.5   |    **0.66** / 0.088 |  1.33 / 0.090 | **0.66** / 0.088 |     0.71 / **0.078** |
| GR00T   |    **1.62** / 0.089 |  1.76 / 0.087 | **1.62** / 0.089 |     3.06 / **0.064** |

Orin Nano:

|         |  ExecuTorch-TensorRT |    ONNX-TensorRT |       Torch-TensorRT |      `torch.compile` |
| ------- | -------------------: | ---------------: | -------------------: | -------------------: |
| ACT     |     **0.13** / 0.004 |     0.15 / 0.004 |     **0.13** / 0.004 |     0.20 / **0.002** |
| SmolVLA |         4.03 / 0.058 | 2.76 / **0.044** |         4.03 / 0.058 | **1.42** / **0.044** |
| pi0.5   |     0.97 / **0.097** | **0.70** / 0.098 |                  OOM |                  OOM |
| GR00T   | **1.79** / **0.085** |     2.14 / 0.091 | **1.79** / **0.085** |                  OOM |

The largest error usually comes from one or two samples where the policy is close to a decision, so the average is the steadier number.

## Good to know

- **Orin Nano memory.** Compiling SmolVLA, pi0.5 or GR00T on the 8 GB board may need swap space.
- **The checks show that the compiled policy matches PyTorch.** They do not show that the robot is safe or that it does the task.
