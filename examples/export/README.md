# Compiled LeRobot policies on an NVIDIA Jetson

Train a policy as usual. Then export it into a compiled folder, and run that folder with `lerobot-rollout`, the same command you use for a PyTorch checkpoint. A compiled policy runs several times faster and uses a fraction of the memory, so the larger policies also fit on a small board.

Supported policies: ACT, SmolVLA, pi0.5 and GR00T N1.7.

## Hardware

- [SO-101](../../docs/source/so101.mdx) arm with a scene camera and a wrist camera
- [Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), to train, export and run
- [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), to export and run the policies that fit in its 8 GB

## Install

Run these commands on each Jetson (JetPack 7.2.1), with [uv](https://docs.astral.sh/uv/):

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

## Export a policy

Export on the same kind of board that will run the policy: a compiled folder only runs on the GPU model that built it. (For ACT and SmolVLA you can also export once and build on each board, see [build_engine.py](build_engine.py).)

```bash
python examples/export/act_executorch_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act_executorch_tensorrt
```

Each policy has one script per route (the way it is compiled). All routes use [TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's compiler for its GPUs. If you are not sure, start with ExecuTorch-TensorRT: it is fast and small, and it runs without PyTorch.

| route               | script name                       | writes                        | needs PyTorch to run |
| ------------------- | --------------------------------- | ----------------------------- | -------------------- |
| ExecuTorch-TensorRT | `<policy>_executorch_tensorrt.py` | an ExecuTorch `.pte` program  | no                   |
| ONNX-TensorRT       | `<policy>_onnx_tensorrt.py`       | a TensorRT `.engine`          | no                   |
| Torch-TensorRT      | `<policy>_torch_tensorrt.py`      | an AOTInductor `.pt2` package | yes                  |

`<policy>` is `act`, `smolvla`, `pi05` or `groot`. Some policies need one or two more arguments:

- **SmolVLA and GR00T** read the camera names and the task from the dataset they were trained on: add `--dataset.repo_id="<user>/so101_dataset"`, and `--dataset.root=<folder>` if the dataset is on your disk.
- **pi0.5** needs the task it was trained on, and the camera slots your robot fills (it fills the rest by itself):

```bash
python examples/export/pi05_executorch_tensorrt.py \
    --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model \
    --task="pick up the block and place it in the cup" \
    --cameras observation.images.base_0_rgb observation.images.left_wrist_0_rgb \
    --output_dir=outputs/export/pi05_executorch_tensorrt
```

pi0.5 and GR00T are too large to compile in one piece on a small board, so their scripts write a chain of smaller programs. `lerobot-rollout` runs the chain for you.

## Run it

Use the same calibrated robot and camera settings as when you recorded the training data:

```bash
lerobot-rollout \
    --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: '/dev/video-wrist', width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: '/dev/video-scene', width: 640, height: 480, fps: 30} }" \
    --duration=30
```

SmolVLA, pi0.5 and GR00T run only the task they were exported for, so pass that same `--task`. If your camera names differ from the policy's, add `--rename_map`, for example for the pi0.5 export above:

```bash
--task="pick up the block and place it in the cup" \
--rename_map='{"observation.images.scene": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
```

Before the robot moves, the rollout checks the compiled policy against the actions PyTorch gave at export time, and stops if they differ.

### Without a robot

[replay_robot](replay_robot) plays back the cameras and joint positions of recorded episodes and accepts every action, so you can try a policy at your desk:

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

## Results

Each policy, on each board, compiled each way, next to `torch.compile` and plain PyTorch. The four policies were trained on the same SO-101 dataset (304 episodes at 30 Hz). Bold marks the best value in a row. OOM means it ran out of memory on the 8 GB Orin Nano. Both boards ran at maximum power with locked GPU clocks.

### Time per chunk

A policy returns a chunk of actions at a time (ACT 100, SmolVLA and pi0.5 50, GR00T 40). `lerobot-rollout` plays the whole chunk, then runs the policy again while the arm holds still, so this time is the pause between chunks. Milliseconds, lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |               **3** |         **3** |              4 |              13 |      16 |
| SmolVLA |              **29** |            30 |             30 |              44 |     165 |
| pi0.5   |                 123 |       **114** |            120 |             124 |     236 |
| GR00T   |              **93** |            95 |             94 |             206 |     211 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |              **25** |        **25** |             26 |              61 |      64 |
| SmolVLA |                 125 |       **121** |            136 |             168 |     761 |
| pi0.5   |                 630 |       **613** |            OOM |             OOM |     OOM |
| GR00T   |             **267** |           271 |            276 |             OOM |     OOM |

Each value is the median of three runs of 100 chunks. Differences of a few milliseconds are within run-to-run noise.

### Memory

Memory the whole board uses while the policy runs, in MiB (the CPU and the GPU share it on a Jetson). Lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |                 540 |       **539** |           1394 |            2407 |    2032 |
| SmolVLA |            **1287** |          1308 |           3176 |            4541 |    4302 |
| pi0.5   |                6047 |      **5928** |           7525 |           12086 |   11596 |
| GR00T   |            **5405** |          5418 |           6254 |           15888 |   15715 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |                 219 |       **218** |            857 |            1584 |    1131 |
| SmolVLA |                 612 |       **484** |           1725 |               - |    2397 |
| pi0.5   |                4962 |      **4893** |            OOM |             OOM |     OOM |
| GR00T   |            **4369** |          4415 |           7703 |             OOM |     OOM |

### Accuracy

How far the compiled policy's actions are from PyTorch's, on 50 dataset samples with the same random noise: largest / average, in degrees (gripper in %). Lower is better. Every cell passed the export's own check.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT |   Torch-TensorRT |      `torch.compile` |
| ------- | ------------------: | ------------: | ---------------: | -------------------: |
| ACT     |        0.14 / 0.004 |  0.14 / 0.004 |     0.14 / 0.004 | **0.05** / **0.001** |
| SmolVLA |    1.72 / **0.048** |  1.98 / 0.049 | 1.72 / **0.048** |     **1.52** / 0.051 |
| pi0.5   |    **0.66** / 0.088 |  1.26 / 0.089 | **0.66** / 0.088 |     0.75 / **0.078** |
| GR00T   |    **1.62** / 0.089 |  1.76 / 0.087 | **1.62** / 0.089 |     3.11 / **0.064** |

Orin Nano:

|         |  ExecuTorch-TensorRT |    ONNX-TensorRT |       Torch-TensorRT |      `torch.compile` |
| ------- | -------------------: | ---------------: | -------------------: | -------------------: |
| ACT     |     **0.13** / 0.004 |     0.15 / 0.004 |     **0.13** / 0.004 |     0.20 / **0.002** |
| SmolVLA |         4.03 / 0.058 |     2.76 / 0.044 |         4.03 / 0.058 | **1.49** / **0.043** |
| pi0.5   |     0.97 / **0.097** | **0.70** / 0.098 |                  OOM |                  OOM |
| GR00T   | **1.79** / **0.085** |     2.14 / 0.090 | **1.79** / **0.085** |                  OOM |

The largest error usually comes from one or two samples where the policy is close to a decision, so the average is the steadier number.

## Good to know

- **Orin Nano memory.** Compiling SmolVLA, pi0.5 or GR00T on the 8 GB board may need swap space.
- **These checks show that the compiled policy matches PyTorch.** They do not show that the robot is safe or that it does the task.
