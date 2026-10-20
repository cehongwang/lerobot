#!/usr/bin/env python

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

"""Write recorded episodes as the frames folder the replay robot plays.

    python examples/export/replay_robot/write_frames.py \
        --dataset.repo_id=<user>/so101_dataset \
        --episodes 0 1 2 3 \
        --output_dir=outputs/replay_frames

The folder holds each camera's frames as JPEG, one file per camera, plus the joint state, the joint and camera
names, and the frame shapes in `index.npz`.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import OBS_IMAGES


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset.repo_id", dest="repo_id", required=True, help="A recorded dataset.")
    parser.add_argument("--dataset.root", dest="root", type=Path, help="Local folder of the dataset.")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0], help="Episodes to play, in order.")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=args.episodes)
    names = list(dataset.meta.features["observation.state"]["names"])
    cameras = [key.removeprefix(f"{OBS_IMAGES}.") for key in dataset.meta.camera_keys]
    state = np.zeros((len(dataset), len(names)), dtype=np.float32)
    offsets = {cam: [0] for cam in cameras}
    shapes = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        files = {cam: stack.enter_context(open(args.output_dir / f"{cam}.jpg.bin", "wb")) for cam in cameras}
        for i in range(len(dataset)):
            item = dataset[i]
            state[i] = item["observation.state"].numpy()
            for cam in cameras:
                rgb = (item[f"{OBS_IMAGES}.{cam}"].permute(1, 2, 0).numpy() * 255).round().clip(0, 255)
                rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                ok, jpeg = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise RuntimeError(f"Could not encode frame {i} of camera {cam}")
                files[cam].write(jpeg.tobytes())
                shapes[cam] = rgb.shape
                offsets[cam].append(offsets[cam][-1] + len(jpeg))
    np.savez(
        args.output_dir / "index.npz",
        state=state,
        names=np.array(names),
        fps=dataset.meta.fps,
        cameras=np.array(cameras),
        **{f"{cam}_shape": np.array(shapes[cam]) for cam in cameras},
        **{f"{cam}_offsets": np.array(offsets[cam], dtype=np.int64) for cam in cameras},
    )
    print(f"Wrote {len(dataset)} frames at {dataset.meta.fps} fps to {args.output_dir}")


if __name__ == "__main__":
    main()
