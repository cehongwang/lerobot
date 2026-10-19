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

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def write_folder(folder: Path, shapes: dict[str, tuple], fps: int, corrupt: str | None = None) -> None:
    """A frames folder as write_frames.py writes it: two frames per camera, of the given shapes."""
    offsets, index = {}, {}
    for cam, shape in shapes.items():
        jpegs = [cv2.imencode(".jpg", np.full(shape, value, np.uint8))[1].tobytes() for value in (40, 200)]
        if cam == corrupt:
            jpegs = [b"not a jpeg"] * 2
        (folder / f"{cam}.jpg.bin").write_bytes(b"".join(jpegs))
        offsets[cam] = np.cumsum([0, *map(len, jpegs)])
        index[f"{cam}_shape"] = np.array(shape)
        index[f"{cam}_offsets"] = offsets[cam]
    np.savez(
        folder / "index.npz",
        state=np.zeros((2, 6), np.float32),
        names=np.array(list("abcdef")),
        fps=fps,
        **index,
    )


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "examples" / "export" / "replay_robot")
    )
    import lerobot_robot_replay

    return lerobot_robot_replay


def test_replay_keeps_each_camera_shape_and_the_recorded_rate(tmp_path, replay):
    write_folder(tmp_path, {"wrist": (4, 6, 3), "scene": (8, 10, 3)}, fps=15)
    robot = replay.ReplayRobot(replay.ReplayRobotConfig(frames_dir=str(tmp_path)))
    assert robot.observation_features["wrist"] == (4, 6, 3)
    assert robot.observation_features["scene"] == (8, 10, 3)
    assert robot._fps == 15
    assert replay.ReplayRobot(replay.ReplayRobotConfig(frames_dir=str(tmp_path), fps=30))._fps == 30


def test_replay_raises_instead_of_hanging_on_a_bad_frame(tmp_path, replay):
    write_folder(tmp_path, {"wrist": (4, 6, 3), "scene": (8, 10, 3)}, fps=15, corrupt="wrist")
    robot = replay.ReplayRobot(replay.ReplayRobotConfig(frames_dir=str(tmp_path)))
    with pytest.raises(RuntimeError, match="replay camera stopped") as caught:
        robot.connect()
    assert "not a valid JPEG" in str(caught.value.__cause__)
    robot._stop.set()
