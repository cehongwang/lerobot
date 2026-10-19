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

"""A robot that replays a recorded SO-101 episode, so `lerobot-rollout` runs without hardware.

It serves the episode's two camera streams and 6-joint state on the wall clock at the recorded rate, like an SO-101
with two OpenCV cameras, and accepts every action, stamping each one so the control loop's real rate can be read back.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots import Robot, RobotConfig

CAMERAS = ("wrist", "scene")


@RobotConfig.register_subclass("replay")
@dataclass
class ReplayRobotConfig(RobotConfig):
    # Folder write_frames.py made: index.npz (state, joint names, frame shape) and one <camera>.jpg.bin per camera.
    frames_dir: str = ""
    # Frames per second to play at; 0 plays at the rate the episodes were recorded.
    fps: int = 0
    start_frame: int = 0
    # File the action and observation timestamps are written to at disconnect; empty writes nothing.
    log_path: str = ""


class ReplayRobot(Robot):
    config_class = ReplayRobotConfig
    name = "replay"

    def __init__(self, config: ReplayRobotConfig):
        super().__init__(config)
        self.config = config
        folder = Path(config.frames_dir)
        index = np.load(folder / "index.npz")
        self._state = index["state"]
        self._motors = [str(name) for name in index["names"]]
        self._shapes = {cam: tuple(int(n) for n in index[f"{cam}_shape"]) for cam in CAMERAS}
        self._fps = config.fps or int(index["fps"])
        # Read one JPEG at a time, so the recording never sits in this process's memory and RSS stays comparable.
        self._jpegs = {
            cam: (os.open(folder / f"{cam}.jpg.bin", os.O_RDONLY), index[f"{cam}_offsets"]) for cam in CAMERAS
        }
        self._latest: dict[str, np.ndarray] = {}
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._threads: list[threading.Thread] = []
        self._t0 = 0.0
        self._connected = False
        self._action_times: list[float] = []
        self._actions: list[list[float]] = []
        self._observation_times: list[float] = []

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**dict.fromkeys(self._motors, float), **self._shapes}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._motors, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _frame_index(self, now: float) -> int:
        return (self.config.start_frame + int((now - self._t0) * self._fps)) % len(self._state)

    def _camera_loop(self, cam: str) -> None:
        try:
            self._play(cam)
        except Exception as error:
            # Kept for connect() and get_observation() to raise, instead of freezing on the last frame.
            self._error = error

    def _play(self, cam: str) -> None:
        # Decodes on its own thread at the camera rate, as an OpenCV camera's read thread does.
        fd, offsets = self._jpegs[cam]
        period = 1.0 / self._fps
        next_time = time.perf_counter()
        while not self._stop.is_set():
            i = self._frame_index(time.perf_counter())
            jpeg = np.frombuffer(os.pread(fd, int(offsets[i + 1] - offsets[i]), int(offsets[i])), np.uint8)
            image = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Frame {i} of the {cam} camera is not a valid JPEG")
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with self._frame_lock:
                self._latest[cam] = frame
            next_time += period
            delay = next_time - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_time = time.perf_counter()

    def connect(self, calibrate: bool = True) -> None:
        self._t0 = time.perf_counter()
        for cam in CAMERAS:
            thread = threading.Thread(
                target=self._camera_loop, args=(cam,), name=f"replay-{cam}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        while len(self._latest) < len(CAMERAS):
            self._raise_camera_error()
            time.sleep(0.005)
        self._connected = True

    def _raise_camera_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("A replay camera stopped") from self._error

    def get_observation(self) -> dict:
        self._raise_camera_error()
        now = time.perf_counter()
        self._observation_times.append(now)
        i = self._frame_index(now)
        observation = {motor: float(self._state[i, j]) for j, motor in enumerate(self._motors)}
        with self._frame_lock:
            observation.update(self._latest)
        return observation

    def send_action(self, action: dict) -> dict:
        self._action_times.append(time.perf_counter())
        self._actions.append([float(action[motor]) for motor in self._motors])
        return dict(action)

    def disconnect(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        for fd, _ in self._jpegs.values():
            os.close(fd)
        self._connected = False
        if self.config.log_path:
            np.savez(
                self.config.log_path,
                action_times=np.array(self._action_times),
                actions=np.array(self._actions, dtype=np.float32).reshape(-1, len(self._motors)),
                observation_times=np.array(self._observation_times),
            )
