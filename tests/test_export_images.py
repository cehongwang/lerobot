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

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from lerobot.inference.export import engine as module
from lerobot.inference.export.images import resize_bicubic, resize_images


@pytest.mark.parametrize("size", [(3, 5), (15, 21), (1, 1), (7, 11)])
def test_bicubic_matches_antialiased_reference(size):
    import torch
    import torch.nn.functional as functional

    image = np.random.default_rng(7).integers(0, 256, (7, 11, 3), dtype=np.uint8)
    original = image.copy()
    expected = (
        functional.interpolate(
            torch.from_numpy(image).permute(2, 0, 1)[None].float(),
            size=size,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .clamp(0, 255)
        .round()[0]
        .permute(1, 2, 0)
        .numpy()
    )
    actual = resize_bicubic(image, size)
    assert actual.dtype == np.uint8 and actual.flags.c_contiguous
    assert np.abs(actual.astype(np.float32) - expected).max() <= 1
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("value", [0, 127, 255])
def test_bicubic_preserves_constant_edges(value):
    image = np.full((3, 7, 3), value, dtype=np.uint8)
    np.testing.assert_array_equal(resize_bicubic(image, (9, 2)), value)


def test_camera_geometry_letterbox_and_unrelated_arrays():
    image = np.full((2, 4, 3), 100, dtype=np.uint8)
    frame = {"camera": image, "state": np.zeros(6, dtype=np.float32)}
    geometry = {
        "cameras": ["camera"],
        "target_size": [4, 4],
        "resize_edge": 4,
        "crop_fraction": None,
        "letterbox": True,
        "image_size": [4, 4],
    }
    result = resize_images(frame, geometry)
    np.testing.assert_array_equal(result["camera"], np.pad(image, ((1, 1), (0, 0), (0, 0))))
    assert result["state"] is frame["state"]
    assert frame["camera"] is image


def test_runtime_resizes_both_startup_and_live_frames(tmp_path, monkeypatch):
    image = np.arange(108, dtype=np.uint8).reshape(6, 6, 3)
    state = np.arange(3, dtype=np.float32)
    expected = state[None]
    frame = {"observation.images.camera": image, "observation.state": state}
    info = {
        "backend": "torch_tensorrt",
        "file": "model.pt2",
        "raw_frame": True,
        "inputs": list(frame),
        "output": "action",
        "test_case": "case.safetensors",
        "tolerance": 0,
        "image_resize": {
            "cameras": ["observation.images.camera"],
            "target_size": [3, 3],
            "resize_edge": 6,
            "crop_fraction": 0.5,
            "letterbox": False,
            "image_size": [3, 3],
        },
    }
    (tmp_path / "export.json").write_text(json.dumps(info))
    save_file({**frame, "expected_actions": expected}, tmp_path / "case.safetensors")
    calls = []

    def program(camera, joints):
        np.testing.assert_array_equal(camera, image[1:4, 1:4])
        np.testing.assert_array_equal(joints, state)
        assert camera.flags.c_contiguous
        calls.append(camera.copy())
        return expected

    monkeypatch.setattr(module, "load_program", lambda *args: program)
    engine = module.ExportInferenceEngine(tmp_path, "pick", "replay")
    np.testing.assert_array_equal(engine.get_action(frame), state)
    assert len(calls) == 2
    assert frame["observation.images.camera"] is image
