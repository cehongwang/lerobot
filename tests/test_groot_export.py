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

import pytest
import torch


@pytest.mark.parametrize(
    "cameras,t,p,m,gh,gw", [(2, 2, 16, 2, 16, 16), (1, 1, 2, 2, 4, 4), (2, 3, 3, 2, 4, 6)]
)
def test_patch_packing_preserves_order_with_at_most_eight_dimensions(monkeypatch, cameras, t, p, m, gh, gw):
    pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    from groot_recipe import GrootObservation

    class Observation(GrootObservation):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.frame_processor = torch.nn.Identity()
            self.cameras = [f"observation.images.camera{i}" for i in range(cameras)]
            self.frame_names = [*self.cameras, "observation.state"]
            self.normalize = self.rescale = self.normalize_state = False
            self.patch_size, self.temporal_patch_size, self.merge_size = p, t, m
            self.grid_h, self.grid_w = gh, gw
            self.max_state_dim = 6

    raw = torch.arange(cameras * gh * p * gw * p * 3).remainder(256).to(torch.uint8)
    raw = raw.reshape(cameras, gh * p, gw * p, 3)
    frame = (*raw.unbind(0), torch.arange(6).float())
    images = (raw.float() / 255).permute(0, 3, 1, 2)
    images = (images.clamp(0, 1) * 255).trunc()
    expected = images[:, None].repeat(1, t, 1, 1, 1)
    expected = expected.reshape(cameras, 1, t, 3, gh // m, m, p, gw // m, m, p)
    expected = expected.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(-1, 3 * t * p * p)

    observation = Observation()
    pixels, state = observation(*frame)
    torch.testing.assert_close(pixels, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, frame[-1][None, None], rtol=0, atol=0)
    program = torch.export.export(observation, frame).run_decompositions()
    ranks = [
        node.meta["val"].ndim
        for node in program.graph.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    ]
    assert max(ranks) <= 8


def observation_for(monkeypatch, frame_cameras, video_keys, rename_map=None):
    """A GrootObservation built from real rename and packing steps, for `frame_cameras` and the checkpoint's `video_keys`."""
    from types import SimpleNamespace

    import numpy as np

    pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    from groot_recipe import GrootObservation

    from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep, GrootN17VLMEncodeStep
    from lerobot.processor import PolicyProcessorPipeline, RenameObservationsProcessorStep

    encode = GrootN17VLMEncodeStep(image_target_size=[8, 8])
    image_processor = SimpleNamespace(
        patch_size=2,
        temporal_patch_size=1,
        merge_size=2,
        do_rescale=False,
        do_normalize=False,
        rescale_factor=1.0,
        image_mean=[0.0] * 3,
        image_std=[1.0] * 3,
    )
    monkeypatch.setattr(
        type(encode), "proc", property(lambda self: SimpleNamespace(image_processor=image_processor))
    )
    pack = GrootN17PackInputsStep(video_modality_keys=video_keys, normalize_min_max=False)
    preprocessor = PolicyProcessorPipeline(
        steps=[RenameObservationsProcessorStep(rename_map=rename_map or {}), pack, encode]
    )
    frame = {f"observation.images.{n}": np.zeros((8, 8, 3), np.uint8) for n in frame_cameras}
    frame["observation.state"] = np.zeros(6, np.float32)
    used = len(video_keys) if video_keys else len(frame_cameras)
    batch = {"image_grid_thw": torch.tensor([[1, 4, 4]] * used)}
    return GrootObservation(preprocessor, frame, batch)


def test_cameras_follow_the_checkpoint_order_and_its_wrist_alias(monkeypatch):
    observation = observation_for(monkeypatch, ["image2", "image"], ["image", "wrist_image"])
    assert observation.cameras == ["observation.images.image", "observation.images.image2"]
    assert observation.image_resize["cameras"] == observation.cameras


def test_resize_geometry_names_the_cameras_the_frame_carries(monkeypatch):
    observation = observation_for(
        monkeypatch, ["front"], ["ego_view"], {"observation.images.front": "observation.images.ego_view"}
    )
    assert observation.cameras == ["observation.images.ego_view"]
    assert observation.image_resize["cameras"] == ["observation.images.front"]
