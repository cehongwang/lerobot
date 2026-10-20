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

"""The ACT recipe: one action chunk of LeRobot's own ACT, as one program.

The program is what `lerobot-rollout` runs: image conversion, the saved preprocessor,
`predict_action_chunk` trimmed to `n_action_steps`, and the saved postprocessor. It takes the robot's
frame as the robot gives it (uint8 HWC images, float32 state) and returns actions in robot units.
"""

import argparse
from collections.abc import Iterator

import numpy as np
import torch
from export import Part, device, frame_from_policy, processors, to_policy_input, write_folder
from torch import Tensor, nn

from lerobot.inference import SyncInferenceEngine
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

TOLERANCE = 0.5


class ACTChunk(nn.Module):
    """One action chunk: the robot's frame in, the actions to play out, in robot units."""

    def __init__(
        self, policy: ACTPolicy, preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.input_names = list(policy.config.input_features)

    def forward(self, *frame: Tensor) -> Tensor:
        observation = {
            name: to_policy_input(name, x) for name, x in zip(self.input_names, frame, strict=True)
        }
        actions = self.policy.predict_action_chunk(self.preprocessor(observation))
        return self.postprocessor(actions[:, : self.policy.config.n_action_steps])[0]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """ACT needs nothing beyond the checkpoint."""


def part_names(args: argparse.Namespace) -> list[str]:
    return ["model"]


class Export:
    """Loads a trained ACT checkpoint, and writes the folder around its compiled program."""

    def __init__(self, args: argparse.Namespace):
        self.policy = ACTPolicy.from_pretrained(args.policy_path).to(device()).eval()
        if self.policy.config.temporal_ensemble_coeff is not None:
            # The ensembler averages chunks across ticks, which one program call per chunk cannot do.
            raise SystemExit(
                "ACT with temporal_ensemble_coeff set cannot be exported: its ensemble spans ticks."
            )
        self.args = args
        self.frame = frame_from_policy(self.policy.config)
        self.expected_actions = self.rollout_actions()

    def parts(self) -> Iterator[Part]:
        module = ACTChunk(self.policy, *processors(self.policy.config, self.args.policy_path))
        inputs = tuple(torch.from_numpy(x).to(device()) for x in self.frame.values())
        yield Part("model", module, inputs, list(self.frame), [ACTION])

    def rollout_actions(self) -> np.ndarray:
        """The chunk `lerobot-rollout` plays with PyTorch for the test frame, one action per tick."""
        engine = SyncInferenceEngine(
            self.policy,
            *processors(self.policy.config, self.args.policy_path),
            {},
            [],
            task="",
            device=str(device()),
            robot_type="",
        )
        engine.reset()
        steps = self.policy.config.n_action_steps
        return np.stack([engine.get_action(dict(self.frame)).numpy() for _ in range(steps)])

    def write(self, programs: dict[str, dict], args: argparse.Namespace) -> None:
        write_folder(
            args,
            self.policy.config,
            self.frame,
            None,
            self.expected_actions,
            programs,
        )
