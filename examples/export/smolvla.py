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

"""The SmolVLA recipe: one action chunk of LeRobot's own SmolVLA, as one program.

As for ACT, the program goes from the robot's frame to actions in robot units. A program cannot tokenize,
so the task's token ids are computed here and held as constants, which ties the folder to that task.
Each chunk starts from random noise, which the program takes as its last input. The robot's camera names
and the task come from the training dataset, because SmolVLA's checkpoint only names placeholder cameras.
"""

import argparse
from collections.abc import Iterator
from copy import copy

import torch
from export import (
    Part,
    add_dataset_arguments,
    dataset_task,
    device,
    frame_from_dataset,
    processors,
    to_policy_input,
    write_folder,
)
from torch import Tensor, nn

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import NewLineTaskProcessorStep, PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.feature_utils import dataset_to_policy_features

# TensorRT reorders bfloat16 math, so actions drift by up to about 4; wrong noise gives 13 or more.
TOLERANCE = 5.0
CUDA_GRAPHS = True


class SmolVLAChunk(nn.Module):
    """One action chunk: the robot's frame and the noise in, the actions to play out, in robot units."""

    def __init__(
        self,
        policy: SmolVLAPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        prompt: dict[str, Tensor],
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.frame_names = frame_names
        self.register_buffer("tokens", prompt[OBS_LANGUAGE_TOKENS])
        self.register_buffer("attention_mask", prompt[OBS_LANGUAGE_ATTENTION_MASK])

    def forward(self, *inputs: Tensor) -> Tensor:
        *frame, noise = inputs
        observation = {
            name: to_policy_input(name, x) for name, x in zip(self.frame_names, frame, strict=True)
        }
        observation[OBS_LANGUAGE_TOKENS] = self.tokens
        observation[OBS_LANGUAGE_ATTENTION_MASK] = self.attention_mask
        actions = self.policy.predict_action_chunk(self.preprocessor(observation), noise=noise[None])
        return self.postprocessor(actions[:, : self.policy.config.n_action_steps])[0]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_dataset_arguments(parser)


def part_names(args: argparse.Namespace) -> list[str]:
    return ["model"]


class Export:
    """Loads a trained SmolVLA checkpoint, and writes the folder around its compiled program."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.policy = SmolVLAPolicy.from_pretrained(args.policy_path).to(device()).eval()
        metadata = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
        self.task = dataset_task(metadata, args.task)
        self.frame = frame_from_dataset(metadata.features)
        # The folder takes the robot's cameras, so its config names them, not the checkpoint's placeholders.
        self.config = copy(self.policy.config)
        features = dataset_to_policy_features(metadata.features)
        self.config.input_features = {name: features[name] for name in self.frame}
        self.noise = torch.randn(self.config.chunk_size, self.config.max_action_dim, device=device())
        self.expected_actions = self.rollout_actions()

    def parts(self) -> Iterator[Part]:
        preprocessor, postprocessor = processors(self.policy.config, self.args.policy_path)
        text = [
            s for s in preprocessor.steps if isinstance(s, NewLineTaskProcessorStep | TokenizerProcessorStep)
        ]
        prompt = PolicyProcessorPipeline(steps=text)(self.observation())
        tensors = PolicyProcessorPipeline(steps=[s for s in preprocessor.steps if s not in text])
        module = SmolVLAChunk(self.policy, tensors, postprocessor, list(self.frame), prompt)
        inputs = (*(torch.from_numpy(x).to(device()) for x in self.frame.values()), self.noise)
        yield Part("model", module, inputs, [*self.frame, "noise"], [ACTION])

    def observation(self) -> dict:
        """The test frame and task as `lerobot-rollout` hands them to the policy's preprocessor."""
        return prepare_observation_for_inference(dict(self.frame), device(), self.task)

    def rollout_actions(self):
        """The chunk the PyTorch policy computes for the test frame, task and noise, as `lerobot-rollout` does."""
        preprocessor, postprocessor = processors(self.policy.config, self.args.policy_path)
        self.policy.reset()
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(
                preprocessor(self.observation()), noise=self.noise[None]
            )
            actions = postprocessor(actions[:, : self.policy.config.n_action_steps])
        return actions[0].cpu().numpy()

    def write(self, programs: dict[str, dict], args: argparse.Namespace) -> None:
        write_folder(
            args,
            self.config,
            self.frame,
            self.noise,
            self.expected_actions,
            programs,
            task=self.task,
            # The program holds the task's token ids.
            task_fixed=True,
            cuda_graphs=CUDA_GRAPHS,
        )
