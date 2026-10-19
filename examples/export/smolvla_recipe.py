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

"""The SmolVLA export recipe: what both backend scripts compile, and the folder they write.

As for ACT, the program is one action chunk of LeRobot's own SmolVLA, from the robot's frame to actions in
robot units. A program cannot tokenize, so the task's token ids are computed here and held as constants,
which ties the folder to that task (`task_fixed` in `export.json`). Each chunk starts from random noise,
which the program takes as its last input. The robot's camera names and the task come from the training
dataset, because SmolVLA's checkpoint only names placeholder cameras.
"""

import argparse
import gc
import json
import time
from copy import copy
from pathlib import Path

import numpy as np
import torch
from act_recipe import TEST_CASE, add_backend_args, gpu_processors, make_output_dir, to_policy_input
from safetensors.numpy import save_file
from torch import Tensor, nn

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import NewLineTaskProcessorStep, PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STR
from lerobot.utils.feature_utils import dataset_to_policy_features

NOISE = "noise"


class SmolVLAChunk(nn.Module):
    """One action chunk: the robot's frame and the noise in, the actions to play out, in robot units."""

    def __init__(
        self,
        policy: SmolVLAPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        tokens: Tensor,
        attention_mask: Tensor,
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.frame_names = frame_names
        self.input_names = [*frame_names, NOISE]
        self.register_buffer("tokens", tokens)
        self.register_buffer("attention_mask", attention_mask)

    def forward(self, *inputs: Tensor) -> Tensor:
        *frame, noise = inputs
        observation = {
            name: to_policy_input(name, x) for name, x in zip(self.frame_names, frame, strict=True)
        }
        observation[OBS_LANGUAGE_TOKENS] = self.tokens
        observation[OBS_LANGUAGE_ATTENTION_MASK] = self.attention_mask
        actions = self.policy.predict_action_chunk(self.preprocessor(observation), noise=noise[None])
        return self.postprocessor(actions[:, : self.policy.config.n_action_steps])[0]


class SmolVLAExport:
    """Loads a trained SmolVLA checkpoint, and writes the exported folder around a compiled program."""

    def __init__(self, args: argparse.Namespace):
        self.output_dir = make_output_dir(args.output_dir, args.job_name)
        self.policy_path = args.policy_path
        self.backend, self.tolerance = args.backend, args.tolerance
        self.policy = SmolVLAPolicy.from_pretrained(self.policy_path).to("cuda").eval()
        self.config = self.policy.config
        preprocessor, postprocessor = self.processors()
        text_steps = [
            s for s in preprocessor.steps if isinstance(s, NewLineTaskProcessorStep | TokenizerProcessorStep)
        ]
        self.text_steps = PolicyProcessorPipeline(steps=text_steps)
        tensor_steps = PolicyProcessorPipeline(steps=[s for s in preprocessor.steps if s not in text_steps])

        metadata = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
        self.task = dataset_task(metadata, args.task)
        self.frame = random_robot_frame(metadata.features)
        # The folder takes the robot's cameras, so its config names them, not the checkpoint's placeholders.
        self.input_features = {
            name: feature
            for name, feature in dataset_to_policy_features(metadata.features).items()
            if name in self.frame
        }
        prompt = self.text_steps(
            prepare_observation_for_inference(dict(self.frame), torch.device("cuda"), self.task)
        )
        self.noise = torch.randn(self.config.chunk_size, self.config.max_action_dim, device="cuda")
        self.inputs = (*(torch.from_numpy(x).cuda() for x in self.frame.values()), self.noise)
        self.module = SmolVLAChunk(
            self.policy,
            tensor_steps,
            postprocessor,
            list(self.frame),
            prompt[OBS_LANGUAGE_TOKENS],
            prompt[OBS_LANGUAGE_ATTENTION_MASK],
        )
        self.input_names = self.module.input_names
        self.start = time.perf_counter()

    def release_policy(self) -> None:
        """Compute the test case's actions, the policy's last use, then free it for the TensorRT step."""
        self.expected_actions = self.rollout_actions()
        del self.policy, self.module
        gc.collect()
        torch.cuda.empty_cache()

    def write(self, program_path: Path, cuda_graphs: bool = False) -> None:
        """Save the test case, the policy config and `export.json` beside the program."""
        case = {**self.frame, NOISE: self.noise.cpu().numpy(), "expected_actions": self.expected_actions}
        save_file(case, self.output_dir / TEST_CASE)
        config = copy(self.config)
        config.input_features = self.input_features
        config.save_pretrained(self.output_dir)
        info = {
            "backend": self.backend,
            "file": program_path.name,
            "inputs": self.input_names,
            "raw_frame": True,
            "task": self.task,
            # The program holds the task's token ids.
            "task_fixed": True,
            "noise_shape": list(self.noise.shape),
            "output": ACTION,
            "test_case": TEST_CASE,
            "tolerance": self.tolerance,
        }
        if cuda_graphs:
            # The ONNX-TensorRT runner then replays the engine from a CUDA graph.
            info["cuda_graphs"] = True
        (self.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"Wrote {self.output_dir} in {time.perf_counter() - self.start:.0f} s")

    def processors(self) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
        """The checkpoint's processors on the GPU, with the tokenizer padding every task to one width."""
        preprocessor, postprocessor = gpu_processors(self.policy.config, self.policy_path)
        # One width for every task, so the program's token inputs keep the shape they were exported with.
        next(s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep)).padding = "max_length"
        return preprocessor, postprocessor

    def rollout_actions(self) -> np.ndarray:
        """The chunk the PyTorch policy computes for the test frame, task and noise, as lerobot-rollout runs it."""
        # Fresh processors: compiling the module leaves its own holding tensors that cannot leave the GPU.
        preprocessor, postprocessor = self.processors()
        self.policy.reset()
        observation = prepare_observation_for_inference(dict(self.frame), torch.device("cuda"), self.task)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(preprocessor(observation), noise=self.noise[None])
            actions = postprocessor(actions[:, : self.policy.config.n_action_steps])
        return actions[0].cpu().numpy()


def random_robot_frame(features: dict[str, dict]) -> dict[str, np.ndarray]:
    """A random observation shaped and named like the robot's in the dataset: uint8 HWC images, float32 state."""
    frame = {}
    for name, feature in features.items():
        if not name.startswith(f"{OBS_STR}."):
            continue
        if feature["dtype"] in ("image", "video"):
            frame[name] = np.random.randint(0, 256, feature["shape"], dtype=np.uint8)
        else:
            frame[name] = np.random.rand(*feature["shape"]).astype(np.float32)
    return frame


def dataset_task(metadata: LeRobotDatasetMetadata, task: str | None) -> str:
    """`task`, which must be one of the dataset's tasks, or the dataset's first task."""
    tasks = [str(t) for t in metadata.tasks.index]
    if task is None:
        return tasks[0]
    if task not in tasks:
        raise ValueError(f"--task must be one of the dataset's tasks: {tasks}.")
    return task


def parse_args(description: str, backend: str) -> argparse.Namespace:
    """The command line every SmolVLA export script shares."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy.path", dest="policy_path", required=True, help="Trained SmolVLA checkpoint."
    )
    parser.add_argument(
        "--dataset.repo_id", dest="dataset", required=True, help="The dataset the policy was trained on."
    )
    parser.add_argument(
        "--dataset.root",
        dest="dataset_root",
        type=Path,
        help="Local copy of that dataset, if not on the Hub.",
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument(
        "--task",
        help="The task to export the policy for, one of the dataset's tasks. Default: its first task.",
    )
    parser.add_argument("--job_name", default=f"smolvla_{backend}", help="Names the default output folder.")
    # TensorRT reorders bfloat16 math, so actions drift by up to about 4; wrong noise gives 13 or more.
    parser.add_argument(
        "--tolerance", type=float, default=5.0, help="Largest allowed action error, robot units."
    )
    add_backend_args(parser, backend)
    return parser.parse_args()
