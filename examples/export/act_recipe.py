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

"""The ACT export recipe: what every backend script compiles, and the folder they write.

The program is one action chunk of LeRobot's own ACT, exactly what `lerobot-rollout` runs: image
conversion, the saved preprocessor, `predict_action_chunk` trimmed to `n_action_steps`, and the saved
postprocessor. It takes the robot's frame as the robot gives it (uint8 HWC images, float32 state) and
returns actions in robot units. The folder also keeps a test case, which the rollout replays first.
"""

import argparse
import datetime as dt
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.inference import SyncInferenceEngine
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

TEST_CASE = "test_case.safetensors"
# What `--export_only` saves instead of an engine, for build_engine.py to compile on each device.
EXPORTED_PROGRAM = "model.pt2"
BUILD_ENGINE_BACKENDS = ("executorch_tensorrt", "onnx_tensorrt")


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


def to_policy_input(name: str, x: Tensor) -> Tensor:
    """What `prepare_observation_for_inference` does to one array of the robot's frame, inside the program."""
    if "image" in name:
        x = (x.float() / 255).permute(2, 0, 1)
    return x.unsqueeze(0)


class ACTExport:
    """Loads a trained ACT checkpoint, and writes the exported folder around a compiled program."""

    def __init__(self, args: argparse.Namespace):
        self.policy = ACTPolicy.from_pretrained(args.policy_path).to("cuda").eval()
        if self.policy.config.temporal_ensemble_coeff is not None:
            # The ensembler averages chunks across ticks, which one program call per chunk cannot do.
            raise ValueError(
                "ACT with temporal_ensemble_coeff set cannot be exported: its ensemble spans ticks."
            )
        self.output_dir = make_output_dir(args.output_dir, args.job_name)
        self.policy_path = args.policy_path
        self.backend, self.tolerance = args.backend, args.tolerance
        self.config = self.policy.config
        self.module = ACTChunk(self.policy, *gpu_processors(self.config, self.policy_path))
        self.input_names = self.module.input_names
        self.frame = random_robot_frame(self.policy)
        self.inputs = tuple(torch.from_numpy(self.frame[name]).cuda() for name in self.input_names)
        self.start = time.perf_counter()

    def release_policy(self) -> None:
        """Compute the test case's actions, the policy's last use, then free it for the TensorRT step."""
        self.expected_actions = self.rollout_actions()
        del self.policy, self.module
        gc.collect()
        torch.cuda.empty_cache()

    def write(self, program_path: Path, cuda_graphs: bool = False) -> None:
        """Save the test case, the policy config, and `export.json` beside the compiled program."""
        save_file({**self.frame, "expected_actions": self.expected_actions}, self.output_dir / TEST_CASE)
        self.config.save_pretrained(self.output_dir)
        info = {
            "backend": self.backend,
            "file": program_path.name,
            "inputs": self.input_names,
            "raw_frame": True,
            "output": ACTION,
            "test_case": TEST_CASE,
            "tolerance": self.tolerance,
        }
        if cuda_graphs:
            # The ONNX-TensorRT runner then replays the engine from a CUDA graph.
            info["cuda_graphs"] = True
        (self.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"Wrote {self.output_dir} in {time.perf_counter() - self.start:.0f} s")

    def rollout_actions(self) -> np.ndarray:
        """The chunk `lerobot-rollout` plays with PyTorch for the test frame, one action per tick."""
        # Fresh processors: compiling the module leaves its own holding tensors that cannot leave the GPU.
        engine = SyncInferenceEngine(
            self.policy,
            *gpu_processors(self.policy.config, self.policy_path),
            {},
            [],
            task="",
            device="cuda",
            robot_type="",
        )
        engine.reset()
        steps = self.policy.config.n_action_steps
        return np.stack([engine.get_action(dict(self.frame)).numpy() for _ in range(steps)])


def save_for_build_engine(export, program: torch.export.ExportedProgram, program_path: Path) -> None:
    """With `--export_only`, save the exported program and the folder, but no engine."""
    torch.export.save(program, export.output_dir / EXPORTED_PROGRAM)
    export.release_policy()
    export.write(program_path)


def add_backend_args(parser: argparse.ArgumentParser, backend: str) -> None:
    """Name the backend, and offer `--export_only` where build_engine.py can finish the folder on each device."""
    parser.set_defaults(backend=backend, export_only=False)
    if backend in BUILD_ENGINE_BACKENDS:
        parser.add_argument(
            "--export_only",
            action="store_true",
            help="Write the folder without the engine, to build it with build_engine.py on each device.",
        )


def gpu_processors(
    config: PreTrainedConfig, policy_path: str
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """The checkpoint's saved processors with every step on the GPU, whatever device they were saved with."""
    return make_pre_post_processors(
        config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )


def random_robot_frame(policy: ACTPolicy) -> dict[str, np.ndarray]:
    """A random observation shaped like the robot's: uint8 HWC images, float32 state."""
    frame = {}
    for name, feature in policy.config.input_features.items():
        if name in policy.config.image_features:
            channels, height, width = feature.shape
            frame[name] = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        else:
            frame[name] = np.random.rand(*feature.shape).astype(np.float32)
    return frame


def make_output_dir(output_dir: Path | None, job_name: str) -> Path:
    """Create the export folder, by default a new dated one under outputs/export, as lerobot-train does."""
    if output_dir is None:
        now = dt.datetime.now()
        output_dir = Path("outputs/export") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{job_name}"
    if output_dir.is_dir():
        raise SystemExit(f"{output_dir} already exists. Pick another --output_dir so nothing is overwritten.")
    output_dir.mkdir(parents=True)
    return output_dir


def parse_args(description: str, backend: str) -> argparse.Namespace:
    """The command line every ACT export script shares."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Trained ACT checkpoint.")
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument("--job_name", default=f"act_{backend}", help="Names the default output folder.")
    parser.add_argument(
        "--tolerance", type=float, default=0.5, help="Largest allowed action error, robot units."
    )
    add_backend_args(parser, backend)
    return parser.parse_args()
