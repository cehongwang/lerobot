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

"""What the policy recipes and the route scripts share: the command line, the part loop, the folder.

A recipe turns one policy into parts a compiler can take: modules with example inputs, and named inputs
and outputs. A whole policy is one part. A policy too large to build at once is a chain of parts, built
one per process, so only one part's weights are ever in memory. A route script compiles each part; the
recipe then writes the test case and `export.json`, which `lerobot-rollout` reads.
"""

import argparse
import dataclasses
import datetime as dt
import gc
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    to_absolute_actions,
)
from lerobot.utils.constants import OBS_STR

POLICIES = ("act", "smolvla", "pi05", "groot")
TEST_CASE = "test_case.safetensors"
NOISE = "noise"
# The same for every policy: they fit the builds into an 8 GB Jetson Orin Nano beside the rest.
TENSORRT_OPTIONS = {
    "min_block_size": 1,
    # Flow-matching time embeddings are computed in float64, which TensorRT does not support.
    "truncate_double": True,
    # On a Jetson the CPU and GPU share memory: CPU memory can swap during the build, GPU memory cannot.
    "offload_module_to_cpu": True,
    # TensorRT's default. On the Orin Nano, level 5 built pi0.5's step program, but it did not load.
    "optimization_level": 3,
    "workspace_size": 1 << 30,
}


@dataclasses.dataclass
class Part:
    """One program of the chunk: a module, its example inputs, and the names of its inputs and outputs."""

    name: str
    module: nn.Module
    inputs: tuple[Tensor, ...]
    input_names: list[str]
    output_names: list[str]


def parse_args(description: str, route: str) -> tuple[argparse.Namespace, Any]:
    """The route's command line with the arguments of the checkpoint's recipe, and that recipe."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Trained policy checkpoint.")
    policy_type = PreTrainedConfig.from_pretrained(parser.parse_known_args()[0].policy_path).type
    if policy_type not in POLICIES:
        raise SystemExit(f"No export recipe for {policy_type!r} policies; supported: {', '.join(POLICIES)}.")
    recipe = importlib.import_module(policy_type)
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument(
        "--tolerance", type=float, default=recipe.TOLERANCE, help="Largest allowed action error, robot units."
    )
    parser.add_argument("--part", help=argparse.SUPPRESS)
    recipe.add_arguments(parser)
    parser.set_defaults(route=route)
    return parser.parse_args(), recipe


def export(
    script: str,
    args: argparse.Namespace,
    recipe: Any,
    compile_part: Callable[[Part, Path], None],
    suffix: str,
) -> None:
    """Compile each part of the checkpoint with `compile_part` into `<part><suffix>`, then write the folder.

    A chain runs `script` again once per part, because a process that built a part keeps some of its memory
    until it exits. That process runs the parts before its own in PyTorch to make its example inputs.
    """
    names = recipe.part_names(args)
    if args.part is None:
        args.output_dir = make_output_dir(args.output_dir, f"{recipe.__name__}_{args.route}")
        if len(names) > 1:
            for name in names:
                command = [
                    sys.executable,
                    script,
                    *sys.argv[1:],
                    f"--output_dir={args.output_dir}",
                    f"--part={name}",
                ]
                subprocess.run(command, check=True)
            return
        args.part = names[0]
    if args.part not in names:
        raise SystemExit(f"Unknown part {args.part!r}; choose from {names}.")
    exporter = recipe.Export(args)
    programs = {}
    for part in exporter.parts():
        programs[part.name] = {
            "file": f"{part.name}{suffix}",
            "inputs": part.input_names,
            "outputs": part.output_names,
        }
        if part.name == args.part:
            print(f"Compiling {part.name}", flush=True)
            compile_part(part, args.output_dir / programs[part.name]["file"])
            break
        del part
        free_memory()
    if args.part == names[-1]:
        exporter.write(programs, args)


def write_folder(
    args: argparse.Namespace,
    config: PreTrainedConfig,
    frame: dict[str, np.ndarray],
    noise: Tensor | None,
    expected_actions: np.ndarray,
    programs: dict[str, dict],
    **info: Any,
) -> None:
    """Save the test case, the policy config and `export.json`, which `lerobot-rollout` reads."""
    case = {**frame, "expected_actions": expected_actions}
    if noise is not None:
        case[NOISE] = noise.cpu().numpy()
        info["noise_shape"] = list(noise.shape)
    save_file(case, args.output_dir / TEST_CASE)
    config.save_pretrained(args.output_dir)
    info = {
        "backend": args.route,
        "programs": programs,
        "inputs": [*frame, *([NOISE] if noise is not None else [])],
        "test_case": TEST_CASE,
        "tolerance": args.tolerance,
        **info,
    }
    (args.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Wrote {args.output_dir}")


def make_output_dir(output_dir: Path | None, job_name: str) -> Path:
    """Create the export folder, by default a new dated one under outputs/export, as lerobot-train does."""
    if output_dir is None:
        now = dt.datetime.now()
        output_dir = Path("outputs/export") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{job_name}"
    if output_dir.is_dir():
        raise SystemExit(f"{output_dir} already exists. Pick another --output_dir so nothing is overwritten.")
    output_dir.mkdir(parents=True)
    return output_dir


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def processors(
    config: PreTrainedConfig, policy_path: str
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """The checkpoint's saved processors with every step on this machine's device."""
    overrides = {"device_processor": {"device": str(device())}}
    return make_pre_post_processors(
        config,
        pretrained_path=policy_path,
        preprocessor_overrides=overrides,
        postprocessor_overrides=overrides,
    )


def to_policy_input(name: str, x: Tensor) -> Tensor:
    """What `prepare_observation_for_inference` does to one array of the robot's frame, inside the program."""
    if "image" in name:
        x = (x.float() / 255).permute(2, 0, 1)
    return x.unsqueeze(0)


def frame_from_policy(config: PreTrainedConfig) -> dict[str, np.ndarray]:
    """A random observation shaped like the policy's inputs: uint8 HWC images, float32 state."""
    frame = {}
    for name, feature in config.input_features.items():
        if name in config.image_features:
            channels, height, width = feature.shape
            frame[name] = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        else:
            frame[name] = np.random.rand(*feature.shape).astype(np.float32)
    return frame


def frame_from_dataset(features: dict[str, dict]) -> dict[str, np.ndarray]:
    """A random observation named and shaped like the robot's in the dataset: uint8 HWC images, float32."""
    frame = {}
    for name, feature in features.items():
        if not name.startswith(f"{OBS_STR}."):
            continue
        shape = feature["shape"]
        if feature["dtype"] in ("image", "video"):
            # The robot's cameras give HWC frames; a dataset may store the shape channel first.
            if feature["names"][2] not in ("channel", "channels"):
                shape = (shape[1], shape[2], shape[0])
            frame[name] = np.random.randint(0, 256, shape, dtype=np.uint8)
        else:
            frame[name] = np.random.rand(*shape).astype(np.float32)
    return frame


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """For a checkpoint naming placeholder cameras: the robot's cameras and tasks come from its dataset."""
    parser.add_argument(
        "--dataset.repo_id", dest="dataset", required=True, help="The dataset the policy was trained on."
    )
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, help="Local copy of that dataset.")
    parser.add_argument("--task", help="One of the dataset's tasks to export for. Default: its first task.")


def dataset_task(metadata: LeRobotDatasetMetadata, task: str | None) -> str:
    """`task`, which must be one of the dataset's tasks, or the dataset's first task."""
    tasks = [str(t) for t in metadata.tasks.index]
    if task is None:
        return tasks[0]
    if task not in tasks:
        raise ValueError(f"--task must be one of the dataset's tasks: {tasks}.")
    return task


def anchored_postprocessor(postprocessor: PolicyProcessorPipeline) -> tuple[PolicyProcessorPipeline, Any]:
    """The postprocessor's steps before its relative-to-absolute step, and that step.

    That step adds the state the preprocessor cached in Python, which an exported program would keep as a
    constant. A program applies it to the state it takes as an input instead, with `absolute_actions`.
    """
    steps = list(postprocessor.steps)
    index = next(
        (i for i, s in enumerate(steps) if isinstance(s, AbsoluteActionsProcessorStep) and s.enabled), None
    )
    if index is None:
        return postprocessor, None
    if not all(isinstance(s, DeviceProcessorStep) for s in steps[index + 1 :]):
        raise ValueError("The relative-to-absolute action step must be the postprocessor's last step.")
    return dataclasses.replace(postprocessor, steps=steps[:index]), steps[index]


def absolute_actions(actions: Tensor, state: Tensor, step: Any) -> Tensor:
    """`step`'s relative-to-absolute conversion of a (batch, steps, joints) chunk, around `state`."""
    if step is None:
        return actions
    return to_absolute_actions(actions, state, step.relative_step._build_mask(actions.shape[-1]))


def free_memory() -> None:
    """Return freed GPU memory to the next part or the build."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
