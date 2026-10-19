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

"""The pi0.5 export recipe: what every backend script compiles, and the folder they write.

The program is one action chunk of LeRobot's own pi0.5, from the robot's raw uint8 HWC cameras, float32
state and noise to actions in robot units. pi0.5 reads the state as bin numbers in its prompt, so the
program digitizes it and builds the prompt from token pieces made at export time; no tokenizer runs at
inference. The program keeps only the vocabulary rows the task's prompt can hold (about 1 GiB less),
which ties the folder to that task (`task_fixed` in `export.json`).

The chunk is a chain of programs (the embeddings, the language model in groups of three layers, one
denoising step, and the actions), each built in its own process from only its own weights, so an 8 GB
board can export it.
"""

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from copy import copy
from pathlib import Path

import torch
from act_recipe import (
    TEST_CASE,
    absolute_actions,
    anchored_postprocessor,
    gpu_processors,
    make_output_dir,
    random_robot_frame,
    to_policy_input,
)
from safetensors.numpy import save_file
from smolvla_recipe import NOISE
from torch import Tensor, nn
from transformers import DynamicCache
from transformers.utils import cached_file

from lerobot.configs import PreTrainedConfig
from lerobot.policies.common.openpi_checkpoint import load_checkpoint_modules
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, get_gemma_config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

LANGUAGE_MODEL = "model.paligemma_with_expert.paligemma.model.language_model"
# The vision encoder, its projector and the token embeddings; the language model's lm_head is never needed.
EMBED_MODULES = (
    "model.paligemma_with_expert.paligemma.model.vision_tower",
    "model.paligemma_with_expert.paligemma.model.multi_modal_projector",
    f"{LANGUAGE_MODEL}.embed_tokens",
)
# The action expert without its lm_head, and the layers around it that `denoise_step` runs.
STEP_MODULES = (
    "model.paligemma_with_expert.gemma_expert.model",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.time_mlp_in",
    "model.time_mlp_out",
)
# The most language model layers whose TensorRT build fits an 8 GB Jetson Orin Nano beside the rest.
LAYERS_PER_PROGRAM = 3
# TensorRT's default optimization level. On the Orin Nano, level 5 built the step program, but it did not load.
TENSORRT_OPTIONS = {
    "min_block_size": 1,
    # The denoising step's time embedding is computed in float64, which TensorRT does not support.
    "truncate_double": True,
    # Where the CPU and GPU share memory, as on a Jetson, CPU memory can swap during the build; GPU memory cannot.
    "offload_module_to_cpu": True,
    "optimization_level": 3,
    "workspace_size": 1 << 30,
}


def free_memory() -> None:
    """Return freed GPU memory, so the next part or the TensorRT build can use it."""
    gc.collect()
    torch.cuda.empty_cache()


def part_names(policy_path: str) -> list[str]:
    """The names of the programs `PI05Export.parts` yields, in order."""
    depth = get_gemma_config(PreTrainedConfig.from_pretrained(policy_path).paligemma_variant).depth
    groups = range(math.ceil(depth / LAYERS_PER_PROGRAM))
    return ["embed", *(f"language_model_{group}" for group in groups), "step", "actions"]


def export_parts(
    args: argparse.Namespace,
    script: str,
    file_name: str,
    compile_part: Callable[[nn.Module, tuple, list[str], list[str], Path], None],
    device_resident: bool = False,
    cuda_graphs: bool = False,
) -> None:
    """Export the chunk one program per process, each loading only its own part of the policy.

    Without `--part`, run `script` again once per part, because a process that built a part keeps some of its
    memory until it exits. With `--part`, build only that part with `compile_part`; the last part also writes
    `export.json`.
    """
    if args.part is None:
        output_dir = make_output_dir(args.output_dir, args.job_name)
        for name in part_names(args.policy_path):
            command = [sys.executable, script, *sys.argv[1:], f"--output_dir={output_dir}", f"--part={name}"]
            subprocess.run(command, check=True)
        return
    export = PI05Export(args)
    programs = {}
    for name, module, inputs, input_names, output_names in export.parts():
        programs[name] = {"file": file_name.format(name=name), "inputs": input_names, "outputs": output_names}
        if name != args.part:
            del module
            free_memory()
            continue
        print(f"{name}: exporting", flush=True)
        compile_part(module, inputs, input_names, output_names, export.output_dir / programs[name]["file"])
        print(f"Wrote {programs[name]['file']}", flush=True)
        if name == "actions":
            export.write(programs, device_resident, cuda_graphs)
        return


class StatePrompt(nn.Module):
    """The fixed task and normalized state as a padded Gemma prompt."""

    def __init__(self, tokenizer: TokenizerProcessorStep, task: str, state_dim: int):
        super().__init__()
        tok = tokenizer.input_tokenizer
        clean = task.strip().replace("_", " ").replace("\n", " ")
        prefix = tok.encode(f"Task: {clean}, State:")
        pieces = [tok.encode(f" {value}", add_special_tokens=False) for value in range(-1, 256)]
        suffix = tok.encode(";\nAction: ", add_special_tokens=False)
        width = max(map(len, pieces))
        if (
            tokenizer.padding_side != "right"
            or len(prefix) + state_dim * width + len(suffix) > tokenizer.max_length
        ):
            raise ValueError("The state prompt must fit in a right-padded tokenizer window.")
        self.pad_id = tok.pad_token_id
        self.register_buffer("edges", -1 + torch.arange(256, dtype=torch.float32) / 128)
        self.register_buffer("prefix", torch.tensor(prefix))
        self.register_buffer("suffix", torch.tensor(suffix))
        self.register_buffer("pieces", torch.tensor([p + [self.pad_id] * (width - len(p)) for p in pieces]))
        self.register_buffer("lengths", torch.tensor(list(map(len, pieces))))
        self.register_buffer("piece_positions", torch.arange(width))
        self.register_buffer("positions", torch.arange(tokenizer.max_length))

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        bins = (state.flatten()[..., None] >= self.edges).sum(-1) - 1
        pieces = self.pieces[bins + 1].flatten()
        valid = (self.piece_positions < self.lengths[bins + 1, None]).flatten()
        tokens = torch.cat((self.prefix, pieces, self.suffix))
        valid = torch.cat(
            (
                torch.ones_like(self.prefix, dtype=torch.bool),
                valid,
                torch.ones_like(self.suffix, dtype=torch.bool),
            )
        )
        # Each valid token owns one output position; unused piece slots own none.
        compact = (valid.cumsum(0)[:, None] - 1 == self.positions) & valid[:, None]
        ids = (tokens[:, None] * compact).sum(0)
        mask = self.positions < valid.sum()
        return torch.where(mask, ids, self.pad_id)[None], mask[None]


class PI05Observation(nn.Module):
    """Convert and normalize the raw frame, then build its state-dependent prompt."""

    def __init__(self, frame_names: list[str], preprocessor: PolicyProcessorPipeline, prompt: StatePrompt):
        super().__init__()
        self.frame_names = frame_names
        self.preprocessor = preprocessor
        self.prompt = prompt

    def forward(self, *frame: Tensor) -> dict[str, Tensor]:
        batch = self.preprocessor(
            {name: to_policy_input(name, x) for name, x in zip(self.frame_names, frame, strict=True)}
        )
        batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK] = self.prompt(batch[OBS_STATE])
        return batch


class PI05Step(nn.Module):
    """LeRobot's `denoise_step`: the prefix's mask and KV cache, the noisy actions and the time in, the velocity out."""

    def __init__(self, model: PI05Pytorch):
        super().__init__()
        self.model = model

    def forward(self, prefix_pad_masks: Tensor, *inputs: Tensor) -> Tensor:
        *cache, x_t, timestep = inputs
        past_key_values = DynamicCache(
            tuple((keys, values, None) for keys, values in zip(cache[::2], cache[1::2], strict=True))
        )
        # In bfloat16 the expert's adaptive norm projections take half the memory, which the Orin Nano needs.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = self.model.denoise_step(prefix_pad_masks, past_key_values, x_t[None], timestep)
        return velocity[0].float()


class PI05Actions(nn.Module):
    """The end of the chunk: the denoised actions and the robot's state in, the actions to play out.

    A policy trained on actions relative to the state turns them back around the state it takes here.
    """

    def __init__(self, policy: PI05Policy, postprocessor: PolicyProcessorPipeline):
        super().__init__()
        self.policy = policy
        self.postprocessor, self.absolute_step = anchored_postprocessor(postprocessor)

    def forward(self, x_0: Tensor, state: Tensor) -> Tensor:
        action_dim = self.policy.config.output_features[ACTION].shape[0]
        actions = self.postprocessor(x_0[None, : self.policy.config.n_action_steps, :action_dim])
        return absolute_actions(actions, state[None], self.absolute_step)[0]


class PI05Embed(nn.Module):
    """The start of the prefix: the raw frame in, the image and prompt embeddings and their mask out."""

    def __init__(self, policy: PI05Policy, preprocessor: PI05Observation):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor

    def forward(self, *observation: Tensor) -> tuple[Tensor, Tensor]:
        batch = self.preprocessor(*observation)
        images, img_masks = self.policy._preprocess_images(batch)
        prefix_embs, prefix_pad_masks, _ = self.policy.model.embed_prefix(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        )
        return prefix_embs, prefix_pad_masks


class PI05LanguageModelLayers(nn.Module):
    """Some of the prefix's language model layers: the hidden states in, the next hidden states and their KV cache out.

    pi0.5's prefix attends both ways inside the prompt, so the mask and positions come from the padding mask alone.
    """

    def __init__(self, model: PI05Pytorch, layers: range):
        super().__init__()
        language_model = model.paligemma_with_expert.paligemma.model.language_model
        language_model.config._attn_implementation = "eager"
        self.layers = nn.ModuleList(language_model.layers[i] for i in layers)
        self.rotary_emb = language_model.rotary_emb
        self.last = layers.stop == len(language_model.layers)

    def forward(self, hidden_states: Tensor, prefix_pad_masks: Tensor) -> tuple[Tensor, ...]:
        att_masks = torch.zeros_like(prefix_pad_masks)
        attention_mask = prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, att_masks))
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        if self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cache = DynamicCache()
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
        # Each layer writes its keys and values at its index in the whole model; a program passes only tensors.
        own = [cache.layers[layer.self_attn.layer_idx] for layer in self.layers]
        keys_values = tuple(tensor for layer in own for tensor in (layer.keys, layer.values))
        # The step reads only the KV cache, so the last layer's attention output and MLP would be dead weight.
        return keys_values if self.last else (hidden_states, *keys_values)


class CompactEmbedding(nn.Module):
    """An embedding that keeps only the rows of `token_ids`, and finds them with a constant table."""

    def __init__(self, embedding: nn.Embedding, token_ids: Tensor):
        super().__init__()
        token_ids = token_ids.to(embedding.weight.device)
        rows = torch.zeros(embedding.num_embeddings, dtype=torch.long, device=token_ids.device)
        rows[token_ids] = torch.arange(len(token_ids), device=token_ids.device)
        self.register_buffer("rows", rows)
        embedding.weight = nn.Parameter(embedding.weight.detach()[token_ids], requires_grad=False)
        embedding.num_embeddings = len(token_ids)
        self.embedding = embedding

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding(self.rows[token_ids])


class PI05Export:
    """Reads a trained pi0.5 checkpoint one part at a time, and writes the exported folder around its programs.

    It loads only the embedding weights, and `parts` loads every later part on its own.
    """

    def __init__(self, args: argparse.Namespace):
        # Each part's process writes into the folder `export_parts` made.
        self.output_dir = args.output_dir
        self.policy_path = args.policy_path
        self.backend, self.tolerance = args.backend, args.tolerance
        self.task = args.task
        self.config = PreTrainedConfig.from_pretrained(self.policy_path)
        self.model_file = cached_file(self.policy_path, "model.safetensors")
        self.policy = self.load_part(*EMBED_MODULES)
        config = self.config = self.policy.config
        if config.use_visual_memory or config.use_proprioceptive_memory:
            raise ValueError("Raw-frame export requires a policy without observation memory.")
        preprocessor, self.postprocessor = gpu_processors(config, self.policy_path)
        tokenizer = next(s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep))
        # The prompt carries the state, so its length can change every step; the program takes one shape.
        tokenizer.padding = "max_length"
        tensor_steps = PolicyProcessorPipeline(
            steps=[
                s
                for s in preprocessor.steps
                if not isinstance(s, Pi05PrepareStateTokenizerProcessorStep | TokenizerProcessorStep)
            ]
        )
        # pi0.5 fills in its empty cameras itself, so the robot does not send them.
        empty_cameras = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(config.empty_cameras)}
        available = set(config.image_features) - empty_cameras
        cameras = available if args.cameras is None else set(args.cameras)
        if not cameras or not cameras <= available:
            raise ValueError(f"Choose cameras from {sorted(available)}.")
        self.frame = {
            name: value
            for name, value in random_robot_frame(self.policy).items()
            if name not in config.image_features or name in cameras
        }
        self.input_features = {name: config.input_features[name] for name in self.frame}
        prompt = StatePrompt(tokenizer, self.task, self.frame[OBS_STATE].size).cuda()
        self.noise = torch.randn(config.chunk_size, config.max_action_dim, device="cuda")
        # The whole policy never loads, so `parts` chains its parts' PyTorch outputs for the test case.
        self.expected_actions = None
        language_model = self.policy.model.paligemma_with_expert.paligemma.model.language_model
        token_ids = torch.cat((prompt.prefix, prompt.pieces.flatten(), prompt.suffix)).unique()
        language_model.embed_tokens = CompactEmbedding(language_model.embed_tokens, token_ids)
        self.inputs = (*(torch.from_numpy(x).cuda() for x in self.frame.values()), self.noise)
        self.observation = PI05Observation(list(self.frame), tensor_steps, prompt)
        self.input_names = [*self.frame, NOISE]
        self.start = time.perf_counter()

    def load_part(self, *modules: str) -> PI05Policy:
        """The policy with only `modules`' weights read from the checkpoint; every other module is None."""
        return load_checkpoint_modules(PI05Policy, self.model_file, self.config, list(modules))

    def parts(self) -> Iterator[tuple[str, nn.Module, tuple, list[str], list[str]]]:
        """Yield the chunk's programs one at a time, each with its example inputs, input names and output names.

        Each part runs once in PyTorch to make the next part's inputs, and is dropped before the next one
        loads, so only one part is ever in memory.
        """
        observation, names = self.inputs[:-1], self.input_names[:-1]
        embed = PI05Embed(self.policy, self.observation)
        del self.policy
        free_memory()
        with torch.no_grad():
            hidden, prefix_pad_masks = embed(*observation)
        yield "embed", embed, observation, names, ["prefix_embs", "prefix_pad_masks"]
        del embed
        hidden_name, cache = "prefix_embs", {}
        depth = get_gemma_config(self.config.paligemma_variant).depth
        for group, first in enumerate(range(0, depth, LAYERS_PER_PROGRAM)):
            layers = range(first, min(first + LAYERS_PER_PROGRAM, depth))
            free_memory()
            layer_group = PI05LanguageModelLayers(
                self.load_part(*(f"{LANGUAGE_MODEL}.layers.{i}" for i in layers)).model, layers
            )
            inputs = (hidden, prefix_pad_masks)
            with torch.no_grad():
                keys_values = layer_group(*inputs)
            cache_names = [f"past_{kind}_{i}" for i in layers for kind in ("key", "value")]
            outputs = cache_names
            if not layer_group.last:
                outputs = [f"hidden_{group}", *cache_names]
                hidden, *keys_values = keys_values
            cache.update(zip(cache_names, keys_values, strict=True))
            yield f"language_model_{group}", layer_group, inputs, [hidden_name, "prefix_pad_masks"], outputs
            del layer_group
            hidden_name = outputs[0]
        free_memory()
        step = PI05Step(self.load_part(*STEP_MODULES).model)
        cache_inputs = (prefix_pad_masks, *cache.values())
        x_t, dt = self.noise, -1.0 / self.config.num_inference_steps
        with torch.no_grad():
            for i in range(self.config.num_inference_steps):
                x_t = x_t + dt * step(*cache_inputs, x_t, torch.tensor([1.0 + i * dt], device="cuda"))
        step_inputs = (*cache_inputs, self.noise, torch.ones(1, device="cuda"))
        yield "step", step, step_inputs, ["prefix_pad_masks", *cache, "x_t", "timestep"], ["v_t"]
        del step
        free_memory()
        actions = PI05Actions(self.load_part(), self.postprocessor)
        state = torch.from_numpy(self.frame[OBS_STATE]).cuda()
        with torch.no_grad():
            self.expected_actions = actions(x_t, state).cpu().numpy()
        yield "actions", actions, (self.noise, state), ["x_t", OBS_STATE], [ACTION]

    def write(
        self, programs: dict[str, dict], device_resident: bool = False, cuda_graphs: bool = False
    ) -> None:
        """Save the raw test case, the policy config and `export.json`, which lists each program's file, inputs and outputs."""
        case = {**self.frame, NOISE: self.noise.cpu().numpy(), "expected_actions": self.expected_actions}
        save_file(case, self.output_dir / TEST_CASE)
        config = copy(self.config)
        config.input_features = self.input_features
        config.save_pretrained(self.output_dir)
        info = {
            "backend": self.backend,
            "programs": programs,
            "device_resident": device_resident,
            "num_steps": self.config.num_inference_steps,
            "inputs": self.input_names,
            "raw_frame": True,
            "task": self.task,
            # The program keeps only the vocabulary rows this task's prompt can hold.
            "task_fixed": True,
            "noise_shape": list(self.noise.shape),
            "output": ACTION,
            "test_case": TEST_CASE,
            "tolerance": self.tolerance,
        }
        if cuda_graphs:
            # The ONNX-TensorRT runner then replays each engine from a CUDA graph.
            info["cuda_graphs"] = True
        (self.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"Wrote {self.output_dir} in {time.perf_counter() - self.start:.0f} s")


def parse_args(description: str, backend: str) -> argparse.Namespace:
    """The command line every pi0.5 export script shares."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Trained pi0.5 checkpoint.")
    parser.add_argument("--task", required=True, help="The task the policy was trained on, as a sentence.")
    parser.add_argument(
        "--cameras", nargs="+", help="Input camera keys; omitted slots use the policy's padding."
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument("--job_name", default=f"pi05_{backend}", help="Names the default output folder.")
    # TensorRT reorders bfloat16 math, so actions drift by up to about 4; wrong noise gives 15 or more.
    parser.add_argument(
        "--tolerance", type=float, default=5.0, help="Largest allowed action error, robot units."
    )
    parser.add_argument(
        "--part", help="Build only this program, in an existing folder. The script sets it itself."
    )
    parser.set_defaults(backend=backend)
    return parser.parse_args()
