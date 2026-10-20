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

"""The pi0.5 recipe: one action chunk of LeRobot's own pi0.5, as a chain of programs.

The chunk goes from the robot's raw uint8 HWC cameras, float32 state and noise to actions in robot units.
pi0.5 reads the state as bin numbers in its prompt, so the program digitizes it and builds the prompt from
token pieces made at export time; no tokenizer runs at inference. The task's tokens are constants, and the
program keeps only the vocabulary rows the prompt can hold (about 1 GiB less), so the folder runs only that
task.

The chain is the embeddings, the language model in groups of three layers, one denoising step, and the
actions. Each part is built in its own process with only its own weights in memory, so an 8 GB board can
export it.
"""

import argparse
import math
from collections.abc import Iterator
from copy import copy

import torch
from export import (
    ChunkActions,
    Part,
    device,
    frame_from_policy,
    free_memory,
    processors,
    to_policy_input,
    write_folder,
)
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

# TensorRT reorders bfloat16 math, so actions drift by up to about 4; wrong noise gives 15 or more.
TOLERANCE = 5.0
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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, help="The task the policy was trained on, as a sentence.")
    parser.add_argument("--cameras", nargs="+", help="The camera slots the robot fills; pi0.5 pads the rest.")


def part_names(args: argparse.Namespace) -> list[str]:
    depth = get_gemma_config(PreTrainedConfig.from_pretrained(args.policy_path).paligemma_variant).depth
    groups = range(math.ceil(depth / LAYERS_PER_PROGRAM))
    return ["embed", *(f"language_model_{group}" for group in groups), "step", "actions"]


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
    """One denoising step of LeRobot's `denoise_step`, with its Euler update.

    The prefix's mask and KV cache, the noisy actions and the time go in; the next noisy actions come out, so
    the rollout runs the steps one after another with no arithmetic between them.
    """

    def __init__(self, model: PI05Pytorch, dt: float):
        super().__init__()
        self.model = model
        self.dt = dt

    def forward(self, prefix_pad_masks: Tensor, *inputs: Tensor) -> Tensor:
        *cache, x_t, timestep = inputs
        past_key_values = DynamicCache(
            tuple((keys, values, None) for keys, values in zip(cache[::2], cache[1::2], strict=True))
        )
        # Every matrix multiplication of the step runs in bfloat16. The checkpoint keeps some weights in
        # float32, such as the expert's adaptive norms, and this halves them, which the Orin Nano needs.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = self.model.denoise_step(prefix_pad_masks, past_key_values, x_t[None], timestep)
        return x_t + self.dt * velocity[0].float()


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


class Export:
    """Reads a trained pi0.5 checkpoint one part at a time, and writes the folder around its programs."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = PreTrainedConfig.from_pretrained(args.policy_path)
        self.config.device = str(device())
        self.model_file = cached_file(args.policy_path, "model.safetensors")
        self.policy = self.load_part(*EMBED_MODULES)
        config = self.policy.config
        if config.use_visual_memory or config.use_proprioceptive_memory:
            raise SystemExit("A pi0.5 policy with observation memory cannot be exported.")
        preprocessor, self.postprocessor = processors(config, args.policy_path)
        tokenizer = next(s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep))
        # The prompt carries the state, so its length can change every step; the program takes one shape.
        tokenizer.padding = "max_length"
        state_steps = Pi05PrepareStateTokenizerProcessorStep | TokenizerProcessorStep
        tensor_steps = PolicyProcessorPipeline(
            steps=[s for s in preprocessor.steps if not isinstance(s, state_steps)]
        )
        # pi0.5 fills in its empty cameras itself, so the robot does not send them.
        empty_cameras = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(config.empty_cameras)}
        available = set(config.image_features) - empty_cameras
        cameras = available if args.cameras is None else set(args.cameras)
        if not cameras or not cameras <= available:
            raise SystemExit(f"Choose --cameras from {sorted(available)}.")
        self.frame = {
            name: value
            for name, value in frame_from_policy(config).items()
            if name not in config.image_features or name in cameras
        }
        self.folder_config = copy(config)
        self.folder_config.input_features = {name: config.input_features[name] for name in self.frame}
        prompt = StatePrompt(tokenizer, args.task, self.frame[OBS_STATE].size).to(device())
        self.noise = torch.randn(config.chunk_size, config.max_action_dim, device=device())
        language_model = self.policy.model.paligemma_with_expert.paligemma.model.language_model
        token_ids = torch.cat((prompt.prefix, prompt.pieces.flatten(), prompt.suffix)).unique()
        language_model.embed_tokens = CompactEmbedding(language_model.embed_tokens, token_ids)
        self.observation = PI05Observation(list(self.frame), tensor_steps, prompt)
        self.dt = -1.0 / config.num_inference_steps
        self.timesteps = [1.0 + step * self.dt for step in range(config.num_inference_steps)]
        self.expected_actions = None

    def load_part(self, *modules: str) -> PI05Policy:
        """The policy with only `modules`' weights read from the checkpoint; every other module is None."""
        return load_checkpoint_modules(PI05Policy, self.model_file, self.config, list(modules))

    def parts(self) -> Iterator[Part]:
        """The chain's parts, each run in PyTorch to make the next part's inputs, one in memory at a time.

        The test case's actions are these parts chained, the denoising step once per step.
        """
        frame = tuple(torch.from_numpy(x).to(device()) for x in self.frame.values())
        embed = PI05Embed(self.policy, self.observation)
        del self.policy
        with torch.no_grad():
            hidden, prefix_pad_masks = embed(*frame)
        yield Part("embed", embed, frame, list(self.frame), ["prefix_embs", "prefix_pad_masks"])
        del embed
        hidden_name, cache = "prefix_embs", {}
        depth = get_gemma_config(self.config.paligemma_variant).depth
        for group, first in enumerate(range(0, depth, LAYERS_PER_PROGRAM)):
            layers = range(first, min(first + LAYERS_PER_PROGRAM, depth))
            free_memory()
            module = PI05LanguageModelLayers(
                self.load_part(*(f"{LANGUAGE_MODEL}.layers.{i}" for i in layers)).model, layers
            )
            inputs = (hidden, prefix_pad_masks)
            with torch.no_grad():
                keys_values = module(*inputs)
            cache_names = [f"past_{kind}_{i}" for i in layers for kind in ("key", "value")]
            outputs = cache_names
            if not module.last:
                outputs = [f"hidden_{group}", *cache_names]
                hidden, *keys_values = keys_values
            cache.update(zip(cache_names, keys_values, strict=True))
            yield Part(f"language_model_{group}", module, inputs, [hidden_name, "prefix_pad_masks"], outputs)
            del module
            hidden_name = outputs[0]
        free_memory()
        step = PI05Step(self.load_part(*STEP_MODULES).model, self.dt)
        cache_inputs = (prefix_pad_masks, *cache.values())
        x_t = self.noise
        with torch.no_grad():
            for timestep in self.timesteps:
                x_t = step(*cache_inputs, x_t, torch.tensor([timestep], device=device()))
        step_inputs = (*cache_inputs, self.noise, torch.ones(1, device=device()))
        yield Part("step", step, step_inputs, ["prefix_pad_masks", *cache, "x_t", "timestep"], ["x_next"])
        del step
        free_memory()
        actions = ChunkActions(
            self.config.n_action_steps, self.config.output_features[ACTION].shape[0], self.postprocessor
        )
        state = torch.from_numpy(self.frame[OBS_STATE]).to(device())
        with torch.no_grad():
            self.expected_actions = actions(x_t, state).cpu().numpy()
        yield actions.part(self.noise, state)

    def write(self, programs: dict[str, dict], args: argparse.Namespace) -> None:
        write_folder(
            args,
            self.folder_config,
            self.frame,
            self.noise,
            self.expected_actions,
            programs,
            steps=["step"],
            timesteps=self.timesteps,
            # The program holds the task's prompt tokens, so the folder runs only this task.
            task=args.task,
        )
