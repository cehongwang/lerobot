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

"""The GR00T N1.7 recipe: one action chunk of LeRobot's own GR00T, as a chain of programs.

The chunk goes from resized uint8 HWC cameras, float32 state and noise to actions in robot units. Qwen3-VL
computes its rope positions and image grid with Python loops no exporter can follow; for one task and one
camera size they never change, so they are computed once here and held as constants, as in NVIDIA's own
GR00T deployment code. The rollout resizes the cameras with NumPy; no tokenizer runs at inference.

The chain is the vision encoder, the language model in groups of six layers, the diffusion model in groups
of eight blocks, and the actions. Each part is built in its own process with only its own weights in
memory, so an 8 GB board can export it.
"""

import argparse
import math
from collections.abc import Iterator
from copy import copy

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from export import (
    ChunkActions,
    Part,
    add_dataset_arguments,
    dataset_frame,
    device,
    free_memory,
    processors,
    to_policy_input,
    write_folder,
)
from torch import Tensor, nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
from transformers.utils import cached_file

from lerobot.configs import PreTrainedConfig
from lerobot.inference.export.images import resize_images
from lerobot.policies.groot.groot_n1_7 import CategorySpecificLinear, GR00TN17Config
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import (
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
)
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RenameObservationsProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

TOLERANCE = 5.0
VISION = "_groot_model.backbone.model.model.visual"
LANGUAGE_MODEL = "_groot_model.backbone.model.model.language_model"
ACTION_HEAD = "_groot_model.action_head"
LANGUAGE_LAYERS_PER_PROGRAM = 6
DIT_BLOCKS_PER_PROGRAM = 8


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_dataset_arguments(parser)


def part_names(args: argparse.Namespace) -> list[str]:
    model = GR00TN17Config.from_pretrained(PreTrainedConfig.from_pretrained(args.policy_path).base_model_path)
    language_groups = math.ceil(model.select_layer / LANGUAGE_LAYERS_PER_PROGRAM)
    step_groups = math.ceil(model.diffusion_model_cfg["num_layers"] / DIT_BLOCKS_PER_PROGRAM)
    return [
        "vision",
        *(f"language_model_{i}" for i in range(language_groups)),
        *(f"step_{i}" for i in range(step_groups)),
        "actions",
    ]


# Fix the image sequence lengths at export time instead of reading cu_seqlens inside the graph.
def vision_attention(attn: nn.Module, hidden_states: Tensor, position_embeddings, chunk_sizes: list[int]):
    seq_length = hidden_states.shape[0]
    qkv = attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1)
    q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    outputs = []
    for q_c, k_c, v_c in zip(q.split(chunk_sizes), k.split(chunk_sizes), v.split(chunk_sizes), strict=True):
        q_c, k_c, v_c = (x.transpose(0, 1)[None] for x in (q_c, k_c, v_c))
        output = F.scaled_dot_product_attention(q_c, k_c, v_c, scale=attn.scaling)
        outputs.append(output[0].transpose(0, 1))
    return attn.proj(torch.cat(outputs, dim=0).reshape(seq_length, -1).contiguous())


class Qwen3VisionForExport(nn.Module):
    """NVIDIA's vision tower wrapper: the position and rotary embeddings of one image grid, computed once."""

    def __init__(self, vision_model: nn.Module, grid_thw: Tensor):
        super().__init__()
        self.chunk_sizes = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
        self.patch_embed = vision_model.patch_embed
        self.blocks = vision_model.blocks
        self.merger = vision_model.merger
        self.deepstack_visual_indexes = vision_model.deepstack_visual_indexes
        self.deepstack_merger_list = vision_model.deepstack_merger_list
        with torch.no_grad():
            pos_embeds = vision_model.fast_pos_embed_interpolate(grid_thw)
            rotary = vision_model.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary, rotary), dim=-1)
        self.register_buffer("_pos_embeds", pos_embeds.contiguous())
        self.register_buffer("_rot_cos", emb.cos().contiguous())
        self.register_buffer("_rot_sin", emb.sin().contiguous())

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states = self.patch_embed(pixel_values) + self._pos_embeds
        position_embeddings = (self._rot_cos, self._rot_sin)
        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = hidden_states + vision_attention(
                block.attn, block.norm1(hidden_states), position_embeddings, self.chunk_sizes
            )
            hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
            if layer_num in self.deepstack_visual_indexes:
                index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(self.deepstack_merger_list[index](hidden_states))
        return self.merger(hidden_states), torch.stack(deepstack_features)


def causal_mask(dtype, device, batch_size: int, seq_len: int, attention_mask: Tensor) -> Tensor:
    """NVIDIA's causal and padding mask for the language model, as an additive float mask."""
    mask_value = torch.finfo(dtype).min * 0.5
    causal = torch.triu(torch.full((seq_len, seq_len), mask_value, device=device, dtype=dtype), 1)
    causal = causal[None, None].expand(batch_size, 1, -1, -1)
    padding = (1.0 - attention_mask[:, None, None, :].to(dtype)) * mask_value
    return causal + padding


def deepstack_add(hidden_states: Tensor, visual_pos_masks: Tensor, visual_embeds: Tensor) -> Tensor:
    """Add a layer's deepstack image features at the image token positions, as Qwen3-VL does."""
    delta = torch.zeros_like(hidden_states).masked_scatter(
        visual_pos_masks.unsqueeze(-1).expand_as(hidden_states), visual_embeds
    )
    return hidden_states + delta


class GrootObservation(nn.Module):
    """The saved preprocessor's tensor steps: the resized frame in, image patches and normalized state out."""

    def __init__(self, preprocessor: PolicyProcessorPipeline, frame: dict[str, np.ndarray], batch: dict):
        super().__init__()
        pack_index = next(
            i for i, s in enumerate(preprocessor.steps) if isinstance(s, GrootN17PackInputsStep)
        )
        pack = preprocessor.steps[pack_index]
        encode = next(s for s in preprocessor.steps if isinstance(s, GrootN17VLMEncodeStep))
        ip = encode.proc.image_processor
        if encode.use_albumentations or pack.video_horizon not in (None, 1):
            raise ValueError("Raw-frame export requires tensor image processing and one observation frame.")
        self.frame_names = list(frame)
        self.frame_processor = PolicyProcessorPipeline(steps=preprocessor.steps[:pack_index])
        sample = self.frame_processor({n: to_policy_input(n, torch.from_numpy(x)) for n, x in frame.items()})
        # The cameras the policy feeds, in its order, as its own packing step picks them.
        self.cameras = pack._ordered_image_keys(sample)
        if not self.cameras:
            raise ValueError("The frame has no camera.")
        target_size = encode.image_target_size
        crop_fraction = encode.crop_fraction
        if crop_fraction is None and encode.image_crop_size and target_size:
            crop_fraction = encode.image_crop_size[0] / target_size[0]
        self.patch_size = ip.patch_size
        self.temporal_patch_size = ip.temporal_patch_size
        self.merge_size = ip.merge_size
        grids = batch["image_grid_thw"].tolist()
        if len(grids) != len(self.cameras) or any(g != grids[0] for g in grids) or grids[0][0] != 1:
            raise ValueError("Raw-frame export requires equal, single-frame camera grids.")
        _, self.grid_h, self.grid_w = grids[0]
        # The engine resizes the frame before the rename step, so the geometry names the frame's cameras.
        renamed = {
            new: old
            for step in self.frame_processor.steps
            if isinstance(step, RenameObservationsProcessorStep)
            for old, new in step.rename_map.items()
        }
        self.image_resize = {
            "cameras": [renamed.get(name, name) for name in self.cameras],
            "target_size": target_size,
            "resize_edge": encode.shortest_image_edge or (target_size[0] if target_size else None),
            "crop_fraction": crop_fraction,
            "letterbox": encode.letter_box_transform,
            "image_size": [self.grid_h * ip.patch_size, self.grid_w * ip.patch_size],
        }
        self.rescale, self.normalize = ip.do_rescale, ip.do_normalize
        self.rescale_factor = ip.rescale_factor
        scale = 1 / ip.rescale_factor if self.rescale else 1
        self.register_buffer("image_mean", (torch.tensor(ip.image_mean) * scale)[None, :, None, None])
        self.register_buffer("image_std", (torch.tensor(ip.image_std) * scale)[None, :, None, None])
        dim = frame[OBS_STATE].size
        if dim > pack.max_state_dim:
            raise ValueError("The frame's state exceeds the checkpoint's state width.")
        self.max_state_dim = pack.max_state_dim
        self.normalize_state = pack.normalize_min_max and pack.stats is not None and OBS_STATE in pack.stats
        self.clip_state = pack.clip_outliers
        if self.normalize_state:
            stats = pack.stats[OBS_STATE]
            lo = torch.as_tensor(stats.get("min", torch.zeros(dim)), dtype=torch.float32).flatten()[:dim]
            hi = torch.as_tensor(stats.get("max", torch.ones(dim)), dtype=torch.float32).flatten()[:dim]
            lo, hi = F.pad(lo, (0, dim - lo.numel())), F.pad(hi, (0, dim - hi.numel()), value=1)
            span = hi - lo
            self.register_buffer("state_min", lo)
            self.register_buffer("state_nonzero", span != 0)
            self.register_buffer("state_span", torch.where(span != 0, span, torch.ones_like(span)))

    def forward(self, *frame: Tensor) -> tuple[Tensor, Tensor]:
        obs = self.frame_processor(
            {n: to_policy_input(n, x) for n, x in zip(self.frame_names, frame, strict=True)}
        )
        images = torch.cat([obs[n] for n in self.cameras])
        images = (images.clamp(0, 1) * 255).trunc()
        if self.normalize:
            images = (images - self.image_mean) / self.image_std
        elif self.rescale:
            images = images * self.rescale_factor
        p, t, m = self.patch_size, self.temporal_patch_size, self.merge_size
        # Combine channel and repeated time to stay within TensorRT's eight-dimension limit.
        patches = images[:, :, None].repeat(1, 1, t, 1, 1)
        patches = patches.reshape(len(self.cameras), 3 * t, self.grid_h // m, m, p, self.grid_w // m, m, p)
        pixels = patches.permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(-1, 3 * t * p * p)
        state = obs[OBS_STATE]
        if self.normalize_state:
            state = torch.where(self.state_nonzero, 2 * (state - self.state_min) / self.state_span - 1, 0)
            if self.clip_state:
                state = state.clamp(-1, 1)
        return pixels, F.pad(state[:, None], (0, self.max_state_dim - state.shape[-1]))


class GrootVision(nn.Module):
    """The vision part: the resized frame in, the image features, deepstack features and normalized state out."""

    def __init__(self, vision: nn.Module, observation: GrootObservation, grid_thw: Tensor, use_bf16: bool):
        super().__init__()
        self.observation = observation
        self.vision = Qwen3VisionForExport(vision, grid_thw)
        self.use_bf16 = use_bf16

    def forward(self, *frame: Tensor) -> tuple[Tensor, ...]:
        pixels, state = self.observation(*frame)
        with torch.autocast(pixels.device.type, torch.bfloat16, enabled=self.use_bf16):
            image_embeds, deepstack = self.vision(pixels)
        return image_embeds, *deepstack.unbind(0), state


def select_embodiment(module: nn.Module, embodiment_id: Tensor) -> None:
    """Keep only this robot's weights: the action head holds one set per embodiment it can be trained on."""
    for layer in module.modules():
        if isinstance(layer, CategorySpecificLinear):
            layer.W = nn.Parameter(layer.W[embodiment_id].detach(), requires_grad=False)
            layer.b = nn.Parameter(layer.b[embodiment_id].detach(), requires_grad=False)


class GrootLanguageModel(nn.Module):
    """Some language model layers; the first group inserts the image features, the last runs the action head's encoders.

    The program takes only what its layers read: the hidden states, the deepstack features added in its layer range,
    and, in the last group, the robot state. An input a program never reads would still be a program input the GPU
    cannot own, since an ExecuTorch export places an input on the device only when the device's backend reads it.
    """

    def __init__(
        self,
        text_model,
        action_head,
        constants: dict,
        start: int,
        stop: int,
        use_bf16: bool,
        num_deepstack: int,
    ):
        super().__init__()
        if not 0 <= start < stop <= len(text_model.layers):
            raise ValueError("Invalid language model layer range.")
        self.start, self.last = start, stop == len(text_model.layers)
        self.deepstack_layers = list(range(start, min(stop, num_deepstack)))
        self.layers = nn.ModuleList(text_model.layers[start:stop])
        self.rotary_emb = text_model.rotary_emb
        self.use_bf16 = use_bf16
        for name in ("attention_mask", "position_ids", "visual_pos_masks"):
            self.register_buffer(name, constants[name])
        if start == 0:
            self.register_buffer("text_embeds", constants["text_embeds"])
            self.register_buffer("image_token_mask", constants["image_token_mask"])
        if self.last:
            self.vlln = action_head.vlln
            self.vl_self_attention = action_head.vl_self_attention
            self.state_encoder = action_head.state_encoder
            self.register_buffer("embodiment_id", torch.zeros_like(constants["embodiment_id"]))
            select_embodiment(self.state_encoder, constants["embodiment_id"])

    def forward(self, hidden: Tensor, *inputs: Tensor):
        """`inputs` holds the deepstack features this layer range adds, in layer order, then the state in the last group."""
        added = len(self.deepstack_layers)
        deepstack = dict(zip(self.deepstack_layers, inputs[:added], strict=True))
        with torch.autocast(hidden.device.type, torch.bfloat16, enabled=self.use_bf16):
            if self.start == 0:
                hidden = self.text_embeds.masked_scatter(
                    self.image_token_mask, hidden.to(self.text_embeds.dtype)
                )
            mask = causal_mask(hidden.dtype, hidden.device, *hidden.shape[:2], self.attention_mask)
            positions = self.rotary_emb(hidden, self.position_ids)
            for index, layer in enumerate(self.layers, self.start):
                hidden = layer(
                    hidden,
                    attention_mask=mask,
                    position_ids=self.position_ids[0],
                    past_key_values=None,
                    position_embeddings=positions,
                )
                if index in deepstack:
                    hidden = deepstack_add(hidden, self.visual_pos_masks, deepstack[index].to(hidden.dtype))
            if self.last:
                (state,) = inputs[added:]
                # GR00T consumes the decoder output before Qwen's final norm.
                features = self.vl_self_attention(self.vlln(hidden))
                state_features = self.state_encoder(state.reshape(state.shape[0], 1, -1), self.embodiment_id)
                return features, state_features
            return hidden


class GrootStep(nn.Module):
    """Some diffusion blocks of one denoising step.

    The first group encodes the noisy actions and the time. The last also takes the noisy actions and returns
    the next ones, with the Euler update inside, so the rollout does no arithmetic between programs.
    """

    def __init__(self, action_head, constants: dict, start: int, stop: int, use_bf16: bool, dt: float):
        super().__init__()
        model = action_head.model
        self.dt = dt
        if not 0 <= start < stop <= len(model.transformer_blocks):
            raise ValueError("Invalid diffusion block range.")
        self.start, self.last = start, stop == len(model.transformer_blocks)
        self.blocks = nn.ModuleList(model.transformer_blocks[start:stop])
        self.interleave = model.config.interleave_self_attention
        self.alternate = action_head.config.use_alternate_vl_dit
        self.attend_text_every = model.attend_text_every_n_blocks if self.alternate else 1
        if self.alternate and not self.interleave:
            raise ValueError("AlternateVLDiT requires interleaved self attention.")
        self.use_bf16 = use_bf16
        self.horizon = action_head.action_horizon
        self.register_buffer("image_mask", constants["image_mask"] & constants["backbone_attention_mask"])
        self.register_buffer("text_mask", ~constants["image_mask"] & constants["backbone_attention_mask"])
        self.register_buffer("embodiment_id", torch.zeros_like(constants["embodiment_id"]))
        if start == 0:
            self.action_encoder = action_head.action_encoder
            self.position_embedding = (
                action_head.position_embedding if action_head.config.add_pos_embed else None
            )
            self.timestep_encoder = model.timestep_encoder
            select_embodiment(self.action_encoder, constants["embodiment_id"])
        if self.last:
            self.norm_out = model.norm_out
            self.proj_out_1 = model.proj_out_1
            self.proj_out_2 = model.proj_out_2
            self.action_decoder = action_head.action_decoder
            select_embodiment(self.action_decoder, constants["embodiment_id"])

    def forward(self, vl_embeds: Tensor, *inputs: Tensor):
        with torch.autocast(vl_embeds.device.type, torch.bfloat16, enabled=self.use_bf16):
            if self.start == 0:
                state_features, x_t, time = inputs
                time = time.long()
                hidden = self.action_encoder(x_t[None], time, self.embodiment_id)
                if self.position_embedding is not None:
                    positions = torch.arange(hidden.shape[1], device=hidden.device)
                    hidden = hidden + self.position_embedding(positions)[None]
                hidden = torch.cat((state_features, hidden), dim=1)
                temb = self.timestep_encoder(time)
            elif self.last:
                hidden, temb, x_t = inputs
            else:
                hidden, temb = inputs
            hidden, vl_embeds = hidden.contiguous(), vl_embeds.contiguous()
            for index, block in enumerate(self.blocks, self.start):
                self_attention = self.interleave and index % 2 == 1
                mask = None
                if self.alternate and not self_attention:
                    mask = self.text_mask if index % (2 * self.attend_text_every) == 0 else self.image_mask
                hidden = block(
                    hidden,
                    encoder_hidden_states=None if self_attention else vl_embeds,
                    encoder_attention_mask=mask,
                    temb=temb,
                )
            if self.last:
                shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
                hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
                prediction = self.action_decoder(self.proj_out_2(hidden), self.embodiment_id)
                return x_t + self.dt * prediction[0, -self.horizon :].float()
            return (hidden, temb) if self.start == 0 else hidden


def check_traceable_decode(postprocessor: PolicyProcessorPipeline, route: str) -> None:
    """Refuse at the start a checkpoint whose action decoding this route cannot compile.

    GR00T N1.7 checkpoints with statistics for their embodiment decode actions in NumPy. The other routes
    trace with torch.export's default non-strict mode, which cannot run NumPy on traced tensors, so the
    export would otherwise fail at its last part after hours. The ONNX exporter traces it into the program,
    except relative actions, which add a state the decoder caches between calls. It is the only recipe check
    that names a route, because the route decides how a program is traced. torch.export's strict mode also
    traces the decoding, but a TensorRT build of that graph is untested, so the routes keep their default.
    """
    decode = next((s for s in postprocessor.steps if isinstance(s, GrootN17ActionDecodeStep)), None)
    if decode is None:
        return
    if decode.use_relative_action:
        raise ValueError(
            "This checkpoint decodes relative actions with GrootN17ActionDecodeStep, which adds the state "
            "it cached from the last observation and cannot be exported."
        )
    if route != "onnx_trt":
        raise ValueError(
            "This checkpoint decodes its actions with GrootN17ActionDecodeStep, which runs in NumPy and "
            "only the ONNX-TensorRT route can export: use onnx_trt.py."
        )


class Export:
    """Reads a trained GR00T N1.7 checkpoint one part at a time, and writes the folder around its programs."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = PreTrainedConfig.from_pretrained(args.policy_path)
        self.config.device = str(device())
        self.model_file = cached_file(args.policy_path, "model.safetensors")
        self.policy = self.load_part(VISION)
        model = self.policy._groot_model
        self.model_config = model.config
        self.horizon = self.policy._action_queue_steps
        preprocessor, self.postprocessor = processors(self.config, args.policy_path)
        check_traceable_decode(self.postprocessor, args.route)
        self.task, self.frame, self.folder_config = dataset_frame(args, self.config)
        self.batch = preprocessor(prepare_observation_for_inference(dict(self.frame), device(), self.task))
        self.frame_processor = GrootObservation(preprocessor, self.frame, self.batch).to(device())
        self.image_resize = self.frame_processor.image_resize
        qwen = model.backbone.model.model
        ids, mask = self.batch["input_ids"], self.batch["attention_mask"]
        image_mask = ids == qwen.config.image_token_id
        positions, _ = qwen.get_rope_index(
            input_ids=ids,
            mm_token_type_ids=self.batch.get("mm_token_type_ids", image_mask.int()),
            image_grid_thw=self.batch["image_grid_thw"],
            attention_mask=mask,
        )
        self.constants = {
            "attention_mask": mask.long(),
            "position_ids": positions,
            "visual_pos_masks": image_mask,
            "image_mask": image_mask,
            "backbone_attention_mask": mask == 1,
            "embodiment_id": self.batch["embodiment_id"],
        }
        self.noise = torch.randn(
            self.model_config.action_horizon, self.model_config.max_action_dim, device=device()
        )
        self.num_steps = self.model_config.num_inference_timesteps
        self.timesteps = [
            int(i / self.num_steps * self.model_config.num_timestep_buckets) for i in range(self.num_steps)
        ]
        self.expected_actions = None

    def load_part(self, *modules: str) -> GrootPolicy:
        """The policy with only `modules`' weights read from the checkpoint, in the checkpoint's own precision."""
        config = copy(self.config)
        # Casting norms, embeddings and category biases changes residual additions outside autocast.
        config.use_bf16 = False
        return GrootPolicy.load_checkpoint_modules(self.model_file, config, list(modules))

    def parts(self) -> Iterator[Part]:
        """The chain's parts, each run in PyTorch to make the next part's inputs, one in memory at a time.

        The test case's actions are these parts chained, the step parts once per denoising step.
        """
        resized = resize_images(self.frame, self.image_resize)
        frame = tuple(torch.from_numpy(x).to(device()) for x in resized.values())
        module = GrootVision(
            self.policy._groot_model.backbone.visual,
            self.frame_processor,
            self.batch["image_grid_thw"],
            self.config.use_bf16,
        ).eval()
        del self.policy
        with torch.no_grad():
            hidden, *deepstack, state = module(*frame)
        deepstack_names = [f"deepstack_{i}" for i in range(len(deepstack))]
        yield Part("vision", module, frame, list(self.frame), ["image_embeds", *deepstack_names, "state"])
        del module
        free_memory()
        policy = self.load_part(f"{LANGUAGE_MODEL}.embed_tokens")
        with torch.no_grad():
            text = policy._groot_model.backbone.language_model.get_input_embeddings()(self.batch["input_ids"])
        self.constants["text_embeds"] = text
        self.constants["image_token_mask"] = self.constants["image_mask"][..., None].expand_as(text)
        del policy
        depth, hidden_name = self.model_config.select_layer, "image_embeds"
        for group, first in enumerate(range(0, depth, LANGUAGE_LAYERS_PER_PROGRAM)):
            stop = min(first + LANGUAGE_LAYERS_PER_PROGRAM, depth)
            modules = [f"{LANGUAGE_MODEL}.layers.{i}" for i in range(first, stop)]
            if stop == depth:
                modules += [f"{ACTION_HEAD}.{n}" for n in ("vlln", "vl_self_attention", "state_encoder")]
            free_memory()
            policy = self.load_part(*modules)
            module = GrootLanguageModel(
                policy._groot_model.backbone.language_model,
                policy._groot_model.action_head,
                self.constants,
                first,
                stop,
                self.config.use_bf16,
                len(deepstack),
            ).eval()
            del policy
            state_input = ((state,), ["state"]) if module.last else ((), [])
            inputs = (hidden, *(deepstack[i] for i in module.deepstack_layers), *state_input[0])
            names = [hidden_name, *(deepstack_names[i] for i in module.deepstack_layers), *state_input[1]]
            with torch.no_grad():
                result = module(*inputs)
            outputs = ["vl_embeds", "state_features"] if module.last else [f"language_hidden_{group}"]
            yield Part(f"language_model_{group}", module, inputs, names, outputs)
            if module.last:
                vl_embeds, state_features = result
            else:
                hidden = result
            hidden_name = outputs[0]
            del module
        actions = self.noise
        depth = self.model_config.diffusion_model_cfg["num_layers"]
        for iteration, bucket in enumerate(self.timesteps):
            timestep = torch.tensor([bucket], dtype=torch.float32, device=device())
            carried, names = (
                (state_features, actions, timestep),
                ["vl_embeds", "state_features", "x_t", "timestep"],
            )
            for group, first in enumerate(range(0, depth, DIT_BLOCKS_PER_PROGRAM)):
                stop = min(first + DIT_BLOCKS_PER_PROGRAM, depth)
                modules = [f"{ACTION_HEAD}.model.transformer_blocks.{i}" for i in range(first, stop)]
                if first == 0:
                    modules += [f"{ACTION_HEAD}.action_encoder", f"{ACTION_HEAD}.model.timestep_encoder"]
                    if self.model_config.add_pos_embed:
                        modules.append(f"{ACTION_HEAD}.position_embedding")
                if stop == depth:
                    modules += [f"{ACTION_HEAD}.model.{n}" for n in ("norm_out", "proj_out_1", "proj_out_2")]
                    modules.append(f"{ACTION_HEAD}.action_decoder")
                free_memory()
                module = GrootStep(
                    self.load_part(*modules)._groot_model.action_head,
                    self.constants,
                    first,
                    stop,
                    self.config.use_bf16,
                    1.0 / self.num_steps,
                ).eval()
                inputs = (vl_embeds, *carried)
                if module.last and first != 0:
                    inputs, names = (*inputs, actions), [*names, "x_t"]
                with torch.no_grad():
                    result = module(*inputs)
                if module.last:
                    outputs = ["x_next"]
                elif first == 0:
                    outputs = [f"step_hidden_{group}", "temb"]
                else:
                    outputs = [f"step_hidden_{group}"]
                # Every denoising step runs the same programs: yield them once, then only finish the test case.
                if iteration == 0:
                    yield Part(f"step_{group}", module, inputs, names, outputs)
                if module.last:
                    actions = result
                else:
                    carried = result if first == 0 else (result, carried[-1])
                    names = ["vl_embeds", f"step_hidden_{group}", "temb"]
                del module
        free_memory()
        module = ChunkActions(self.horizon, self.config.output_features[ACTION].shape[0], self.postprocessor)
        state = torch.from_numpy(self.frame[OBS_STATE]).to(device())
        with torch.no_grad():
            self.expected_actions = module(actions, state).float().cpu().numpy()
        yield module.part(actions, state)

    def write(self, programs: dict[str, dict], args: argparse.Namespace) -> None:
        write_folder(
            args,
            self.folder_config,
            self.frame,
            self.noise,
            self.expected_actions,
            programs,
            steps=[name for name in programs if name.startswith("step_")],
            timesteps=self.timesteps,
            image_resize=self.image_resize,
            task=self.task,
            # The task's token ids and rope positions are constants inside the programs.
            task_fixed=True,
        )
