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

"""Checkpoint loading shared by the openpi-derived policies that remap keys (pi0, pi05, pi0_fast)."""

import torch
from safetensors import safe_open
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import (
    T,
    _load_state_dict_into_meta_model,
    _parameters_on_meta,
    _shares_parameters,
)


def load_complete_checkpoint(
    policy_cls: type[T], model_file: str, config: PreTrainedConfig, **kwargs
) -> T | None:
    """Build `policy_cls` with its parameters on the meta device and stream `model_file` straight into them.

    This skips randomly initializing weights that the checkpoint replaces, and never holds a second full copy
    of them. Returns None, before reading any weight, when the file cannot be read, it misses a key the policy
    has or holds one with another shape, or the policy shares a parameter or computed a buffer from one.
    `from_pretrained` then builds the policy and calls `load_state_dict`. Errors while reading the weights
    raise. Weights the policy does not have are skipped. The regular path gives the same weights: it copies
    every other weight, then catches the error about the extra ones.

    Only a class that sets `_supports_meta_load` in its own body takes this path, so a subclass has to opt in.
    That promises the constructor never reads, moves or holds on to a parameter, that
    `_fix_pytorch_state_dict_keys` handles each key on its own and only renames, copies or drops values (this
    path calls it once per key and uses only names and shapes), and that `_prepare_pretrained_state_dict`,
    which this path skips, changes nothing in a complete checkpoint.
    """
    # Read from the class itself, so that a subclass does not inherit its parent's promise.
    if not vars(policy_cls).get("_supports_meta_load", False):
        return None
    try:
        checkpoint = safe_open(model_file, framework="pt", device="cpu")
    except Exception:
        return None
    with checkpoint:
        with _parameters_on_meta():
            policy = policy_cls(config, **kwargs)

        def remap(key: str, shape: list[int]) -> dict[str, torch.Size]:
            fixed = policy._fix_pytorch_state_dict_keys({key: torch.empty(shape, device="meta")}, config)
            return {k if k.startswith("model.") else f"model.{k}": v.shape for k, v in fixed.items()}

        # A file key gives zero, one or two model names: the fixes drop some keys and copy lm_head. Read the
        # keys in the order `load_file` returns them, offset order since safetensors 0.6, so that when two keys
        # fill one weight, such as a stored embedding and the lm_head copied into it, the same one is kept.
        keys = checkpoint.offset_keys() if hasattr(checkpoint, "offset_keys") else checkpoint.keys()
        names = {key: remap(key, checkpoint.get_slice(key).get_shape()) for key in keys}
        expected = {name: tensor.shape for name, tensor in policy.state_dict().items()}
        unexpected = sorted(name for fixed in names.values() for name in fixed if name not in expected)
        names = {key: {n: s for n, s in fixed.items() if n in expected} for key, fixed in names.items()}
        shapes = {name: shape for fixed in names.values() for name, shape in fixed.items()}
        if (
            shapes != expected
            or _shares_parameters(policy)
            or any(buffer.is_meta for buffer in policy.buffers())
        ):
            return None
        tensors = ((name, checkpoint.get_tensor(key)) for key, fixed in names.items() for name in fixed)
        _load_state_dict_into_meta_model(policy, tensors, config.device)
    # Buffers the constructor computed, like rotary tables, are not in the checkpoint and are still on the CPU.
    policy.model.to(config.device)
    if unexpected:
        print(f"Unexpected keys when loading state dict: {len(unexpected)} keys")
        for name in unexpected[:5]:
            print(f"  - {name}")
        if len(unexpected) > 5:
            print(f"  ... and {len(unexpected) - 5} more")
    else:
        print("All keys loaded successfully!")
    return policy


def load_checkpoint_modules(
    policy_cls: type[T], model_file: str, config: PreTrainedConfig, modules: list[str], **kwargs
) -> T:
    """Build `policy_cls` on the meta device and load only the weights of `modules` from `model_file`.

    For exporting one part of a policy too large for the machine's memory: only the listed submodules,
    like `model.paligemma_with_expert.gemma_expert`, get real tensors, read one at a time from the file.
    Every other submodule holding parameters is replaced by None, as the policy already does for modules
    it never runs, so code that reaches one fails instead of reading random weights, and an export of
    the part saves only its own weights. Raises if a listed submodule has a parameter the file does not
    hold. The same promises as `load_complete_checkpoint` apply. The policy comes back in evaluation
    mode, as from `from_pretrained`, since a policy's own `train` may reach a module that is now None.
    """
    if not vars(policy_cls).get("_supports_meta_load", False):
        raise ValueError(f"{policy_cls.__name__} does not support loading on the meta device.")
    prefixes = tuple(f"{name}." for name in modules)
    with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
        with _parameters_on_meta():
            policy = policy_cls(config, **kwargs)
        targets = policy.state_dict(keep_vars=True)
        # A file key gives zero, one or two model names (lm_head is also copied), and the last one wins, as in a full load.
        sources = {}
        for key in checkpoint.offset_keys() if hasattr(checkpoint, "offset_keys") else checkpoint.keys():
            for name in policy._fix_pytorch_state_dict_keys({key: torch.empty(0, device="meta")}, config):
                name = name if name.startswith("model.") else f"model.{name}"
                if name.startswith(prefixes) and name in targets:
                    sources[name] = key
        # A CPU copy first, because a device copy straight from the memory-mapped file can keep its pages.
        tensors = {
            name: checkpoint.get_tensor(key).to(dtype=targets[name].dtype, copy=True).to(config.device)
            for name, key in sources.items()
        }
        policy.load_state_dict(tensors, strict=False, assign=True)
    for name in modules:
        missing = [p for p, param in policy.get_submodule(name).named_parameters() if param.is_meta]
        if missing:
            raise KeyError(f"{model_file} does not hold {name}.{missing[0]} and {len(missing) - 1} more")
    policy.eval()
    _drop_unloaded(policy)
    # Buffers the constructor computed, like rotary tables, are not in the checkpoint and are still on the CPU.
    return policy.to(config.device)


def _drop_unloaded(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        parameters = list(child.parameters())
        if parameters and all(parameter.is_meta for parameter in parameters):
            setattr(module, name, None)
        else:
            _drop_unloaded(child)
