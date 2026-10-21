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

"""Write a pi0.5 checkpoint's weights as `executorch_trt.py --one_process` maps them.

Run it once, on any machine with room for the whole policy on the CPU, then use the file as the policy
folder's `model.safetensors`:

    python examples/export/pi05_mapping_file.py \
        --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model --output=model.safetensors

It holds the weights as lerobot loads them, with the vision tower in bfloat16 as `--one_process` holds it,
and without the two lm_heads, which no exported program reads.
"""

import argparse
from copy import copy

import torch
from torch_tensorrt.dynamo.tools import save_weights_for_mapping
from transformers.utils import cached_file

from lerobot.configs import PreTrainedConfig
from lerobot.policies.common.openpi_checkpoint import load_checkpoint_modules
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy.path", dest="policy_path", required=True, help="Trained pi0.5 checkpoint.")
parser.add_argument("--output", required=True, help="The safetensors file to write.")
args = parser.parse_args()
config = copy(PreTrainedConfig.from_pretrained(args.policy_path))
config.device = "cpu"
model_file = cached_file(args.policy_path, "model.safetensors")
policy = load_checkpoint_modules(PI05Policy, model_file, config, ["model"])
model = policy.model.paligemma_with_expert
model.paligemma.model.vision_tower.to(torch.bfloat16)
model.paligemma.lm_head = None
model.gemma_expert.lm_head = None
save_weights_for_mapping(policy, args.output)
print(f"Wrote {args.output}")
