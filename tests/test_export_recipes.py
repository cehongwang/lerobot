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

"""Each recipe exported from start to end on the CPU, with a small random policy.

The compiler is replaced by a runner that replays the module the recipe hands it, so the test covers the
recipe's parts, the folder it writes and the rollout engine's startup check, without a GPU.
"""

import importlib
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.inference.export import ExportInferenceEngine, engine as export_engine
from tests.utils import require_hf_token

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "export"


@pytest.fixture
def recipes(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    yield
    for name in ("export", "act", "smolvla", "pi05", "groot"):
        sys.modules.pop(name, None)


class Replay:
    """Stands in for a compiled program: runs the module the recipe exported, on host arrays."""

    def __init__(self, module):
        self.module = module.eval()

    def run_device(self, *inputs):
        with torch.no_grad():
            outputs = self.module(*(torch.as_tensor(np.asarray(x)) for x in inputs))
        return outputs if isinstance(outputs, tuple) else (outputs,)

    def __call__(self, *inputs):
        (output,) = self.run_device(*inputs)
        return output.numpy()

    def upload(self, value):
        return value


def export_with_replay(recipe_name, args, monkeypatch):
    """Export through the shared part loop, compiling each part into a `Replay` the engine then loads."""
    export = importlib.import_module("export")
    recipe = importlib.import_module(recipe_name)
    compiled = {}

    def compile_part(part, path):
        # The rollout hands every input on the GPU; the export places there only the inputs a program reads.
        program = torch.export.export(part.module, part.inputs)
        user_inputs = set(program.graph_signature.user_inputs)
        unread = [n.name for n in program.graph.nodes if n.name in user_inputs and not n.users]
        assert not unread, f"{part.name} takes inputs it never reads: {unread}"
        path.write_bytes(b"")
        compiled[path.name] = Replay(part.module)

    names = recipe.part_names(args)
    for name in names:
        args.part = name
        export.export("unused", args, recipe, compile_part, ".bin")
    monkeypatch.setattr(
        export_engine,
        "load_programs",
        lambda folder, info: {name: compiled[p["file"]] for name, p in info["programs"].items()},
    )
    return ExportInferenceEngine(args.output_dir, task=getattr(args, "task", None) or "")


def act_checkpoint(path):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(0)
    config = ACTConfig(
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        pretrained_backbone_weights=None,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    stats = {
        "observation.images.front": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
        "observation.state": {"mean": torch.zeros(3), "std": torch.ones(3)},
        "action": {"mean": torch.tensor([1.0, -1.0, 0.0]), "std": torch.tensor([2.0, 2.0, 2.0])},
    }
    policy = ACTPolicy(config)
    policy.save_pretrained(path)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return path


def test_act_exports_and_reproduces_its_test_case(recipes, tmp_path, monkeypatch):
    checkpoint = act_checkpoint(tmp_path / "checkpoint")
    args = Namespace(
        policy_path=str(checkpoint), output_dir=tmp_path / "folder", tolerance=1e-5, route="onnx_trt"
    )
    args.output_dir.mkdir()
    engine = export_with_replay("act", args, monkeypatch)
    info = __import__("json").loads((args.output_dir / "export.json").read_text())
    assert list(info["programs"]) == ["model"]
    assert info["inputs"] == ["observation.images.front", "observation.state"]
    assert "noise_shape" not in info
    frame = {
        "observation.images.front": np.zeros((32, 32, 3), np.uint8),
        "observation.state": np.zeros(3, np.float32),
    }
    actions = [engine.get_action(frame) for _ in range(4)]
    assert all(action.shape == (3,) for action in actions)


def test_act_with_temporal_ensembling_is_refused(recipes, monkeypatch):
    from types import SimpleNamespace

    act = importlib.import_module("act")
    policy = SimpleNamespace(config=SimpleNamespace(temporal_ensemble_coeff=0.01))
    policy.to = lambda device: policy
    policy.eval = lambda: policy
    monkeypatch.setattr(act.ACTPolicy, "from_pretrained", lambda path: policy)
    with pytest.raises(SystemExit, match="temporal_ensemble_coeff"):
        act.Export(Namespace(policy_path="checkpoint"))


def tiny_smolvlm(path):
    """A local copy of SmolVLM2's config, tokenizer and processor, with a small vision and language model."""
    import json
    import shutil

    from huggingface_hub import snapshot_download

    # SmolVLM2 is public; only its config, tokenizer and processor files are needed, not its weights.
    source = Path(
        snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", allow_patterns=["*.json", "*.txt"])
    )
    path.mkdir()
    for file in source.iterdir():
        if file.is_file() and file.suffix in (".json", ".txt"):
            shutil.copy(file, path / file.name)
    config = json.loads((path / "config.json").read_text())
    config["text_config"].update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    # A 64-pixel image in 4-pixel patches: a 16 x 16 grid, which the connector shrinks by its scale factor.
    config["vision_config"].update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=64,
        patch_size=4,
    )
    (path / "config.json").write_text(json.dumps(config))
    return path


def dataset_metadata(features, tasks):
    from types import SimpleNamespace

    import pandas as pd

    return SimpleNamespace(features=features, tasks=pd.DataFrame(index=tasks))


def smolvla_checkpoint(path, vlm):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import RenameObservationsProcessorStep

    torch.manual_seed(0)
    config = SmolVLAConfig(
        device="cpu",
        vlm_model_name=str(vlm),
        load_vlm_weights=False,
        num_vlm_layers=2,
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=6,
        max_action_dim=6,
        num_steps=2,
        resize_imgs_with_padding=(64, 64),
        input_features={
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    stats = {
        "observation.state": {"mean": torch.zeros(3), "std": torch.ones(3)},
        "action": {"mean": torch.tensor([1.0, -1.0, 0.0]), "std": torch.tensor([2.0, 2.0, 2.0])},
    }
    SmolVLAPolicy(config).save_pretrained(path)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    # As lerobot-train saves it: the robot's camera names map to the checkpoint's placeholders.
    rename = next(s for s in preprocessor.steps if isinstance(s, RenameObservationsProcessorStep))
    rename.rename_map = {"observation.images.front": "observation.images.camera1"}
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return path


def test_smolvla_exports_for_the_robots_cameras_and_reproduces_its_test_case(recipes, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    vlm = tiny_smolvlm(tmp_path / "vlm")
    checkpoint = smolvla_checkpoint(tmp_path / "checkpoint", vlm)
    features = {
        # The robot names its camera differently from the checkpoint's placeholder, and stores it channel first.
        "observation.images.front": {
            "dtype": "video",
            "shape": (3, 32, 32),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        "action": {"dtype": "float32", "shape": (3,), "names": None},
    }
    metadata = dataset_metadata(features, ["pick the cup", "open the drawer"])
    monkeypatch.setattr(
        importlib.import_module("export"), "LeRobotDatasetMetadata", lambda repo_id, root=None: metadata
    )
    args = Namespace(
        policy_path=str(checkpoint),
        output_dir=tmp_path / "folder",
        tolerance=1e-4,
        route="executorch_trt",
        dataset="user/dataset",
        dataset_root=None,
        task="open the drawer",
    )
    args.output_dir.mkdir()
    engine = export_with_replay("smolvla", args, monkeypatch)
    info = __import__("json").loads((args.output_dir / "export.json").read_text())
    assert info["inputs"] == ["observation.images.front", "observation.state", "noise"]
    assert info["task"] == "open the drawer"
    assert info["noise_shape"] == [4, 6]
    frame = {
        "observation.images.front": np.zeros((32, 32, 3), np.uint8),
        "observation.state": np.zeros(3, np.float32),
    }
    assert engine.get_action(frame).shape == (3,)


def pi05_checkpoint(path, monkeypatch):
    """A pi0.5 checkpoint with a four-layer Gemma, so the chain splits its language model into two groups."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05
    from tests.policies.pi0_pi05.utils.meta_load import save_checkpoint, use_tiny_backbone

    use_tiny_backbone(monkeypatch, modeling_pi05)
    tiny = modeling_pi05.GemmaConfig(width=16, depth=4, mlp_dim=32, num_heads=1, num_kv_heads=1, head_dim=16)
    monkeypatch.setattr(modeling_pi05, "get_gemma_config", lambda variant: tiny)
    paligemma = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma

    def small_projector(config):
        # pi0.5 fixes the image projector at Gemma 2B's width; the four-layer Gemma here is narrower.
        config.vision_config.projection_dim = tiny.width
        config.vision_config.intermediate_size = 32
        return paligemma(config=config)

    monkeypatch.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma", small_projector)
    config = PI05Config(
        device="cpu",
        dtype=torch.float32,
        image_resolution=(28, 28),
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=8,
        max_action_dim=8,
        num_inference_steps=2,
        tokenizer_max_length=48,
        input_features={
            "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    renames = {f"time_mlp_in.{name}": f"action_time_mlp_in.{name}" for name in ("weight", "bias")}
    save_checkpoint(PI05Policy, config, path, renames)
    config.save_pretrained(path)
    stats = {
        "observation.state": {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
            "q01": -torch.ones(3),
            "q99": torch.ones(3),
        },
        "action": {"mean": torch.zeros(3), "std": torch.ones(3), "q01": -torch.ones(3), "q99": torch.ones(3)},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return path, config


@require_hf_token
def test_pi05_chain_reproduces_the_policy(recipes, tmp_path, monkeypatch):
    """The chained parts compute what `PI05Policy.predict_action_chunk` computes, from the same frame and noise."""
    pytest.importorskip("transformers")
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.utils import prepare_observation_for_inference

    torch.manual_seed(0)
    checkpoint, config = pi05_checkpoint(tmp_path / "checkpoint", monkeypatch)
    task = "pick up the cube"
    args = Namespace(
        policy_path=str(checkpoint),
        output_dir=tmp_path / "folder",
        tolerance=1e-4,
        route="executorch_trt",
        task=task,
        cameras=None,
    )
    args.output_dir.mkdir()
    engine = export_with_replay("pi05", args, monkeypatch)
    info = __import__("json").loads((args.output_dir / "export.json").read_text())
    assert list(info["programs"]) == ["embed", "language_model_0", "language_model_1", "step", "actions"]
    assert info["steps"] == ["step"] and info["timesteps"] == [1.0, 0.5]

    case = __import__("safetensors.numpy", fromlist=["load_file"]).load_file(
        args.output_dir / "test_case.safetensors"
    )
    policy = PI05Policy.from_pretrained(checkpoint, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    frame = {name: case[name] for name in ("observation.images.base_0_rgb", "observation.state")}
    with torch.no_grad():
        batch = preprocessor(prepare_observation_for_inference(dict(frame), torch.device("cpu"), task))
        actions = policy.predict_action_chunk(batch, noise=torch.from_numpy(case["noise"])[None])
        expected = postprocessor(actions[:, : config.n_action_steps])[0].numpy()
    np.testing.assert_allclose(case["expected_actions"], expected, rtol=0, atol=1e-4)
    assert engine.get_action(frame).shape == (3,)


@require_hf_token
def test_pi05_prompt_matches_the_policy_prompt_on_bin_edges(recipes):
    """The program's prompt equals pi0.5's own prompt step, for states on the bin edges and between them."""
    pytest.importorskip("transformers")
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import TokenizerProcessorStep, TransitionKey

    pi05 = importlib.import_module("pi05")
    tokenizer = TokenizerProcessorStep(
        tokenizer_name="google/paligemma-3b-pt-224", max_length=64, padding_side="right", padding="max_length"
    )
    state = torch.tensor([[-1.0, -1.0 + 1 / 128, 0.0, 0.5 - 1e-6, 1.0, 0.3]])
    prompt = pi05.StatePrompt(tokenizer, "pick up the cube", state.shape[1])
    tokens, mask = prompt(state)
    transition = {
        TransitionKey.OBSERVATION: {"observation.state": state},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick up the cube"]},
    }
    transition = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=6)(transition)
    expected = tokenizer(transition)[TransitionKey.OBSERVATION]
    torch.testing.assert_close(tokens, expected["observation.language.tokens"])
    torch.testing.assert_close(mask, expected["observation.language.attention_mask"].bool())
