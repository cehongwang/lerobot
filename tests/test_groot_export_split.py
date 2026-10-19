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

import importlib
import json
import os
import runpy
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def test_checkpoint_split_matches_chunk(monkeypatch, tmp_path):
    checkpoint = os.environ.get("GROOT_SPLIT_CHECKPOINT")
    dataset = os.environ.get("GROOT_SPLIT_DATASET")
    if not checkpoint or not dataset:
        pytest.skip("Set GROOT_SPLIT_CHECKPOINT and GROOT_SPLIT_DATASET for the checkpoint check.")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    from groot_recipe import GrootChunk, GrootExport, free_memory, part_names

    from lerobot.inference.export.engine import ProgramChain
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    class Program:
        def __init__(self, module):
            self.module = module

        def run_device(self, *inputs):
            with torch.no_grad():
                return self.module(*(torch.as_tensor(x, device="cuda") for x in inputs))

        def __call__(self, *inputs):
            return self.run_device(*inputs).float().cpu().numpy()

    torch.manual_seed(14)
    np.random.seed(14)
    args = Namespace(
        task=None,
        output_dir=tmp_path,
        policy_path=checkpoint,
        backend="eager",
        tolerance=0.0,
        dataset=dataset,
        dataset_root=os.environ.get("GROOT_SPLIT_DATASET_ROOT"),
    )
    export = GrootExport(args, split=True)
    programs, metadata = {}, {}
    for name, module, inputs, input_names, output_names in export.parts():
        programs[name] = Program(module)
        metadata[name] = {"file": f"{name}.pte", "inputs": input_names, "outputs": output_names}
        assert len(inputs) == len(input_names)
        with torch.no_grad():
            outputs = module(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        assert len(outputs) == len(output_names)
        print(f"CHECKED {name}", flush=True)
        del module, outputs
        free_memory()
    assert list(programs) == part_names(checkpoint)
    export.write(metadata)
    info = json.loads((tmp_path / "export.json").read_text())
    chain = ProgramChain(programs, info)
    chained = torch.from_numpy(chain(*(x.cpu().numpy() for x in export.inputs)))
    actual = torch.from_numpy(export.expected_actions)
    torch.testing.assert_close(chained, actual, rtol=0, atol=0)
    del chain, programs
    free_memory()
    policy = GrootPolicy.from_pretrained(checkpoint).cuda().eval()
    reference = GrootChunk(policy, export.batch, export.postprocessor, export.frame_processor).eval()
    with torch.no_grad():
        expected = reference(*export.inputs).float().cpu()
    difference = (actual - expected).abs().max().item()
    print(f"SPLIT_VS_CHUNK max_abs={difference:.9g} shape={tuple(actual.shape)}", flush=True)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(chained, expected, rtol=0, atol=0)


@pytest.fixture
def recipe(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    return importlib.import_module("groot_recipe")


def action_pipeline(*steps):
    """A postprocessor that takes and returns an action tensor, as the policies' own do."""
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    return PolicyProcessorPipeline(
        steps=list(steps), to_transition=policy_action_to_transition, to_output=transition_to_policy_action
    )


def tiny_head(alternate=True, interleave=True, num_layers=5):
    from lerobot.policies.groot.groot_n1_7 import GR00TN17ActionHead, GR00TN17Config

    config = GR00TN17Config(
        backbone_embedding_dim=32,
        hidden_size=16,
        input_embedding_dim=32,
        max_state_dim=6,
        max_action_dim=4,
        action_horizon=3,
        max_num_embodiments=3,
        max_seq_len=16,
        num_inference_timesteps=3,
        use_alternate_vl_dit=alternate,
        diffusion_model_cfg={
            "num_layers": num_layers,
            "num_attention_heads": 4,
            "attention_head_dim": 8,
            "output_dim": 16,
            "dropout": 0.0,
            "positional_embeddings": None,
            "interleave_self_attention": interleave,
        },
        vl_self_attention_cfg={
            "num_layers": 1,
            "num_attention_heads": 4,
            "attention_head_dim": 8,
            "dropout": 0.0,
            "positional_embeddings": None,
        },
    )
    return GR00TN17ActionHead(config).eval()


def constants():
    image_mask = torch.tensor([[False, True, True, False, False]])
    return {
        "image_mask": image_mask,
        "backbone_attention_mask": torch.tensor([[True, True, True, True, False]]),
        "embodiment_id": torch.tensor([2]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "position_ids": torch.arange(5)[None, None].expand(3, 1, -1),
        "visual_pos_masks": image_mask,
        "image_token_mask": image_mask[..., None].expand(1, 5, 32),
        "text_embeds": torch.randn(1, 5, 32),
    }


def test_vision_part_preserves_deepstack_and_state(recipe):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    vision = Qwen3VLVisionModel(
        Qwen3VLVisionConfig(
            depth=3,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1, 2],
        )
    ).eval()

    class Observation(torch.nn.Module):
        def forward(self, pixels, state):
            return pixels, state

    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    pixels, state = torch.randn(32, 24), torch.randn(1, 1, 6)
    part = recipe.GrootVision(vision, Observation(), grid, False).eval()
    reference = recipe.Qwen3VisionForExport(vision, grid).eval()
    with torch.no_grad():
        images, deepstack = reference(pixels)
        actual = part(pixels, state)
        captured = torch.export.export(part, (pixels, state)).module()(pixels, state)
    for got, saved, expected in zip(actual, captured, (images, *deepstack, state), strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)


def test_language_slices_preserve_global_deepstack_indices_and_pre_norm_output(recipe):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    text = Qwen3VLTextModel(
        Qwen3VLTextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
        )
    ).eval()
    head, fixed = tiny_head(), constants()
    images, state = torch.randn(2, 32), torch.randn(1, 1, 6)
    deepstack = tuple(torch.randn(2, 32) for _ in range(3))
    with torch.no_grad():
        embeddings = fixed["text_embeds"].masked_scatter(fixed["image_token_mask"], images)
        expected = recipe.LLMForExport(text)(
            embeddings, fixed["attention_mask"], fixed["position_ids"], fixed["visual_pos_masks"], *deepstack
        )
        expected = head.vl_self_attention(head.vlln(expected))
        expected_state = head.state_encoder(state, fixed["embodiment_id"])
        hidden = images
        for start, stop in ((0, 1), (1, 3), (3, 4)):
            part = recipe.GrootLanguageModel(text, head, fixed, start, stop, False, len(deepstack)).eval()
            inputs = (
                hidden,
                *(deepstack[i] for i in part.deepstack_layers),
                *((state,) if part.last else ()),
            )
            hidden = part(*inputs)
            saved = torch.export.export(part, inputs).module()(*inputs)
            torch.testing.assert_close(saved, hidden, rtol=0, atol=0)
    torch.testing.assert_close(hidden, (expected, expected_state), rtol=0, atol=0)


def test_vision_wrapper_matches_the_hugging_face_vision_model(recipe):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.manual_seed(14)
    vision = Qwen3VLVisionModel(
        Qwen3VLVisionConfig(
            depth=3,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1, 2],
        )
    ).eval()
    grid, pixels = torch.tensor([[1, 4, 4], [1, 4, 4]]), torch.randn(32, 24)
    with torch.no_grad():
        expected = vision(pixels, grid)
        images, deepstack = recipe.Qwen3VisionForExport(vision, grid).eval()(pixels)
    torch.testing.assert_close(images, expected.pooler_output, rtol=0, atol=1e-6)
    torch.testing.assert_close(deepstack, torch.stack(expected.deepstack_features), rtol=0, atol=1e-6)


def test_language_wrapper_matches_the_hugging_face_text_model(recipe):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(14)
    text = Qwen3VLTextModel(
        Qwen3VLTextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
        )
    ).eval()
    fixed = constants()
    deepstack = [torch.randn(2, 32) for _ in range(3)]
    embeddings = fixed["text_embeds"].masked_scatter(fixed["image_token_mask"], torch.randn(2, 32))
    last = []
    # The wrapper returns the last layer's output, before the model's final norm.
    text.layers[-1].register_forward_hook(lambda module, args, output: last.append(output))
    with torch.no_grad():
        text(
            inputs_embeds=embeddings,
            attention_mask=fixed["attention_mask"],
            position_ids=fixed["position_ids"],
            visual_pos_masks=fixed["visual_pos_masks"],
            deepstack_visual_embeds=deepstack,
        )
        actual = recipe.LLMForExport(text)(
            embeddings, fixed["attention_mask"], fixed["position_ids"], fixed["visual_pos_masks"], *deepstack
        )
    expected = last[0][0] if isinstance(last[0], tuple) else last[0]
    valid = fixed["attention_mask"][0].bool()
    torch.testing.assert_close(actual[:, valid], expected[:, valid], rtol=0, atol=1e-5)


def test_language_slices_take_only_the_inputs_their_layers_read(recipe):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    text = Qwen3VLTextModel(
        Qwen3VLTextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
        )
    ).eval()
    head, fixed = tiny_head(), constants()
    slices = {
        (start, stop): recipe.GrootLanguageModel(text, head, fixed, start, stop, False, 3)
        for start, stop in ((0, 2), (2, 6), (6, 8))
    }
    assert [part.deepstack_layers for part in slices.values()] == [[0, 1], [2], []]
    assert [part.last for part in slices.values()] == [False, False, True]


@pytest.mark.parametrize("alternate,interleave", [(False, False), (False, True), (True, True)])
def test_step_slices_and_program_chain_match_action_head(recipe, alternate, interleave):
    from transformers.feature_extraction_utils import BatchFeature

    from lerobot.inference.export.engine import ProgramChain

    torch.manual_seed(14)
    head, fixed = tiny_head(alternate, interleave), constants()
    features, state, noise = torch.randn(1, 5, 32), torch.randn(1, 1, 32), torch.randn(3, 4)
    with torch.no_grad():
        expected = head.get_action_with_features(
            features,
            state,
            fixed["embodiment_id"],
            BatchFeature(
                {
                    "image_mask": fixed["image_mask"],
                    "backbone_attention_mask": fixed["backbone_attention_mask"],
                }
            ),
            BatchFeature({}),
            noise=noise[None],
        )["action_pred"][0]
    split_head = deepcopy(head)
    parts = [
        recipe.GrootStep(split_head, fixed, start, stop, False).eval()
        for start, stop in ((0, 2), (2, 3), (3, 5))
    ]
    for part in parts:
        for layer in part.modules():
            if isinstance(layer, recipe.CategorySpecificLinear):
                assert layer.W.shape[0] == layer.b.shape[0] == 1
    calls = []

    class Program:
        def __init__(self, module, name):
            self.module, self.name = module, name

        def __call__(self, *arrays):
            calls.append(self.name)
            with torch.no_grad():
                result = self.module(*(torch.from_numpy(x) for x in arrays))
            if isinstance(result, tuple):
                return tuple(x.numpy() for x in result)
            return result.numpy()

    programs = {f"step_{i}": Program(part, f"step_{i}") for i, part in enumerate(parts)}
    programs["actions"] = Program(recipe.GrootActions(2, 3, action_pipeline()), "actions")
    names = {
        "step_0": (["vl", "state", "x_t", "time"], ["hidden", "temb"]),
        "step_1": (["vl", "hidden", "temb"], ["hidden"]),
        "step_2": (["vl", "hidden", "temb"], ["velocity"]),
        "actions": (["x_t", "robot_state"], ["actions"]),
    }
    info = {
        "inputs": ["vl", "state", "robot_state", "noise"],
        "num_steps": 3,
        "dt": 1 / 3,
        "timesteps": [0, 333, 666],
        "programs": {
            name: {"inputs": inputs, "outputs": outputs} for name, (inputs, outputs) in names.items()
        },
    }
    chain = ProgramChain(programs, info)
    actual = chain.initialize(features.numpy(), state.numpy(), np.zeros(6, np.float32), noise.numpy())
    np.testing.assert_allclose(actual, expected[:2, :3].numpy(), rtol=0, atol=1e-6)
    assert calls == ["step_0", "step_1", "step_2"] * 3 + ["actions"]
    inputs = (features, state, noise, torch.tensor([333.0]))
    with torch.no_grad():
        for part in parts:
            result = part(*inputs)
            saved = torch.export.export(part, inputs).module()(*inputs)
            torch.testing.assert_close(saved, result, rtol=0, atol=0)
            if not part.last:
                hidden, temb = result if part.start == 0 else (result, inputs[-1])
                inputs = (features, hidden, temb)


def test_diffusion_parts_export_to_onnx_and_run_as_chain(recipe, monkeypatch, tmp_path):
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    from lerobot.inference.export.engine import ProgramChain
    from lerobot.utils.constants import ACTION

    class Prefix(torch.nn.Module):
        last = True
        deepstack_layers = []

        def forward(self, features, state):
            return features, state

    torch.manual_seed(14)
    head = tiny_head(num_layers=25)
    features, state, noise = torch.randn(1, 5, 32), torch.randn(1, 1, 32), torch.randn(3, 4)
    text = SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(32, 32))
    export = recipe.GrootExport.__new__(recipe.GrootExport)
    export.__dict__.update(
        policy=SimpleNamespace(_groot_model=SimpleNamespace(backbone=SimpleNamespace(visual=None))),
        frame_processor=None,
        batch={"image_grid_thw": None, "input_ids": torch.zeros(1, 5, dtype=torch.long)},
        config=SimpleNamespace(use_bf16=False, output_features={ACTION: SimpleNamespace(shape=(3,))}),
        model_config=SimpleNamespace(
            select_layer=1, diffusion_model_cfg={"num_layers": 25}, add_pos_embed=False
        ),
        constants=constants(),
        inputs=(features, state, noise),
        input_names=["features", "state", "noise"],
        noise=noise,
        timesteps=[0, 333, 666],
        num_steps=3,
        horizon=2,
        postprocessor=action_pipeline(),
        frame={"observation.state": np.zeros(3, np.float32)},
    )
    # Keep the prefix tiny while exercising the real recipe's diffusion modules and names.
    monkeypatch.setattr(recipe, "GrootVision", lambda *args: Prefix())
    monkeypatch.setattr(recipe, "GrootLanguageModel", lambda *args: Prefix())
    monkeypatch.setattr(
        export,
        "load_part",
        lambda *args: SimpleNamespace(
            _groot_model=SimpleNamespace(
                backbone=SimpleNamespace(language_model=text), action_head=deepcopy(head)
            )
        ),
    )
    sessions = {}

    def check_engine(onnx_path, engine_path, **options):
        onnx.checker.check_model(onnx.load(onnx_path), full_check=True)
        sessions[engine_path.stem] = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

    monkeypatch.setitem(sys.modules, "build_engine", SimpleNamespace(build_engine=check_engine))
    script = Path(recipe.__file__).with_name("groot_onnx_tensorrt.py")
    compile_part = runpy.run_path(str(script))["compile_part"]
    programs, metadata = {}, {}

    class Program:
        def __init__(self, session, input_names):
            self.session, self.input_names = session, input_names

        def __call__(self, *arrays):
            outputs = self.session.run(None, dict(zip(self.input_names, arrays, strict=True)))
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

    for name, module, inputs, input_names, output_names in export.parts():
        if not (name.startswith("step_") or name == "actions"):
            continue
        compile_part(module, inputs, input_names, output_names, tmp_path / f"{name}.engine")
        session = sessions[name]
        assert [value.name for value in session.get_inputs()] == input_names
        assert [value.name for value in session.get_outputs()] == output_names
        programs[name] = Program(session, input_names)
        metadata[name] = {"inputs": input_names, "outputs": output_names}
        with torch.no_grad():
            expected = module(*inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = session.run(None, dict(zip(input_names, [x.numpy() for x in inputs], strict=True)))
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-5)
    assert list(programs) == ["step_0", "step_1", "step_2", "step_3", "actions"]
    info = {
        "inputs": ["vl_embeds", "state_features", "observation.state", "noise"],
        "programs": metadata,
        "num_steps": export.num_steps,
        "timesteps": export.timesteps,
        "dt": 1 / export.num_steps,
    }
    robot_state = export.frame["observation.state"]
    actual = ProgramChain(programs, info)(features.numpy(), state.numpy(), robot_state, noise.numpy())
    np.testing.assert_allclose(actual, export.expected_actions, rtol=1e-5, atol=1e-5)


def test_export_parts_uses_a_fresh_process_for_each_part(recipe, monkeypatch, tmp_path):
    names = ["vision", "language_model_0", "step_0", "step_1", "actions"]
    calls = []
    monkeypatch.setattr(recipe, "part_names", lambda path: names)
    monkeypatch.setattr(recipe.subprocess, "run", lambda command, check: calls.append((command, check)))
    monkeypatch.setattr(sys, "argv", ["export", "--policy.path=checkpoint"])
    output = tmp_path / "export"
    args = Namespace(part=None, policy_path="checkpoint", output_dir=output, job_name="groot")
    recipe.export_parts(args, "export", "{name}.pte", None)
    assert output.is_dir()
    assert calls == [
        (
            [
                sys.executable,
                "export",
                "--policy.path=checkpoint",
                f"--output_dir={output}",
                f"--part={name}",
            ],
            True,
        )
        for name in names
    ]


@pytest.mark.parametrize("device_resident", [False, True])
def test_export_parts_saves_the_device_contract(recipe, monkeypatch, tmp_path, device_resident):
    export = recipe.GrootExport.__new__(recipe.GrootExport)
    export.__dict__.update(
        output_dir=tmp_path,
        backend="executorch_tensorrt",
        frame={"observation.state": np.zeros(2, np.float32)},
        noise=torch.zeros(3, 4),
        expected_actions=np.zeros((3, 2), np.float32),
        config=SimpleNamespace(save_pretrained=lambda path: None),
        input_features={},
        input_names=["observation.state", "noise"],
        image_resize=None,
        task="pick",
        tolerance=5.0,
        num_steps=2,
        timesteps=[0, 500],
        start=recipe.time.perf_counter(),
    )
    monkeypatch.setattr(recipe, "part_names", lambda path: ["actions"])
    monkeypatch.setattr(recipe, "GrootExport", lambda args, split: export)
    monkeypatch.setattr(
        export,
        "parts",
        lambda: iter([("actions", torch.nn.Identity(), (export.noise,), ["x_t"], ["action"])]),
    )
    compiled = []
    args = Namespace(part="actions", policy_path="checkpoint")
    recipe.export_parts(
        args, "export", "{name}.pte", lambda *args: compiled.append(args[-1]), device_resident=device_resident
    )
    info = json.loads((tmp_path / "export.json").read_text())
    assert compiled == [tmp_path / "actions.pte"]
    assert info["device_resident"] is device_resident
    assert info["programs"]["actions"]["file"] == "actions.pte"
    assert info["dt"] == 0.5 and info["timesteps"] == [0, 500]
    # The task is baked into the programs and the frame goes in raw, so the engine must know both.
    assert info["task"] == "pick" and info["task_fixed"] is True
    assert info["raw_frame"] is True


def test_part_loader_preserves_precision_without_changing_policy_config(recipe, monkeypatch):
    calls = []
    export = recipe.GrootExport.__new__(recipe.GrootExport)
    export.config = SimpleNamespace(use_bf16=True, dtype=torch.float32)
    export.model_file = "checkpoint"
    monkeypatch.setattr(
        recipe.GrootPolicy,
        "load_checkpoint_modules",
        lambda path, config, modules: calls.append((path, config, modules)),
    )
    export.load_part(recipe.VISION)
    assert export.config.use_bf16
    path, config, modules = calls[0]
    assert path == export.model_file and modules == [recipe.VISION]
    assert not config.use_bf16 and config.dtype == torch.float32


def test_actions_apply_the_postprocessor_after_slicing(recipe):
    from lerobot.processor import ProcessorStep, TransitionKey

    class ScaleByHorizon(ProcessorStep):
        """Depends on the chunk's length, so slicing after it gives a different answer than before it."""

        def __call__(self, transition):
            actions = transition[TransitionKey.ACTION]
            return {**transition, TransitionKey.ACTION: actions * actions.shape[1] + 3}

        def transform_features(self, features):
            return features

    part = recipe.GrootActions(2, 3, action_pipeline(ScaleByHorizon()))
    values = torch.arange(20).reshape(4, 5).float()
    state = torch.zeros(6)
    torch.testing.assert_close(part(values, state), values[:2, :3] * 2 + 3)
    saved = torch.export.export(part, (values, state)).module()(values, state)
    torch.testing.assert_close(saved, part(values, state))


def test_numpy_action_decoding_is_refused_before_any_part_is_built(recipe):
    from lerobot.policies.groot.processor_groot import (
        GrootActionUnpackUnnormalizeStep,
        GrootN17ActionDecodeStep,
    )

    recipe.check_traceable_decode(action_pipeline(GrootActionUnpackUnnormalizeStep()))
    with pytest.raises(ValueError, match="cannot be exported"):
        recipe.check_traceable_decode(action_pipeline(GrootN17ActionDecodeStep()))


def test_task_must_be_one_of_the_dataset_tasks(recipe):
    import pandas as pd

    smolvla = importlib.import_module("smolvla_recipe")
    metadata = SimpleNamespace(tasks=pd.DataFrame(index=["pick the cup", "open the drawer"]))
    assert smolvla.dataset_task(metadata, None) == "pick the cup"
    assert smolvla.dataset_task(metadata, "open the drawer") == "open the drawer"
    with pytest.raises(ValueError, match="one of the dataset's tasks"):
        smolvla.dataset_task(metadata, "close the drawer")
    assert recipe.dataset_task is smolvla.dataset_task


@pytest.mark.parametrize("policy", ["pi05", "groot"])
def test_relative_actions_follow_the_robot_state_after_export(recipe, policy):
    from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep, TransitionKey

    relative = RelativeActionsProcessorStep(enabled=True)
    relative({TransitionKey.OBSERVATION: {"observation.state": torch.full((1, 3), 50.0)}})
    postprocessor = action_pipeline(AbsoluteActionsProcessorStep(enabled=True, relative_step=relative))
    if policy == "groot":
        part = recipe.GrootActions(2, 3, postprocessor)
    else:
        pi05 = importlib.import_module("pi05_recipe")
        config = SimpleNamespace(n_action_steps=2, output_features={"action": SimpleNamespace(shape=(3,))})
        part = pi05.PI05Actions(SimpleNamespace(config=config), postprocessor)
    offsets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    program = torch.export.export(part, (offsets, torch.full((3,), 50.0))).module()
    moved = torch.tensor([-5.0, 0.0, 5.0])
    torch.testing.assert_close(program(offsets, moved), offsets[:2] + moved)


@pytest.mark.parametrize("backend", ["onnx_tensorrt", "torch_tensorrt"])
def test_split_script_launches_every_recipe_part(recipe, monkeypatch, tmp_path, backend):
    monkeypatch.setattr(
        recipe.PreTrainedConfig, "from_pretrained", lambda path: SimpleNamespace(base_model_path="base")
    )
    monkeypatch.setattr(
        recipe.GR00TN17Config,
        "from_pretrained",
        lambda path: SimpleNamespace(select_layer=16, diffusion_model_cfg={"num_layers": 25}),
    )
    monkeypatch.setitem(sys.modules, "build_engine", SimpleNamespace(build_engine=MagicMock()))
    monkeypatch.setitem(sys.modules, "torch_tensorrt", MagicMock())
    load = MagicMock(side_effect=AssertionError("The parent must not load the whole policy"))
    monkeypatch.setattr(recipe, "GrootExport", load)
    calls = MagicMock()
    monkeypatch.setattr(recipe.subprocess, "run", calls)
    script = Path(recipe.__file__).with_name(f"groot_{backend}.py")
    output = tmp_path / "export"
    argv = [str(script), "--policy.path=checkpoint", "--dataset.repo_id=dataset", f"--output_dir={output}"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(script), run_name="__main__")
    names = [
        "vision",
        *(f"language_model_{i}" for i in range(3)),
        *(f"step_{i}" for i in range(4)),
        "actions",
    ]
    assert calls.call_count == len(names)
    for call, name in zip(calls.call_args_list, names, strict=True):
        assert call.args == ([sys.executable, *argv, f"--output_dir={output}", f"--part={name}"],)
        assert call.kwargs == {"check": True}
    load.assert_not_called()


@pytest.mark.parametrize("backend,suffix", [("onnx_tensorrt", ".engine"), ("torch_tensorrt", ".pt2")])
def test_split_script_compiles_named_parts_and_writes_chain(recipe, monkeypatch, tmp_path, backend, suffix):
    class Part(torch.nn.Module):
        def forward(self, x):
            return x + 1, x * 2

    names = ["vision", "language_model_0", "step_0", "actions"]
    module = Part().eval()
    inputs = (torch.arange(4, dtype=torch.float32),)
    input_names, output_names = ["x_t"], ["hidden", "state"]
    export = recipe.GrootExport.__new__(recipe.GrootExport)
    export.__dict__.update(
        output_dir=tmp_path,
        backend=backend,
        frame={"observation.state": np.zeros(2, np.float32)},
        noise=inputs[0],
        expected_actions=np.ones(4, np.float32),
        config=SimpleNamespace(save_pretrained=lambda path: None),
        input_features={},
        input_names=["observation.state", "noise"],
        image_resize=None,
        task="pick",
        tolerance=5.0,
        num_steps=2,
        timesteps=[0, 500],
        start=recipe.time.perf_counter(),
    )
    monkeypatch.setattr(recipe, "part_names", lambda path: names)
    load = MagicMock(return_value=export)
    monkeypatch.setattr(recipe, "GrootExport", load)
    monkeypatch.setattr(
        export, "parts", lambda: iter((name, module, inputs, input_names, output_names) for name in names)
    )
    events = []
    monkeypatch.setattr(module, "cpu", lambda: events.append("cpu"))
    monkeypatch.setattr(recipe, "free_memory", lambda: events.append("free"))
    onnx = MagicMock(side_effect=lambda *args, **kwargs: events.append("onnx"))
    monkeypatch.setattr(torch.onnx, "export", onnx)
    build = MagicMock(side_effect=lambda *args, **kwargs: events.append("build"))
    monkeypatch.setitem(sys.modules, "build_engine", SimpleNamespace(build_engine=build))
    trt = MagicMock()
    monkeypatch.setitem(sys.modules, "torch_tensorrt", trt)
    script = Path(recipe.__file__).with_name(f"groot_{backend}.py")
    for name in names:
        monkeypatch.setattr(
            sys,
            "argv",
            [str(script), "--policy.path=checkpoint", "--dataset.repo_id=dataset", f"--part={name}"],
        )
        runpy.run_path(str(script), run_name="__main__")
        assert load.call_args.kwargs == {"split": True}
        if name != "actions":
            assert not (tmp_path / "export.json").exists()
    if backend == "onnx_tensorrt":
        assert onnx.call_count == build.call_count == len(names)
        for name, call in zip(names, onnx.call_args_list, strict=True):
            assert call.args == (module, inputs, tmp_path / f"{name}.onnx")
            assert call.kwargs == {"dynamo": True, "input_names": input_names, "output_names": output_names}
        for name, call in zip(names, build.call_args_list, strict=True):
            assert call.args == (tmp_path / f"{name}.onnx", tmp_path / f"{name}.engine")
            assert call.kwargs == {
                "workspace_gib": recipe.TENSORRT_OPTIONS["workspace_size"] / (1 << 30),
                "optimization_level": recipe.TENSORRT_OPTIONS["optimization_level"],
            }
        assert events == [
            event for i in range(len(names)) for event in [*(["free"] * i), "onnx", "cpu", "free", "build"]
        ]
    else:
        assert trt.dynamo.compile.call_count == trt.save.call_count == len(names)
        for call in trt.dynamo.compile.call_args_list:
            assert call.kwargs == {"arg_inputs": inputs, **recipe.TENSORRT_OPTIONS}
            torch.testing.assert_close(call.args[0].module()(*inputs), module(*inputs))
        for name, call in zip(names, trt.save.call_args_list, strict=True):
            assert call.args == (trt.dynamo.compile.return_value, str(tmp_path / f"{name}.pt2"))
            assert call.kwargs == {"output_format": "aot_inductor", "arg_inputs": inputs}
    info = json.loads((tmp_path / "export.json").read_text())
    assert info["backend"] == backend
    assert info["device_resident"] is (backend == "torch_tensorrt")
    assert info["programs"] == {
        name: {"file": f"{name}{suffix}", "inputs": input_names, "outputs": output_names} for name in names
    }
    assert "file" not in info
    assert info["dt"] == 0.5 and info["timesteps"] == [0, 500]


def test_part_names_cover_remainders_and_reject_unknown_parts(recipe, monkeypatch):
    monkeypatch.setattr(
        recipe.PreTrainedConfig, "from_pretrained", lambda path: SimpleNamespace(base_model_path="base")
    )
    monkeypatch.setattr(
        recipe.GR00TN17Config,
        "from_pretrained",
        lambda path: SimpleNamespace(select_layer=16, diffusion_model_cfg={"num_layers": 25}),
    )
    assert recipe.part_names("checkpoint") == [
        "vision",
        "language_model_0",
        "language_model_1",
        "language_model_2",
        "step_0",
        "step_1",
        "step_2",
        "step_3",
        "actions",
    ]
    with pytest.raises(ValueError, match="Unknown part"):
        recipe.export_parts(Namespace(part="bad", policy_path="checkpoint"), "export", "{name}.pte", None)
