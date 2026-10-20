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
import runpy
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def recipe(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    return importlib.import_module("groot")


def whole_language_model(
    recipe, text, embeddings, attention_mask, position_ids, visual_pos_masks, *deepstack
):
    """All layers in one run, with the recipe's mask and deepstack add: the last layer's output before the norm."""
    hidden = embeddings
    mask = recipe.causal_mask(hidden.dtype, hidden.device, *hidden.shape[:2], attention_mask)
    positions = text.rotary_emb(hidden, position_ids)
    for index, layer in enumerate(text.layers):
        hidden = layer(
            hidden,
            attention_mask=mask,
            position_ids=position_ids[0],
            past_key_values=None,
            position_embeddings=positions,
        )
        if index < len(deepstack):
            hidden = recipe.deepstack_add(hidden, visual_pos_masks, deepstack[index].to(hidden.dtype))
    return hidden


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
        expected = whole_language_model(
            recipe,
            text,
            embeddings,
            fixed["attention_mask"],
            fixed["position_ids"],
            fixed["visual_pos_masks"],
            *deepstack,
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


def test_language_model_layers_match_the_hugging_face_text_model(recipe):
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
    # GR00T reads the last layer's output, before the model's final norm.
    text.layers[-1].register_forward_hook(lambda module, args, output: last.append(output))
    with torch.no_grad():
        text(
            inputs_embeds=embeddings,
            attention_mask=fixed["attention_mask"],
            position_ids=fixed["position_ids"],
            visual_pos_masks=fixed["visual_pos_masks"],
            deepstack_visual_embeds=deepstack,
        )
        actual = whole_language_model(
            recipe,
            text,
            embeddings,
            fixed["attention_mask"],
            fixed["position_ids"],
            fixed["visual_pos_masks"],
            *deepstack,
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
        recipe.GrootStep(split_head, fixed, start, stop, False, 1 / 3).eval()
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

        def run_device(self, *values):
            calls.append(self.name)
            with torch.no_grad():
                result = self.module(*(torch.as_tensor(x) for x in values))
            return result if isinstance(result, tuple) else (result,)

        def __call__(self, *values):
            (output,) = self.run_device(*values)
            return output.numpy()

        def upload(self, value):
            return value

    programs = {f"step_{i}": Program(part, f"step_{i}") for i, part in enumerate(parts)}
    programs["actions"] = Program(recipe.GrootActions(2, 3, action_pipeline()), "actions")
    names = {
        "step_0": (["vl", "state", "x_t", "timestep"], ["hidden", "temb"]),
        "step_1": (["vl", "hidden", "temb"], ["hidden"]),
        "step_2": (["vl", "hidden", "temb", "x_t"], ["x_next"]),
        "actions": (["x_t", "robot_state"], ["actions"]),
    }
    info = {
        "inputs": ["vl", "state", "robot_state", "noise"],
        "steps": ["step_0", "step_1", "step_2"],
        "timesteps": [0, 333, 666],
        "programs": {
            name: {"inputs": inputs, "outputs": outputs} for name, (inputs, outputs) in names.items()
        },
    }
    chain = ProgramChain(programs, info)
    actual = chain(features.numpy(), state.numpy(), np.zeros(6, np.float32), noise.numpy())
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
                inputs = (
                    (features, hidden, temb)
                    if not parts[parts.index(part) + 1].last
                    else (features, hidden, temb, noise)
                )


def test_diffusion_parts_export_to_onnx_and_run_as_chain(recipe, monkeypatch, tmp_path):
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    from lerobot.inference.export.engine import ProgramChain
    from lerobot.utils.constants import ACTION

    class Prefix(torch.nn.Module):
        last = True
        deepstack_layers = []

        def forward(self, features, state, robot_state=None):
            return features, state

    torch.manual_seed(14)
    head = tiny_head(num_layers=25)
    features, state, noise = torch.randn(1, 5, 32), torch.randn(1, 1, 32), torch.randn(3, 4)
    text = SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(32, 32))
    export = recipe.Export.__new__(recipe.Export)
    export.__dict__.update(
        policy=SimpleNamespace(_groot_model=SimpleNamespace(backbone=SimpleNamespace(visual=None))),
        frame_processor=None,
        image_resize=None,
        batch={"image_grid_thw": None, "input_ids": torch.zeros(1, 5, dtype=torch.long)},
        config=SimpleNamespace(use_bf16=False, output_features={ACTION: SimpleNamespace(shape=(3,))}),
        model_config=SimpleNamespace(
            select_layer=1, diffusion_model_cfg={"num_layers": 25}, add_pos_embed=False
        ),
        constants=constants(),
        noise=noise,
        timesteps=[0, 333, 666],
        num_steps=3,
        horizon=2,
        postprocessor=action_pipeline(),
        frame={
            "features": features.numpy(),
            "state": state.numpy(),
            "observation.state": np.zeros(3, np.float32),
        },
    )
    monkeypatch.setattr(recipe, "resize_images", lambda frame, geometry: frame)
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

    def check_engine(onnx_path, engine_path):
        onnx.checker.check_model(onnx.load(onnx_path), full_check=True)
        sessions[engine_path.stem] = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

    monkeypatch.setitem(sys.modules, "build_engine", SimpleNamespace(build_engine=check_engine))
    compile_part = runpy.run_path(str(Path(recipe.__file__).with_name("onnx_trt.py")))["compile_part"]
    programs, metadata = {}, {}

    class Program:
        def __init__(self, session, input_names):
            self.session, self.input_names = session, input_names

        def run_device(self, *arrays):
            return tuple(
                self.session.run(None, dict(zip(self.input_names, map(np.asarray, arrays), strict=True)))
            )

        def __call__(self, *arrays):
            (output,) = self.run_device(*arrays)
            return output

        def upload(self, value):
            return value

    for part in export.parts():
        if not (part.name.startswith("step_") or part.name == "actions"):
            continue
        compile_part(part, tmp_path / f"{part.name}.engine")
        session = sessions[part.name]
        assert [value.name for value in session.get_inputs()] == part.input_names
        assert [value.name for value in session.get_outputs()] == part.output_names
        programs[part.name] = Program(session, part.input_names)
        metadata[part.name] = {"inputs": part.input_names, "outputs": part.output_names}
        with torch.no_grad():
            expected = part.module(*part.inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = session.run(None, dict(zip(part.input_names, [x.numpy() for x in part.inputs], strict=True)))
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-5)
    assert list(programs) == ["step_0", "step_1", "step_2", "step_3", "actions"]
    info = {
        "inputs": ["vl_embeds", "state_features", "observation.state", "noise"],
        "programs": metadata,
        "steps": ["step_0", "step_1", "step_2", "step_3"],
        "timesteps": export.timesteps,
    }
    robot_state = export.frame["observation.state"]
    actual = ProgramChain(programs, info)(features.numpy(), state.numpy(), robot_state, noise.numpy())
    np.testing.assert_allclose(actual, export.expected_actions, rtol=1e-5, atol=1e-5)


def test_part_loader_preserves_precision_without_changing_policy_config(recipe, monkeypatch):
    calls = []
    export = recipe.Export.__new__(recipe.Export)
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


def n17_decoder(relative=False):
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep

    rep = {"rep": "relative", "state_key": "arm"} if relative else {"rep": "absolute"}
    return GrootN17ActionDecodeStep(
        env_action_dim=3,
        raw_stats={"action": {"arm": {"min": [-2.0, -1.0, 0.0], "max": [2.0, 1.0, 10.0]}}},
        modality_config={"action": {"modality_keys": ["arm"], "action_configs": [rep]}},
        use_relative_action=relative,
    )


@pytest.mark.parametrize("backend", ["executorch_trt", "torch_trt", "executorch_cuda", "onnx_trt"])
def test_numpy_action_decoding_is_refused_only_where_it_cannot_compile(recipe, backend):
    from lerobot.policies.groot.processor_groot import GrootActionUnpackUnnormalizeStep

    recipe.check_traceable_decode(action_pipeline(GrootActionUnpackUnnormalizeStep()), backend)
    if backend == "onnx_trt":
        recipe.check_traceable_decode(action_pipeline(n17_decoder()), backend)
    else:
        with pytest.raises(ValueError, match="only the ONNX-TensorRT route"):
            recipe.check_traceable_decode(action_pipeline(n17_decoder()), backend)
    with pytest.raises(ValueError, match="relative actions"):
        recipe.check_traceable_decode(action_pipeline(n17_decoder(relative=True)), backend)


def test_numpy_action_decoding_exports_to_onnx_and_follows_new_actions(recipe, tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    module = recipe.GrootActions(2, 3, action_pipeline(n17_decoder())).eval()
    actions, state = torch.rand(4, 3) * 2 - 1, torch.zeros(3)
    path = tmp_path / "actions.onnx"
    torch.onnx.export(
        module, (actions, state), path, dynamo=True, input_names=["x", "state"], output_names=["a"]
    )
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    moved = torch.rand(4, 3) * 2 - 1
    with torch.no_grad():
        expected = module(moved, state).numpy()
    np.testing.assert_allclose(
        session.run(None, {"x": moved.numpy(), "state": state.numpy()})[0], expected, atol=1e-6
    )


@pytest.mark.parametrize(
    "shape,names",
    [((12, 16, 3), ["height", "width", "channels"]), ((3, 12, 16), ["channels", "height", "width"])],
)
def test_robot_frame_cameras_are_hwc_whatever_the_dataset_layout(recipe, shape, names):
    smolvla = importlib.import_module("export")
    features = {
        "observation.images.front": {"dtype": "video", "shape": shape, "names": names},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    frame = smolvla.frame_from_dataset(features)
    assert frame["observation.images.front"].shape == (12, 16, 3)
    assert frame["observation.images.front"].dtype == np.uint8
    assert frame["observation.state"].shape == (6,) and "action" not in frame


def test_task_must_be_one_of_the_dataset_tasks(recipe):
    import pandas as pd

    smolvla = importlib.import_module("export")
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
        pi05 = importlib.import_module("pi05")
        config = SimpleNamespace(n_action_steps=2, output_features={"action": SimpleNamespace(shape=(3,))})
        part = pi05.PI05Actions(SimpleNamespace(config=config), postprocessor)
    offsets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    program = torch.export.export(part, (offsets, torch.full((3,), 50.0))).module()
    moved = torch.tensor([-5.0, 0.0, 5.0])
    torch.testing.assert_close(program(offsets, moved), offsets[:2] + moved)


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("policy", ["pi05", "groot"])
def test_actions_program_takes_the_state_only_when_it_reads_it(recipe, policy, relative):
    """An input the program never reads stays on the host, where the rollout cannot hand it one."""
    from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep

    relative_step = RelativeActionsProcessorStep(enabled=True)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
    postprocessor = action_pipeline(absolute_step) if relative else action_pipeline()
    if policy == "groot":
        part = recipe.GrootActions(2, 3, postprocessor)
    else:
        pi05 = importlib.import_module("pi05")
        config = SimpleNamespace(n_action_steps=2, output_features={"action": SimpleNamespace(shape=(3,))})
        part = pi05.PI05Actions(SimpleNamespace(config=config), postprocessor)
    inputs, names = importlib.import_module("export").actions_inputs(
        part.absolute_step, torch.zeros(3, 3), torch.zeros(3)
    )
    program = torch.export.export(part, inputs)
    user_inputs = set(program.graph_signature.user_inputs)
    unread = [n.name for n in program.graph.nodes if n.name in user_inputs and not n.users]
    assert unread == []
    assert names == (["x_t", "observation.state"] if relative else ["x_t"])


def test_part_names_cover_remainders_and_reject_unknown_parts(recipe, monkeypatch):
    monkeypatch.setattr(
        recipe.PreTrainedConfig, "from_pretrained", lambda path: SimpleNamespace(base_model_path="base")
    )
    monkeypatch.setattr(
        recipe.GR00TN17Config,
        "from_pretrained",
        lambda path: SimpleNamespace(select_layer=16, diffusion_model_cfg={"num_layers": 25}),
    )
    assert recipe.part_names(Namespace(policy_path="checkpoint")) == [
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
    export = importlib.import_module("export")
    with pytest.raises(SystemExit, match="Unknown part"):
        export.export("script", Namespace(part="bad", policy_path="checkpoint"), recipe, None, ".pte")
