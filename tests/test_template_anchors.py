"""Run with `python -m pytest tests/test_template_anchors.py` on Apple Silicon."""

import json

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from fast_dllm_mlx.generate import (
    DreamGenerationConfig,
    _apply_template_anchors,
    _compiled_select_confident_updates_greedy,
    _encode_template_anchors,
    _load_template_anchors,
    diffusion_generate,
    diffusion_generate_step,
    sample_tokens,
    stream_diffusion_generate,
)
from fast_dllm_mlx.model import Model, ModelArgs


class TinyDream:
    """Run the real Dream attention/cache path without downloading weights."""

    def __init__(self):
        mx.random.seed(42)
        self.inner = Model(
            ModelArgs(
                model_type="dream",
                hidden_size=32,
                num_hidden_layers=2,
                intermediate_size=64,
                num_attention_heads=2,
                rms_norm_eps=1e-6,
                vocab_size=32,
                max_position_embeddings=64,
                num_key_value_heads=2,
                rope_theta=10000,
                rope_scaling=None,
                tie_word_embeddings=False,
            )
        )
        self.layers = self.inner.layers
        self.inputs = []

    def __call__(self, tokens, **kwargs):
        self.inputs.append((kwargs.get("cache_position", 0), np.array(tokens)))
        logits = self.inner(tokens, **kwargs)
        # Keep the randomly initialized test model from predicting its mask ID.
        return mx.where(mx.arange(32) == 31, -1e9, logits)


def config(**overrides):
    return DreamGenerationConfig(
        max_length=10,
        max_new_tokens=8,
        steps=16,
        block_length=4,
        temperature=0,
        threshold=1.1,
        mask_token_id=31,
        use_chat_template=False,
        **overrides,
    )


@pytest.fixture
def tokenizer():
    vocab = {f"t{i}": i for i in range(32)}
    vocab.update({"[UNK]": 0, "Analysis:": 3, "Answer:": 7, "[MASK]": 31})
    # Unique IDs make this a real fast tokenizer, rather than an encode stub.
    for token, index in list(vocab.items()):
        if token.startswith("t") and index in (0, 3, 7, 31):
            del vocab[token]
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", mask_token="[MASK]"
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_anchors_survive_every_forward_and_cross_block_boundaries(compiled):
    model = TinyDream()
    anchors = {0: [3, 4], 3: [5, 6], 7: [7]}
    prompt = mx.array([[1, 2], [8, 9]])
    result = diffusion_generate_step(
        prompt, model, config(use_compile=compiled), anchor_token_ids=anchors
    )
    np.testing.assert_array_equal(np.array(result[:, :2]), np.array(prompt))
    assert not bool(mx.any(result == 31).item())
    fixed = {
        2 + offset + i: token
        for offset, ids in anchors.items()
        for i, token in enumerate(ids)
    }
    for position, token in fixed.items():
        assert np.all(np.array(result[:, position]) == token)
    # Includes global cache prefill and local block replacement passes.
    assert any(start == 2 for start, _ in model.inputs)
    assert any(start == 6 for start, _ in model.inputs)
    for start, tokens in model.inputs:
        for position, token in fixed.items():
            if start <= position < start + tokens.shape[1]:
                assert np.all(tokens[:, position - start] == token)


def test_fully_anchored_response_is_preserved():
    model = TinyDream()
    result = diffusion_generate_step(
        mx.array([[1, 2]]), model, config(), anchor_token_ids={0: list(range(3, 11))}
    )
    assert result.tolist() == [list(range(1, 11))]


def test_empty_anchors_match_default_generation():
    model = TinyDream()
    prompt = mx.array([[1, 2]])
    default = diffusion_generate_step(prompt, model, config())
    empty = diffusion_generate_step(prompt, model, config(), anchor_token_ids={})
    np.testing.assert_array_equal(np.array(default), np.array(empty))


def test_sampling_confidence_matches_selected_token_probability():
    logits = mx.array([[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]])
    confidence, tokens = sample_tokens(logits, temperature=0.4, key=mx.random.key(1))
    probs = np.array(mx.softmax(logits / 0.4, axis=-1))
    expected = np.take_along_axis(probs, np.array(tokens)[..., None], axis=-1)[..., 0]
    assert confidence.shape == tokens.shape == (1, 2)
    np.testing.assert_allclose(np.array(confidence), expected, rtol=1e-6)


def test_compiled_updates_preserve_batch_shape_between_calls():
    for batch_size in (2, 1, 3):
        tokens = mx.full((batch_size, 4), 31, dtype=mx.int32)
        logits = mx.zeros((batch_size, 4, 32))
        updated = _compiled_select_confident_updates_greedy(tokens, logits, 31, 1.1)
        assert updated.shape == tokens.shape
        assert mx.sum(updated != 31, axis=-1).tolist() == [1] * batch_size


def test_sampled_generation_preserves_anchors():
    settings = config(use_compile=False)
    settings.temperature = 0.4
    result = diffusion_generate_step(
        mx.array([[1, 2]]), TinyDream(), settings, anchor_token_ids={0: [3], 6: [7]}
    )
    assert result[0, 2].item() == 3
    assert result[0, 8].item() == 7
    assert not bool(mx.any(result == 31).item())


def test_text_anchors_work_in_both_public_generation_apis(tokenizer):
    model = TinyDream()
    settings = config(template_anchors={0: "Analysis:", 6: "Answer:"})
    text = diffusion_generate(model, tokenizer, [1, 2], settings)
    (response,) = stream_diffusion_generate(model, tokenizer, [1, 2], settings)
    assert text == response.text
    assert text.split()[0] == "Analysis:"
    assert text.split()[6] == "Answer:"
    assert response.sequences[0, 2].item() == 3
    assert response.sequences[0, 8].item() == 7
    assert response.generation_tokens == 8


@pytest.mark.parametrize(
    "anchors, message",
    [
        ({-1: [3]}, "non-negative"),
        ({True: [3]}, "non-negative"),
        ({"1": [3]}, "non-negative"),
        ({7: [3, 4]}, "exceeds"),
        ({0: [3, 4], 1: [5]}, "overlap"),
        ({0: [31]}, "mask token"),
        ({0: []}, "non-empty"),
        ({0: [-1]}, "token ID"),
    ],
)
def test_invalid_token_anchors_fail_before_inference(anchors, message):
    model = TinyDream()
    with pytest.raises(ValueError, match=message):
        diffusion_generate_step(
            mx.array([[1, 2]]), model, config(), anchor_token_ids=anchors
        )
    assert model.inputs == []


def test_anchor_cannot_fit_when_there_is_no_generation_space():
    with pytest.raises(ValueError, match="exceeds"):
        _apply_template_anchors(mx.array([[1, 2]]), 2, {0: [3]}, 31)


@pytest.mark.parametrize("anchors", [["text"], {"0": "text"}, {0: ""}, {0: 3}])
def test_text_anchor_validation(tokenizer, anchors):
    with pytest.raises(ValueError):
        _encode_template_anchors(tokenizer, anchors)


def test_tokenizer_does_not_insert_special_tokens(tokenizer):
    assert _encode_template_anchors(tokenizer, {0: "Analysis:", 6: "Answer:"}) == {
        0: [3],
        6: [7],
    }


@pytest.mark.parametrize(
    "value", [["text"], {"-1": "text"}, {"1.5": "text"}, {"1": "a", "01": "b"}]
)
def test_cli_rejects_invalid_offsets(tmp_path, value):
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        _load_template_anchors(path)


def test_cli_loads_integer_offsets(tmp_path):
    path = tmp_path / "anchors.json"
    path.write_text('{"0": "Analysis:", "6": "Answer:"}')
    assert _load_template_anchors(path) == {0: "Analysis:", 6: "Answer:"}
