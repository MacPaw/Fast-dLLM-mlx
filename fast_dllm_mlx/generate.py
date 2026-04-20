import argparse
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
from transformers import AutoTokenizer

from .cache import make_dual_prompt_cache
from .model import Model, ModelArgs


def _get_model_classes(config: dict):
    del config
    return Model, ModelArgs


def _resolve_model_path(path_or_repo: str) -> Path:
    model_path = Path(path_or_repo)
    if model_path.exists():
        return model_path
    return Path(
        snapshot_download(
            path_or_repo,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "*.jinja",
                "*.model",
                "*.py",
                "*.tiktoken",
            ],
        )
    )


def load(
    path_or_repo: str,
    tokenizer_config: Optional[dict] = None,
    model_config: Optional[dict] = None,
    trust_remote_code: bool = True,
    lazy: bool = False,
):
    model_path = _resolve_model_path(path_or_repo)
    model, config = load_model(
        model_path,
        lazy=lazy,
        model_config=model_config,
        get_model_classes=_get_model_classes,
    )

    tokenizer_kwargs = dict(tokenizer_config or {})
    tokenizer_kwargs.setdefault("trust_remote_code", trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@dataclass
class DreamGenerationConfig:
    temperature: float = 0.4
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_length: int = 256
    max_new_tokens: Optional[int] = None
    eps: float = 1e-3
    steps: int = 256
    alg: str = "confidence_threshold"
    alg_temp: Optional[float] = None
    threshold: float = 0.9
    block_length: Optional[int] = 32
    dual_cache: bool = True
    use_compile: bool = True
    mask_token_id: Optional[int] = 151666
    pad_token_id: Optional[int] = 151643
    bos_token_id: Optional[int] = 151643
    eos_token_id: Optional[int] = 151643
    num_return_sequences: int = 1
    use_chat_template: bool = True


@dataclass
class DreamGenerationResponse:
    text: str
    sequences: mx.array
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None
    history: Optional[List[mx.array]] = None


@dataclass
class DreamGenerator:
    model: nn.Module
    tokenizer: Any

    def generate(
        self,
        prompt: Union[str, List[int]],
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> str:
        return diffusion_generate(
            self.model,
            self.tokenizer,
            prompt,
            generation_config=generation_config,
            **kwargs,
        )


def top_p_logits(logits: mx.array, top_p: float) -> mx.array:
    sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
    sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
    cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = mx.concatenate(
        [
            mx.zeros_like(sorted_indices_to_remove[:, :1]),
            sorted_indices_to_remove[:, :-1],
        ],
        axis=-1,
    )

    mask = mx.zeros_like(logits, dtype=mx.bool_)
    mask = mx.put_along_axis(mask, sorted_indices, sorted_indices_to_remove, axis=-1)
    return mx.where(mask, -mx.inf, logits)


def top_k_logits(logits: mx.array, top_k: int) -> mx.array:
    top_k = min(top_k, logits.shape[-1])
    kth_largest = mx.sort(logits, axis=-1)[:, -top_k][:, None]
    return mx.where(logits < kth_largest, -mx.inf, logits)


def sample_tokens(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
    key: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = mx.softmax(logits, axis=-1)
    if temperature > 0:
        if key is None:
            key = mx.random.key(int(time.time()))
        x0 = mx.random.categorical(logits, axis=-1, key=key)
        confidence = probs[mx.arange(probs.shape[0]), x0]
    else:
        confidence = mx.max(probs, axis=-1)
        x0 = mx.argmax(probs, axis=-1)

    if margin_confidence:
        sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]

    if neg_entropy:
        epsilon = 1e-10
        log_probs = mx.log(probs + epsilon)
        confidence = mx.sum(probs * log_probs, axis=-1)

    return confidence, x0


def _prepare_prompt(
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    generation_config: DreamGenerationConfig,
) -> mx.array:
    if isinstance(prompt, mx.array):
        return prompt if prompt.ndim == 2 else prompt[None, :]

    if isinstance(prompt, str):
        if generation_config.use_chat_template and getattr(
            tokenizer, "chat_template", None
        ):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    return mx.array(prompt)[None, :]


def _shift_logits(logits: mx.array) -> mx.array:
    if logits.shape[1] > 1:
        return mx.concatenate([logits[:, :1], logits[:, :-1]], axis=1)
    return logits


def _to_numpy(arr: mx.array) -> np.ndarray:
    return np.asarray(arr.tolist())


def _scatter_true(indices: mx.array, width: int) -> mx.array:
    mask = mx.zeros((indices.shape[0], width), dtype=mx.bool_)
    values = mx.ones(indices.shape, dtype=mx.bool_)
    return mx.put_along_axis(mask, indices, values, axis=-1)


def _select_confident_updates(
    x_block: mx.array,
    logits: mx.array,
    mask_token_id: int,
    threshold: float,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    key: mx.array,
) -> mx.array:
    mask_index = x_block == mask_token_id
    if not bool(mx.any(mask_index).item()):
        return x_block

    confidence, x0 = sample_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        key=key,
    )
    x0 = mx.where(mask_index, x0, x_block)
    neg_inf = mx.full(confidence.shape, -mx.inf, dtype=confidence.dtype)
    confidence = mx.where(mask_index, confidence, neg_inf)
    transfer_index = mask_index & (confidence >= threshold)
    max_conf_idx = mx.argmax(confidence, axis=-1, keepdims=True)
    force_mask = _scatter_true(max_conf_idx, x_block.shape[1])
    has_mask = mx.broadcast_to(mx.any(mask_index, axis=-1, keepdims=True), transfer_index.shape)
    transfer_index = transfer_index | (force_mask & has_mask)
    return mx.where(transfer_index, x0, x_block)


@partial(mx.compile, shapeless=True)
def _compiled_select_confident_updates_greedy(
    x_block: mx.array,
    logits: mx.array,
    mask_token_id: int,
    threshold: float,
) -> mx.array:
    mask_index = x_block == mask_token_id
    probs = mx.softmax(logits, axis=-1)
    confidence = mx.max(probs, axis=-1)
    x0 = mx.argmax(probs, axis=-1)
    x0 = mx.where(mask_index, x0, x_block)
    neg_inf = mx.full(confidence.shape, -mx.inf, dtype=confidence.dtype)
    confidence = mx.where(mask_index, confidence, neg_inf)
    transfer_index = mask_index & (confidence >= threshold)
    max_conf_idx = mx.argmax(confidence, axis=-1, keepdims=True)
    force_mask = mx.put_along_axis(
        mx.zeros_like(mask_index),
        max_conf_idx,
        mx.ones(max_conf_idx.shape, dtype=mx.bool_),
        axis=-1,
    )
    has_mask = mx.broadcast_to(mx.any(mask_index, axis=-1, keepdims=True), transfer_index.shape)
    transfer_index = transfer_index | (force_mask & has_mask)
    return mx.where(transfer_index, x0, x_block)


def diffusion_generate_step(
    prompt: mx.array,
    model: nn.Module,
    generation_config: DreamGenerationConfig,
) -> mx.array:
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    block_length = generation_config.block_length
    dual_cache = generation_config.dual_cache
    threshold = generation_config.threshold
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    batch_size = prompt.shape[0]
    pad_length = max_length - prompt.shape[1]
    if pad_length > 0:
        mask_tokens = mx.full(
            (batch_size, pad_length), mask_token_id, dtype=prompt.dtype
        )
        x = mx.concatenate([prompt, mask_tokens], axis=1)
    else:
        x = prompt[:, :max_length]

    gen_length = max_length - prompt.shape[1]
    if gen_length <= 0:
        return x

    if block_length is None:
        block_length = gen_length
    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        )

    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        )
    steps_per_block = steps // num_blocks

    if not dual_cache:
        raise NotImplementedError(
            "This first MLX Fast-dLLM version only supports dual_cache=True."
        )

    prompt_length = prompt.shape[1]
    key = mx.random.key(int(time.time() * 1000))

    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = block_start + block_length

        caches = make_dual_prompt_cache(model)
        full_logits = model(x, mask="full", cache=caches)
        full_logits = _shift_logits(full_logits)
        _, x0 = sample_tokens(
            full_logits[:, block_start:block_end, :],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        x_block = x[:, block_start:block_end]
        block_mask = x_block == mask_token_id
        if bool(mx.any(block_mask[:, :1]).item()):
            first_token = mx.where(block_mask[:, :1], x0[:, :1], x_block[:, :1])
            x = mx.concatenate(
                [x[:, :block_start], first_token, x[:, block_start + 1 :]],
                axis=1,
            )

        for _ in range(steps_per_block):
            x_block = x[:, block_start:block_end]
            if not bool(mx.any(x_block == mask_token_id).item()):
                break

            local_logits = model(
                x_block,
                mask="full",
                cache=caches,
                cache_position=block_start,
                replace_cache=True,
            )
            local_logits = _shift_logits(local_logits)

            if (
                generation_config.use_compile
                and temperature == 0
                and top_p is None
                and top_k is None
            ):
                updated_block = _compiled_select_confident_updates_greedy(
                    x_block,
                    local_logits,
                    mask_token_id,
                    threshold,
                )
            else:
                key, sample_key = mx.random.split(key)
                updated_block = _select_confident_updates(
                    x_block,
                    local_logits,
                    mask_token_id=mask_token_id,
                    threshold=threshold,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    key=sample_key,
                )

            if not bool(mx.any(updated_block != x_block).item()):
                break

            x = mx.concatenate(
                [x[:, :block_start], updated_block, x[:, block_end:]],
                axis=1,
            )
            mx.eval(x)

    return x


def diffusion_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    generation_config: Optional[DreamGenerationConfig] = None,
    **kwargs,
) -> str:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    generation_config = generation_config or DreamGenerationConfig()
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)

    prompt_array = _prepare_prompt(tokenizer, prompt, generation_config)
    if generation_config.mask_token_id is None:
        generation_config.mask_token_id = (
            tokenizer.mask_token_id or tokenizer.unk_token_id
        )

    input_length = prompt_array.shape[1]
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = input_length + generation_config.max_new_tokens

    sequence = diffusion_generate_step(prompt_array, model, generation_config)
    generated = _to_numpy(sequence[0, input_length:])
    return tokenizer.decode(generated.tolist(), skip_special_tokens=True)


def stream_diffusion_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    generation_config: Optional[DreamGenerationConfig] = None,
    **kwargs,
) -> Generator[DreamGenerationResponse, None, None]:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    generation_config = generation_config or DreamGenerationConfig()
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)

    prompt_array = _prepare_prompt(tokenizer, prompt, generation_config)
    if generation_config.mask_token_id is None:
        generation_config.mask_token_id = (
            tokenizer.mask_token_id or tokenizer.unk_token_id
        )
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = (
            prompt_array.shape[1] + generation_config.max_new_tokens
        )

    start_time = time.perf_counter()
    sequence = diffusion_generate_step(prompt_array, model, generation_config)
    current_time = time.perf_counter()
    generated_tokens = sequence[0, prompt_array.shape[1] :]
    generated_text = tokenizer.decode(
        _to_numpy(generated_tokens).tolist(), skip_special_tokens=True
    )

    yield DreamGenerationResponse(
        text=generated_text,
        sequences=sequence,
        prompt_tokens=prompt_array.shape[1],
        prompt_tps=prompt_array.shape[1] / max(current_time - start_time, 1e-6),
        generation_tokens=generated_tokens.shape[0],
        generation_tps=generated_tokens.shape[0] / max(current_time - start_time, 1e-6),
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="complete",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    config = DreamGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_length=args.block_length,
        threshold=args.threshold,
    )
    print(diffusion_generate(model, tokenizer, args.prompt, generation_config=config))


if __name__ == "__main__":
    main()
