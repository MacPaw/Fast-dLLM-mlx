# Copyright 2026 MacPaw Way Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
from transformers import AutoTokenizer

from .model import Model, ModelArgs, make_compiled_softcap

_GENERATION_STREAM = None


def _get_generation_stream():
    global _GENERATION_STREAM
    if _GENERATION_STREAM is None:
        _GENERATION_STREAM = mx.new_stream(mx.default_device())
    return _GENERATION_STREAM


@contextmanager
def _diffusion_generation_runtime(
    *,
    use_generation_stream: bool,
    wired_limit: Optional[int],
):
    previous_wired_limit = None
    try:
        if wired_limit is not None:
            previous_wired_limit = mx.set_wired_limit(int(wired_limit))
        if use_generation_stream:
            with mx.stream(_get_generation_stream()):
                yield
        else:
            yield
    finally:
        if previous_wired_limit is not None:
            mx.set_wired_limit(previous_wired_limit)


def _wired_limit_bytes(wired_limit_gb: Optional[float]) -> Optional[int]:
    if wired_limit_gb is None:
        return None
    return int(wired_limit_gb * (1024**3))


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


def _resolve_model_path_local(path_or_repo: str) -> Path:
    model_path = Path(path_or_repo)
    if model_path.exists():
        return model_path
    return Path(
        snapshot_download(
            path_or_repo,
            local_files_only=True,
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
    local_files_only: bool = False,
):
    model_path = (
        _resolve_model_path_local(path_or_repo)
        if local_files_only
        else _resolve_model_path(path_or_repo)
    )
    model, _ = load_model(
        model_path,
        lazy=lazy,
        model_config=model_config,
        get_model_classes=_get_model_classes,
    )

    tokenizer_kwargs = dict(tokenizer_config or {})
    tokenizer_kwargs.setdefault("trust_remote_code", trust_remote_code)
    tokenizer_kwargs.setdefault("local_files_only", local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@dataclass
class DiffusionGemmaGenerationConfig:
    max_new_tokens: Optional[int] = None
    max_canvases: int = 8
    min_canvas_length: int = 64
    max_canvas_length: Optional[int] = None
    full_canvas: bool = False
    max_denoising_steps: int = 48
    t_min: float = 0.4
    t_max: float = 0.8
    entropy_bound: float = 0.1
    confidence_threshold: float = 0.005
    stability_threshold: int = 1
    eos_token_ids: Optional[List[int]] = None
    use_chat_template: bool = True
    enable_thinking: Optional[bool] = None
    use_compile: bool = False
    precompute_decoder_masks: bool = False
    use_generation_stream: bool = True
    wired_limit: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class DiffusionGemmaGenerationResponse:
    text: str
    sequences: mx.array
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None


@dataclass
class DiffusionGemmaGenerator:
    model: nn.Module
    tokenizer: Any

    def generate(
        self,
        prompt: Union[str, List[int], mx.array],
        generation_config: Optional[DiffusionGemmaGenerationConfig] = None,
        **kwargs,
    ) -> str:
        return diffusion_generate(
            self.model,
            self.tokenizer,
            prompt,
            generation_config=generation_config,
            **kwargs,
        )


def _prepare_prompt(
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    generation_config: DiffusionGemmaGenerationConfig,
) -> mx.array:
    if isinstance(prompt, mx.array):
        return prompt if prompt.ndim == 2 else prompt[None, :]

    if isinstance(prompt, str):
        has_chat_template = bool(
            getattr(tokenizer, "has_chat_template", False)
            or getattr(tokenizer, "chat_template", None)
        )
        if generation_config.use_chat_template and has_chat_template:
            template_kwargs = {}
            if generation_config.enable_thinking is not None:
                template_kwargs["enable_thinking"] = generation_config.enable_thinking
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    return mx.array(prompt)[None, :]


def _to_numpy(arr: mx.array) -> np.ndarray:
    return np.asarray(arr.tolist())


def _token_entropy(logits: mx.array) -> mx.array:
    logp = nn.log_softmax(logits, axis=-1)
    return -mx.sum(mx.exp(logp) * logp, axis=-1)


def _make_token_entropy_function(*, use_compile: bool):
    if use_compile:
        return mx.compile(_token_entropy, shapeless=True), True
    return _token_entropy, False


def _entropy_bound_accept(
    current: mx.array,
    denoiser: mx.array,
    logits: mx.array,
    entropy_bound: float,
):
    entropy = _token_entropy(logits)
    order = mx.argsort(entropy, axis=-1)
    sorted_entropy = mx.take_along_axis(entropy, order, axis=-1)
    cumulative = mx.cumsum(sorted_entropy, axis=-1)
    selected_sorted = (cumulative - sorted_entropy) <= entropy_bound
    inverse = mx.argsort(order, axis=-1)
    accepted_mask = mx.take_along_axis(selected_sorted, inverse, axis=-1)
    accepted = mx.where(accepted_mask, denoiser, current)
    return accepted, accepted_mask


def _make_entropy_bound_accept_function(
    entropy_bound: float,
    *,
    use_compile: bool,
):
    def entropy_bound_accept(current, denoiser, logits):
        return _entropy_bound_accept(current, denoiser, logits, entropy_bound)

    if use_compile:
        return mx.compile(entropy_bound_accept, shapeless=True), False
    return _entropy_bound_accept, True


def _make_key(config: DiffusionGemmaGenerationConfig):
    if config.seed is None:
        return None
    return mx.random.key(config.seed)


def _make_decoder_logits_functions(
    model: Model,
    cache,
    softcap: float,
    *,
    decoder_attention_mask=None,
    use_compile: bool,
):
    softcap_fn = getattr(model, "_softcap", None)
    if softcap_fn is None:
        softcap_fn = make_compiled_softcap(float(softcap))

    def decoder_logits_without_self_conditioning(canvas):
        hidden = model.model.decoder(
            canvas,
            cache=cache,
            self_conditioning_logits=None,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = model.model.decoder.embed_tokens.as_linear(hidden)
        return softcap_fn(logits)

    def decoder_logits_with_self_conditioning(canvas, self_conditioning_logits):
        hidden = model.model.decoder(
            canvas,
            cache=cache,
            self_conditioning_logits=self_conditioning_logits,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = model.model.decoder.embed_tokens.as_linear(hidden)
        return softcap_fn(logits)

    if not use_compile:
        return (
            decoder_logits_without_self_conditioning,
            decoder_logits_with_self_conditioning,
        )

    return (
        mx.compile(decoder_logits_without_self_conditioning, shapeless=True),
        mx.compile(decoder_logits_with_self_conditioning, shapeless=True),
    )


def _denoise_one_canvas(
    model: Model,
    cache,
    batch_size: int,
    canvas_length: int,
    next_key,
    config: DiffusionGemmaGenerationConfig,
    progress_callback=None,
    stats_callback=None,
) -> mx.array:
    vocab_size = model.config.vocab_size
    softcap = float(model.config.final_logit_softcapping)
    use_compile = config.use_compile

    def random_canvas():
        return mx.random.randint(
            0,
            vocab_size,
            (batch_size, canvas_length),
            key=next_key(),
        )

    canvas = random_canvas()
    decoder_attention_mask = None
    if config.precompute_decoder_masks:
        decoder_attention_mask = model.model.decoder._make_decoder_masks(
            canvas[..., None],
            cache,
            decoder_attention_mask=None,
        )
    decoder_logits_without_sc, decoder_logits_with_sc = _make_decoder_logits_functions(
        model,
        cache,
        softcap,
        decoder_attention_mask=decoder_attention_mask,
        use_compile=use_compile,
    )
    entropy_bound_accept, entropy_bound_accept_needs_bound = (
        _make_entropy_bound_accept_function(
            config.entropy_bound,
            use_compile=use_compile,
        )
    )
    token_entropy, token_entropy_is_compiled = _make_token_entropy_function(
        use_compile=use_compile,
    )
    self_conditioning_logits = None
    history = mx.full(
        (config.stability_threshold, batch_size, canvas_length),
        -1,
        dtype=canvas.dtype,
    )
    argmax_canvas = canvas
    denoising_steps = 0

    for step in range(config.max_denoising_steps):
        denoising_steps += 1
        if progress_callback is not None:
            progress_callback(step + 1, config.max_denoising_steps)
        try:
            if self_conditioning_logits is None:
                logits = decoder_logits_without_sc(canvas)
            else:
                logits = decoder_logits_with_sc(canvas, self_conditioning_logits)
        except Exception:
            if not use_compile:
                raise
            use_compile = False
            decoder_logits_without_sc, decoder_logits_with_sc = (
                _make_decoder_logits_functions(
                    model,
                    cache,
                    softcap,
                    decoder_attention_mask=decoder_attention_mask,
                    use_compile=False,
                )
            )
            entropy_bound_accept, entropy_bound_accept_needs_bound = (
                _make_entropy_bound_accept_function(
                    config.entropy_bound,
                    use_compile=False,
                )
            )
            token_entropy, token_entropy_is_compiled = _make_token_entropy_function(
                use_compile=False,
            )
            if self_conditioning_logits is None:
                logits = decoder_logits_without_sc(canvas)
            else:
                logits = decoder_logits_with_sc(canvas, self_conditioning_logits)
        temperature = config.t_min + (config.t_max - config.t_min) * (
            (config.max_denoising_steps - step) / config.max_denoising_steps
        )
        scaled_logits = logits / temperature
        denoiser = mx.random.categorical(scaled_logits, axis=-1, key=next_key())
        try:
            if entropy_bound_accept_needs_bound:
                _, accept_mask = entropy_bound_accept(
                    canvas,
                    denoiser,
                    scaled_logits,
                    config.entropy_bound,
                )
            else:
                _, accept_mask = entropy_bound_accept(canvas, denoiser, scaled_logits)
        except Exception:
            if not use_compile or entropy_bound_accept_needs_bound:
                raise
            entropy_bound_accept, entropy_bound_accept_needs_bound = (
                _make_entropy_bound_accept_function(
                    config.entropy_bound,
                    use_compile=False,
                )
            )
            _, accept_mask = entropy_bound_accept(
                canvas,
                denoiser,
                scaled_logits,
                config.entropy_bound,
            )
        argmax_canvas = mx.argmax(logits, axis=-1)

        if config.stability_threshold == 0:
            stable = mx.ones((batch_size,), dtype=mx.bool_)
        else:
            stable = mx.all(mx.all(history == argmax_canvas[None], axis=-1), axis=0)
            history = mx.roll(history, -1, axis=0)
            history[-1] = argmax_canvas

        try:
            entropy = token_entropy(scaled_logits)
        except Exception:
            if not use_compile or not token_entropy_is_compiled:
                raise
            token_entropy, token_entropy_is_compiled = _make_token_entropy_function(
                use_compile=False,
            )
            entropy = token_entropy(scaled_logits)
        confident = mx.mean(entropy, axis=-1) < config.confidence_threshold
        if bool(mx.all(stable & confident).item()):
            break

        canvas = mx.where(accept_mask, denoiser, random_canvas())
        self_conditioning_logits = scaled_logits

    if stats_callback is not None:
        stats_callback(
            canvas_length=canvas_length,
            denoising_steps=denoising_steps,
        )
    return argmax_canvas


def diffusion_generate_ids(
    model: Model,
    prompt_ids: mx.array,
    generation_config: Optional[DiffusionGemmaGenerationConfig] = None,
    **kwargs,
) -> mx.array:
    generation_config = generation_config or DiffusionGemmaGenerationConfig()
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)

    blocks = list(
        stream_diffusion_generate_ids(
            model,
            prompt_ids,
            generation_config=generation_config,
            progress_callback=kwargs.get("progress_callback"),
            stats_callback=kwargs.get("stats_callback"),
        )
    )
    return mx.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]


def _canvas_length_for_block(
    model_canvas_length: int,
    generated_tokens: int,
    config,
) -> int:
    if config.full_canvas or config.max_new_tokens is None:
        return model_canvas_length

    remaining = max(0, int(config.max_new_tokens) - generated_tokens)
    if config.min_canvas_length <= 0:
        raise ValueError("min_canvas_length must be a positive integer.")
    if config.max_canvas_length is not None and config.max_canvas_length <= 0:
        raise ValueError("max_canvas_length must be a positive integer.")
    max_canvas_length = min(
        model_canvas_length,
        int(config.max_canvas_length or model_canvas_length),
    )
    min_canvas_length = min(max_canvas_length, int(config.min_canvas_length))
    return min(max_canvas_length, max(remaining, min_canvas_length))


def _max_canvas_length_for_generation(model_canvas_length: int, config) -> int:
    if config.full_canvas or config.max_new_tokens is None:
        return model_canvas_length
    if config.max_canvas_length is not None and config.max_canvas_length <= 0:
        raise ValueError("max_canvas_length must be a positive integer.")
    return min(
        model_canvas_length,
        int(config.max_canvas_length or model_canvas_length),
    )


def stream_diffusion_generate_ids(
    model: Model,
    prompt_ids: mx.array,
    generation_config: Optional[DiffusionGemmaGenerationConfig] = None,
    progress_callback=None,
    stats_callback=None,
    **kwargs,
) -> Generator[mx.array, None, None]:
    generation_config = generation_config or DiffusionGemmaGenerationConfig()
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)

    batch_size = prompt_ids.shape[0]
    model_canvas_length = model.config.canvas_length
    max_canvas_length = _max_canvas_length_for_generation(
        model_canvas_length,
        generation_config,
    )
    if generation_config.max_new_tokens is None:
        n_blocks = 1
    else:
        n_blocks = max(
            1,
            min(
                generation_config.max_canvases,
                -(-int(generation_config.max_new_tokens) // max_canvas_length),
            ),
        )

    key = _make_key(generation_config)

    def next_key():
        nonlocal key
        if key is None:
            return None
        key, subkey = mx.random.split(key)
        return subkey

    with _diffusion_generation_runtime(
        use_generation_stream=generation_config.use_generation_stream,
        wired_limit=generation_config.wired_limit,
    ):
        cache = model.make_cache()
        _, cache = model.model.encoder(prompt_ids, cache=cache)
        eos_set = set(generation_config.eos_token_ids or [])
        eos_ids = mx.array(list(eos_set), dtype=prompt_ids.dtype) if eos_set else None
        generated_tokens = 0

        for block_idx in range(n_blocks):
            canvas_length = _canvas_length_for_block(
                model_canvas_length,
                generated_tokens,
                generation_config,
            )
            canvas = _denoise_one_canvas(
                model,
                cache,
                batch_size,
                canvas_length,
                next_key,
                generation_config,
                progress_callback=progress_callback,
                stats_callback=stats_callback,
            )
            mx.eval(canvas)
            yield canvas
            generated_tokens += canvas.shape[1]

            mx.clear_cache()
            if eos_ids is not None and bool(mx.any(canvas[..., None] == eos_ids).item()):
                break
            if block_idx != n_blocks - 1:
                _, cache = model.model.encoder(canvas, cache=cache)


def diffusion_generate(
    model: Model,
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    generation_config: Optional[DiffusionGemmaGenerationConfig] = None,
    **kwargs,
) -> str:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    generation_config = generation_config or DiffusionGemmaGenerationConfig()
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)

    prompt_ids = _prepare_prompt(tokenizer, prompt, generation_config)
    if generation_config.eos_token_ids is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        generation_config.eos_token_ids = [] if eos_token_id is None else [eos_token_id]

    sequence = diffusion_generate_ids(
        model,
        prompt_ids,
        generation_config=generation_config,
        progress_callback=kwargs.get("progress_callback"),
        stats_callback=kwargs.get("stats_callback"),
    )
    tokens = _to_numpy(sequence[0]).tolist()
    if generation_config.max_new_tokens is not None:
        tokens = tokens[: generation_config.max_new_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def stream_diffusion_generate(
    model: Model,
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    generation_config: Optional[DiffusionGemmaGenerationConfig] = None,
    **kwargs,
) -> Generator[DiffusionGemmaGenerationResponse, None, None]:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    generation_config = generation_config or DiffusionGemmaGenerationConfig()
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)

    prompt_ids = _prepare_prompt(tokenizer, prompt, generation_config)
    if generation_config.eos_token_ids is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        generation_config.eos_token_ids = [] if eos_token_id is None else [eos_token_id]

    start_time = time.perf_counter()
    emitted = 0
    for canvas in stream_diffusion_generate_ids(
        model,
        prompt_ids,
        generation_config=generation_config,
        stats_callback=kwargs.get("stats_callback"),
    ):
        current_time = time.perf_counter()
        tokens = _to_numpy(canvas[0]).tolist()
        if generation_config.max_new_tokens is not None:
            remaining = generation_config.max_new_tokens - emitted
            tokens = tokens[:remaining]
        emitted += len(tokens)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        yield DiffusionGemmaGenerationResponse(
            text=text,
            sequences=canvas,
            prompt_tokens=prompt_ids.shape[1],
            prompt_tps=prompt_ids.shape[1] / max(current_time - start_time, 1e-6),
            generation_tokens=len(tokens),
            generation_tps=emitted / max(current_time - start_time, 1e-6),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="canvas",
        )
        if (
            generation_config.max_new_tokens is not None
            and emitted >= generation_config.max_new_tokens
        ):
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-denoising-steps", type=int, default=48)
    parser.add_argument("--min-canvas-length", type=int, default=64)
    parser.add_argument("--max-canvas-length", type=int)
    parser.add_argument("--full-canvas", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Compile the repeated DiffusionGemma decoder logits calls.",
    )
    parser.add_argument(
        "--precompute-decoder-masks",
        action="store_true",
        help="Precompute decoder masks once per canvas.",
    )
    parser.add_argument(
        "--no-generation-stream",
        action="store_true",
        help="Run generation on the default MLX stream.",
    )
    parser.add_argument(
        "--wired-limit-gb",
        type=float,
        help="Temporarily set the MLX wired memory limit during generation.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--think",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking in chat templates that support enable_thinking.",
    )
    thinking_group.add_argument(
        "--no-think",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking in chat templates that support enable_thinking.",
    )
    parser.set_defaults(enable_thinking=None)
    args = parser.parse_args()

    if args.verbose:
        print(f"Loading {args.model}...", flush=True)
    tic = time.perf_counter()
    model, tokenizer = load(args.model, local_files_only=args.local_files_only)
    if args.verbose:
        print(f"Loaded in {time.perf_counter() - tic:.2f}s", flush=True)
    config = DiffusionGemmaGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_denoising_steps=args.max_denoising_steps,
        min_canvas_length=args.min_canvas_length,
        max_canvas_length=args.max_canvas_length,
        full_canvas=args.full_canvas,
        enable_thinking=args.enable_thinking,
        use_compile=args.use_compile,
        precompute_decoder_masks=args.precompute_decoder_masks,
        use_generation_stream=not args.no_generation_stream,
        wired_limit=_wired_limit_bytes(args.wired_limit_gb),
        seed=args.seed,
    )
    if args.verbose:
        print("Generating...", flush=True)

    def progress(step, total):
        if args.verbose:
            print(f"denoise step {step}/{total}", flush=True)

    print(
        diffusion_generate(
            model,
            tokenizer,
            args.prompt,
            generation_config=config,
            progress_callback=progress,
        )
    )


if __name__ == "__main__":
    main()
