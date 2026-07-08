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
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper

from diffusion_gemma_mlx.generate import (
    DiffusionGemmaGenerationResponse,
    _canvas_length_for_block,
    _diffusion_generation_runtime,
    _make_decoder_logits_functions,
    _max_canvas_length_for_generation,
    _prepare_prompt,
    _to_numpy,
    _wired_limit_bytes,
    load,
)
from diffusion_gemma_mlx.model import Model, make_compiled_softcap
from fast_dllm_mlx.generate import top_k_logits, top_p_logits


@dataclass
class FastDiffusionGemmaGenerationConfig:
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    max_canvases: int = 8
    min_canvas_length: int = 64
    max_canvas_length: Optional[int] = None
    full_canvas: bool = False
    steps: int = 48
    threshold: float = 0.9
    min_tokens_per_step: int = 1
    temperature_schedule: bool = True
    schedule_t_min: float = 0.4
    schedule_t_max: float = 0.8
    use_self_conditioning: bool = False
    self_conditioning_top_k: Optional[int] = 256
    use_chat_template: bool = True
    enable_thinking: Optional[bool] = None
    use_compile: bool = False
    precompute_decoder_masks: bool = False
    early_stop: bool = True
    use_generation_stream: bool = True
    wired_limit: Optional[int] = None
    eos_token_ids: Optional[List[int]] = None
    seed: Optional[int] = None


@dataclass
class FastDiffusionGemmaGenerator:
    model: nn.Module
    tokenizer: Any

    def generate(
        self,
        prompt: Union[str, List[int], mx.array],
        generation_config: Optional[FastDiffusionGemmaGenerationConfig] = None,
        **kwargs,
    ) -> str:
        return diffusion_generate(
            self.model,
            self.tokenizer,
            prompt,
            generation_config=generation_config,
            **kwargs,
        )


def _sample_tokens(
    logits: mx.array,
    *,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    key,
) -> Tuple[mx.array, mx.array]:
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    if temperature <= 0:
        tokens = mx.argmax(logits, axis=-1)
        token_logits = mx.take_along_axis(
            logits,
            mx.expand_dims(tokens, -1),
            axis=-1,
        ).squeeze(-1)
        confidence = mx.exp(token_logits - mx.logsumexp(logits, axis=-1))
    else:
        probs = mx.softmax(logits, axis=-1)
        tokens = mx.random.categorical(logits, axis=-1, key=key)
        confidence = mx.take_along_axis(
            probs,
            mx.expand_dims(tokens, -1),
            axis=-1,
        ).squeeze(-1)
    return confidence, tokens


def _can_compile_sample_tokens(config: FastDiffusionGemmaGenerationConfig) -> bool:
    return config.temperature <= 0 and config.top_p is None and config.top_k is None


def _make_sample_tokens_function(
    config: FastDiffusionGemmaGenerationConfig,
    *,
    use_compile: bool,
):
    def greedy_sample_tokens(logits):
        tokens = mx.argmax(logits, axis=-1)
        token_logits = mx.take_along_axis(
            logits,
            mx.expand_dims(tokens, -1),
            axis=-1,
        ).squeeze(-1)
        confidence = mx.exp(token_logits - mx.logsumexp(logits, axis=-1))
        return confidence, tokens

    if use_compile and _can_compile_sample_tokens(config):
        return mx.compile(greedy_sample_tokens, shapeless=True), False
    return _sample_tokens, True


def _scatter_true(indices: mx.array, width: int) -> mx.array:
    mask = mx.zeros((indices.shape[0], width), dtype=mx.bool_)
    values = mx.ones(indices.shape, dtype=mx.bool_)
    return mx.put_along_axis(mask, indices, values, axis=-1)


def _select_updates(
    canvas: mx.array,
    logits: mx.array,
    unresolved: mx.array,
    config: FastDiffusionGemmaGenerationConfig,
    key,
    sample_tokens_fn=None,
) -> Tuple[mx.array, mx.array, mx.array]:
    if sample_tokens_fn is None:
        confidence, proposal = _sample_tokens(
            logits,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            key=key,
        )
    else:
        confidence, proposal = sample_tokens_fn(logits)
    neg_inf = mx.full(confidence.shape, -mx.inf, dtype=confidence.dtype)
    confidence = mx.where(unresolved, confidence, neg_inf)
    transfer = unresolved & (confidence >= config.threshold)

    for _ in range(max(config.min_tokens_per_step, 0)):
        has_unresolved = mx.broadcast_to(
            mx.any(unresolved & ~transfer, axis=-1, keepdims=True),
            transfer.shape,
        )
        next_idx = mx.argmax(
            mx.where(transfer, neg_inf, confidence), axis=-1, keepdims=True
        )
        force = _scatter_true(next_idx, canvas.shape[1])
        transfer = transfer | (force & has_unresolved)

    next_canvas = mx.where(transfer, proposal, canvas)
    next_unresolved = unresolved & ~transfer
    return next_canvas, next_unresolved, proposal


def _make_key(config: FastDiffusionGemmaGenerationConfig):
    if config.seed is None:
        return None
    return mx.random.key(config.seed)


_SC_TOP_K_MIN_CANVAS_LENGTH = 128


def _topk_soft_conditioning_embeddings(
    model: Model,
    logits: mx.array,
    top_k: int,
) -> mx.array:
    """Approximate the self-conditioning soft embedding with the top-k tokens.

    The exact path softmaxes the full vocab and matmuls against the whole
    embedding table (~28% of a denoising step); the feedback distribution is
    extremely peaked, so the truncated, renormalized softmax over the top-k
    logits loses ~1% of the signal at k=256 while skipping most of that work.
    Runs in float16: the tail precision does not survive the softmax anyway.
    """
    decoder = model.model.decoder
    embed_tokens = decoder.embed_tokens
    logits = logits.astype(mx.float16)
    indices = mx.argpartition(logits, kth=-top_k, axis=-1)[..., -top_k:]
    top_logits = mx.take_along_axis(logits, indices, axis=-1)
    probs = mx.softmax(top_logits, axis=-1, precise=True)
    if isinstance(embed_tokens, nn.QuantizedEmbedding):
        rows = mx.dequantize(
            embed_tokens.weight[indices],
            embed_tokens.scales[indices],
            embed_tokens.biases[indices],
            group_size=embed_tokens.group_size,
            bits=embed_tokens.bits,
        )
    else:
        rows = embed_tokens.weight[indices]
    soft = (probs[..., None].astype(rows.dtype) * rows).sum(axis=-2)
    # _embed_canvas applies embed_scale on the logits path but takes
    # precomputed embeddings as-is, so scale here.
    return soft * decoder.embed_scale


def _make_decoder_logits_sc_embeddings_function(
    model: Model,
    cache,
    softcap: float,
    *,
    decoder_attention_mask=None,
):
    softcap_fn = getattr(model, "_softcap", None)
    if softcap_fn is None:
        softcap_fn = make_compiled_softcap(float(softcap))

    def decoder_logits_with_sc_embeddings(canvas, sc_embeddings):
        hidden = model.model.decoder(
            canvas,
            cache=cache,
            self_conditioning_embeddings=sc_embeddings,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = model.model.decoder.embed_tokens.as_linear(hidden)
        return softcap_fn(logits)

    return decoder_logits_with_sc_embeddings


def _denoise_one_canvas_fast(
    model: Model,
    cache,
    batch_size: int,
    canvas_length: int,
    next_key,
    config: FastDiffusionGemmaGenerationConfig,
    stats_callback=None,
) -> mx.array:
    softcap = float(model.config.final_logit_softcapping)
    use_compile = config.use_compile
    canvas = mx.random.randint(
        0,
        model.config.vocab_size,
        (batch_size, canvas_length),
        key=next_key(),
    )
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
    sc_top_k = config.self_conditioning_top_k if config.use_self_conditioning else None
    if sc_top_k is not None and not 0 < sc_top_k < model.config.vocab_size:
        sc_top_k = None
    # The top-k savings scale with canvas length while its fixed overheads and
    # slight step inflation do not: measured +16-18% at canvas 128/256 but a
    # wash at 64, so fall back to the exact softmax for small canvases.
    if sc_top_k is not None and canvas_length < _SC_TOP_K_MIN_CANVAS_LENGTH:
        sc_top_k = None
    decoder_logits_with_sc_emb = None
    if sc_top_k is not None:
        decoder_logits_with_sc_emb = _make_decoder_logits_sc_embeddings_function(
            model,
            cache,
            softcap,
            decoder_attention_mask=decoder_attention_mask,
        )
    sample_tokens, sample_tokens_needs_args = _make_sample_tokens_function(
        config,
        use_compile=use_compile,
    )
    unresolved = mx.ones((batch_size, canvas_length), dtype=mx.bool_)
    self_conditioning_logits = None
    committed = canvas
    denoising_steps = 0
    # Linear temperature schedule from the checkpoint's generation_config
    # (t_max on the first step down to ~t_min on the last). Dividing logits by
    # t < 1 sharpens confidences so tokens cross the threshold in fewer steps.
    schedule_temperatures = None
    if config.temperature_schedule:
        t_min, t_max = config.schedule_t_min, config.schedule_t_max
        schedule_temperatures = [
            t_min + (t_max - t_min) * ((config.steps - step) / config.steps)
            for step in range(config.steps)
        ]

    for step in range(config.steps):
        denoising_steps += 1
        if config.early_stop and not bool(mx.any(unresolved).item()):
            break

        self_conditioning_embeddings = None
        if self_conditioning_logits is not None and sc_top_k is not None:
            self_conditioning_embeddings = _topk_soft_conditioning_embeddings(
                model,
                self_conditioning_logits,
                sc_top_k,
            )
        try:
            if self_conditioning_logits is None:
                logits = decoder_logits_without_sc(canvas)
            elif self_conditioning_embeddings is not None:
                logits = decoder_logits_with_sc_emb(
                    canvas, self_conditioning_embeddings
                )
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
            sample_tokens, sample_tokens_needs_args = _make_sample_tokens_function(
                config,
                use_compile=False,
            )
            if self_conditioning_logits is None:
                logits = decoder_logits_without_sc(canvas)
            elif self_conditioning_embeddings is not None:
                logits = decoder_logits_with_sc_emb(
                    canvas, self_conditioning_embeddings
                )
            else:
                logits = decoder_logits_with_sc(canvas, self_conditioning_logits)
        if schedule_temperatures is not None:
            logits = logits / schedule_temperatures[step]
        try:
            if sample_tokens_needs_args:
                canvas, unresolved, committed = _select_updates(
                    canvas,
                    logits,
                    unresolved,
                    config,
                    next_key(),
                )
            else:
                canvas, unresolved, committed = _select_updates(
                    canvas,
                    logits,
                    unresolved,
                    config,
                    None,
                    sample_tokens_fn=sample_tokens,
                )
        except Exception:
            if not use_compile or sample_tokens_needs_args:
                raise
            sample_tokens, sample_tokens_needs_args = _make_sample_tokens_function(
                config,
                use_compile=False,
            )
            canvas, unresolved, committed = _select_updates(
                canvas,
                logits,
                unresolved,
                config,
                next_key(),
            )
        self_conditioning_logits = logits if config.use_self_conditioning else None

    if stats_callback is not None:
        stats_callback(
            canvas_length=canvas_length,
            denoising_steps=denoising_steps,
        )
    return mx.where(unresolved, committed, canvas)


def stream_diffusion_generate_ids(
    model: Model,
    prompt_ids: mx.array,
    generation_config: Optional[FastDiffusionGemmaGenerationConfig] = None,
    stats_callback=None,
    **kwargs,
) -> Generator[mx.array, None, None]:
    generation_config = generation_config or FastDiffusionGemmaGenerationConfig()
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)

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
            canvas = _denoise_one_canvas_fast(
                model,
                cache,
                prompt_ids.shape[0],
                canvas_length,
                next_key,
                generation_config,
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


def diffusion_generate_ids(
    model: Model,
    prompt_ids: mx.array,
    generation_config: Optional[FastDiffusionGemmaGenerationConfig] = None,
    **kwargs,
) -> mx.array:
    stats_callback = kwargs.pop("stats_callback", None)
    blocks = list(
        stream_diffusion_generate_ids(
            model,
            prompt_ids,
            generation_config=generation_config,
            stats_callback=stats_callback,
            **kwargs,
        )
    )
    return mx.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]


def diffusion_generate(
    model: Model,
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    generation_config: Optional[FastDiffusionGemmaGenerationConfig] = None,
    **kwargs,
) -> str:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    generation_config = generation_config or FastDiffusionGemmaGenerationConfig()
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
    )
    tokens = _to_numpy(sequence[0]).tolist()
    if generation_config.max_new_tokens is not None:
        tokens = tokens[: generation_config.max_new_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def stream_diffusion_generate(
    model: Model,
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    generation_config: Optional[FastDiffusionGemmaGenerationConfig] = None,
    **kwargs,
) -> Generator[DiffusionGemmaGenerationResponse, None, None]:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    generation_config = generation_config or FastDiffusionGemmaGenerationConfig()
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
        yield DiffusionGemmaGenerationResponse(
            text=tokenizer.decode(tokens, skip_special_tokens=True),
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
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--min-canvas-length", type=int, default=64)
    parser.add_argument("--max-canvas-length", type=int)
    parser.add_argument("--full-canvas", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use-self-conditioning",
        action="store_true",
        help="Feed previous-step logits back into Fast DiffusionGemma denoising.",
    )
    parser.add_argument(
        "--self-conditioning-top-k",
        type=int,
        default=256,
        help=(
            "Approximate self-conditioning with the top-k token embeddings. "
            "0 uses the exact softmax over the full vocabulary."
        ),
    )
    parser.add_argument(
        "--no-temperature-schedule",
        action="store_true",
        help="Disable the linear denoising temperature schedule.",
    )
    parser.add_argument("--schedule-t-min", type=float, default=0.4)
    parser.add_argument("--schedule-t-max", type=float, default=0.8)
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
        "--no-early-stop",
        action="store_true",
        help="Run all denoising steps without per-step completion synchronization.",
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

    model, tokenizer = load(args.model)
    config = FastDiffusionGemmaGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_canvas_length=args.min_canvas_length,
        max_canvas_length=args.max_canvas_length,
        full_canvas=args.full_canvas,
        steps=args.steps,
        threshold=args.threshold,
        temperature=args.temperature,
        temperature_schedule=not args.no_temperature_schedule,
        schedule_t_min=args.schedule_t_min,
        schedule_t_max=args.schedule_t_max,
        use_self_conditioning=args.use_self_conditioning,
        self_conditioning_top_k=args.self_conditioning_top_k or None,
        enable_thinking=args.enable_thinking,
        use_compile=args.use_compile,
        precompute_decoder_masks=args.precompute_decoder_masks,
        early_stop=not args.no_early_stop,
        use_generation_stream=not args.no_generation_stream,
        wired_limit=_wired_limit_bytes(args.wired_limit_gb),
        seed=args.seed,
    )
    print(diffusion_generate(model, tokenizer, args.prompt, generation_config=config))


if __name__ == "__main__":
    main()
