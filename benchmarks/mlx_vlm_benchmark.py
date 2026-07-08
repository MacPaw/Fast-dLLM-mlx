# Copyright 2026 MacPaw Way Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from benchmarks.utils import (
    BenchmarkResult,
    PromptSpec,
    encode_token_count,
    generated_text_token_count,
    get_tokenizer,
    load_prompts,
    print_summary,
    write_csv,
    write_json,
)

DEFAULT_MODEL_ID = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark mlx-vlm generation on prompts/*.txt."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"mlx-vlm model id or local path. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional text or JSON file containing prompts. Falls back to prompts/*.txt.",
    )
    parser.add_argument(
        "--image",
        nargs="+",
        default=None,
        help="Optional image path/URL(s) passed to mlx-vlm for every prompt.",
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        default=None,
        help="Optional audio path/URL(s) passed to mlx-vlm for every prompt.",
    )
    parser.add_argument(
        "--video",
        nargs="+",
        default=None,
        help="Optional video path/URL(s) passed to mlx-vlm for every prompt.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to sample from video inputs.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        "--max-denoising-steps",
        type=int,
        default=None,
        help="Maximum DiffusionGemma denoising steps per canvas.",
    )
    parser.add_argument(
        "--diffusion-sampler",
        choices=("confidence-threshold", "entropy-bound"),
        default="confidence-threshold",
        help="DiffusionGemma sampler used by mlx-vlm.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold for mlx-vlm's confidence-threshold sampler.",
    )
    parser.add_argument(
        "--diffusion-min-canvas-length",
        type=int,
        default=None,
        help="Minimum active DiffusionGemma canvas length.",
    )
    parser.add_argument(
        "--diffusion-max-canvas-length",
        type=int,
        default=None,
        help="Maximum active DiffusionGemma canvas length.",
    )
    parser.add_argument(
        "--diffusion-full-canvas",
        action="store_true",
        help="Always denoise a full checkpoint-sized canvas.",
    )
    parser.add_argument(
        "--diffusion-static-cache",
        action="store_true",
        help="Use mlx-vlm's static cache path for diffusion generation.",
    )
    parser.add_argument(
        "--diffusion-compile",
        action="store_true",
        help="Enable mlx-vlm's compiled diffusion decoder path.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--kv-bits", type=float, default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument(
        "--kv-quant-scheme",
        choices=("uniform", "turboquant"),
        default="uniform",
    )
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Use raw prompt text instead of mlx-vlm chat template rendering.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Skip special tokens in mlx-vlm detokenization.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True while loading the model.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision to load.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a short warmup generation before measured prompts.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print generated text for each prompt.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("mlx_vlm_benchmark_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("mlx_vlm_benchmark_results.json"),
        help="JSON output path.",
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
    return parser


def load_mlx_vlm():
    try:
        from mlx_vlm import load
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as exc:
        raise SystemExit(
            "mlx-vlm is not installed. Install it with `uv add mlx-vlm` "
            "or run this benchmark in an environment that already has mlx-vlm."
        ) from exc

    return load, stream_generate, apply_chat_template


def normalize_media(values: list[str] | None) -> list[str] | None:
    return values or None


def build_prompt(
    *,
    processor,
    config,
    prompt: str,
    args: argparse.Namespace,
    apply_chat_template,
) -> str:
    if args.disable_chat_template:
        return prompt

    template_kwargs: dict[str, Any] = {}
    if args.enable_thinking is not None:
        template_kwargs["enable_thinking"] = args.enable_thinking
    if args.video is not None:
        template_kwargs["video"] = args.video
        template_kwargs["fps"] = args.fps

    return apply_chat_template(
        processor,
        config,
        prompt,
        num_images=len(args.image or []),
        num_audios=len(args.audio or []),
        **template_kwargs,
    )


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "diffusion_sampler": args.diffusion_sampler,
        "max_kv_size": args.max_kv_size,
        "kv_bits": args.kv_bits,
        "kv_group_size": args.kv_group_size,
        "kv_quant_scheme": args.kv_quant_scheme,
        "skip_special_tokens": args.skip_special_tokens,
        "fps": args.fps,
    }
    if args.max_denoising_steps is not None:
        kwargs["max_denoising_steps"] = args.max_denoising_steps
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold
        kwargs["diffusion_threshold"] = args.threshold
    if args.diffusion_min_canvas_length is not None:
        kwargs["diffusion_min_canvas_length"] = args.diffusion_min_canvas_length
    if args.diffusion_max_canvas_length is not None:
        kwargs["diffusion_max_canvas_length"] = args.diffusion_max_canvas_length
    if args.diffusion_full_canvas:
        kwargs["diffusion_full_canvas"] = True
    if args.diffusion_static_cache:
        kwargs["diffusion_static_cache"] = True
    if args.diffusion_compile:
        kwargs["diffusion_compile"] = True
    if args.prefill_step_size is not None:
        kwargs["prefill_step_size"] = args.prefill_step_size
    if args.enable_thinking is not None:
        kwargs["enable_thinking"] = args.enable_thinking
    return kwargs


def benchmark_one_prompt(
    *,
    model,
    processor,
    prompt_spec: PromptSpec,
    prompt_index: int,
    model_id: str,
    args: argparse.Namespace,
    stream_generate,
    apply_chat_template,
) -> BenchmarkResult:
    if args.seed is not None:
        import mlx.core as mx

        mx.random.seed(args.seed)

    config = getattr(model, "config", None)
    prompt = build_prompt(
        processor=processor,
        config=config,
        prompt=prompt_spec.text,
        args=args,
        apply_chat_template=apply_chat_template,
    )
    tokenizer = get_tokenizer(processor)
    prompt_tokens = encode_token_count(tokenizer, prompt, add_special_tokens=False)

    pieces: list[str] = []
    first_token_time: float | None = None
    diffusion_canvas_tokens = None
    diffusion_denoising_steps = None
    diffusion_work_tokens = None

    start_time = time.perf_counter()
    for response in stream_generate(
        model,
        processor,
        prompt,
        image=normalize_media(args.image),
        audio=normalize_media(args.audio),
        video=normalize_media(args.video),
        **generation_kwargs(args),
    ):
        if first_token_time is None and response.text:
            first_token_time = time.perf_counter()
        pieces.append(response.text)
        diffusion_canvas_tokens = getattr(
            response, "diffusion_canvas_tokens", diffusion_canvas_tokens
        )
        diffusion_denoising_steps = getattr(
            response, "diffusion_denoising_steps", diffusion_denoising_steps
        )
        diffusion_work_tokens = getattr(
            response, "diffusion_work_tokens", diffusion_work_tokens
        )
        if args.print_response:
            print(response.text, end="", flush=True)
    end_time = time.perf_counter()

    if args.print_response:
        print()
        if not "".join(pieces):
            print("[no text generated]")

    generated_text = "".join(pieces)
    generated_tokens = generated_text_token_count(tokenizer, generated_text)
    total_time = end_time - start_time
    ttft = None if first_token_time is None else first_token_time - start_time
    decode_window = None
    if first_token_time is not None and end_time > first_token_time:
        decode_window = end_time - first_token_time

    generation_toks_per_s = (
        generated_tokens / decode_window
        if decode_window and generated_tokens > 0
        else None
    )
    overall_toks_per_s = (
        generated_tokens / total_time
        if generated_tokens > 0 and total_time > 0
        else None
    )

    return BenchmarkResult(
        model_label="mlx-vlm",
        model_id=model_id,
        device="mlx",
        prompt_index=prompt_index,
        prompt_type=prompt_spec.prompt_type,
        prompt_chars=len(prompt_spec.text),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        time_to_first_token_s=ttft,
        generation_tokens_per_s=generation_toks_per_s,
        overall_tokens_per_s=overall_toks_per_s,
        total_time_s=total_time,
        diffusion_canvas_tokens=diffusion_canvas_tokens,
        diffusion_denoising_steps=diffusion_denoising_steps,
        diffusion_work_tokens=diffusion_work_tokens,
    )


def warmup_model(
    *,
    model,
    processor,
    args: argparse.Namespace,
    stream_generate,
    apply_chat_template,
) -> None:
    warmup_prompt = build_prompt(
        processor=processor,
        config=getattr(model, "config", None),
        prompt="Write one short sentence about Apple Silicon inference.",
        args=args,
        apply_chat_template=apply_chat_template,
    )
    kwargs = generation_kwargs(args)
    kwargs["max_tokens"] = min(args.max_new_tokens, 32)
    for _ in stream_generate(
        model,
        processor,
        warmup_prompt,
        image=normalize_media(args.image),
        audio=normalize_media(args.audio),
        video=normalize_media(args.video),
        **kwargs,
    ):
        pass


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    load, stream_generate, apply_chat_template = load_mlx_vlm()
    prompts = load_prompts(args.prompt_file)

    print(f"Loading {args.model} with mlx-vlm...")
    model, processor = load(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )

    if args.warmup:
        print("Running warmup...")
        warmup_model(
            model=model,
            processor=processor,
            args=args,
            stream_generate=stream_generate,
            apply_chat_template=apply_chat_template,
        )

    results: list[BenchmarkResult] = []
    for prompt_index, prompt_spec in enumerate(prompts, start=1):
        print(
            f"Benchmarking mlx-vlm on mlx, "
            f"{prompt_spec.prompt_type}, prompt {prompt_index}/{len(prompts)}..."
        )
        results.append(
            benchmark_one_prompt(
                model=model,
                processor=processor,
                prompt_spec=prompt_spec,
                prompt_index=prompt_index,
                model_id=args.model,
                args=args,
                stream_generate=stream_generate,
                apply_chat_template=apply_chat_template,
            )
        )

    write_csv(args.csv, results)
    write_json(args.json, results)
    print_summary(results)
    print()
    print(f"CSV written to {args.csv}")
    print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()
