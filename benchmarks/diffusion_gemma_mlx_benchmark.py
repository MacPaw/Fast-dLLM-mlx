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


from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmarks.utils import (
    BenchmarkResult,
    PromptSpec,
    generated_text_token_count,
    load_prompts,
    print_summary,
    write_csv,
    write_json,
)
from diffusion_gemma_mlx import (
    DiffusionGemmaGenerationConfig,
    DiffusionGemmaGenerator,
    load,
    stream_diffusion_generate,
)
from diffusion_gemma_mlx.generate import _prepare_prompt, _wired_limit_bytes

DEFAULT_MODEL_ID = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def add_thinking_args(parser: argparse.ArgumentParser) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark classic DiffusionGemma MLX inference on prompts/*.txt."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"DiffusionGemma model id or local path. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional text or JSON file containing prompts. Falls back to prompts/*.txt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of diffusion slots to generate per prompt.",
    )
    parser.add_argument(
        "--max-denoising-steps",
        type=int,
        default=48,
        help="Maximum denoising steps per canvas.",
    )
    parser.add_argument(
        "--max-canvases",
        type=int,
        default=8,
        help="Maximum number of generated canvases.",
    )
    parser.add_argument("--min-canvas-length", type=int, default=64)
    parser.add_argument("--max-canvas-length", type=int, default=None)
    parser.add_argument(
        "--full-canvas",
        action="store_true",
        help="Always denoise full checkpoint-sized canvases.",
    )
    parser.add_argument("--t-min", type=float, default=0.4)
    parser.add_argument("--t-max", type=float, default=0.8)
    parser.add_argument("--entropy-bound", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.005)
    parser.add_argument("--stability-threshold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Compile repeated DiffusionGemma decoder logits calls.",
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
        default=None,
        help="Temporarily set the MLX wired memory limit during generation.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Use raw prompt text instead of tokenizer chat templates.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from the local Hugging Face cache only.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a short warmup generation before measured prompts.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the generated text for each prompt.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("diffusion_gemma_mlx_benchmark_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("diffusion_gemma_mlx_benchmark_results.json"),
        help="JSON output path.",
    )
    add_thinking_args(parser)
    return parser


def build_generation_config(
    args: argparse.Namespace,
    max_new_tokens: int,
) -> DiffusionGemmaGenerationConfig:
    return DiffusionGemmaGenerationConfig(
        max_new_tokens=max_new_tokens,
        max_canvases=args.max_canvases,
        min_canvas_length=args.min_canvas_length,
        max_canvas_length=args.max_canvas_length,
        full_canvas=args.full_canvas,
        max_denoising_steps=args.max_denoising_steps,
        t_min=args.t_min,
        t_max=args.t_max,
        entropy_bound=args.entropy_bound,
        confidence_threshold=args.confidence_threshold,
        stability_threshold=args.stability_threshold,
        use_chat_template=not args.disable_chat_template,
        enable_thinking=args.enable_thinking,
        use_compile=args.use_compile,
        precompute_decoder_masks=args.precompute_decoder_masks,
        use_generation_stream=not args.no_generation_stream,
        wired_limit=_wired_limit_bytes(args.wired_limit_gb),
        seed=args.seed,
    )


def benchmark_one_prompt(
    *,
    model,
    tokenizer,
    prompt_spec: PromptSpec,
    prompt_index: int,
    model_id: str,
    args: argparse.Namespace,
) -> BenchmarkResult:
    config = build_generation_config(args, args.max_new_tokens)

    pieces: list[str] = []
    prompt_tokens = int(_prepare_prompt(tokenizer, prompt_spec.text, config).shape[1])
    diffusion_stats = {
        "canvas_tokens": 0,
        "denoising_steps": 0,
        "work_tokens": 0,
    }

    def stats_callback(*, canvas_length: int, denoising_steps: int) -> None:
        diffusion_stats["canvas_tokens"] += canvas_length
        diffusion_stats["denoising_steps"] += denoising_steps
        diffusion_stats["work_tokens"] += canvas_length * denoising_steps

    start_time = time.perf_counter()
    for response in stream_diffusion_generate(
        model,
        tokenizer,
        prompt_spec.text,
        generation_config=config,
        stats_callback=stats_callback,
    ):
        pieces.append(response.text)
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
    overall_toks_per_s = (
        generated_tokens / total_time
        if generated_tokens > 0 and total_time > 0
        else None
    )

    return BenchmarkResult(
        model_label="diffusion-gemma-mlx",
        model_id=model_id,
        device="mlx",
        prompt_index=prompt_index,
        prompt_type=prompt_spec.prompt_type,
        prompt_chars=len(prompt_spec.text),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        time_to_first_token_s=None,
        generation_tokens_per_s=None,
        overall_tokens_per_s=overall_toks_per_s,
        total_time_s=total_time,
        diffusion_canvas_tokens=diffusion_stats["canvas_tokens"],
        diffusion_denoising_steps=diffusion_stats["denoising_steps"],
        diffusion_work_tokens=diffusion_stats["work_tokens"],
    )


def warmup_model(generator: DiffusionGemmaGenerator, args: argparse.Namespace) -> None:
    generator.generate(
        "Write one short sentence about masked diffusion language models.",
        generation_config=build_generation_config(
            args,
            min(args.max_new_tokens, 32),
        ),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file)

    print(f"Loading {args.model} with diffusion_gemma_mlx...")
    model, tokenizer = load(args.model, local_files_only=args.local_files_only)
    generator = DiffusionGemmaGenerator(model=model, tokenizer=tokenizer)

    if args.warmup:
        print("Running warmup...")
        warmup_model(generator, args)

    results: list[BenchmarkResult] = []
    for prompt_index, prompt_spec in enumerate(prompts, start=1):
        print(
            f"Benchmarking diffusion-gemma-mlx on mlx, "
            f"{prompt_spec.prompt_type}, prompt {prompt_index}/{len(prompts)}..."
        )
        results.append(
            benchmark_one_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt_spec=prompt_spec,
                prompt_index=prompt_index,
                model_id=args.model,
                args=args,
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
