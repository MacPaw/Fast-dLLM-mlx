from __future__ import annotations

import argparse
import time
from pathlib import Path

from fast_dllm_mlx import DreamGenerationConfig, DreamGenerator, load
from benchmarks.utils import (
    BenchmarkResult,
    PromptSpec,
    load_prompts,
    print_summary,
    write_csv,
    write_json,
)

DEFAULT_MODEL_ID = "mlx-community/DiffuCoder-7B-cpGRPO-8bit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Fast-dLLM MLX Dream inference on prompts/*.txt."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Fast-dLLM MLX model id or local path. Default: {DEFAULT_MODEL_ID}",
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
        "--steps",
        type=int,
        default=128,
        help="Total Fast-dLLM denoising steps.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Semi-autoregressive block length.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for finalizing easy tokens in parallel.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through trust_remote_code=True for tokenizer loading.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a short warmup generation before measured prompts.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the final generated text for each prompt.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("fast_dllm_mlx_benchmark_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("fast_dllm_mlx_benchmark_results.json"),
        help="JSON output path.",
    )
    return parser


def maybe_apply_chat_template(tokenizer, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return prompt

    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def benchmark_one_prompt(
    *,
    generator: DreamGenerator,
    tokenizer,
    prompt_spec: PromptSpec,
    prompt_index: int,
    model_id: str,
    max_new_tokens: int,
    steps: int,
    block_length: int,
    threshold: float,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    print_response: bool,
) -> BenchmarkResult:
    rendered_prompt = maybe_apply_chat_template(tokenizer, prompt_spec.text)
    prompt_token_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    prompt_tokens = len(prompt_token_ids)

    start_time = time.perf_counter()
    generated_text = generator.generate(
        prompt_spec.text,
        generation_config=DreamGenerationConfig(
            max_new_tokens=max_new_tokens,
            steps=steps,
            block_length=block_length,
            threshold=threshold,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            dual_cache=True,
        ),
    )
    end_time = time.perf_counter()

    total_time = end_time - start_time
    generated_tokens = max_new_tokens
    overall_toks_per_s = generated_tokens / total_time if total_time > 0 else None

    if print_response:
        print(generated_text if generated_text else "[no text generated]")

    return BenchmarkResult(
        model_label="fast-dllm-mlx",
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
    )


def warmup_model(generator: DreamGenerator) -> None:
    generator.generate(
        "Write one short sentence about masked diffusion language models.",
        generation_config=DreamGenerationConfig(
            max_new_tokens=32,
            steps=32,
            block_length=32,
            threshold=0.9,
            temperature=0.0,
            top_p=None,
            top_k=None,
            dual_cache=True,
        ),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file)

    print(f"Loading {args.model} with fast_dllm_mlx...")
    model, tokenizer = load(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    generator = DreamGenerator(model=model, tokenizer=tokenizer)

    if args.warmup:
        print("Running warmup...")
        warmup_model(generator)

    results: list[BenchmarkResult] = []
    for prompt_index, prompt_spec in enumerate(prompts, start=1):
        print(
            f"Benchmarking fast-dllm-mlx on mlx, "
            f"{prompt_spec.prompt_type}, prompt {prompt_index}/{len(prompts)}..."
        )
        result = benchmark_one_prompt(
            generator=generator,
            tokenizer=tokenizer,
            prompt_spec=prompt_spec,
            prompt_index=prompt_index,
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            block_length=args.block_length,
            threshold=args.threshold,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            print_response=args.print_response,
        )
        results.append(result)

    write_csv(args.csv, results)
    write_json(args.json, results)
    print_summary(results)
    print()
    print(f"CSV written to {args.csv}")
    print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()
