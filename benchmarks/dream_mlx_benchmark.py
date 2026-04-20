from __future__ import annotations

import argparse
import time
from pathlib import Path

from dream_mlx import DreamGenerationConfig, DreamGenerator, load
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
        description="Benchmark Dream MLX inference on prompts/*.txt."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Dream MLX model id or local path. Default: {DEFAULT_MODEL_ID}",
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
        default=20,
        help="Number of diffusion denoising steps.",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Enable safe opt-in mx.compile helpers inside dream_mlx.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through trust_remote_code=True for tokenizer loading.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the final generated text for each prompt.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dream_mlx_benchmark_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("dream_mlx_benchmark_results.json"),
        help="JSON output path.",
    )
    return parser


def benchmark_one_prompt(
    *,
    generator: DreamGenerator,
    tokenizer,
    prompt_spec: PromptSpec,
    prompt_index: int,
    model_id: str,
    max_new_tokens: int,
    steps: int,
    use_compile: bool,
    print_response: bool,
) -> BenchmarkResult:
    rendered_prompt = prompt_spec.text
    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        try:
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_spec.text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered_prompt = prompt_spec.text

    prompt_token_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    prompt_tokens = len(prompt_token_ids)

    start_time = time.perf_counter()
    generated_text = generator.generate(
        prompt_spec.text,
        generation_config=DreamGenerationConfig(
            max_new_tokens=max_new_tokens,
            steps=steps,
            use_compile=use_compile,
        ),
    )
    end_time = time.perf_counter()

    total_time = end_time - start_time
    generated_tokens = max_new_tokens
    overall_toks_per_s = generated_tokens / total_time if total_time > 0 else None

    if print_response:
        print(generated_text if generated_text else "[no text generated]")

    return BenchmarkResult(
        model_label="dream-mlx",
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file)

    print(f"Loading {args.model} with dream_mlx...")
    model, tokenizer = load(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    generator = DreamGenerator(model=model, tokenizer=tokenizer)

    results: list[BenchmarkResult] = []
    for prompt_index, prompt_spec in enumerate(prompts, start=1):
        print(
            f"Benchmarking dream-mlx on mlx, "
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
            use_compile=args.use_compile,
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
