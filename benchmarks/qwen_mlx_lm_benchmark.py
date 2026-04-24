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

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from benchmarks.utils import (
    BenchmarkResult,
    PromptSpec,
    load_prompts,
    print_summary,
    write_csv,
    write_json,
)

DEFAULT_MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-8bit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark an MLX LM model on the prompts used by main.py."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"MLX model id or local path. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional text or JSON file containing prompts. Falls back to prompts/*.txt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional MLX sampling seed.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Use the raw prompt text instead of tokenizer chat templates.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a short warmup generation before the measured prompts.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the generated text for each prompt.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("mlx_lm_qwen_benchmark_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("mlx_lm_qwen_benchmark_results.json"),
        help="JSON output path.",
    )
    return parser


def maybe_apply_chat_template(
    tokenizer, prompt: str, disable_chat_template: bool
) -> str:
    if disable_chat_template or not getattr(tokenizer, "has_chat_template", False):
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
    model,
    tokenizer,
    prompt_spec: PromptSpec,
    prompt_index: int,
    model_id: str,
    max_new_tokens: int,
    temp: float,
    top_p: float,
    seed: int | None,
    disable_chat_template: bool,
    print_response: bool,
) -> BenchmarkResult:
    rendered_prompt = maybe_apply_chat_template(
        tokenizer, prompt_spec.text, disable_chat_template
    )

    prompt_token_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    prompt_tokens = len(prompt_token_ids)

    generation_kwargs = {
        "max_tokens": max_new_tokens,
        "sampler": make_sampler(temp=temp, top_p=top_p),
    }
    if seed is not None:
        mx.random.seed(seed)

    pieces: list[str] = []
    first_token_time: float | None = None
    last_response = None

    start_time = time.perf_counter()
    for response in stream_generate(
        model, tokenizer, rendered_prompt, **generation_kwargs
    ):
        last_response = response
        if response.text:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            pieces.append(response.text)
            if print_response:
                print(response.text, end="", flush=True)
    end_time = time.perf_counter()

    if print_response:
        print()

    generated_text = "".join(pieces)
    generated_tokens = (
        0 if last_response is None else int(last_response.generation_tokens)
    )
    ttft = None if first_token_time is None else first_token_time - start_time

    decode_window = None
    if first_token_time is not None and end_time > first_token_time:
        decode_window = end_time - first_token_time

    total_time = end_time - start_time
    generation_toks_per_s = (
        generated_tokens / decode_window
        if decode_window and generated_tokens > 0
        else None
    )
    overall_toks_per_s = generated_tokens / total_time if generated_tokens > 0 else None

    if print_response and not generated_text:
        print("[no text generated]")

    return BenchmarkResult(
        model_label="mlx-lm-qwen",
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
    )


def warmup_model(model, tokenizer, disable_chat_template: bool) -> None:
    warmup_prompt = (
        "Write a short paragraph about Apple Silicon language model inference."
    )
    rendered_prompt = maybe_apply_chat_template(
        tokenizer, warmup_prompt, disable_chat_template
    )
    for _ in stream_generate(
        model,
        tokenizer,
        rendered_prompt,
        max_tokens=32,
        sampler=make_sampler(temp=0.0, top_p=1.0),
    ):
        pass


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file)

    print(f"Loading {args.model} with mlx_lm...")
    model, tokenizer = load(args.model)

    if args.warmup:
        print("Running warmup...")
        warmup_model(model, tokenizer, args.disable_chat_template)

    results: list[BenchmarkResult] = []
    for prompt_index, prompt_spec in enumerate(prompts, start=1):
        print(
            f"Benchmarking mlx-lm-qwen on mlx, "
            f"{prompt_spec.prompt_type}, prompt {prompt_index}/{len(prompts)}..."
        )
        result = benchmark_one_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt_spec=prompt_spec,
            prompt_index=prompt_index,
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
            temp=args.temp,
            top_p=args.top_p,
            seed=args.seed,
            disable_chat_template=args.disable_chat_template,
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
