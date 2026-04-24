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

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"


@dataclass
class BenchmarkResult:
    model_label: str
    model_id: str
    device: str
    prompt_index: int
    prompt_type: str
    prompt_chars: int
    prompt_tokens: int
    generated_tokens: int
    time_to_first_token_s: float | None
    generation_tokens_per_s: float | None
    overall_tokens_per_s: float | None
    total_time_s: float


@dataclass
class PromptSpec:
    prompt_type: str
    text: str


def load_prompts(prompt_file: Path | None) -> list[PromptSpec]:
    if prompt_file is None:
        prompt_files = sorted(PROMPTS_DIR.glob("*.txt"))
        if not prompt_files:
            raise ValueError(f"No default prompt files found in {PROMPTS_DIR}")

        prompts: list[PromptSpec] = []
        for path in prompt_files:
            prompt_type = path.stem
            prompts.extend(
                PromptSpec(prompt_type=prompt_type, text=line.strip())
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

        if not prompts:
            raise ValueError(
                f"Default prompt files in {PROMPTS_DIR} did not contain any prompts."
            )
        return prompts

    raw = prompt_file.read_text(encoding="utf-8").strip()
    if prompt_file.suffix.lower() == ".json":
        data = json.loads(raw)
        if not isinstance(data, list) or not all(
            isinstance(item, str) for item in data
        ):
            raise ValueError("JSON prompt file must be a list of strings.")
        return [PromptSpec(prompt_type=prompt_file.stem, text=item) for item in data]

    prompts = [
        PromptSpec(prompt_type=prompt_file.stem, text=line.strip())
        for line in raw.splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError("Prompt file did not contain any prompts.")
    return prompts


def write_csv(path: Path, results: Iterable[BenchmarkResult]) -> None:
    rows = [asdict(result) for result in results]
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, results: Iterable[BenchmarkResult]) -> None:
    data = [asdict(result) for result in results]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_summary(results: list[BenchmarkResult]) -> None:
    print()
    print("Benchmark summary")
    print("=" * 100)
    header = (
        f"{'model':20} {'device':6} {'type':8} {'prompt':6} {'ttft(s)':>10} "
        f"{'gen tok/s':>12} {'overall tok/s':>14} {'total(s)':>10} {'gen toks':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        ttft = (
            f"{result.time_to_first_token_s:.3f}"
            if result.time_to_first_token_s is not None
            else "n/a"
        )
        gen_tps = (
            f"{result.generation_tokens_per_s:.2f}"
            if result.generation_tokens_per_s is not None
            else "n/a"
        )
        overall_tps = (
            f"{result.overall_tokens_per_s:.2f}"
            if result.overall_tokens_per_s is not None
            else "n/a"
        )
        print(
            f"{result.model_label:20} {result.device:6} {result.prompt_type:8} "
            f"{result.prompt_index:6d} {ttft:>10} {gen_tps:>12} "
            f"{overall_tps:>14} {result.total_time_s:>10.3f} "
            f"{result.generated_tokens:>10d}"
        )

    aggregates: dict[tuple[str, str, str], list[float]] = {}
    overall_aggregates: dict[tuple[str, str], list[float]] = {}
    for result in results:
        if result.overall_tokens_per_s is None:
            continue
        aggregates.setdefault(
            (result.model_label, result.device, result.prompt_type), []
        ).append(result.overall_tokens_per_s)
        overall_aggregates.setdefault((result.model_label, result.device), []).append(
            result.overall_tokens_per_s
        )

    if not aggregates:
        return

    print()
    print("Average overall tok/s by prompt type")
    print("-" * 100)
    for (model_label, device, prompt_type), values in sorted(aggregates.items()):
        print(
            f"{model_label:20} {device:6} {prompt_type:8} "
            f"{sum(values) / len(values):10.2f}"
        )

    print()
    print("Average overall tok/s overall")
    print("-" * 100)
    for (model_label, device), values in sorted(overall_aggregates.items()):
        print(
            f"{model_label:20} {device:6} {'overall':8} "
            f"{sum(values) / len(values):10.2f}"
        )
