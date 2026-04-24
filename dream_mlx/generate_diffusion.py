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
from functools import partial
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.models import cache
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
from transformers import AutoTokenizer

try:
    from .model import Model, ModelArgs
except ImportError:
    from model import Model, ModelArgs


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
    """Configuration for Dream diffusion generation."""

    temperature: float = 0.4
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_length: int = 256
    max_new_tokens: Optional[int] = None

    # Diffusion specific params
    eps: float = 1e-3
    steps: int = 512
    alg: str = "origin"  # 'origin', 'maskgit_plus', 'topk_margin', 'entropy'
    alg_temp: Optional[float] = None
    use_compile: bool = False

    # Special tokens
    mask_token_id: Optional[int] = 151666
    pad_token_id: Optional[int] = 151643
    bos_token_id: Optional[int] = 151643
    eos_token_id: Optional[int] = 151643

    # Output control
    num_return_sequences: int = 1
    use_chat_template: bool = True


@dataclass
class DreamGenerationResponse:
    """Response from Dream generation."""

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
    """Compatibility wrapper around the functional Dream diffusion API."""

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


@partial(mx.compile, shapeless=True)
def _compiled_top_k_logits(logits: mx.array, top_k: int) -> mx.array:
    top_k = min(top_k, logits.shape[-1])
    kth_largest = mx.sort(logits, axis=-1)[:, -top_k][:, None]
    return mx.where(logits < kth_largest, -mx.inf, logits)


@partial(mx.compile, shapeless=True)
def _compiled_greedy_sample(logits: mx.array) -> Tuple[mx.array, mx.array]:
    probs = mx.softmax(logits, axis=-1)
    return mx.max(probs, axis=-1), mx.argmax(probs, axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state, shapeless=True)
def _compiled_random_sample(logits: mx.array) -> Tuple[mx.array, mx.array]:
    probs = mx.softmax(logits, axis=-1)
    x0 = mx.random.categorical(logits, axis=-1)
    confidence = probs[mx.arange(probs.shape[0]), x0]
    return confidence, x0


def top_p_logits(logits: mx.array, top_p: float) -> mx.array:
    """Apply top-p filtering to logits."""
    sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]  # Sort descending
    sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]

    cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift indices to keep first token above threshold
    sorted_indices_to_remove = mx.concatenate(
        [
            mx.zeros_like(sorted_indices_to_remove[:, :1]),
            sorted_indices_to_remove[:, :-1],
        ],
        axis=-1,
    )

    # Create mask for original indices
    mask = mx.zeros_like(logits, dtype=mx.bool_)
    batch_indices = mx.arange(logits.shape[0])[:, None]
    # Note: This also needs to be fixed for MLX, but keeping for now
    # mask = mask.at[batch_indices, sorted_indices].set(sorted_indices_to_remove)

    return mx.where(mask, -mx.inf, logits)


def top_k_logits(logits: mx.array, top_k: int) -> mx.array:
    """Apply top-k filtering to logits."""
    return _compiled_top_k_logits(logits, top_k)


def sample_tokens(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
    key: Optional[mx.array] = None,
    use_compile: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Sample tokens from logits with various confidence measures."""

    if temperature > 0:
        logits = logits / temperature

    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)

    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    if temperature > 0:
        if key is None:
            if use_compile:
                confidence, x0 = _compiled_random_sample(logits)
            else:
                key = mx.random.key(int(time.time()))
                probs = mx.softmax(logits, axis=-1)
                x0 = mx.random.categorical(logits, axis=-1, key=key)
                confidence = probs[mx.arange(probs.shape[0]), x0]
        else:
            probs = mx.softmax(logits, axis=-1)
            x0 = mx.random.categorical(logits, axis=-1, key=key)
            confidence = probs[mx.arange(probs.shape[0]), x0]
    else:
        if use_compile and not margin_confidence and not neg_entropy:
            confidence, x0 = _compiled_greedy_sample(logits)
        else:
            probs = mx.softmax(logits, axis=-1)
            confidence = mx.max(probs, axis=-1)
            x0 = mx.argmax(probs, axis=-1)

    if margin_confidence or neg_entropy:
        probs = mx.softmax(logits, axis=-1)

    if margin_confidence:
        sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]  # Descending
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = mx.log(probs + epsilon)
        confidence = mx.sum(probs * log_probs, axis=-1)

    return confidence, x0


def custom_nonzero(arr: mx.array) -> mx.array:
    """
    Returns indices of True values in a boolean array.
    Equivalent to numpy's np.argwhere(arr) but for MLX.
    """
    if arr.size == 0:
        return mx.array([], dtype=mx.int32).reshape(0, arr.ndim)

    indices = np.argwhere(np.array(arr))
    if indices.size == 0:
        return mx.array([], dtype=mx.int32).reshape(0, arr.ndim)
    return mx.array(indices, dtype=mx.int32)


def _shift_logits(logits: mx.array) -> mx.array:
    if logits.shape[1] == 0:
        return logits
    if logits.shape[1] == 1:
        return logits
    return mx.concatenate([logits[:, :1], logits[:, :-1]], axis=1)


def diffusion_generate_step(
    prompt: mx.array,
    model: nn.Module,
    generation_config: DreamGenerationConfig,
    prompt_cache: Optional[Any] = None,
) -> mx.array:
    # Extract config parameters
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    eps = generation_config.eps
    alg = generation_config.alg
    alg_temp = generation_config.alg_temp
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    batch_size = prompt.shape[0]

    # Pad input to max_length with mask tokens
    pad_length = max_length - prompt.shape[1]
    if pad_length > 0:
        mask_tokens = mx.full(
            (batch_size, pad_length), mask_token_id, dtype=prompt.dtype
        )
        x = mx.concatenate([prompt, mask_tokens], axis=1)
    else:
        x = prompt[:, :max_length]

    # Create timestep schedule
    timesteps = mx.linspace(1, eps, steps + 1)

    # Initialize random key
    key = mx.random.key(int(time.time() * 1000))

    # Diffusion sampling loop
    for i in range(steps):
        # Find masked positions manually
        mask_index = x == mask_token_id

        # Check if there are any masked tokens left
        if not mx.any(mask_index):
            break  # No more masked tokens

        # Forward pass through model
        logits = model(x, cache=prompt_cache)

        # Apply logits shifting: [B, L, V] -> [B, L, V]
        if logits.shape[1] == x.shape[1] + 1:
            logits = logits[:, :-1]  # Remove last position
        logits = _shift_logits(logits)

        # Get current timestep values
        t = timesteps[i]
        s = timesteps[i + 1]

        # Manually find masked positions and extract logits
        masked_positions = []
        masked_logits_list = []

        # Iterate through all positions to find masked ones
        for b in range(batch_size):
            for pos in range(x.shape[1]):
                # Check if this position is masked
                if x[b, pos].item() == mask_token_id:
                    masked_positions.append((b, pos))
                    masked_logits_list.append(logits[b, pos])

        if len(masked_positions) == 0:
            continue

        # Stack the logits for masked positions
        masked_logits = mx.stack(masked_logits_list)

        # Split keys for this step
        key, sample_key, transfer_key = mx.random.split(key, 3)

        # Convert to list for easier manipulation
        x_list = x.tolist()

        if alg == "origin":
            # Original algorithm: probabilistic transfer
            p_transfer = 1 - s / t if i < steps - 1 else 1

            # Sample new tokens
            _, new_tokens = sample_tokens(
                masked_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                key=sample_key,
                use_compile=generation_config.use_compile,
            )

            # Apply transfer with probability
            transfer_probs = mx.random.uniform(shape=new_tokens.shape, key=transfer_key)
            update_mask = transfer_probs < p_transfer

            # Update tokens where transfer condition is met
            for idx, (b, pos) in enumerate(masked_positions):
                if update_mask[idx].item():
                    x_list[b][pos] = int(new_tokens[idx].item())

        else:
            # Confidence-based algorithms
            if alg == "maskgit_plus":
                confidence, new_tokens = sample_tokens(
                    masked_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    key=sample_key,
                    use_compile=generation_config.use_compile,
                )
            elif alg == "topk_margin":
                confidence, new_tokens = sample_tokens(
                    masked_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    margin_confidence=True,
                    key=sample_key,
                    use_compile=generation_config.use_compile,
                )
            elif alg == "entropy":
                confidence, new_tokens = sample_tokens(
                    masked_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=True,
                    key=sample_key,
                    use_compile=generation_config.use_compile,
                )
            else:
                raise ValueError(f"Unknown algorithm: {alg}")

            # Calculate number of tokens to update
            num_masked = len(masked_positions)
            num_update = int(num_masked * (1 - s / t)) if i < steps - 1 else num_masked

            if num_update > 0:
                # Get top confidence positions
                sorted_indices = mx.argsort(-confidence)
                top_indices = sorted_indices[:num_update]

                # Update tokens in sequence
                for idx in top_indices:
                    idx_val = int(idx.item())
                    if idx_val < len(masked_positions):
                        b, pos = masked_positions[idx_val]
                        x_list[b][pos] = int(new_tokens[idx_val].item())

        # Convert back to MLX array
        x = mx.array(x_list)

    # Return the final result after all diffusion steps
    return x


def stream_diffusion_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, mx.array, List[int]],
    generation_config: Optional[DreamGenerationConfig] = None,
    **kwargs,
) -> Generator[DreamGenerationResponse, None, None]:
    """
    Generate text using Dream's diffusion process with step-by-step streaming.
    """

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Prepare generation config
    if generation_config is None:
        generation_config = DreamGenerationConfig()

    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)

    # Encode prompt
    if not isinstance(prompt, mx.array):
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
        prompt = mx.array(prompt)[None, :]  # Add batch dimension

    # Set up special tokens
    if generation_config.mask_token_id is None:
        generation_config.mask_token_id = (
            tokenizer.mask_token_id or tokenizer.unk_token_id
        )

    # Prepare max length
    input_length = prompt.shape[1]
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_length

    # Generate with step-by-step streaming
    tic = time.perf_counter()

    # Disable cache reuse across denoising steps. The sequence changes each step.
    prompt_cache = None

    # Call the streaming diffusion function
    for step_result in diffusion_generate_step_streaming(
        prompt=prompt,
        model=model,
        generation_config=generation_config,
        prompt_cache=prompt_cache,
        tokenizer=tokenizer,
        input_length=input_length,
        start_time=tic,
    ):
        yield step_result


def diffusion_generate_step_streaming(
    prompt: mx.array,
    model: nn.Module,
    generation_config: DreamGenerationConfig,
    prompt_cache: Optional[Any],
    tokenizer,
    input_length: int,
    start_time: float,
) -> Generator[DreamGenerationResponse, None, None]:
    """
    Streaming version of diffusion generation that yields after each step.
    """

    def safe_decode_tokens(tokens, tokenizer, mask_token_id):
        """Safely decode tokens, handling mask tokens and None values."""
        decoded_tokens = []
        for t in tokens:
            token_id = t.item()
            if token_id == mask_token_id:
                # Use a placeholder for mask tokens
                decoded_tokens.append("[MASK]")
            else:
                try:
                    # Try to decode individual token
                    decoded_token = tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    )
                    decoded_tokens.append(decoded_token)
                except (TypeError, ValueError):
                    # If decoding fails, use a placeholder
                    decoded_tokens.append(f"[UNK_{token_id}]")
        return "".join(decoded_tokens)

    # Extract config parameters
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    eps = generation_config.eps
    alg = generation_config.alg
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    batch_size = prompt.shape[0]

    # Pad input to max_length with mask tokens
    pad_length = max_length - prompt.shape[1]
    if pad_length > 0:
        mask_tokens = mx.full(
            (batch_size, pad_length), mask_token_id, dtype=prompt.dtype
        )
        x = mx.concatenate([prompt, mask_tokens], axis=1)
    else:
        x = prompt[:, :max_length]

    # Create timestep schedule
    timesteps = mx.linspace(1, eps, steps + 1)

    # Initialize random key
    key = mx.random.key(int(time.time() * 1000))

    # Yield initial state (all masked)
    current_time = time.perf_counter()
    generated_tokens = x[0, input_length:]
    generated_text = safe_decode_tokens(generated_tokens, tokenizer, mask_token_id)

    yield DreamGenerationResponse(
        text=generated_text,
        sequences=x,
        prompt_tokens=input_length,
        prompt_tps=input_length / (current_time - start_time)
        if current_time > start_time
        else 0,
        generation_tokens=generated_tokens.shape[0],
        generation_tps=0,
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason=f"step_0/{steps}",
    )

    # Diffusion sampling loop
    for i in range(steps):
        # Find masked positions manually
        mask_index = x == mask_token_id

        # Check if there are any masked tokens left
        if not mx.any(mask_index):
            break  # No more masked tokens

        # Forward pass through model
        logits = model(x, cache=prompt_cache)

        # Apply logits shifting: [B, L, V] -> [B, L, V]
        if logits.shape[1] == x.shape[1] + 1:
            logits = logits[:, :-1]  # Remove last position
        logits = _shift_logits(logits)

        # Get current timestep values
        t = timesteps[i]
        s = timesteps[i + 1]

        # Manually find masked positions and extract logits
        masked_positions = []
        masked_logits_list = []

        # Iterate through all positions to find masked ones
        for b in range(batch_size):
            for pos in range(x.shape[1]):
                # Check if this position is masked
                if x[b, pos].item() == mask_token_id:
                    masked_positions.append((b, pos))
                    masked_logits_list.append(logits[b, pos])

        if len(masked_positions) == 0:
            continue

        # Stack the logits for masked positions
        masked_logits = mx.stack(masked_logits_list)

        # Split keys for this step
        key, sample_key, transfer_key = mx.random.split(key, 3)

        # Convert to list for easier manipulation
        x_list = x.tolist()

        if alg == "origin":
            # Original algorithm: probabilistic transfer
            p_transfer = 1 - s / t if i < steps - 1 else 1

            # Sample new tokens
            _, new_tokens = sample_tokens(
                masked_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                key=sample_key,
                use_compile=generation_config.use_compile,
            )

            # Apply transfer with probability
            transfer_probs = mx.random.uniform(shape=new_tokens.shape, key=transfer_key)
            update_mask = transfer_probs < p_transfer

            # Update tokens where transfer condition is met
            for idx, (b, pos) in enumerate(masked_positions):
                if update_mask[idx].item():
                    x_list[b][pos] = int(new_tokens[idx].item())

        else:
            # Confidence-based algorithms
            if alg == "maskgit_plus":
                confidence, new_tokens = sample_tokens(
                    masked_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    key=sample_key,
                    use_compile=generation_config.use_compile,
                )
            elif alg == "topk_margin":
                confidence, new_tokens = sample_tokens(
                    masked_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    margin_confidence=True,
                    key=sample_key,
                    use_compile=generation_config.use_compile,
                )
            elif alg == "entropy":
                confidence, new_tokens = sample_tokens(
                    masked_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=True,
                    key=sample_key,
                    use_compile=generation_config.use_compile,
                )
            else:
                raise ValueError(f"Unknown algorithm: {alg}")

            # Calculate number of tokens to update
            num_masked = len(masked_positions)
            num_update = int(num_masked * (1 - s / t)) if i < steps - 1 else num_masked

            if num_update > 0:
                # Get top confidence positions
                sorted_indices = mx.argsort(-confidence)
                top_indices = sorted_indices[:num_update]

                # Update tokens in sequence
                for idx in top_indices:
                    idx_val = int(idx.item())
                    if idx_val < len(masked_positions):
                        b, pos = masked_positions[idx_val]
                        x_list[b][pos] = int(new_tokens[idx_val].item())

        # Convert back to MLX array
        x = mx.array(x_list)

        # Yield intermediate result after each step
        current_time = time.perf_counter()
        generated_tokens = x[0, input_length:]
        generated_text = safe_decode_tokens(generated_tokens, tokenizer, mask_token_id)

        # Count remaining masked tokens
        remaining_masked = mx.sum(x == mask_token_id).item()

        yield DreamGenerationResponse(
            text=generated_text,
            sequences=x,
            prompt_tokens=input_length,
            prompt_tps=input_length / (current_time - start_time)
            if current_time > start_time
            else 0,
            generation_tokens=generated_tokens.shape[0],
            generation_tps=generated_tokens.shape[0] / (current_time - start_time)
            if current_time > start_time
            else 0,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=f"step_{i + 1}/{steps}_masked_{remaining_masked}",
        )

    # Final result - use skip_special_tokens=True for clean output
    current_time = time.perf_counter()
    generated_tokens = x[0, input_length:]

    # For final result, try to decode normally first
    try:
        generated_text = tokenizer.decode(
            generated_tokens.tolist(), skip_special_tokens=True
        )
    except (TypeError, ValueError):
        # If that fails, use safe decoding
        generated_text = safe_decode_tokens(generated_tokens, tokenizer, mask_token_id)
        # Remove [MASK] tokens from final output
        generated_text = generated_text.replace("[MASK]", "")

    for token_text in ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|dlm_pad|>"):
        generated_text = generated_text.replace(token_text, "")
    generated_text = generated_text.strip()

    yield DreamGenerationResponse(
        text=generated_text,
        sequences=x,
        prompt_tokens=input_length,
        prompt_tps=input_length / (current_time - start_time)
        if current_time > start_time
        else 0,
        generation_tokens=generated_tokens.shape[0],
        generation_tps=generated_tokens.shape[0] / (current_time - start_time)
        if current_time > start_time
        else 0,
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="complete",
    )


def diffusion_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, List[int]],
    live_preview: bool = True,
    **kwargs,
) -> str:
    """
    Generate a complete response using Dream's diffusion process with live updates.
    """
    if live_preview:
        print("=" * 50)
        print("🎭 DIFFUSION GENERATION - LIVE VIEW")
        print("=" * 50)
        print(f"Prompt: {prompt}")
        print("=" * 50)

    response = None

    for i, response in enumerate(
        stream_diffusion_generate(model, tokenizer, prompt, **kwargs)
    ):
        if not live_preview:
            continue

        # Clear the current line and move cursor to beginning
        print(f"\r\033[K", end="")

        # Show step info
        step_info = f"[{response.finish_reason}] "

        # Show current state with masked tokens visible
        current_text = response.text

        # Color coding: green for unmasked, red for masked
        colored_text = ""
        if "[MASK]" in current_text:
            # Replace mask tokens with colored placeholders
            colored_text = current_text.replace("[MASK]", "\033[91m[MASK]\033[0m")
        else:
            colored_text = f"\033[92m{current_text}\033[0m"

        # Truncate text if too long for terminal display
        max_display_length = 100
        if len(current_text) > max_display_length:
            display_text = current_text[:max_display_length] + "..."
            if "[MASK]" in display_text:
                display_text = display_text.replace("[MASK]", "\033[91m[MASK]\033[0m")
            else:
                display_text = f"\033[92m{display_text}\033[0m"
            colored_text = display_text

        print(f"{step_info}{colored_text}", end="", flush=True)

        # Avoid throttling generation during live preview.

    if response is None:
        raise RuntimeError("Generation produced no response.")

    if live_preview:
        print(f"\r\033[K", end="")
        print("✅ Generation complete!")
        print("\n" + "=" * 50)
        print("📊 GENERATION STATS")
        print("=" * 50)
        print(f"Prompt tokens: {response.prompt_tokens}")
        print(f"Generated tokens: {response.generation_tokens}")
        print(f"Generation TPS: {response.generation_tps:.3f}")
        print(f"Peak memory: {response.peak_memory:.3f} GB")
        print("=" * 50)
        print("📝 FINAL RESULT:")
        print("=" * 50)
        print(response.text)
        print("=" * 50)

    return response.text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dream diffusion inference with MLX.")
    parser.add_argument("--model", required=True, help="HF repo id or local model path")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument(
        "--alg",
        choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
        default="origin",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--no-live-preview", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer_config = {"trust_remote_code": args.trust_remote_code}
    model, tokenizer = load(
        args.model,
        tokenizer_config=tokenizer_config,
        trust_remote_code=args.trust_remote_code,
    )
    final_text = diffusion_generate(
        model,
        tokenizer,
        args.prompt,
        live_preview=not args.no_live_preview,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        use_compile=args.use_compile,
        use_chat_template=not args.no_chat_template,
    )
    print("FINAL_TEXT_START")
    print(final_text)
    print("FINAL_TEXT_END")


if __name__ == "__main__":
    main()
