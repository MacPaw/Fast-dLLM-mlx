from __future__ import annotations

import argparse
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
import transformers.modeling_rope_utils as rope_utils
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput
import transformers.utils as transformers_utils


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", "origin")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", True)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)


class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        assert inputs is not None
        assert generation_config is not None

        input_ids = inputs
        attention_mask = kwargs.pop("attention_mask", None)
        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self._sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        device = input_ids.device
        mask_token_id = generation_config.mask_token_id
        max_length = generation_config.max_new_tokens + input_ids.shape[1]
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        return_dict_in_generate = generation_config.return_dict_in_generate
        histories = [] if generation_config.output_history else None

        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(
                attention_mask,
                (0, max_length - attention_mask.shape[1]),
                value=1.0,
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        generation_logits_hook_func = lambda i, x, logits: logits
        generation_tokens_hook_func = lambda i, x, logits: x

        for i in range(steps):
            mask_index = x == mask_token_id
            outputs = self(x, attention_mask=attention_mask, tok_idx=tok_idx)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
                _, x0[transfer_index_t_s] = sample_tokens(
                    mask_logits[transfer_index_t_s],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                x[mask_index] = x0.clone()
            else:
                if alg == "maskgit_plus":
                    confidence, x0 = sample_tokens(
                        mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                    )
                elif alg == "topk_margin":
                    confidence, x0 = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        margin_confidence=True,
                    )
                elif alg == "entropy":
                    confidence, x0 = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        neg_entropy=True,
                    )
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")

                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = (
                    int(num_mask_token * (1 - s / t))
                    if i < steps - 1
                    else int(num_mask_token)
                )
                full_confidence = torch.full_like(
                    x, -torch.inf, device=device, dtype=logits.dtype
                )
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        full_confidence = F.softmax(full_confidence / alg_temp, dim=-1)
                        transfer_index = torch.multinomial(
                            full_confidence, num_samples=number_transfer_tokens
                        )
                    x_ = torch.zeros_like(x, device=device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = (
                        torch.arange(x.size(0), device=device)
                        .unsqueeze(1)
                        .expand_as(transfer_index)
                    )
                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]

            x = generation_tokens_hook_func(i, x, logits)
            if histories is not None:
                histories.append(x.clone())

        if return_dict_in_generate:
            return DreamModelOutput(sequences=x, history=histories)
        return x


def maybe_apply_chat_template(tokenizer, prompt: str, disable_chat_template: bool):
    if disable_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return tokenizer(prompt, return_tensors="pt")
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer(prompt, return_tensors="pt")


def load_model_and_tokenizer(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
):
    if not hasattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
        def is_flash_attn_greater_or_equal_2_10() -> bool:
            return False

        transformers_utils.is_flash_attn_greater_or_equal_2_10 = (
            is_flash_attn_greater_or_equal_2_10
        )

    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        def compute_default_rope_parameters(
            config=None,
            device=None,
            seq_len: int | None = None,
            layer_type: str | None = None,
        ):
            del seq_len, layer_type
            if config is None:
                raise ValueError("config is required for default RoPE compatibility")

            base = getattr(config, "rope_theta", 10000.0)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64)
                    .to(device=device, dtype=torch.float)
                    / dim
                )
            )
            return inv_freq, 1.0

        rope_utils.ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    ).to(device).eval()
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    return model, tokenizer


def decode_generation(tokenizer, input_ids: torch.Tensor, sequences: torch.Tensor) -> str:
    generated_ids = sequences[0][len(input_ids[0]) :].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Dream diffusion generation in Torch.")
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--alg", choices=["origin", "maskgit_plus", "topk_margin", "entropy"], default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-chat-template", action="store_true")
    return parser


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def resolve_dtype(device: str, dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model,
        device=device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    inputs = maybe_apply_chat_template(tokenizer, args.prompt, args.disable_chat_template)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    generation_config = DreamGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
        mask_token_id=getattr(model.config, "mask_token_id", None),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_history=False,
    )

    with torch.inference_mode():
        output = model.diffusion_generate(
            inputs["input_ids"],
            generation_config=generation_config,
            attention_mask=inputs.get("attention_mask"),
        )

    print(decode_generation(tokenizer, inputs["input_ids"], output.sequences))


if __name__ == "__main__":
    main()
