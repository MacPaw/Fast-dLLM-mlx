from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import KVCache


class DualKVCache(KVCache):
    """KV cache that supports both append and in-place block replacement."""

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        position: Optional[int] = None,
        replace: bool = False,
    ):
        if not replace:
            return super().update_and_fetch(keys, values)

        if self.keys is None or self.values is None:
            raise ValueError("DualKVCache must be prefetched before block replacement.")
        if position is None:
            raise ValueError("position is required when replace=True.")

        end = position + keys.shape[2]
        if end > self.offset:
            raise ValueError(
                f"Replacement span [{position}, {end}) exceeds cache size {self.offset}."
            )

        self.keys[..., position:end, :] = keys
        self.values[..., position:end, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


def make_dual_prompt_cache(model: nn.Module) -> List[DualKVCache]:
    return [DualKVCache() for _ in range(len(model.layers))]

