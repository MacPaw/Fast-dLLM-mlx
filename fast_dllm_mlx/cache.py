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
