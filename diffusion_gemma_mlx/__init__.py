# Copyright 2026 MacPaw Way Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

__all__ = [
    "DiffusionGemmaGenerationConfig",
    "DiffusionGemmaGenerationResponse",
    "DiffusionGemmaGenerator",
    "Model",
    "ModelArgs",
    "diffusion_generate",
    "diffusion_generate_ids",
    "load",
    "stream_diffusion_generate",
    "stream_diffusion_generate_ids",
]


def __getattr__(name):
    if name in {
        "Model",
        "ModelArgs",
    }:
        from . import model

        return getattr(model, name)
    if name in __all__:
        from . import generate

        return getattr(generate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
