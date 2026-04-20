__all__ = [
    "DreamGenerationConfig",
    "DreamGenerator",
    "DreamGenerationResponse",
    "diffusion_generate",
    "stream_diffusion_generate",
    "load",
]


def __getattr__(name):
    if name in __all__:
        from . import generate_diffusion

        return getattr(generate_diffusion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
