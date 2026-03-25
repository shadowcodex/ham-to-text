"""Denoiser registry and NoOp denoiser."""

from __future__ import annotations

from typing import Any

import numpy as np

from ham_to_text import ModelLoadError
from ham_to_text.config import PipelineConfig


class NoOpDenoiser:
    name: str = "none"

    def __init__(self, config: PipelineConfig) -> None:
        pass

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        return audio, sample_rate


_REGISTRY: dict[str, type] = {"none": NoOpDenoiser}


def register_denoiser(name: str, cls: type) -> None:
    _REGISTRY[name] = cls


def get_denoiser(name: str, config: PipelineConfig) -> Any:
    if name not in _REGISTRY:
        raise ModelLoadError(
            f"Unknown denoiser: {name}. Available: {list(_REGISTRY.keys())}. "
            f"For DeepFilterNet: pip install ham-to-text[deepfilter]"
        )
    return _REGISTRY[name](config)


def registered_denoisers() -> list[str]:
    return list(_REGISTRY.keys())
