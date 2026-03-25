"""Pipeline stage protocol and stage exports."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PipelineStage(Protocol):
    name: str

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Process audio, return (processed_audio, sample_rate)."""
        ...


# Auto-discover optional denoiser plugins so they register themselves.
for _mod in ("deepfilter", "noisereduce_stage"):
    try:
        importlib.import_module(f"{__name__}.{_mod}")
    except ImportError:
        pass
