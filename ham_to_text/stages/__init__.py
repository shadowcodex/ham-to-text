"""Pipeline stage protocol and stage exports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PipelineStage(Protocol):
    name: str

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Process audio, return (processed_audio, sample_rate)."""
        ...
