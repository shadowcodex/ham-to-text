"""Noisereduce denoiser stage — spectral gating for narrowband audio."""

from __future__ import annotations

import logging

import numpy as np

from ham_to_text.config import PipelineConfig

logger = logging.getLogger(__name__)

try:
    import noisereduce as nr

    class NoisereduceDenoiser:
        name: str = "noisereduce"

        def __init__(self, config: PipelineConfig) -> None:
            self._config = config
            logger.info("Noisereduce denoiser initialized")

        def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
            reduced = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=self._config.nr_stationary,
                prop_decrease=self._config.nr_prop_decrease,
                n_fft=self._config.nr_n_fft,
                time_constant_s=self._config.nr_time_constant_s,
            )
            return reduced.astype(np.float32), sample_rate

    from ham_to_text.stages.denoise import register_denoiser
    register_denoiser("noisereduce", NoisereduceDenoiser)
    logger.debug("Noisereduce denoiser registered")

except ImportError:
    pass
