"""DeepFilterNet 3 denoiser stage (optional — requires deepfilternet + PyTorch)."""

from __future__ import annotations

import logging
import math

import numpy as np

from ham_to_text import ModelLoadError
from ham_to_text.config import PipelineConfig

logger = logging.getLogger(__name__)

try:
    import torch
    from df.enhance import enhance, init_df
    from scipy.signal import resample_poly

    class DeepFilterDenoiser:
        name: str = "deepfilter"

        def __init__(self, config: PipelineConfig) -> None:
            self._config = config
            try:
                self._model, self._df_state, _ = init_df()
            except Exception as e:
                raise ModelLoadError(f"Failed to load DeepFilterNet: {e}") from e
            logger.info("DeepFilterNet 3 model loaded")

        def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
            if sample_rate != 48000:
                g = math.gcd(sample_rate, 48000)
                up = 48000 // g
                down = sample_rate // g
                audio = resample_poly(audio, up, down).astype(np.float32)

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            enhanced = enhance(
                self._model,
                self._df_state,
                audio_tensor,
                atten_lim_db=self._config.dfn_attenuation_limit,
            )
            enhanced_np = enhanced.squeeze(0).numpy()

            out = resample_poly(enhanced_np, 1, 3).astype(np.float32)
            return out, 16000

    from ham_to_text.stages.denoise import register_denoiser
    register_denoiser("deepfilter", DeepFilterDenoiser)
    logger.debug("DeepFilterNet denoiser registered")

except ImportError:
    pass
