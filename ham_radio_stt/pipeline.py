"""Pipeline — composes stages and runs them in sequence."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from ham_radio_stt import AudioProcessingError
from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.result import TranscriptionResult
from ham_radio_stt.stages.sox_preprocess import SoxPreprocess
from ham_radio_stt.stages.denoise import get_denoiser
from ham_radio_stt.transcribe import WhisperTranscriber

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        logger.info("Initializing pipeline...")

        self._sox_stage = SoxPreprocess(config)
        self._denoiser = get_denoiser(config.denoiser, config)
        self._stages = [self._sox_stage, self._denoiser]
        self._transcriber = WhisperTranscriber(config)

        stage_names = [s.name for s in self._stages]
        logger.info("Pipeline stages: %s -> transcribe", " -> ".join(stage_names))

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_source: str = "audio",
    ) -> TranscriptionResult:
        for stage in self._stages:
            audio, sample_rate = stage.process(audio, sample_rate)

        return self._transcriber.transcribe(audio, sample_rate, audio_source)

    def transcribe_file(self, path: str | Path) -> TranscriptionResult:
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        audio, sr = sf.read(str(path), dtype="float32")
        return self.transcribe_audio(audio, sr, audio_source=f"file:{path}")
