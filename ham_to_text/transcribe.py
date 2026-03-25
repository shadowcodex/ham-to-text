"""Whisper transcription wrapper using faster-whisper."""

from __future__ import annotations

import logging
import time

import numpy as np
from faster_whisper import WhisperModel

from ham_to_text import ModelLoadError
from ham_to_text.config import PipelineConfig
from ham_to_text.result import TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        try:
            logger.info("Loading Whisper model: %s", config.whisper_model)
            start = time.monotonic()
            self._model = WhisperModel(
                config.whisper_model,
                device=config.whisper_device,
                compute_type=config.whisper_compute_type,
            )
            elapsed = time.monotonic() - start
            logger.info("Whisper model loaded in %.1fs", elapsed)
        except Exception as e:
            raise ModelLoadError(f"Failed to load Whisper model: {e}") from e

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_source: str,
    ) -> TranscriptionResult:
        cfg = self._config
        start = time.monotonic()

        segments_gen, info = self._model.transcribe(
            audio,
            language=cfg.whisper_language,
            beam_size=cfg.whisper_beam_size,
            best_of=cfg.whisper_best_of,
            temperature=cfg.whisper_temperature,
            initial_prompt=cfg.whisper_initial_prompt,
            vad_filter=cfg.vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=cfg.vad_min_silence_duration_ms,
                speech_pad_ms=cfg.vad_speech_pad_ms,
            ),
            condition_on_previous_text=False,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        # Materialize once — generator is lazy and stateful
        segments = list(segments_gen)

        processing_time = time.monotonic() - start
        duration = info.duration

        text = " ".join(s.text.strip() for s in segments).strip()
        segment_dicts = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments
        ]

        avg_log_prob = (
            sum(s.avg_logprob for s in segments) / len(segments)
            if segments
            else 0.0
        )
        no_speech_prob = (
            sum(s.no_speech_prob for s in segments) / len(segments)
            if segments
            else 1.0
        )

        return TranscriptionResult(
            text=text,
            segments=segment_dicts,
            language=info.language,
            duration_s=duration,
            processing_time_s=processing_time,
            real_time_factor=processing_time / duration if duration > 0 else 0.0,
            audio_source=audio_source,
            avg_log_prob=avg_log_prob,
            no_speech_prob=no_speech_prob,
        )
