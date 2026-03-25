"""Pipeline — composes stages and runs them in sequence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf

from ham_to_text import AudioProcessingError
from ham_to_text.config import PipelineConfig
from ham_to_text.result import TranscriptionResult
from ham_to_text.stages.sox_preprocess import SoxPreprocess
from ham_to_text.stages.denoise import get_denoiser
from ham_to_text.transcribe import WhisperTranscriber

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

    def _vad_segment(self, audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        """Use faster-whisper's VAD to find speech segments. Returns list of (start_sample, end_sample)."""
        max_segment_samples = 30 * sample_rate  # 30s max per spec

        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except ImportError:
            # Fallback: split into max_segment_samples chunks
            return self._split_long_segments([(0, len(audio))], max_segment_samples)

        vad_options = VadOptions(
            min_silence_duration_ms=self._config.vad_min_silence_duration_ms,
            speech_pad_ms=self._config.vad_speech_pad_ms,
        )
        timestamps = get_speech_timestamps(audio, vad_options)

        if not timestamps:
            return [(0, len(audio))]

        segments = [(int(ts["start"]), int(ts["end"])) for ts in timestamps]

        # Enforce 30s max segment boundary
        return self._split_long_segments(segments, max_segment_samples)

    @staticmethod
    def _split_long_segments(
        segments: list[tuple[int, int]], max_samples: int
    ) -> list[tuple[int, int]]:
        """Split any segment exceeding max_samples into chunks."""
        result = []
        for start, end in segments:
            while end - start > max_samples:
                result.append((start, start + max_samples))
                start += max_samples
            result.append((start, end))
        return result

    def transcribe_file_progressive(
        self,
        path: str | Path,
    ) -> Generator[TranscriptionResult, None, None]:
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        audio, sr = sf.read(str(path), dtype="float32")

        # Pass 1: SoX preprocess entire file
        processed, processed_sr = self._sox_stage.process(audio, sr)

        # Pass 2: VAD segmentation
        segments = self._vad_segment(processed, processed_sr)

        # Pass 3: Transcribe each segment
        for i, (start, end) in enumerate(segments):
            chunk = processed[start:end]
            is_final = i == len(segments) - 1

            # Run remaining stages (denoiser etc.) on chunk
            stage_audio = chunk
            stage_sr = processed_sr
            for stage in self._stages[1:]:
                stage_audio, stage_sr = stage.process(stage_audio, stage_sr)

            offset_s = start / processed_sr
            result = self._transcriber.transcribe(
                stage_audio, stage_sr,
                audio_source=f"file:{path}",
            )
            result.segment_index = i
            result.offset_s = offset_s
            result.is_final = is_final
            yield result

    def transcribe_file(self, path: str | Path) -> TranscriptionResult:
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        audio, sr = sf.read(str(path), dtype="float32")
        return self.transcribe_audio(audio, sr, audio_source=f"file:{path}")
