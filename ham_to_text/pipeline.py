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


def _save_debug_audio(
    debug_dir: Path, index: int, name: str, audio: np.ndarray, sample_rate: int
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{index:02d}_{name}.wav"
    sf.write(str(path), audio, sample_rate)
    logger.info("Debug audio saved: %s", path)


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        logger.info("Initializing pipeline...")

        self._sox_stage = SoxPreprocess(config)
        self._denoiser = get_denoiser(config.denoiser, config)
        self._stages = [self._sox_stage, self._denoiser]
        self._transcriber = WhisperTranscriber(config)
        self._debug_dir = Path(config.debug_audio_dir) if config.debug_audio_dir else None

        stage_names = [s.name for s in self._stages]
        logger.info("Pipeline stages: %s -> transcribe", " -> ".join(stage_names))

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_source: str = "audio",
    ) -> TranscriptionResult:
        debug_dir = self._debug_dir
        if debug_dir:
            _save_debug_audio(debug_dir, 0, "input", audio, sample_rate)

        for i, stage in enumerate(self._stages):
            audio, sample_rate = stage.process(audio, sample_rate)
            if debug_dir:
                _save_debug_audio(debug_dir, i + 1, stage.name, audio, sample_rate)

        return self._transcriber.transcribe(audio, sample_rate, audio_source)

    def _vad_segment(self, audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        """Use webrtcvad to find speech segments. Returns list of (start_sample, end_sample)."""
        max_segment_samples = 30 * sample_rate

        try:
            import webrtcvad
        except ImportError:
            logger.warning("webrtcvad not available, processing entire file as one segment")
            return self._split_long_segments([(0, len(audio))], max_segment_samples)

        cfg = self._config
        vad = webrtcvad.Vad(cfg.vad_aggressiveness)

        frame_ms = cfg.vad_frame_ms
        frame_samples = int(sample_rate * frame_ms / 1000)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        # Classify each frame as speech or not
        speech_flags = []
        for i in range(0, len(pcm) - frame_samples + 1, frame_samples):
            frame = pcm[i : i + frame_samples].tobytes()
            try:
                speech_flags.append(vad.is_speech(frame, sample_rate))
            except Exception:
                speech_flags.append(False)

        if not any(speech_flags):
            return [(0, len(audio))]

        # Merge consecutive speech frames into segments, bridging short silence gaps
        min_silence_frames = max(1, cfg.vad_min_silence_ms // frame_ms)
        pad_frames = max(0, cfg.vad_speech_pad_ms // frame_ms)

        segments: list[tuple[int, int]] = []
        in_speech = False
        seg_start = 0
        silence_count = 0

        for i, is_speech in enumerate(speech_flags):
            if is_speech:
                if not in_speech:
                    seg_start = max(0, i - pad_frames)
                    in_speech = True
                silence_count = 0
            elif in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    seg_end = min(len(speech_flags), i - silence_count + 1 + pad_frames)
                    segments.append((
                        seg_start * frame_samples,
                        min(seg_end * frame_samples, len(audio)),
                    ))
                    in_speech = False
                    silence_count = 0

        # Close final segment
        if in_speech:
            seg_end = min(len(speech_flags), len(speech_flags) + pad_frames)
            segments.append((
                seg_start * frame_samples,
                min(seg_end * frame_samples, len(audio)),
            ))

        # Energy-based fallback: recover gaps with significant audio energy
        segments = self._recover_gaps(audio, segments, sample_rate)

        logger.info("VAD found %d segments", len(segments))
        return self._split_long_segments(segments, max_segment_samples)

    def _recover_gaps(
        self,
        audio: np.ndarray,
        segments: list[tuple[int, int]],
        sample_rate: int,
    ) -> list[tuple[int, int]]:
        """Fill VAD gaps that contain significant energy (likely missed speech)."""
        rms_threshold = self._config.vad_energy_threshold
        min_gap_samples = int(0.5 * sample_rate)
        recovered = []

        # Check gap before first segment
        if segments and segments[0][0] > min_gap_samples:
            gap = audio[: segments[0][0]]
            if np.sqrt(np.mean(gap**2)) >= rms_threshold:
                recovered.append((0, segments[0][0]))

        # Check gaps between segments
        for i in range(len(segments) - 1):
            gap_start = segments[i][1]
            gap_end = segments[i + 1][0]
            if gap_end - gap_start < min_gap_samples:
                continue
            gap = audio[gap_start:gap_end]
            if np.sqrt(np.mean(gap**2)) >= rms_threshold:
                recovered.append((gap_start, gap_end))

        # Check gap after last segment
        if segments and len(audio) - segments[-1][1] > min_gap_samples:
            gap = audio[segments[-1][1] :]
            if np.sqrt(np.mean(gap**2)) >= rms_threshold:
                recovered.append((segments[-1][1], len(audio)))

        if recovered:
            logger.info("Recovered %d gap segments via energy fallback", len(recovered))

        merged = sorted(segments + recovered, key=lambda s: s[0])
        return merged

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

        debug_dir = self._debug_dir
        if debug_dir:
            _save_debug_audio(debug_dir, 0, "input", audio, sr)

        # Pass 1: SoX preprocess entire file
        processed, processed_sr = self._sox_stage.process(audio, sr)
        if debug_dir:
            _save_debug_audio(debug_dir, 1, self._sox_stage.name, processed, processed_sr)

        # Pass 2: VAD segmentation
        segments = self._vad_segment(processed, processed_sr)

        # Pass 3: Transcribe each segment, feeding prior context forward
        recent_texts: list[str] = []
        max_context_segments = self._config.whisper_context_segments

        for i, (start, end) in enumerate(segments):
            chunk = processed[start:end]
            is_final = i == len(segments) - 1

            # Run remaining stages (denoiser etc.) on chunk
            stage_audio = chunk
            stage_sr = processed_sr
            for j, stage in enumerate(self._stages[1:], start=2):
                stage_audio, stage_sr = stage.process(stage_audio, stage_sr)
                if debug_dir:
                    _save_debug_audio(
                        debug_dir, j, f"{stage.name}_seg{i:03d}", stage_audio, stage_sr
                    )

            prior_text = " ".join(recent_texts[-max_context_segments:])

            offset_s = start / processed_sr
            result = self._transcriber.transcribe(
                stage_audio, stage_sr,
                audio_source=f"file:{path}",
                prior_text=prior_text,
            )
            result.segment_index = i
            result.offset_s = offset_s
            result.is_final = is_final

            if result.text.strip():
                recent_texts.append(result.text.strip())

            yield result

    def transcribe_file(self, path: str | Path) -> TranscriptionResult:
        path = Path(path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        audio, sr = sf.read(str(path), dtype="float32")
        return self.transcribe_audio(audio, sr, audio_source=f"file:{path}")
