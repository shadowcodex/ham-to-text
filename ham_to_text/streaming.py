"""Streaming session — capture audio from device, VAD-gated flush, transcribe."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone

import numpy as np

from ham_to_text import StreamError
from ham_to_text.config import PipelineConfig
from ham_to_text.result import TranscriptionResult

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except ImportError:
    sd = None


class StreamingSession:
    def __init__(
        self,
        pipeline,
        config: PipelineConfig,
        callback,
    ) -> None:
        if sd is None:
            raise StreamError(
                "sounddevice not installed. Run: pip install ham-to-text[stream]"
            )

        self._pipeline = pipeline
        self._config = config
        self._callback = callback
        self._running = False
        self._paused = False
        self._chunk_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        chunk_samples = int(config.stream_sample_rate * config.stream_chunk_duration_s)
        self._chunk_samples = chunk_samples
        self._silence_threshold = 0.01

        self._capture_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None

    def _calibrate(self) -> float:
        """Capture 1s of ambient audio and compute silence threshold."""
        logger.info("Calibrating silence threshold (1 second)...")
        samples = []
        duration = 0.0
        chunk_dur = self._config.stream_chunk_duration_s

        try:
            with sd.InputStream(
                samplerate=self._config.stream_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_samples,
                device=self._config.stream_input_device,
            ) as stream:
                while duration < 1.0:
                    data, _ = stream.read(self._chunk_samples)
                    samples.append(data.flatten())
                    duration += chunk_dur
        except Exception as e:
            raise StreamError(f"Failed to calibrate: {e}") from e

        ambient = np.concatenate(samples)
        rms = float(np.sqrt(np.mean(ambient**2)))
        threshold = 3.0 * rms
        logger.info("Silence threshold calibrated: %.6f (ambient RMS: %.6f)", threshold, rms)
        return threshold

    def __enter__(self) -> StreamingSession:
        self._silence_threshold = self._calibrate()
        self._running = True
        self._stop_event.clear()

        self._process_thread = threading.Thread(
            target=self._processing_loop,
            name="stt-processing",
            daemon=True,
        )
        self._process_thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=10)

    def pause(self) -> None:
        self._paused = True
        self._running = False

    def resume(self) -> None:
        self._paused = False
        self._running = True

    @property
    def is_running(self) -> bool:
        return self._running

    def _processing_loop(self) -> None:
        cfg = self._config
        rolling_buffer: list[np.ndarray] = []
        buffer_duration = 0.0
        silence_duration = 0.0

        try:
            with sd.InputStream(
                samplerate=cfg.stream_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_samples,
                device=cfg.stream_input_device,
            ) as stream:
                while not self._stop_event.is_set():
                    if self._paused:
                        time.sleep(0.05)
                        continue

                    data, overflowed = stream.read(self._chunk_samples)
                    if overflowed:
                        logger.warning("Audio input overflow detected")

                    chunk = data.flatten()
                    rolling_buffer.append(chunk)
                    chunk_dur = len(chunk) / cfg.stream_sample_rate
                    buffer_duration += chunk_dur

                    rms = float(np.sqrt(np.mean(chunk**2)))
                    if rms < self._silence_threshold:
                        silence_duration += chunk_dur
                    else:
                        silence_duration = 0.0

                    should_flush = False
                    if silence_duration >= cfg.stream_silence_timeout_s and buffer_duration > silence_duration:
                        should_flush = True
                    elif buffer_duration >= cfg.stream_buffer_duration_s:
                        should_flush = True

                    if should_flush:
                        audio = np.concatenate(rolling_buffer)
                        rolling_buffer.clear()
                        buffer_duration = 0.0
                        silence_duration = 0.0
                        self._flush(audio, cfg.stream_sample_rate)

                if rolling_buffer:
                    audio = np.concatenate(rolling_buffer)
                    if len(audio) > 0:
                        self._flush(audio, cfg.stream_sample_rate)

        except Exception as e:
            logger.error("Streaming error: %s", e)
            self._running = False

    def _flush(self, audio: np.ndarray, sample_rate: int) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            result = self._pipeline.transcribe_audio(
                audio, sample_rate,
                audio_source=f"stream:{timestamp}",
            )
            result.timestamp = timestamp
            self._callback(result)
        except Exception as e:
            logger.error("Transcription error during flush: %s", e)
