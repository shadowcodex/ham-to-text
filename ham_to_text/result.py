"""TranscriptionResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResult:
    text: str
    segments: list[dict]
    language: str
    duration_s: float
    processing_time_s: float
    real_time_factor: float
    audio_source: str
    avg_log_prob: float
    no_speech_prob: float

    # File mode
    segment_index: Optional[int] = None
    offset_s: Optional[float] = None
    is_final: Optional[bool] = None

    # Stream mode
    timestamp: Optional[str] = None

    def is_likely_valid(self) -> bool:
        return (
            self.no_speech_prob < 0.6
            and self.avg_log_prob > -1.0
            and len(self.text.strip()) > 0
        )

    def to_json_dict(self) -> dict:
        d: dict = {"type": "transcription"}
        d["text"] = self.text
        d["is_valid"] = self.is_likely_valid()
        d["duration_s"] = self.duration_s
        d["processing_time_s"] = self.processing_time_s
        d["real_time_factor"] = self.real_time_factor
        d["avg_log_prob"] = self.avg_log_prob
        d["no_speech_prob"] = self.no_speech_prob
        d["segments"] = self.segments

        if self.segment_index is not None:
            d["segment_index"] = self.segment_index
        if self.offset_s is not None:
            d["offset_s"] = self.offset_s
        if self.is_final is not None:
            d["is_final"] = self.is_final
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp

        return d
