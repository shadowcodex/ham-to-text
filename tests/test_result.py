import pytest
from ham_to_text.result import TranscriptionResult


def _make_result(**overrides) -> TranscriptionResult:
    defaults = dict(
        text="CQ CQ this is W1AW",
        segments=[{"start": 0.0, "end": 3.0, "text": "CQ CQ this is W1AW"}],
        language="en",
        duration_s=3.0,
        processing_time_s=1.5,
        real_time_factor=0.5,
        audio_source="file:test.wav",
        avg_log_prob=-0.2,
        no_speech_prob=0.03,
    )
    defaults.update(overrides)
    return TranscriptionResult(**defaults)


class TestIsLikelyValid:
    def test_valid_speech(self):
        result = _make_result()
        assert result.is_likely_valid() is True

    def test_high_no_speech_prob(self):
        result = _make_result(no_speech_prob=0.8)
        assert result.is_likely_valid() is False

    def test_low_confidence(self):
        result = _make_result(avg_log_prob=-1.5)
        assert result.is_likely_valid() is False

    def test_empty_text(self):
        result = _make_result(text="")
        assert result.is_likely_valid() is False

    def test_whitespace_only_text(self):
        result = _make_result(text="   ")
        assert result.is_likely_valid() is False

    def test_boundary_no_speech_prob_exclusive(self):
        result = _make_result(no_speech_prob=0.6)
        assert result.is_likely_valid() is False

    def test_just_below_no_speech_threshold(self):
        result = _make_result(no_speech_prob=0.59)
        assert result.is_likely_valid() is True

    def test_boundary_avg_log_prob_exclusive(self):
        result = _make_result(avg_log_prob=-1.0)
        assert result.is_likely_valid() is False

    def test_just_above_log_prob_threshold(self):
        result = _make_result(avg_log_prob=-0.99)
        assert result.is_likely_valid() is True


class TestToJsonDict:
    def test_file_mode_fields(self):
        result = _make_result(segment_index=0, offset_s=0.0, is_final=True)
        d = result.to_json_dict()
        assert d["type"] == "transcription"
        assert d["segment_index"] == 0
        assert d["offset_s"] == 0.0
        assert d["is_final"] is True
        assert "timestamp" not in d

    def test_stream_mode_fields(self):
        result = _make_result(timestamp="2026-03-14T18:30:01Z")
        d = result.to_json_dict()
        assert d["type"] == "transcription"
        assert d["timestamp"] == "2026-03-14T18:30:01Z"
        assert "segment_index" not in d

    def test_is_valid_included(self):
        result = _make_result()
        d = result.to_json_dict()
        assert d["is_valid"] is True
