import json
import subprocess
import sys

import pytest

from ham_radio_stt.cli import format_json_line, format_error_json


class TestFormatJsonLine:
    def test_transcription_has_type(self):
        from ham_radio_stt.result import TranscriptionResult
        result = TranscriptionResult(
            text="CQ", segments=[], language="en",
            duration_s=1.0, processing_time_s=0.5,
            real_time_factor=0.5, audio_source="file:test.wav",
            avg_log_prob=-0.2, no_speech_prob=0.03,
        )
        line = format_json_line(result)
        parsed = json.loads(line)
        assert parsed["type"] == "transcription"
        assert parsed["text"] == "CQ"
        assert "is_valid" in parsed

    def test_file_mode_fields(self):
        from ham_radio_stt.result import TranscriptionResult
        result = TranscriptionResult(
            text="CQ", segments=[], language="en",
            duration_s=1.0, processing_time_s=0.5,
            real_time_factor=0.5, audio_source="file:test.wav",
            avg_log_prob=-0.2, no_speech_prob=0.03,
            segment_index=0, offset_s=0.0, is_final=True,
        )
        line = format_json_line(result)
        parsed = json.loads(line)
        assert parsed["segment_index"] == 0
        assert parsed["is_final"] is True


class TestFormatErrorJson:
    def test_error_has_type(self):
        line = format_error_json("Device not found", "STREAM_ERROR")
        parsed = json.loads(line)
        assert parsed["type"] == "error"
        assert parsed["error"] == "Device not found"
        assert parsed["code"] == "STREAM_ERROR"


class TestCliEntryPoint:
    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "ham_radio_stt", "--version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_no_args_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "ham_radio_stt"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0 or "usage" in result.stderr.lower() or "usage" in result.stdout.lower()

    def test_file_missing_path_exits_2(self):
        result = subprocess.run(
            [sys.executable, "-m", "ham_radio_stt", "file"],
            capture_output=True, text=True,
        )
        assert result.returncode == 2
