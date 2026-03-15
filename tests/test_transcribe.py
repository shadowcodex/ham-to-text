import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.transcribe import WhisperTranscriber
from ham_radio_stt.result import TranscriptionResult
from ham_radio_stt import ModelLoadError


class TestWhisperTranscriber:
    def test_transcribe_returns_result(self):
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 3.0
        mock_segment.text = " CQ CQ this is W1AW"
        mock_segment.avg_logprob = -0.2
        mock_segment.no_speech_prob = 0.03

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 3.0

        with patch("ham_radio_stt.transcribe.WhisperModel") as MockModel:
            instance = MockModel.return_value
            instance.transcribe.return_value = (iter([mock_segment]), mock_info)

            transcriber = WhisperTranscriber(PipelineConfig())
            audio = np.zeros(16000 * 3, dtype=np.float32)
            result = transcriber.transcribe(audio, 16000, audio_source="file:test.wav")

        assert isinstance(result, TranscriptionResult)
        assert "CQ CQ this is W1AW" in result.text
        assert result.language == "en"
        assert result.duration_s == 3.0
        assert result.avg_log_prob == pytest.approx(-0.2)
        assert result.no_speech_prob == pytest.approx(0.03)
        assert result.processing_time_s >= 0
        assert result.audio_source == "file:test.wav"

    def test_empty_segments_returns_empty_text(self):
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 2.0

        with patch("ham_radio_stt.transcribe.WhisperModel") as MockModel:
            instance = MockModel.return_value
            instance.transcribe.return_value = (iter([]), mock_info)

            transcriber = WhisperTranscriber(PipelineConfig())
            audio = np.zeros(16000 * 2, dtype=np.float32)
            result = transcriber.transcribe(audio, 16000, audio_source="file:test.wav")

        assert result.text == ""
        assert result.segments == []

    def test_model_load_failure_raises(self):
        with patch("ham_radio_stt.transcribe.WhisperModel", side_effect=Exception("download failed")):
            with pytest.raises(ModelLoadError, match="download failed"):
                WhisperTranscriber(PipelineConfig())
