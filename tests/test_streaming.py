import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ham_to_text.config import PipelineConfig
from ham_to_text.streaming import StreamingSession
from ham_to_text.result import TranscriptionResult
from ham_to_text import StreamError


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.transcribe_audio.return_value = TranscriptionResult(
        text="CQ CQ",
        segments=[{"start": 0.0, "end": 1.0, "text": "CQ CQ"}],
        language="en",
        duration_s=1.0,
        processing_time_s=0.5,
        real_time_factor=0.5,
        audio_source="stream:test",
        avg_log_prob=-0.2,
        no_speech_prob=0.03,
    )
    return pipeline


class TestStreamingSession:
    def test_starts_and_stops(self, mock_pipeline):
        config = PipelineConfig()
        results = []

        with patch("ham_to_text.streaming.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value.__enter__ = MagicMock(return_value=mock_stream)
            mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)
            # Mock read to return silence
            mock_stream.read.return_value = (np.zeros((22050, 1), dtype=np.float32), False)

            session = StreamingSession(mock_pipeline, config, results.append)
            with session:
                assert session.is_running
                time.sleep(0.2)
                session.stop()

            assert not session.is_running

    def test_pause_and_resume(self, mock_pipeline):
        config = PipelineConfig()

        with patch("ham_to_text.streaming.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value.__enter__ = MagicMock(return_value=mock_stream)
            mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)
            mock_stream.read.return_value = (np.zeros((22050, 1), dtype=np.float32), False)

            session = StreamingSession(mock_pipeline, config, lambda r: None)
            with session:
                session.pause()
                assert not session.is_running
                session.resume()
                assert session.is_running
                session.stop()

    def test_missing_sounddevice_raises(self, mock_pipeline):
        config = PipelineConfig()
        with patch("ham_to_text.streaming.sd", None):
            with pytest.raises(StreamError):
                StreamingSession(mock_pipeline, config, lambda r: None)
