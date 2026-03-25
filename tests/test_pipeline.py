from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ham_to_text.config import PipelineConfig
from ham_to_text.pipeline import Pipeline
from ham_to_text.result import TranscriptionResult


@pytest.fixture
def mock_pipeline():
    with patch("ham_to_text.pipeline.WhisperTranscriber") as MockTranscriber:
        mock_result = TranscriptionResult(
            text="CQ CQ",
            segments=[{"start": 0.0, "end": 1.0, "text": "CQ CQ"}],
            language="en",
            duration_s=1.0,
            processing_time_s=0.5,
            real_time_factor=0.5,
            audio_source="file:test.wav",
            avg_log_prob=-0.2,
            no_speech_prob=0.03,
        )
        MockTranscriber.return_value.transcribe.return_value = mock_result

        with patch("ham_to_text.pipeline.SoxPreprocess") as MockSox:
            MockSox.return_value.process.return_value = (
                np.zeros(16000, dtype=np.float32),
                16000,
            )
            MockSox.return_value.name = "sox_preprocess"
            with patch("ham_to_text.pipeline.get_denoiser") as MockDenoiser:
                MockDenoiser.return_value.process.return_value = (
                    np.zeros(16000, dtype=np.float32),
                    16000,
                )
                MockDenoiser.return_value.name = "none"
                pipeline = Pipeline(PipelineConfig())
                yield pipeline


class TestPipeline:
    def test_transcribe_audio(self, mock_pipeline):
        audio = np.zeros(16000, dtype=np.float32)
        result = mock_pipeline.transcribe_audio(audio, 16000)
        assert isinstance(result, TranscriptionResult)
        assert result.text == "CQ CQ"

    def test_transcribe_file(self, mock_pipeline):
        with patch("soundfile.read", return_value=(np.zeros(16000, dtype=np.float32), 16000)):
            with patch("pathlib.Path.exists", return_value=True):
                result = mock_pipeline.transcribe_file("test.wav")
        assert isinstance(result, TranscriptionResult)

    def test_transcribe_file_not_found(self, mock_pipeline):
        from ham_to_text import AudioProcessingError
        with pytest.raises(AudioProcessingError, match="not found"):
            mock_pipeline.transcribe_file("/nonexistent/file.wav")

    def test_stages_run_in_order(self):
        call_order = []

        class FakeStage1:
            name = "stage1"
            def __init__(self, config): pass
            def process(self, audio, sr):
                call_order.append("stage1")
                return audio, sr

        class FakeStage2:
            name = "stage2"
            def __init__(self, config): pass
            def process(self, audio, sr):
                call_order.append("stage2")
                return audio, sr

        with patch("ham_to_text.pipeline.WhisperTranscriber") as MockTranscriber:
            mock_result = TranscriptionResult(
                text="test", segments=[], language="en",
                duration_s=1.0, processing_time_s=0.5,
                real_time_factor=0.5, audio_source="test",
                avg_log_prob=-0.2, no_speech_prob=0.03,
            )
            MockTranscriber.return_value.transcribe.return_value = mock_result

            with patch("ham_to_text.pipeline.SoxPreprocess", FakeStage1):
                with patch("ham_to_text.pipeline.get_denoiser", return_value=FakeStage2(None)):
                    pipeline = Pipeline(PipelineConfig())
                    pipeline.transcribe_audio(np.zeros(16000, dtype=np.float32), 16000)

        assert call_order == ["stage1", "stage2"]


class TestProgressiveTranscription:
    def test_transcribe_file_progressive_yields_results(self):
        with patch("ham_to_text.pipeline.WhisperTranscriber") as MockTranscriber:
            def make_result(*args, **kwargs):
                return TranscriptionResult(
                    text="CQ", segments=[{"start": 0.0, "end": 1.0, "text": "CQ"}],
                    language="en", duration_s=1.0, processing_time_s=0.5,
                    real_time_factor=0.5, audio_source="file:test.wav",
                    avg_log_prob=-0.2, no_speech_prob=0.03,
                )
            MockTranscriber.return_value.transcribe.side_effect = make_result

            with patch("ham_to_text.pipeline.SoxPreprocess") as MockSox:
                MockSox.return_value.process.return_value = (
                    np.zeros(16000 * 5, dtype=np.float32), 16000,
                )
                MockSox.return_value.name = "sox_preprocess"
                with patch("ham_to_text.pipeline.get_denoiser") as MockDenoiser:
                    MockDenoiser.return_value.process.return_value = (
                        np.zeros(16000 * 2, dtype=np.float32), 16000,
                    )
                    MockDenoiser.return_value.name = "none"
                    with patch("ham_to_text.pipeline.Pipeline._vad_segment") as mock_vad:
                        mock_vad.return_value = [
                            (0, 16000 * 2),
                            (16000 * 3, 16000 * 5),
                        ]
                        with patch("soundfile.read", return_value=(np.zeros(44100 * 5, dtype=np.float32), 44100)):
                            with patch("pathlib.Path.exists", return_value=True):
                                pipeline = Pipeline(PipelineConfig())
                                results = list(pipeline.transcribe_file_progressive("test.wav"))

            assert len(results) == 2
            assert results[0].segment_index == 0
            assert results[0].is_final is False
            assert results[1].segment_index == 1
            assert results[1].is_final is True

    def test_split_long_segments(self):
        segments = [(0, 100)]
        result = Pipeline._split_long_segments(segments, 30)
        assert result == [(0, 30), (30, 60), (60, 90), (90, 100)]

    def test_split_short_segments_unchanged(self):
        segments = [(0, 20), (30, 50)]
        result = Pipeline._split_long_segments(segments, 30)
        assert result == [(0, 20), (30, 50)]
