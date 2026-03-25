import pytest
from pathlib import Path

from ham_to_text.result import TranscriptionResult


@pytest.mark.real_audio
@pytest.mark.slow
@pytest.mark.requires_sox
class TestRealAudio:
    def test_transcription_produces_result(self, real_audio_files):
        from ham_to_text.pipeline import Pipeline
        from ham_to_text.config import PipelineConfig

        config = PipelineConfig(whisper_model="tiny")
        pipeline = Pipeline(config)

        for audio_path in real_audio_files:
            result = pipeline.transcribe_file(str(audio_path))
            assert isinstance(result, TranscriptionResult)
            assert result.duration_s > 0
            assert result.processing_time_s > 0

            print(f"\n--- {audio_path.name} ---")
            print(f"Text: {result.text}")
            print(f"Duration: {result.duration_s:.1f}s")
            print(f"RTF: {result.real_time_factor:.2f}x")
            print(f"Confidence: {result.avg_log_prob:.2f}")
            print(f"No-speech prob: {result.no_speech_prob:.2f}")
            print(f"Valid: {result.is_likely_valid()}")
