"""Ham Radio STT — Offline speech-to-text for ham radio audio."""

__version__ = "0.1.0"


class HamSTTError(Exception):
    """Base exception for ham-radio-stt."""


class AudioProcessingError(HamSTTError):
    """SoX failed, bad audio format, or corrupt file."""


class ModelLoadError(HamSTTError):
    """Model download failed, disk full, or missing dependency."""


class StreamError(HamSTTError):
    """Audio device not found or disconnected."""


class ConfigError(HamSTTError):
    """Bad TOML config or invalid parameter values."""


from ham_radio_stt.config import PipelineConfig
from ham_radio_stt.result import TranscriptionResult

__all__ = [
    "__version__",
    "PipelineConfig",
    "TranscriptionResult",
    "HamSTTError",
    "AudioProcessingError",
    "ModelLoadError",
    "StreamError",
    "ConfigError",
]
