"""PipelineConfig dataclass and TOML config loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional

from ham_to_text import ConfigError

_SECTION_PREFIX = {
    "whisper": "whisper_",
    "sox": "sox_",
    "streaming": "stream_",
    "deepfilter": "dfn_",
    "vad": "vad_",
}

_SPECIAL_KEYS = {
    ("denoiser", "name"): "denoiser",
    ("whisper", "model"): "whisper_model",
    ("whisper", "device"): "whisper_device",
    ("sox", "norm_level_db"): "sox_norm_level_db",
}


@dataclass
class PipelineConfig:
    denoiser: str = "none"
    whisper_model: str = "distil-large-v3"
    whisper_language: str = "en"
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    whisper_temperature: float = 0.0
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"
    whisper_initial_prompt: str = (
        "Ham radio communication. "
        "CQ de W1AW, 73, QSL, QRM, QRN, QSB, QTH, over, copy, roger, "
        "phonetic alphabet: Alpha Bravo Charlie Delta Echo Foxtrot Golf "
        "Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec "
        "Romeo Sierra Tango Uniform Victor Whiskey X-ray Yankee Zulu."
    )
    sox_highpass_hz: int = 200
    sox_lowpass_hz: int = 3400
    sox_norm_level_db: float = -3.0
    target_sample_rate: int = 16000
    target_channels: int = 1
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500
    vad_speech_pad_ms: int = 200
    stream_chunk_duration_s: float = 0.5
    stream_buffer_duration_s: float = 30.0
    stream_silence_timeout_s: float = 1.5
    stream_sample_rate: int = 44100
    stream_input_device: Optional[int] = None
    dfn_attenuation_limit: float = 100.0
    dfn_post_filter: bool = True


def _field_names() -> set[str]:
    return {f.name for f in fields(PipelineConfig)}


def _toml_key_to_field(section: str, key: str) -> str:
    special = _SPECIAL_KEYS.get((section, key))
    if special:
        return special
    prefix = _SECTION_PREFIX.get(section)
    if prefix:
        return f"{prefix}{key}"
    return key


def load_config_from_toml(path: Path) -> tuple[PipelineConfig, set[str]]:
    """Load config from TOML. Returns (config, set_of_field_names_that_were_set)."""
    if not path.exists():
        return PipelineConfig(), set()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e

    valid_fields = _field_names()
    overrides: dict = {}

    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            field_name = _toml_key_to_field(section, key)
            if field_name not in valid_fields:
                raise ConfigError(
                    f"Unknown config key [{section}] {key} "
                    f"(mapped to '{field_name}') in {path}"
                )
            overrides[field_name] = value

    return replace(PipelineConfig(), **overrides), set(overrides.keys())


def global_config_path() -> Path:
    return Path.home() / ".config" / "hamstt" / "config.toml"


def load_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict] = None,
) -> PipelineConfig:
    config, _ = load_config_from_toml(global_config_path())

    local = Path("hamstt.toml")
    if local.exists():
        local_config, local_fields = load_config_from_toml(local)
        changes = {f: getattr(local_config, f) for f in local_fields}
        config = replace(config, **changes)

    if config_file:
        file_config, file_fields = load_config_from_toml(config_file)
        changes = {f: getattr(file_config, f) for f in file_fields}
        config = replace(config, **changes)

    if cli_overrides:
        valid_fields = _field_names()
        for key in cli_overrides:
            if key not in valid_fields:
                raise ConfigError(f"Unknown CLI config key: {key}")
        config = replace(config, **cli_overrides)

    return config
