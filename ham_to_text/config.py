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
    "noisereduce": "nr_",
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
    whisper_context_segments: int = 5
    whisper_initial_prompt: str = (
        "CQ CQ CQ, this is AAR 3 Oscar Foxtrot calling any Army MARS station, over. "
        "AAR 3 Oscar Foxtrot, this is AFA 9 Tango Lima, good evening, you're five by nine, over. "
        "Roger, good copy. I have a routine precedence MARS message, 12 groups, over. "
        "Ready to copy, send your message, over. "
        "Line 1, routine. Line 6, from commanding officer Fort Liberty. "
        "Line 7, to director Army Emergency Management. "
        "Break. Text reads: request status update on DSCA exercise participants. "
        "All counties report SITREP by 1800 Zulu, date-time group 1-5-0-9-0-0 Zulu March. "
        "I authenticate Bravo Foxtrot. Read back, over. "
        "I read back. Routine from Fort Liberty to director Army Emergency Management. "
        "Request status update DSCA exercise. SITREP by 1800 Zulu. Authenticate Bravo Foxtrot. "
        "That is correct. Nothing further, 73, AAR 3 Oscar Foxtrot clear. "
        "All stations, this is net control. COMEX is complete, ENDEX ENDEX ENDEX. "
        "Check out in order. AFA 9 Tango Lima, QRT, 73. "
        # Vocabulary: MARS branches and callsign prefixes
        "Air Force MARS, Army MARS, AFMARS, AMARS, MARSRADIO, BELL RINGER, "
        "AFA, AFF, AFN, AAR, AAT, AEM, REACH. "
        # Vocabulary: prowords and procedure words (ACP-125)
        "Over, out, roger, wilco, copy, negative, affirmative, acknowledge, confirm, "
        "correction, disregard, break, break break, this is, go ahead, standby, "
        "wait one, wait out, say again, I say again, read back, I read back, "
        "how do you read, nothing heard, words twice, I spell, figures, "
        "more to follow, flash, immediate, priority, routine, "
        "silence, silence lifted, execute, no play, COMEX, ENDEX. "
        # Vocabulary: phone patch and net operations
        "Phone patch, morale call, official patch, terminate the patch, "
        "unsecured line, DSN, PBX, SELCAL, propagation, "
        "directed net, free net, net control station, check in, check out, clear the net. "
        # Vocabulary: military ranks
        "Private, Corporal, Specialist, "
        "Sergeant, Staff Sergeant, Sergeant First Class, Master Sergeant, "
        "First Sergeant, Sergeant Major, Command Sergeant Major, "
        "Airman, Airman First Class, Senior Airman, Technical Sergeant, "
        "Chief Master Sergeant, Senior Master Sergeant, "
        "Second Lieutenant, First Lieutenant, Captain, Major, "
        "Lieutenant Colonel, Colonel, Brigadier General, Major General, "
        "Lieutenant General, General. "
        # Vocabulary: logistics and equipment
        "K-loader, 463L pallet, forklift, payloader, payload, "
        "hospitality kit, rolling stock, bulk cargo, load plan, "
        "C-5, C-17, C-130, C-130J, KC-135, KC-10, KC-46, "
        "sortie, tail number, chalk, mission number, ETA, "
        "base ops, base operations, flight line, POL. "
        # Vocabulary: radio and comms
        "HF, VHF, UHF, SSB, USB, LSB, AM, FM, ALE, COMSEC, "
        "QRM, QRN, QSB, QSL, QSO, QSY, QTH, QRZ, QRT, QRX, CQ, de, 73. "
        # Vocabulary: message format
        "Date-time group, DTG, Zulu, UNCLAS, SITREP, BT. "
        # Vocabulary: signal reports
        "Five by nine, five by five, loud and clear, Lima Charlie, "
        "weak but readable, unreadable. "
        # Vocabulary: phonetic alphabet
        "Alpha, Bravo, Charlie, Delta, Echo, Foxtrot, Golf, "
        "Hotel, India, Juliet, Kilo, Lima, Mike, November, Oscar, Papa, Quebec, "
        "Romeo, Sierra, Tango, Uniform, Victor, Whiskey, X-ray, Yankee, Zulu. "
        # Vocabulary: phonetic numbers
        "Zero, Wun, Too, Tree, Fower, Fife, Six, Seven, Ait, Niner."
    )
    sox_highpass_hz: int = 200
    sox_lowpass_hz: int = 3400
    sox_eq_center_hz: int = 1800
    sox_eq_boost_db: float = 6.0
    sox_norm_level_db: float = -3.0
    target_sample_rate: int = 16000
    target_channels: int = 1
    vad_filter: bool = True
    vad_aggressiveness: int = 0
    vad_frame_ms: int = 30
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 300
    vad_energy_threshold: float = 0.02
    stream_chunk_duration_s: float = 0.5
    stream_buffer_duration_s: float = 30.0
    stream_silence_timeout_s: float = 1.5
    stream_sample_rate: int = 44100
    stream_input_device: Optional[int] = None
    dfn_attenuation_limit: float = 100.0
    dfn_post_filter: bool = True
    nr_stationary: bool = False
    nr_prop_decrease: float = 0.75
    nr_n_fft: int = 512
    nr_time_constant_s: float = 2.0
    debug_audio_dir: Optional[str] = None


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
