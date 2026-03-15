import pytest
from pathlib import Path
from ham_radio_stt.config import PipelineConfig, load_config_from_toml


class TestPipelineConfigDefaults:
    def test_default_model(self):
        config = PipelineConfig()
        assert config.whisper_model == "distil-large-v3"

    def test_default_denoiser(self):
        config = PipelineConfig()
        assert config.denoiser == "none"

    def test_default_sample_rate(self):
        config = PipelineConfig()
        assert config.target_sample_rate == 16000

    def test_default_silence_timeout(self):
        config = PipelineConfig()
        assert config.stream_silence_timeout_s == 1.5


class TestLoadConfigFromToml:
    def test_load_whisper_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[whisper]\nmodel = "small"\nbeam_size = 3\n')
        config, set_fields = load_config_from_toml(toml_file)
        assert config.whisper_model == "small"
        assert config.whisper_beam_size == 3
        assert set_fields == {"whisper_model", "whisper_beam_size"}
        assert config.whisper_language == "en"

    def test_load_denoiser_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[denoiser]\nname = "deepfilter"\n')
        config, set_fields = load_config_from_toml(toml_file)
        assert config.denoiser == "deepfilter"
        assert "denoiser" in set_fields

    def test_load_sox_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[sox]\nhighpass_hz = 300\n")
        config, _ = load_config_from_toml(toml_file)
        assert config.sox_highpass_hz == 300
        assert config.sox_lowpass_hz == 3400

    def test_load_streaming_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[streaming]\nsilence_timeout_s = 3.0\n")
        config, _ = load_config_from_toml(toml_file)
        assert config.stream_silence_timeout_s == 3.0

    def test_load_deepfilter_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[deepfilter]\nattenuation_limit = 80.0\npost_filter = false\n")
        config, _ = load_config_from_toml(toml_file)
        assert config.dfn_attenuation_limit == 80.0
        assert config.dfn_post_filter is False

    def test_load_vad_section(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[vad]\nmin_silence_duration_ms = 300\n")
        config, _ = load_config_from_toml(toml_file)
        assert config.vad_min_silence_duration_ms == 300

    def test_nonexistent_file_returns_defaults(self):
        config, set_fields = load_config_from_toml(Path("/nonexistent/path.toml"))
        assert config == PipelineConfig()
        assert set_fields == set()

    def test_invalid_toml_raises_config_error(self, tmp_path):
        from ham_radio_stt import ConfigError
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text("this is not valid toml [[[")
        with pytest.raises(ConfigError):
            load_config_from_toml(toml_file)

    def test_unknown_key_raises_config_error(self, tmp_path):
        from ham_radio_stt import ConfigError
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[whisper]\nnonexistent_key = 42\n")
        with pytest.raises(ConfigError):
            load_config_from_toml(toml_file)

    def test_partial_toml_does_not_clobber_earlier_layers(self, tmp_path):
        from dataclasses import replace
        global_toml = tmp_path / "global.toml"
        global_toml.write_text('[whisper]\nmodel = "small"\n')
        local_toml = tmp_path / "local.toml"
        local_toml.write_text("[sox]\nhighpass_hz = 300\n")

        global_config, _ = load_config_from_toml(global_toml)
        local_config, local_fields = load_config_from_toml(local_toml)
        changes = {f: getattr(local_config, f) for f in local_fields}
        merged = replace(global_config, **changes)

        assert merged.whisper_model == "small"
        assert merged.sox_highpass_hz == 300
