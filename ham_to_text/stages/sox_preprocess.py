"""Stage 1: SoX preprocessing — bandpass filter, compand, normalize, resample."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ham_to_text import AudioProcessingError
from ham_to_text.config import PipelineConfig


class SoxPreprocess:
    name: str = "sox_preprocess"

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._sox_path = shutil.which("sox")
        if self._sox_path is None:
            import platform
            system = platform.system()
            install_hint = {
                "Darwin": "brew install sox",
                "Linux": "sudo apt install sox",
                "Windows": "choco install sox",
            }.get(system, "Install SoX and ensure it is on your PATH")
            raise AudioProcessingError(
                f"SoX not found on PATH. Install it: {install_hint}"
            )

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        cfg = self._config
        input_tmp = None
        output_tmp = None
        try:
            input_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            input_tmp.close()
            output_tmp.close()

            sf.write(input_tmp.name, audio, sample_rate)

            cmd = [
                self._sox_path,
                input_tmp.name,
                "--rate", str(cfg.target_sample_rate),
                "--channels", str(cfg.target_channels),
                "--encoding", "signed-integer",
                "--bits", "16",
                output_tmp.name,
                "highpass", str(cfg.sox_highpass_hz),
                "lowpass", str(cfg.sox_lowpass_hz),
            ]

            if cfg.sox_eq_boost_db != 0:
                cmd += [
                    "equalizer", str(cfg.sox_eq_center_hz), "1.5q",
                    str(cfg.sox_eq_boost_db),
                ]

            cmd += [
                "compand", "0.01,0.2",
                "-60,-60,-30,-10,0,-3",
                "-3", "-60", "0.1",
                "norm", str(cfg.sox_norm_level_db),
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode(errors="replace") if e.stderr else "unknown error"
                raise AudioProcessingError(f"SoX failed: {stderr}") from e

            processed, out_sr = sf.read(output_tmp.name, dtype="float32")
            return processed, out_sr

        finally:
            for tmp in (input_tmp, output_tmp):
                if tmp is not None:
                    Path(tmp.name).unlink(missing_ok=True)
