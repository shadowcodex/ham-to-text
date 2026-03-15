"""CLI entry point for ham-radio-stt."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

import ham_radio_stt
from ham_radio_stt.config import load_config
from ham_radio_stt.result import TranscriptionResult

logger = logging.getLogger("ham_radio_stt")


def format_json_line(result: TranscriptionResult) -> str:
    return json.dumps(result.to_json_dict(), ensure_ascii=False)


def format_error_json(error: str, code: str) -> str:
    return json.dumps({"type": "error", "error": error, "code": code})


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _build_config(args: argparse.Namespace):
    cli_overrides: dict = {}
    if hasattr(args, "model") and args.model is not None:
        cli_overrides["whisper_model"] = args.model
    if hasattr(args, "denoiser") and args.denoiser is not None:
        cli_overrides["denoiser"] = args.denoiser
    if hasattr(args, "device") and args.device is not None:
        cli_overrides["stream_input_device"] = args.device

    config_file = getattr(args, "config", None)
    config_path = Path(config_file) if config_file else None

    return load_config(config_file=config_path, cli_overrides=cli_overrides or None)


def _cmd_file(args: argparse.Namespace) -> int:
    config = _build_config(args)
    use_json = getattr(args, "json", False)

    try:
        from ham_radio_stt.pipeline import Pipeline
        pipeline = Pipeline(config)
    except ham_radio_stt.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), "MODEL_LOAD_ERROR"), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return _exit_code_for(e)

    path = Path(args.path)
    if not path.exists():
        msg = f"File not found: {path}"
        if use_json:
            print(format_error_json(msg, "AUDIO_PROCESSING_ERROR"), flush=True)
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return 1

    try:
        result = pipeline.transcribe_file(str(path))
        result.segment_index = 0
        result.offset_s = 0.0
        result.is_final = True

        if use_json:
            print(format_json_line(result), flush=True)
        else:
            print(result.text)

    except ham_radio_stt.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), type(e).__name__.upper()), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    config = _build_config(args)
    use_json = getattr(args, "json", False)

    try:
        from ham_radio_stt.pipeline import Pipeline
        from ham_radio_stt.streaming import StreamingSession
        pipeline = Pipeline(config)
    except ham_radio_stt.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), "MODEL_LOAD_ERROR"), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return _exit_code_for(e)

    def on_result(result: TranscriptionResult) -> None:
        if use_json:
            print(format_json_line(result), flush=True)
        else:
            if result.is_likely_valid():
                print(result.text, flush=True)

    try:
        import time
        with StreamingSession(pipeline, config, on_result) as session:
            signal.signal(signal.SIGINT, lambda *_: session.stop())
            while session.is_running:
                time.sleep(0.1)
    except ham_radio_stt.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), type(e).__name__.upper()), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def _cmd_devices(args: argparse.Namespace) -> int:
    use_json = getattr(args, "json", False)
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if use_json:
            device_list = []
            for i, d in enumerate(devices):
                device_list.append({
                    "index": i,
                    "name": d["name"],
                    "max_input_channels": d["max_input_channels"],
                    "max_output_channels": d["max_output_channels"],
                    "default_samplerate": d["default_samplerate"],
                })
            print(json.dumps(device_list, indent=2))
        else:
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    print(f"  {i}: {d['name']} (inputs: {d['max_input_channels']})")
    except ImportError:
        msg = "sounddevice not installed. Run: pip install ham-radio-stt[stream]"
        if use_json:
            print(format_error_json(msg, "MISSING_DEPENDENCY"), flush=True)
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return 3

    return 0


def _exit_code_for(error: Exception) -> int:
    if isinstance(error, ham_radio_stt.ConfigError):
        return 2
    if isinstance(error, ham_radio_stt.ModelLoadError):
        return 3
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ham-radio-stt",
        description="Offline speech-to-text for ham radio audio",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {ham_radio_stt.__version__}")
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default: WARNING)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    file_parser = subparsers.add_parser("file", help="Transcribe an audio file")
    file_parser.add_argument("path", help="Path to audio file")
    file_parser.add_argument("--json", action="store_true", help="Output as JSONL")
    file_parser.add_argument("--model", help="Whisper model name")
    file_parser.add_argument("--denoiser", help="Denoiser name")
    file_parser.add_argument("--config", help="Path to TOML config file")

    stream_parser = subparsers.add_parser("stream", help="Stream from audio device")
    stream_parser.add_argument("--json", action="store_true", help="Output as JSONL")
    stream_parser.add_argument("--device", type=int, help="Audio device index")
    stream_parser.add_argument("--model", help="Whisper model name")
    stream_parser.add_argument("--denoiser", help="Denoiser name")
    stream_parser.add_argument("--config", help="Path to TOML config file")

    devices_parser = subparsers.add_parser("devices", help="List audio devices")
    devices_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, OSError):
        pass

    handlers = {
        "file": _cmd_file,
        "stream": _cmd_stream,
        "devices": _cmd_devices,
    }

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
