"""CLI entry point for ham-to-text."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

import ham_to_text
from ham_to_text.config import load_config
from ham_to_text.result import TranscriptionResult

logger = logging.getLogger("ham_to_text")


def format_json_line(result: TranscriptionResult) -> str:
    return json.dumps(result.to_json_dict(), ensure_ascii=False)


def format_error_json(error: str, code: str) -> str:
    return json.dumps({"type": "error", "error": error, "code": code})


def _load_dotenv() -> None:
    """Load .env file from current directory if it exists. No dependencies needed."""
    import os
    env_path = Path(".env")
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def _setup_logging(level: str) -> None:
    import os

    log_level = getattr(logging, level.upper(), logging.WARNING)

    # Force configuration even if root logger already has handlers
    root = logging.getLogger()
    root.setLevel(log_level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(log_level)

    # Set our own logger explicitly
    ham_logger = logging.getLogger("ham_to_text")
    ham_logger.setLevel(log_level)

    # Force huggingface_hub download progress bars to show
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    try:
        import huggingface_hub
        huggingface_hub.utils.logging.set_verbosity_info()
        huggingface_hub.utils.logging.enable_progress_bars()
    except (ImportError, AttributeError):
        pass


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


def _status(msg: str, use_json: bool) -> None:
    """Print a status message to stderr (unless in JSON mode, keep stderr clean too)."""
    if not use_json:
        print(msg, file=sys.stderr, flush=True)


def _cmd_file(args: argparse.Namespace) -> int:
    config = _build_config(args)
    use_json = getattr(args, "json", False)

    _status(f"Loading model: {config.whisper_model} (this may download ~1.5GB on first run)...", use_json)
    try:
        from ham_to_text.pipeline import Pipeline
        pipeline = Pipeline(config)
    except ham_to_text.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), "MODEL_LOAD_ERROR"), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return _exit_code_for(e)

    _status("Model loaded.", use_json)

    path = Path(args.path)
    if not path.exists():
        msg = f"File not found: {path}"
        if use_json:
            print(format_error_json(msg, "AUDIO_PROCESSING_ERROR"), flush=True)
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return 1

    _status(f"Processing: {path.name}", use_json)
    try:
        for result in pipeline.transcribe_file_progressive(str(path)):
            if use_json:
                print(format_json_line(result), flush=True)
            else:
                # Human mode: always print text. JSON consumers use is_valid to filter.
                if result.text.strip():
                    print(result.text, flush=True)
    except ham_to_text.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), type(e).__name__.upper()), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    config = _build_config(args)
    use_json = getattr(args, "json", False)

    _status(f"Loading model: {config.whisper_model} (this may download ~1.5GB on first run)...", use_json)
    try:
        from ham_to_text.pipeline import Pipeline
        from ham_to_text.streaming import StreamingSession
        pipeline = Pipeline(config)
    except ham_to_text.HamSTTError as e:
        if use_json:
            print(format_error_json(str(e), "MODEL_LOAD_ERROR"), flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return _exit_code_for(e)

    _status("Model loaded. Starting stream (Ctrl+C to stop)...", use_json)

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
    except ham_to_text.HamSTTError as e:
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
        msg = "sounddevice not installed. Run: pip install ham-to-text[stream]"
        if use_json:
            print(format_error_json(msg, "MISSING_DEPENDENCY"), flush=True)
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return 3

    return 0


def _exit_code_for(error: Exception) -> int:
    if isinstance(error, ham_to_text.ConfigError):
        return 2
    if isinstance(error, ham_to_text.ModelLoadError):
        return 3
    return 1


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        prog="ham-to-text",
        description="Offline speech-to-text for ham radio audio",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {ham_to_text.__version__}")
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
