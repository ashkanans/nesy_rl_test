from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_cb import get_arg_parser, train


def _load_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix in {".yml", ".yaml"}:
        import yaml

        payload = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        payload = json.loads(path.read_text())
    else:
        raise ValueError("Config file must use .yaml/.yml or .json")

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Top-level config payload must be a mapping/object.")
    return payload


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args(argv)

    parser = get_arg_parser(add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML/JSON config. CLI flags override config values.",
    )
    config_values = _load_config(known.config)
    if config_values:
        parser.set_defaults(**config_values)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
