from __future__ import annotations

import argparse
import asyncio
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcRow:
    pid: int
    etime: str
    cmd: str


_ENTRY_RE = re.compile(
    r"python .*scripts/(cb_tt_matrix|cb_dt_matrix|run_baselines|evaluate|train|train_dt|eval_dt)\.py"
)


def _run(cmd: list[str], timeout: int = 8) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    return int(proc.returncode), proc.stdout, proc.stderr


def _read_ps_rows() -> list[ProcRow]:
    rc, out, _ = _run(["ps", "-eo", "pid,etime,args", "--no-headers"])
    if rc != 0:
        return []
    rows: list[ProcRow] = []
    for line in out.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+(.*)$", line.strip())
        if m is None:
            continue
        cmd = m.group(3)
        if _ENTRY_RE.search(cmd) is None:
            continue
        rows.append(ProcRow(pid=int(m.group(1)), etime=m.group(2), cmd=cmd))
    rows.sort(key=lambda r: r.pid)
    return rows


def _get_arg(tokens: list[str], flag: str, default: str | None = None) -> str | None:
    try:
        i = tokens.index(flag)
    except ValueError:
        return default
    if i + 1 >= len(tokens):
        return default
    return tokens[i + 1]


def _get_multi(tokens: list[str], flag: str) -> list[str]:
    try:
        i = tokens.index(flag) + 1
    except ValueError:
        return []
    vals: list[str] = []
    while i < len(tokens) and not tokens[i].startswith("--"):
        vals.append(tokens[i])
        i += 1
    return vals


def _gpu_count() -> int:
    rc, out, _ = _run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], timeout=6
    )
    if rc != 0:
        return 0
    return len([x for x in out.splitlines() if x.strip() != ""])


def _alpha_from_baseline_name(name: str) -> str:
    if name == "vanilla":
        return "0.0"
    if name.startswith("logic_alpha"):
        return name[len("logic_alpha") :]
    return "-"


def _phase_alpha_from_run_baselines(base_run_dir: str | None, epochs: int) -> tuple[str, str]:
    if not base_run_dir:
        return "-", "unknown"
    root = Path(base_run_dir)
    if not root.exists():
        return "-", "unknown"
    candidates = [
        d
        for d in root.iterdir()
        if d.is_dir() and (d.name == "vanilla" or d.name.startswith("logic_alpha"))
    ]
    if not candidates:
        return "-", "setup"

    cur = max(candidates, key=lambda p: p.stat().st_mtime)
    alpha = _alpha_from_baseline_name(cur.name)
    final_ckpt = cur / f"cb_state_{max(0, epochs - 1)}.pt"
    metrics_path = cur / "metrics.json"

    if not final_ckpt.exists():
        return alpha, "training"
    if not metrics_path.exists():
        return alpha, "evaluation"
    return alpha, "finished_baseline"


def _format_cb_lines(rows: list[ProcRow]) -> str:
    gpu_count = _gpu_count()
    out_lines: list[str] = [
        f"gpu_available = {1 if gpu_count > 0 else 0} (gpu_count={gpu_count})"
    ]

    formatted: list[str] = []
    for row in rows:
        try:
            tokens = shlex.split(row.cmd)
        except ValueError:
            continue

        script_tok = next(
            (t for t in tokens if "scripts/" in t and t.endswith(".py")),
            "",
        )
        script = Path(script_tok).name

        env = _get_arg(tokens, "--env", "")
        if script not in {"cb_tt_matrix.py", "cb_dt_matrix.py"} and env != "cb":
            continue

        semantics = (_get_arg(tokens, "--cb_state_semantics", "pre") or "pre").upper()
        spec = _get_arg(tokens, "--spec", "-") or "-"
        seed = _get_arg(tokens, "--seed", "-") or "-"
        mode = _get_arg(tokens, "--decoding_mode", "-") or "-"
        if mode == "-":
            mode = _get_arg(tokens, "--dt_mode", "-") or "-"
        alpha = _get_arg(tokens, "--alpha", "-") or "-"
        phase = "unknown"

        if script in {"cb_tt_matrix.py", "cb_dt_matrix.py"}:
            alphas = _get_multi(tokens, "--alphas")
            alpha = f"sweep({','.join(alphas)})" if alphas else "-"
            phase = "orchestrator"
            spec = "-"
            seed = "-"
            mode = "-"
        elif script == "run_baselines.py":
            base_run_dir = _get_arg(tokens, "--base_run_dir", None)
            epochs_str = _get_arg(tokens, "--epochs", "30") or "30"
            try:
                epochs = int(epochs_str)
            except ValueError:
                epochs = 30
            a2, phase = _phase_alpha_from_run_baselines(base_run_dir, epochs)
            if a2 != "-":
                alpha = a2
            elif _get_multi(tokens, "--alphas"):
                alpha = f"sweep({','.join(_get_multi(tokens, '--alphas'))})"
        elif script == "evaluate.py":
            phase = "evaluation"
            ckpt = _get_arg(tokens, "--checkpoint", "") or ""
            if "/vanilla/" in ckpt:
                alpha = "0.0"
            else:
                m = re.search(r"/logic_alpha([^/]+)/", ckpt)
                if m is not None:
                    alpha = m.group(1)
        elif script == "eval_dt.py":
            phase = "evaluation"
            ckpt = _get_arg(tokens, "--checkpoint", "") or ""
            if "/vanilla/" in ckpt:
                alpha = "0.0"
            else:
                m = re.search(r"/logic_alpha([^/]+)/", ckpt)
                if m is not None:
                    alpha = m.group(1)
        elif script == "train.py":
            phase = "training"
            if alpha == "-" and _get_multi(tokens, "--alphas"):
                alpha = f"sweep({','.join(_get_multi(tokens, '--alphas'))})"
        elif script == "train_dt.py":
            phase = "training"
            la = _get_arg(tokens, "--logic_alpha", None)
            if la is not None:
                alpha = la

        formatted.append(
            f"CB - {semantics} - {spec} - Seed {seed} - Alpha {alpha} ({phase}) "
            f"[mode={mode}, job={script}, pid={row.pid}, t={row.etime}]"
        )

    if not formatted:
        out_lines.append("No active CB experiment process found.")
    else:
        for i, line in enumerate(formatted, 1):
            out_lines.append(f"{i}) {line}")

    return "\n".join(out_lines)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def _send_with_telethon(text: str, args: argparse.Namespace) -> None:
    try:
        from telethon import TelegramClient
    except Exception as exc:
        raise RuntimeError("Telethon is required. Install with `pip install telethon`.") from exc

    client = TelegramClient(args.telethon_session, args.telegram_api_id, args.telegram_api_hash)
    await client.start(phone=args.telegram_phone)
    try:
        await client.send_message(entity=args.chat_id, message=text)
    finally:
        await client.disconnect()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CB-focused hourly Telegram status sender with compact alpha/phase report."
    )
    p.add_argument("--telegram-api-id", type=int, default=None)
    p.add_argument("--telegram-api-hash", type=str, default=None)
    p.add_argument("--telegram-phone", type=str, default=None)
    p.add_argument("--chat-id", type=int, default=None)
    p.add_argument(
        "--telethon-session",
        type=str,
        default=os.getenv("TELETHON_SESSION", "runs/_monitor/user_session"),
    )
    p.add_argument("--interval-sec", type=int, default=3600)
    p.add_argument("--once", action="store_true", help="Send one snapshot then exit.")
    p.add_argument("--dry-run", action="store_true", help="Print snapshot text and exit.")
    args = p.parse_args(argv)

    if args.telegram_api_id is None:
        raw = os.getenv("TELEGRAM_API_ID")
        args.telegram_api_id = int(raw) if raw else None
    if args.telegram_api_hash is None:
        args.telegram_api_hash = os.getenv("TELEGRAM_API_HASH")
    if args.telegram_phone is None:
        args.telegram_phone = os.getenv("TELEGRAM_PHONE")
    if args.chat_id is None:
        raw = os.getenv("TELEGRAM_GROUP_ID") or os.getenv("GROUP_ID") or os.getenv("TELEGRAM_CHAT_ID")
        args.chat_id = int(raw) if raw else None
    return args


async def _loop(args: argparse.Namespace) -> None:
    while True:
        text = _format_cb_lines(_read_ps_rows())
        if args.dry_run:
            print(text)
        else:
            await _send_with_telethon(text=text, args=args)
        if args.once:
            return
        await asyncio.sleep(max(60, int(args.interval_sec)))


def _validate_send_args(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    if args.telegram_api_id is None:
        _require_env("TELEGRAM_API_ID")
    if not args.telegram_api_hash:
        _require_env("TELEGRAM_API_HASH")
    if not args.telegram_phone:
        _require_env("TELEGRAM_PHONE")
    if args.chat_id is None:
        _require_env("TELEGRAM_GROUP_ID")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_send_args(args)
    try:
        asyncio.run(_loop(args))
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as exc:
        print(f"[telegram-cb-hourly-status] error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
