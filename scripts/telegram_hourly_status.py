from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ENTRYPOINT_RE = re.compile(
    r"\bscripts/(?P<script>"
    r"train|evaluate|run_baselines|sweep|eval_suite|train_dt|eval_dt|fl02_tt_decoding_compare"
    r")\.py\b"
)
FLAG_RE = re.compile(
    r"--(?P<flag>base_run_dir|run_dir|sweep_run_dir|suite_run_dir|output_root|env|seed)\s+"
    r"(?P<value>[^\s;]+)"
)


@dataclass
class ExperimentProc:
    pid: int
    etime: str
    script: str
    env: str | None
    seed: str | None
    out_dir: str | None
    cmd: str


def _run(cmd: list[str], timeout: int = 8) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return int(proc.returncode), proc.stdout, proc.stderr


def _safe_json_load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _safe_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


class TelegramBotClient:
    def __init__(self, token: str):
        self._base = f"https://api.telegram.org/bot{token}"

    def _call(self, method: str, payload: dict) -> dict:
        url = f"{self._base}/{method}"
        body = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Telegram API HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Telegram API network error: {e}") from e

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid Telegram response: {raw}") from e
        if not parsed.get("ok"):
            raise RuntimeError(f"Telegram API error: {parsed}")
        return parsed

    def send_message(self, chat_id: int, text: str) -> None:
        self._call("sendMessage", {"chat_id": str(chat_id), "text": text, "disable_web_page_preview": "true"})

    def get_updates(self, offset: int | None, timeout_sec: int = 0, limit: int = 100) -> dict:
        payload: dict[str, str] = {"timeout": str(timeout_sec), "limit": str(limit)}
        if offset is not None:
            payload["offset"] = str(offset)
        return self._call("getUpdates", payload)


def _to_bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value is not None and value != "":
            return value
    return None


def _to_bot_style_chat_id(entity) -> int:
    # Bot API chat_id style:
    # - basic group chat: -<id>
    # - supergroup/channel: -100<id>
    cls_name = entity.__class__.__name__.lower()
    if "channel" in cls_name:
        return int(f"-100{int(entity.id)}")
    return -int(entity.id)


async def _telethon_find_group_id(
    *,
    group_title: str,
    api_id: int,
    api_hash: str,
    phone: str,
    session_path: str,
) -> int | None:
    try:
        from telethon import TelegramClient
    except Exception as exc:
        raise RuntimeError(
            "Telethon bootstrap requested but Telethon is unavailable. Install it with "
            "`pip install telethon` in the container."
        ) from exc

    client = TelegramClient(session_path, api_id, api_hash)
    await client.start(phone=phone)
    try:
        async for dialog in client.iter_dialogs():
            entity = dialog.entity
            title = getattr(entity, "title", None)
            if title == group_title and dialog.is_group:
                return _to_bot_style_chat_id(entity)
    finally:
        await client.disconnect()
    return None


async def _telethon_create_or_find_group_id(
    *,
    group_title: str,
    api_id: int,
    api_hash: str,
    phone: str,
    session_path: str,
) -> int:
    try:
        from telethon import TelegramClient
        from telethon.tl.functions.messages import CreateChatRequest
    except Exception as exc:
        raise RuntimeError(
            "Telethon bootstrap requested but Telethon is unavailable. Install it with "
            "`pip install telethon` in the container."
        ) from exc

    existing = await _telethon_find_group_id(
        group_title=group_title,
        api_id=api_id,
        api_hash=api_hash,
        phone=phone,
        session_path=session_path,
    )
    if existing is not None:
        return existing

    client = TelegramClient(session_path, api_id, api_hash)
    await client.start(phone=phone)
    try:
        created = await client(CreateChatRequest(users=[], title=group_title))
        entity = created.chats[0] if getattr(created, "chats", None) else None
        if entity is None:
            raise RuntimeError("Telethon group creation returned no chat entity.")
        return _to_bot_style_chat_id(entity)
    finally:
        await client.disconnect()


async def _telethon_send_message(
    *,
    api_id: int,
    api_hash: str,
    phone: str,
    session_path: str,
    entity_id: int,
    text: str,
) -> None:
    try:
        from telethon import TelegramClient
    except Exception as exc:
        raise RuntimeError(
            "Telethon send requested but Telethon is unavailable. Install it with "
            "`pip install telethon` in the container."
        ) from exc

    client = TelegramClient(session_path, api_id, api_hash)
    await client.start(phone=phone)
    try:
        await client.send_message(entity=entity_id, message=text)
    finally:
        await client.disconnect()


def _normalize_out_dir(value: str) -> str | None:
    value = value.strip().strip("'").strip('"')
    if not value:
        return None
    if value.startswith("$"):
        return None
    return value


def _parse_proc_line(line: str) -> ExperimentProc | None:
    line = line.rstrip()
    if not line:
        return None
    m = re.match(r"^\s*(\d+)\s+(\S+)\s+(.*)$", line)
    if not m:
        return None
    pid = int(m.group(1))
    etime = m.group(2)
    cmd = m.group(3)

    sm = ENTRYPOINT_RE.search(cmd)
    if sm is None:
        return None
    script = sm.group("script")

    found: dict[str, str] = {}
    for fm in FLAG_RE.finditer(cmd):
        found[fm.group("flag")] = fm.group("value")
    env = found.get("env")
    seed = found.get("seed")

    out_dir = None
    for key in ["base_run_dir", "run_dir", "sweep_run_dir", "suite_run_dir", "output_root"]:
        if key in found:
            out_dir = _normalize_out_dir(found[key])
            if out_dir:
                break

    return ExperimentProc(
        pid=pid,
        etime=etime,
        script=script,
        env=env,
        seed=seed,
        out_dir=out_dir,
        cmd=cmd,
    )


def get_running_experiments() -> list[ExperimentProc]:
    rc, out, _ = _run(["ps", "-eo", "pid,etime,args", "--no-headers"])
    if rc != 0:
        return []
    rows: list[ExperimentProc] = []
    for line in out.splitlines():
        parsed = _parse_proc_line(line)
        if parsed is not None:
            rows.append(parsed)

    direct = [r for r in rows if re.search(r"(^|\s)python(\d+(\.\d+)?)?(\s|$)", r.cmd) is not None]
    if direct:
        return direct
    return rows


def _read_meminfo() -> tuple[int | None, int | None]:
    total_kb = None
    avail_kb = None
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
    except Exception:
        return None, None
    return total_kb, avail_kb


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    u = 0
    while x >= 1024.0 and u < len(units) - 1:
        x /= 1024.0
        u += 1
    return f"{x:.1f}{units[u]}"


def get_system_snapshot(disk_path: str) -> dict:
    chosen_disk_path = disk_path
    if not os.path.exists(chosen_disk_path):
        chosen_disk_path = os.getcwd()
    if not os.path.exists(chosen_disk_path):
        chosen_disk_path = "/"

    load1, load5, _ = os.getloadavg()
    cpu_count = os.cpu_count() or 0

    mem_total_kb, mem_avail_kb = _read_meminfo()
    mem_used_kb = None
    if mem_total_kb is not None and mem_avail_kb is not None:
        mem_used_kb = mem_total_kb - mem_avail_kb

    usage = shutil.disk_usage(chosen_disk_path)
    return {
        "load1": load1,
        "load5": load5,
        "cpu_count": cpu_count,
        "mem_total_kb": mem_total_kb,
        "mem_used_kb": mem_used_kb,
        "disk_total": int(usage.total),
        "disk_used": int(usage.used),
        "disk_free": int(usage.free),
        "disk_path": chosen_disk_path,
    }


def _is_cmd_available(name: str) -> bool:
    rc, _, _ = _run(["bash", "-lc", f"command -v {name}"])
    return rc == 0


def get_gpu_snapshot() -> list[dict]:
    if not _is_cmd_available("nvidia-smi"):
        return []

    rc, out, _ = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        return []

    gpus: list[dict] = []
    for raw in out.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 7:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        gpus.append(
            {
                "index": idx,
                "name": parts[1],
                "util": parts[2],
                "mem_used": parts[3],
                "mem_total": parts[4],
                "power": parts[5],
                "temp": parts[6],
            }
        )

    # Attach process count/memory per GPU when available.
    uuid_map: dict[str, int] = {}
    rc_uuid, out_uuid, _ = _run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    if rc_uuid == 0:
        for line in out_uuid.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) != 2:
                continue
            try:
                uuid_map[cols[1]] = int(cols[0])
            except ValueError:
                pass

    proc_count: dict[int, int] = {g["index"]: 0 for g in gpus}
    proc_mem: dict[int, int] = {g["index"]: 0 for g in gpus}
    rc_app, out_app, _ = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc_app == 0:
        for line in out_app.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) != 3:
                continue
            gpu_uuid = cols[1]
            if gpu_uuid not in uuid_map:
                continue
            idx = uuid_map[gpu_uuid]
            proc_count[idx] = proc_count.get(idx, 0) + 1
            try:
                proc_mem[idx] = proc_mem.get(idx, 0) + int(float(cols[2]))
            except ValueError:
                pass

    for g in gpus:
        idx = g["index"]
        g["proc_count"] = proc_count.get(idx, 0)
        g["proc_mem"] = proc_mem.get(idx, 0)
    return gpus


def _extract_chat_from_update(update: dict) -> dict | None:
    for key in ["message", "channel_post", "edited_message", "edited_channel_post"]:
        payload = update.get(key)
        if isinstance(payload, dict) and isinstance(payload.get("chat"), dict):
            return payload["chat"]
    for key in ["my_chat_member", "chat_member"]:
        payload = update.get(key)
        if isinstance(payload, dict) and isinstance(payload.get("chat"), dict):
            return payload["chat"]
    return None


def ensure_chat_id(
    client: TelegramBotClient,
    explicit_chat_id: int | None,
    group_title: str | None,
    state: dict,
) -> tuple[int, dict]:
    if explicit_chat_id is not None:
        state["chat_id"] = int(explicit_chat_id)
        return int(explicit_chat_id), state

    saved = state.get("chat_id")
    if isinstance(saved, int):
        return saved, state

    offset = state.get("update_offset")
    if not isinstance(offset, int):
        offset = None

    updates = client.get_updates(offset=offset, timeout_sec=0, limit=100)
    result = updates.get("result", [])
    max_update = offset
    candidate_chat_id = None
    for upd in result:
        if isinstance(upd, dict) and isinstance(upd.get("update_id"), int):
            uid = int(upd["update_id"])
            if max_update is None or uid >= max_update:
                max_update = uid + 1
        chat = _extract_chat_from_update(upd if isinstance(upd, dict) else {})
        if not chat:
            continue
        ctype = str(chat.get("type", ""))
        title = str(chat.get("title", ""))
        cid = chat.get("id")
        if not isinstance(cid, int):
            continue
        if ctype in {"group", "supergroup"}:
            if group_title is None or title == group_title:
                candidate_chat_id = cid

    if max_update is not None:
        state["update_offset"] = int(max_update)
    if candidate_chat_id is not None:
        state["chat_id"] = int(candidate_chat_id)
        return int(candidate_chat_id), state

    if group_title is None:
        raise RuntimeError(
            "Could not auto-detect TELEGRAM_CHAT_ID. Set TELEGRAM_CHAT_ID explicitly."
        )
    raise RuntimeError(
        "Could not find target group in bot updates. "
        f"Create/add a group named '{group_title}', add the bot to it, and send one message."
    )


def _require_telethon_settings(args) -> None:
    if args.telegram_api_id is None or not args.telegram_api_hash or not args.telegram_phone:
        raise RuntimeError(
            "Telethon mode requires TELEGRAM_API_ID/API_ID, TELEGRAM_API_HASH/API_HASH, "
            "and TELEGRAM_PHONE/PHONE."
        )


def ensure_telethon_group_id(args, state: dict) -> tuple[int, dict]:
    if args.chat_id is not None:
        state["chat_id"] = int(args.chat_id)
        return int(args.chat_id), state

    saved = state.get("chat_id")
    if isinstance(saved, int):
        return saved, state

    if not args.group_title:
        raise RuntimeError(
            "TELEGRAM_GROUP_ID/GROUP_ID or --chat-id is missing, and group title was not provided."
        )

    _require_telethon_settings(args)
    found = asyncio.run(
        _telethon_find_group_id(
            group_title=args.group_title,
            api_id=int(args.telegram_api_id),
            api_hash=str(args.telegram_api_hash),
            phone=str(args.telegram_phone),
            session_path=args.telethon_session,
        )
    )
    if found is None and args.create_group_if_missing:
        found = asyncio.run(
            _telethon_create_or_find_group_id(
                group_title=args.group_title,
                api_id=int(args.telegram_api_id),
                api_hash=str(args.telegram_api_hash),
                phone=str(args.telegram_phone),
                session_path=args.telethon_session,
            )
        )
    if found is None:
        raise RuntimeError(
            "Could not find target group in Telethon dialogs. Set TELEGRAM_GROUP_ID/GROUP_ID "
            "or pass --create-group-if-missing with --group-title."
        )

    state["chat_id"] = int(found)
    return int(found), state


def _build_report_text(
    jobs: list[ExperimentProc],
    system: dict,
    gpus: list[dict],
    max_jobs: int,
) -> str:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: list[str] = [f"NESY RL status | {now}"]

    lines.append(f"running_experiments: {len(jobs)}")
    if jobs:
        for j in jobs[:max_jobs]:
            parts = [f"- {j.script}.py", f"pid={j.pid}", f"t={j.etime}"]
            if j.env:
                parts.append(f"env={j.env}")
            if j.seed:
                parts.append(f"seed={j.seed}")
            if j.out_dir:
                parts.append(f"dir={j.out_dir}")
            lines.append(" ".join(parts))
        hidden = len(jobs) - min(len(jobs), max_jobs)
        if hidden > 0:
            lines.append(f"- ... +{hidden} more")

    mem_total_kb = system.get("mem_total_kb")
    mem_used_kb = system.get("mem_used_kb")
    if isinstance(mem_total_kb, int) and isinstance(mem_used_kb, int):
        mem_used_gb = mem_used_kb / (1024 * 1024)
        mem_total_gb = mem_total_kb / (1024 * 1024)
        mem_pct = (mem_used_kb / max(1, mem_total_kb)) * 100.0
        mem_str = f"{mem_used_gb:.1f}/{mem_total_gb:.1f}GB ({mem_pct:.0f}%)"
    else:
        mem_str = "n/a"

    disk_total = int(system["disk_total"])
    disk_used = int(system["disk_used"])
    disk_pct = (disk_used / max(1, disk_total)) * 100.0
    lines.append(
        "system: "
        f"load1={system['load1']:.2f} load5={system['load5']:.2f} "
        f"cpu_cores={system['cpu_count']} ram={mem_str} "
        f"disk={_fmt_bytes(disk_used)}/{_fmt_bytes(disk_total)} ({disk_pct:.0f}%) "
        f"path={system['disk_path']}"
    )

    if gpus:
        lines.append(f"gpus: {len(gpus)}")
        for g in gpus:
            lines.append(
                "- "
                f"gpu{g['index']} {g['name']} util={g['util']}% "
                f"mem={g['mem_used']}/{g['mem_total']}MiB "
                f"pwr={g['power']}W temp={g['temp']}C "
                f"procs={g['proc_count']} proc_mem={g['proc_mem']}MiB"
            )
    else:
        lines.append("gpus: nvidia-smi unavailable or no visible GPUs")

    return "\n".join(lines)


def send_status_once(args) -> str:
    jobs = get_running_experiments()
    system = get_system_snapshot(args.disk_path)
    gpus = get_gpu_snapshot()
    text = _build_report_text(jobs, system, gpus, args.max_jobs)

    if args.dry_run:
        return text

    state_path = Path(args.state_file)
    state = _safe_json_load(state_path)

    if args.telegram_mode == "telethon":
        _require_telethon_settings(args)
        chat_id, state = ensure_telethon_group_id(args, state)
        _safe_json_dump(state_path, state)
        asyncio.run(
            _telethon_send_message(
                api_id=int(args.telegram_api_id),
                api_hash=str(args.telegram_api_hash),
                phone=str(args.telegram_phone),
                session_path=args.telethon_session,
                entity_id=int(chat_id),
                text=text,
            )
        )
        return text

    if args.telegram_mode == "bot":
        if not args.bot_token:
            raise RuntimeError("Bot mode requires TELEGRAM_BOT_TOKEN (or --bot-token).")
        client = TelegramBotClient(args.bot_token)
        chat_id, state = ensure_chat_id(client, args.chat_id, args.group_title, state)
        _safe_json_dump(state_path, state)
        client.send_message(chat_id=chat_id, text=text)
        return text

    raise RuntimeError(f"Unknown telegram mode: {args.telegram_mode}")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Hourly Telegram status reporter for running NESY-RL experiments.",
    )
    p.add_argument(
        "--telegram-mode",
        choices=["telethon", "bot"],
        default=_first_env("TELEGRAM_MODE") or "telethon",
        help="Transport for sending to Telegram. Default matches YoutubeBot style: telethon.",
    )
    p.add_argument("--bot-token", type=str, default=os.getenv("TELEGRAM_BOT_TOKEN"))
    chat_id_env = _first_env("TELEGRAM_GROUP_ID", "GROUP_ID", "TELEGRAM_CHAT_ID")
    p.add_argument("--chat-id", type=int, default=int(chat_id_env) if chat_id_env else None)
    p.add_argument("--group-title", type=str, default=os.getenv("TELEGRAM_GROUP_TITLE"))
    p.add_argument(
        "--create-group-if-missing",
        action=argparse.BooleanOptionalAction,
        default=_to_bool_env(os.getenv("TELEGRAM_CREATE_GROUP_IF_MISSING"), default=True),
        help="In Telethon mode, create group once if not found by --group-title.",
    )
    api_id_env = _first_env("TELEGRAM_API_ID", "API_ID")
    p.add_argument("--telegram-api-id", type=int, default=int(api_id_env) if api_id_env else None)
    p.add_argument("--telegram-api-hash", type=str, default=_first_env("TELEGRAM_API_HASH", "API_HASH"))
    p.add_argument("--telegram-phone", type=str, default=_first_env("TELEGRAM_PHONE", "PHONE"))
    p.add_argument(
        "--telethon-session",
        type=str,
        default=_first_env("TELETHON_SESSION", "TELEGRAM_SESSION") or "user_session",
    )
    p.add_argument(
        "--state-file",
        type=str,
        default=os.getenv("TELEGRAM_STATE_FILE", "runs/_monitor/telegram_status_state.json"),
    )
    p.add_argument("--interval-sec", type=int, default=3600)
    p.add_argument("--max-jobs", type=int, default=8)
    p.add_argument("--disk-path", type=str, default="/workspace/nesy_rl")
    p.add_argument("--once", action="store_true", help="Send one snapshot then exit.")
    p.add_argument("--dry-run", action="store_true", help="Print snapshot text, do not send.")
    p.add_argument(
        "--ensure-group",
        action="store_true",
        help=(
            "Resolve and persist chat id using provided --chat-id or group discovery, "
            "then send a short initialization message."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    if args.ensure_group:
        if args.dry_run:
            print("Dry-run: ensure-group skipped Telegram API calls.")
            return
        state_path = Path(args.state_file)
        state = _safe_json_load(state_path)
        if args.telegram_mode == "telethon":
            _require_telethon_settings(args)
            chat_id, state = ensure_telethon_group_id(args, state)
            _safe_json_dump(state_path, state)
            asyncio.run(
                _telethon_send_message(
                    api_id=int(args.telegram_api_id),
                    api_hash=str(args.telegram_api_hash),
                    phone=str(args.telegram_phone),
                    session_path=args.telethon_session,
                    entity_id=int(chat_id),
                    text="NESY RL monitor initialized.",
                )
            )
            print(f"Initialized monitor chat_id={chat_id} (telethon mode).")
            return

        if args.telegram_mode == "bot":
            if not args.bot_token:
                raise RuntimeError("Bot mode requires TELEGRAM_BOT_TOKEN (or --bot-token).")
            client = TelegramBotClient(args.bot_token)
            chat_id, state = ensure_chat_id(client, args.chat_id, args.group_title, state)
            _safe_json_dump(state_path, state)
            client.send_message(chat_id=chat_id, text="NESY RL monitor initialized.")
            print(f"Initialized monitor chat_id={chat_id} (bot mode).")
            return

        raise RuntimeError(f"Unknown telegram mode: {args.telegram_mode}")
        return

    if args.once:
        text = send_status_once(args)
        if args.dry_run:
            print(text)
        return

    while True:
        try:
            text = send_status_once(args)
            if args.dry_run:
                print(text)
        except KeyboardInterrupt:
            print("Stopped by user.")
            return
        except Exception as exc:
            print(f"[telegram-hourly-status] error: {exc}", file=sys.stderr)
        time.sleep(max(30, int(args.interval_sec)))


if __name__ == "__main__":
    main()
