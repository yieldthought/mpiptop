#!/usr/bin/env python3
"""mpiptop: visualize MPI python stacks across hosts using py-spy."""

from __future__ import annotations

import argparse
import colorsys
import dataclasses
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import termios
import time
import textwrap
import tty
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclasses.dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    args: str


@dataclasses.dataclass(frozen=True)
class RankInfo:
    rank: int
    host: str


@dataclasses.dataclass(frozen=True)
class ProgramSelector:
    module: Optional[str]
    script: Optional[str]
    display: str


@dataclasses.dataclass(frozen=True)
class State:
    prte_pid: int
    rankfile: str
    ranks: List[RankInfo]
    selector: ProgramSelector


@dataclasses.dataclass(frozen=True)
class RankProcess:
    pid: int
    cmdline: str
    rss_kb: Optional[int]
    python_exe: Optional[str]
    env: Dict[str, str]


@dataclasses.dataclass(frozen=True)
class ThreadBlock:
    header: str
    stack: List[str]


@dataclasses.dataclass(frozen=True)
class ParsedPySpy:
    details: List[str]
    threads: List[ThreadBlock]


@dataclasses.dataclass(frozen=True)
class RankSnapshot:
    output: Optional[str]
    error: Optional[str]
    stack_lines: List[str]
    details: List[str]


@dataclasses.dataclass
class SessionEvent:
    timestamp: float
    ranks: Dict[int, Dict[str, object]]


@dataclasses.dataclass
class TimelineLevel:
    start: int
    end: int
    selected: int = 0
    buckets: List[Tuple[int, int]] = dataclasses.field(default_factory=list)


PUNCT_STYLE = "grey62"
BORDER_STYLE = "grey62"
KEY_STYLE = "#7ad7ff"
HEADER_HEIGHT = 3
SESSION_VERSION = 1
SESSION_LOG_FILE = "session.jsonl"
SESSION_METADATA_FILE = "metadata.json"
SESSION_EVENTS_FILE = "events.jsonl"
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
HEARTBEAT_INTERVAL = 60
DIVERGENCE_THRESHOLD = 0.5
DIVERGENCE_INTERVAL = 60
ENV_KEYS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYTHONHOME",
    "HOME",
)


REMOTE_FINDER_SCRIPT = r"""
import json
import os

TARGET = os.environ.get("MPIPTOP_TARGET", "")
MODULE = os.environ.get("MPIPTOP_MODULE", "")
ENV_KEYS = [
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYTHONHOME",
    "HOME",
]


def read_cmdline(pid):
    with open(f"/proc/{pid}/cmdline", "rb") as f:
        data = f.read().split(b"\0")
    return [x.decode(errors="ignore") for x in data if x]


def read_env(pid):
    with open(f"/proc/{pid}/environ", "rb") as f:
        data = f.read().split(b"\0")
    env = {}
    for item in data:
        if b"=" in item:
            k, v = item.split(b"=", 1)
            env[k.decode(errors="ignore")] = v.decode(errors="ignore")
    return env


def select_env_subset(env):
    return {key: env[key] for key in ENV_KEYS if key in env}


def read_rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except Exception:
        return None
    return None


def read_exe(pid):
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except Exception:
        return ""


def matches(cmd):
    if not cmd:
        return False
    exe = os.path.basename(cmd[0])
    if "python" not in exe:
        return False
    if MODULE:
        try:
            idx = cmd.index("-m")
        except ValueError:
            return False
        return idx + 1 < len(cmd) and cmd[idx + 1] == MODULE
    if TARGET:
        for arg in cmd:
            if arg == TARGET or arg.endswith("/" + TARGET) or os.path.basename(arg) == os.path.basename(TARGET):
                return True
        return False
    return True


results = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        cmd = read_cmdline(pid)
        if not matches(cmd):
            continue
        env = read_env(pid)
        rank = env.get("OMPI_COMM_WORLD_RANK") or env.get("PMIX_RANK") or env.get("PMI_RANK")
        if rank is None:
            continue
        results.append(
            [
                int(rank),
                int(pid),
                " ".join(cmd),
                read_rss_kb(pid),
                read_exe(pid),
                select_env_subset(env),
            ]
        )
    except Exception:
        continue

print(json.dumps(results))
"""


def iso_timestamp(value: Optional[float] = None) -> str:
    ts = time.time() if value is None else value
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def default_session_path() -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.abspath(f"mpiptop-session-{stamp}.jsonl")


def normalize_session_path(path: str) -> Tuple[str, str]:
    if path.endswith(".jsonl") or (os.path.exists(path) and os.path.isfile(path)):
        base_dir = os.path.dirname(path) or "."
        return base_dir, path
    return path, os.path.join(path, SESSION_LOG_FILE)


def ensure_session_path(path: str) -> Tuple[str, str]:
    base_dir, log_path = normalize_session_path(path)
    if os.path.exists(path):
        if os.path.isdir(path):
            if os.listdir(path):
                if os.path.exists(log_path) or os.path.exists(os.path.join(path, SESSION_METADATA_FILE)):
                    return base_dir, log_path
                raise SystemExit(f"record path exists and is not empty: {path}")
        elif os.path.isfile(path):
            return base_dir, log_path
        else:
            raise SystemExit(f"record path exists and is not a file or directory: {path}")
    else:
        if log_path.endswith(".jsonl"):
            os.makedirs(base_dir, exist_ok=True)
        else:
            os.makedirs(base_dir, exist_ok=True)
    return base_dir, log_path


def write_session_metadata(log_path: str, state: State, refresh: int, pythonpath: str) -> None:
    payload = {
        "version": SESSION_VERSION,
        "created_at": iso_timestamp(),
        "refresh": refresh,
        "rankfile": state.rankfile,
        "prte_pid": state.prte_pid,
        "selector": dataclasses.asdict(state.selector),
        "ranks": [dataclasses.asdict(rank) for rank in state.ranks],
        "pythonpath": pythonpath,
        "record_on_change": True,
    }
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        return
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "metadata", "data": payload}) + "\n")


def load_session_metadata(path: str) -> Dict[str, object]:
    base_dir, log_path = normalize_session_path(path)
    metadata_path = os.path.join(base_dir, SESSION_METADATA_FILE)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if not os.path.exists(log_path):
        raise SystemExit(f"metadata not found in {path}")
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("type") == "metadata":
                payload = data.get("data")
                if isinstance(payload, dict):
                    return payload
            if isinstance(data, dict) and "version" in data and "ranks" in data:
                return data
    raise SystemExit(f"metadata not found in {log_path}")


def read_last_event(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        pos = handle.tell()
        if pos == 0:
            return None
        chunk = b""
        while pos > 0:
            step = min(4096, pos)
            pos -= step
            handle.seek(pos)
            chunk = handle.read(step) + chunk
            if b"\n" in chunk:
                break
        lines = [line for line in chunk.splitlines() if line.strip()]
        while lines:
            raw = lines.pop().decode("utf-8", errors="ignore")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("type") == "metadata":
                continue
            if isinstance(data, dict) and data.get("type") == "event":
                payload = data.get("data")
                if isinstance(payload, dict):
                    return payload
            return data
        return None


def load_session_events(path: str) -> List[SessionEvent]:
    base_dir, log_path = normalize_session_path(path)
    events_path = os.path.join(base_dir, SESSION_EVENTS_FILE)
    if not os.path.exists(events_path) and not os.path.exists(log_path):
        raise SystemExit(f"events not found in {path}")
    path_to_read = events_path if os.path.exists(events_path) else log_path
    events: List[SessionEvent] = []
    with open(path_to_read, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("type") == "metadata":
                continue
            if isinstance(data, dict) and data.get("type") == "event":
                data = data.get("data", {})
            if not isinstance(data, dict):
                continue
            timestamp = float(data.get("t", 0.0))
            ranks_raw = data.get("ranks", {})
            ranks: Dict[int, Dict[str, object]] = {}
            for key, value in ranks_raw.items():
                try:
                    rank_id = int(key)
                except (TypeError, ValueError):
                    continue
                ranks[rank_id] = value
            events.append(SessionEvent(timestamp=timestamp, ranks=ranks))
    return events


def signature_from_snapshot(snapshot: Optional[RankSnapshot]) -> str:
    if snapshot is None:
        return "missing"
    if snapshot.error:
        return f"error:{snapshot.error}"
    if snapshot.output is None:
        return "missing"
    digest = hashlib.sha1(snapshot.output.encode("utf-8", errors="ignore")).hexdigest()
    return digest


def snapshot_signature(ranks: List[RankInfo], snapshots: Dict[int, RankSnapshot]) -> Dict[int, str]:
    signature: Dict[int, str] = {}
    for info in ranks:
        signature[info.rank] = signature_from_snapshot(snapshots.get(info.rank))
    return signature


def signature_from_event(event: Dict[str, object]) -> Optional[Dict[int, str]]:
    ranks = event.get("ranks", {})
    if not isinstance(ranks, dict):
        return None
    signature: Dict[int, str] = {}
    for key, payload in ranks.items():
        try:
            rank_id = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            signature[rank_id] = "missing"
            continue
        if payload.get("error"):
            signature[rank_id] = f"error:{payload.get('error')}"
        elif payload.get("py_spy"):
            digest = hashlib.sha1(
                str(payload.get("py_spy")).encode("utf-8", errors="ignore")
            ).hexdigest()
            signature[rank_id] = digest
        else:
            signature[rank_id] = "missing"
    return signature


class RecordSession:
    def __init__(self, path: str, state: State, refresh: int, pythonpath: str):
        self.base_dir, self.log_path = ensure_session_path(path)
        write_session_metadata(self.log_path, state, refresh, pythonpath)
        self.handle = open(self.log_path, "a", encoding="utf-8")
        self.event_count = 0
        self.last_signature: Optional[Dict[int, str]] = None
        last_event = read_last_event(self.log_path)
        if last_event:
            self.last_signature = signature_from_event(last_event)
            self.event_count = self._count_events()

    def _count_events(self) -> int:
        if not os.path.exists(self.log_path):
            return 0
        count = 0
        with open(self.log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("type") == "metadata":
                    continue
                count += 1
        return count

    def record_if_changed(
        self,
        state: State,
        rank_to_proc: Dict[int, RankProcess],
        snapshots: Dict[int, RankSnapshot],
    ) -> bool:
        signature = snapshot_signature(state.ranks, snapshots)
        if self.last_signature is not None and signature == self.last_signature:
            return False
        payload: Dict[str, object] = {"t": time.time(), "ranks": {}}
        ranks_payload: Dict[str, object] = {}
        for info in state.ranks:
            rank = info.rank
            proc = rank_to_proc.get(rank)
            snapshot = snapshots.get(rank)
            entry: Dict[str, object] = {"host": info.host}
            if proc is not None:
                entry["pid"] = proc.pid
                entry["cmdline"] = proc.cmdline
                entry["rss_kb"] = proc.rss_kb
            if snapshot is None:
                entry["error"] = "No data"
            elif snapshot.error:
                entry["error"] = snapshot.error
            elif snapshot.output is not None:
                entry["py_spy"] = snapshot.output
            else:
                entry["error"] = "No data"
            ranks_payload[str(rank)] = entry
        payload["ranks"] = ranks_payload
        self.handle.write(json.dumps({"type": "event", "data": payload}) + "\n")
        self.handle.flush()
        self.last_signature = signature
        self.event_count += 1
        return True

    def close(self) -> None:
        try:
            self.handle.close()
        except Exception:
            pass

def read_ps() -> List[Proc]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    procs: List[Proc] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        args = parts[2]
        procs.append(Proc(pid=pid, ppid=ppid, args=args))
    return procs


def select_env_subset(env: Dict[str, str]) -> Dict[str, str]:
    return {key: env[key] for key in ENV_KEYS if key in env and env[key]}


def find_prterun(procs: Sequence[Proc], prterun_pid: Optional[int]) -> Proc:
    if prterun_pid is not None:
        for proc in procs:
            if proc.pid == prterun_pid:
                return proc
        raise SystemExit(f"prterun/mpirun pid {prterun_pid} not found")

    matcher = re.compile(r"(?:^|/)(prterun|mpirun|orterun)\b")
    candidates = [proc for proc in procs if matcher.search(proc.args)]
    if not candidates:
        raise SystemExit("no prterun/mpirun process found")

    with_rankfile = [proc for proc in candidates if find_rankfile_path(proc.args)]
    if with_rankfile:
        candidates = with_rankfile

    candidates.sort(key=lambda p: p.pid, reverse=True)
    return candidates[0]


def find_rankfile_path(args: str) -> Optional[str]:
    match = re.search(r"rankfile:file=([^\s]+)", args)
    if match:
        return match.group(1)
    match = re.search(r"--rankfile\s+([^\s]+)", args)
    if match:
        return match.group(1)
    match = re.search(r"\s-rf\s+([^\s]+)", args)
    if match:
        return match.group(1)
    return None


def parse_rankfile(path: str) -> List[RankInfo]:
    if not os.path.exists(path):
        raise SystemExit(f"rankfile not found: {path}")
    ranks: List[RankInfo] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"rank\s+(\d+)\s*=\s*([^\s]+)", line)
            if not match:
                continue
            rank = int(match.group(1))
            host = match.group(2)
            ranks.append(RankInfo(rank=rank, host=host))
    if not ranks:
        raise SystemExit(f"no ranks parsed from {path}")
    ranks.sort(key=lambda r: r.rank)
    return ranks


def build_children_map(procs: Sequence[Proc]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {}
    for proc in procs:
        children.setdefault(proc.ppid, []).append(proc.pid)
    return children


def find_descendants(children: Dict[int, List[int]], root_pid: int) -> List[int]:
    stack = [root_pid]
    seen = set()
    descendants: List[int] = []
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for child in children.get(pid, []):
            descendants.append(child)
            stack.append(child)
    return descendants


def is_python_process(args: str) -> bool:
    first = args.split(maxsplit=1)[0] if args else ""
    base = os.path.basename(first)
    return "python" in base


def select_program(procs: Sequence[Proc], descendants: Iterable[int]) -> Optional[Proc]:
    descendant_set = set(descendants)
    candidates = [proc for proc in procs if proc.pid in descendant_set and is_python_process(proc.args)]
    candidates = [proc for proc in candidates if "py-spy" not in proc.args]
    if not candidates:
        return None
    def score(proc: Proc) -> Tuple[int, int]:
        has_py = 1 if ".py" in proc.args else 0
        return (has_py, len(proc.args))
    candidates.sort(key=score, reverse=True)
    return candidates[0]


def parse_python_selector(args: str) -> ProgramSelector:
    if not args:
        return ProgramSelector(module=None, script=None, display="")
    try:
        parts = shlex.split(args)
    except ValueError:
        parts = args.split()
    module = None
    script = None
    if "-m" in parts:
        idx = parts.index("-m")
        if idx + 1 < len(parts):
            module = parts[idx + 1]
    for token in parts[1:]:
        if token.startswith("-"):
            continue
        script = token
        break
    display = " ".join(parts)
    return ProgramSelector(module=module, script=script, display=display)


def selector_score(selector: ProgramSelector) -> Tuple[int, int, int, int]:
    if not selector.display:
        return (0, 0, 0, 0)
    has_script = 1 if selector.script else 0
    has_module = 1 if selector.module else 0
    display = f" {selector.display} "
    has_python_target = 1 if ".py" in selector.display or " -m " in display else 0
    return (has_script, has_module, has_python_target, len(selector.display))


def best_selector_from_procs(procs: Iterable[RankProcess]) -> Optional[ProgramSelector]:
    best: Optional[ProgramSelector] = None
    best_score = selector_score(best or ProgramSelector(module=None, script=None, display=""))
    for proc in procs:
        candidate = parse_python_selector(proc.cmdline)
        score = selector_score(candidate)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def extract_python_exe(cmdline: str) -> Optional[str]:
    if not cmdline:
        return None
    try:
        parts = shlex.split(cmdline)
    except ValueError:
        parts = cmdline.split()
    if not parts:
        return None
    exe = parts[0]
    if "python" in os.path.basename(exe):
        return exe
    return None


def matches_python_cmd(cmd: List[str], selector: ProgramSelector) -> bool:
    if not cmd:
        return False
    exe = os.path.basename(cmd[0])
    if "python" not in exe:
        return False
    if selector.module:
        try:
            idx = cmd.index("-m")
        except ValueError:
            return False
        return idx + 1 < len(cmd) and cmd[idx + 1] == selector.module
    if selector.script:
        target = selector.script
        base_target = os.path.basename(target)
        for arg in cmd:
            if arg == target or arg.endswith("/" + target) or os.path.basename(arg) == base_target:
                return True
        return False
    return True


def find_rank_pids_local(
    selector: ProgramSelector,
) -> List[Tuple[int, int, str, Optional[int], Optional[str], Dict[str, str]]]:
    results: List[Tuple[int, int, str, Optional[int], Optional[str], Dict[str, str]]] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmd = [x.decode(errors="ignore") for x in handle.read().split(b"\0") if x]
            if not matches_python_cmd(cmd, selector):
                continue
            with open(f"/proc/{pid}/environ", "rb") as handle:
                env_items = [x for x in handle.read().split(b"\0") if x]
            env: Dict[str, str] = {}
            for item in env_items:
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                env[key.decode(errors="ignore")] = value.decode(errors="ignore")
            rank = env.get("OMPI_COMM_WORLD_RANK") or env.get("PMIX_RANK") or env.get("PMI_RANK")
            if rank is None:
                continue
            rss_kb = read_rss_kb(int(pid))
            cmdline = " ".join(cmd)
            try:
                exe_path = os.readlink(f"/proc/{pid}/exe")
            except Exception:
                exe_path = ""
            python_exe = exe_path or extract_python_exe(cmdline)
            env_subset = select_env_subset(env)
            results.append((int(rank), int(pid), cmdline, rss_kb, python_exe, env_subset))
        except Exception:
            continue
    return results


def read_rss_kb(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except Exception:
        return None
    return None


def run_ssh(host: str, command: str, timeout: int = 8) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            host,
            command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def find_rank_pids_remote(
    host: str, selector: ProgramSelector
) -> Tuple[List[Tuple[int, int, str, Optional[int], Optional[str], Dict[str, str]]], Optional[str]]:
    env_prefix = build_env_prefix(
        {
            "MPIPTOP_TARGET": selector.script or "",
            "MPIPTOP_MODULE": selector.module or "",
        }
    )
    remote_cmd = f"{env_prefix}python3 - <<'PY'\n{REMOTE_FINDER_SCRIPT}\nPY"
    try:
        result = run_ssh(host, remote_cmd)
    except subprocess.TimeoutExpired:
        return [], f"ssh timeout to {host}"
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout).strip()
        msg = stderr or f"ssh failed ({result.returncode})"
        return [], f"{host}: {msg}"
    try:
        data = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return [], f"{host}: invalid json from remote"
    parsed: List[Tuple[int, int, str, Optional[int], Optional[str], Dict[str, str]]] = []
    for entry in data:
        env_subset: Dict[str, str] = {}
        python_exe: Optional[str] = None
        if len(entry) >= 6:
            r, p, cmd, rss_kb, python_exe, env_subset = entry[:6]
        elif len(entry) >= 5:
            r, p, cmd, rss_kb, python_exe = entry[:5]
            env_subset = {}
        elif len(entry) >= 4:
            r, p, cmd, rss_kb = entry[:4]
            python_exe = None
            env_subset = {}
        else:
            r, p, cmd = entry
            rss_kb = None
            python_exe = None
            env_subset = {}
        rss_value = int(rss_kb) if rss_kb is not None else None
        parsed.append((int(r), int(p), str(cmd), rss_value, python_exe, env_subset or {}))
    return parsed, None


def run_py_spy(
    host: str,
    proc: RankProcess,
    pythonpath: str,
    install_attempted: set,
    timeout: int = 8,
) -> Tuple[Optional[str], Optional[str]]:
    env_vars = merge_env(proc, pythonpath, os.environ.copy() if is_local_host(host) else None)
    env_prefix = build_env_prefix(env_vars)
    py_spy_path = None
    if proc.python_exe:
        py_spy_path = os.path.join(os.path.dirname(proc.python_exe), "py-spy")
        if is_local_host(host) and not os.access(py_spy_path, os.X_OK):
            py_spy_path = None
    if not py_spy_path:
        venv = env_vars.get("VIRTUAL_ENV")
        if venv:
            py_spy_path = os.path.join(venv, "bin", "py-spy")
            if is_local_host(host) and not os.access(py_spy_path, os.X_OK):
                py_spy_path = None

    def run_dump() -> subprocess.CompletedProcess:
        if is_local_host(host):
            env = env_vars
            cmd = [py_spy_path or "py-spy", "dump", "-p", str(proc.pid)]
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        if py_spy_path:
            spy_cmd = shlex.quote(py_spy_path)
            script = (
                f"if [ -x {spy_cmd} ]; then {spy_cmd} dump -p {proc.pid}; "
                f"else py-spy dump -p {proc.pid}; fi"
            )
            remote_cmd = f"{env_prefix}sh -lc {shlex.quote(script)}"
        else:
            remote_cmd = f"{env_prefix}py-spy dump -p {proc.pid}"
        return run_ssh(host, remote_cmd, timeout=timeout)

    def missing_py_spy(exc: Optional[BaseException], result: Optional[subprocess.CompletedProcess]) -> bool:
        if isinstance(exc, FileNotFoundError):
            return True
        if result is None:
            return False
        stderr = (result.stderr or result.stdout or "").lower()
        return result.returncode == 127 or "py-spy: command not found" in stderr

    def ensure_installed() -> Optional[str]:
        venv = env_vars.get("VIRTUAL_ENV")
        venv_python = os.path.join(venv, "bin", "python") if venv else None
        python_exe = proc.python_exe or venv_python or (sys.executable if is_local_host(host) else "python3")
        env_prefix_install = build_env_prefix(env_vars if not is_local_host(host) else {})
        if is_local_host(host):
            env = env_vars
            try:
                result = subprocess.run(
                    [python_exe, "-m", "pip", "install", "py-spy"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                )
            except Exception as exc:
                return str(exc)
            if result.returncode == 0:
                return None
            retry = subprocess.run(
                [python_exe, "-m", "pip", "install", "--user", "py-spy"],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            if retry.returncode == 0:
                return None
            pip_error = (retry.stderr or retry.stdout or "").strip() or "pip install py-spy failed"
            if should_try_uv(pip_error):
                uv_error = uv_install_local(python_exe, env, pip_error)
                if uv_error is None:
                    return None
                fallback_error = pip_user_install_local(env)
                if fallback_error is None:
                    return None
                return f"{uv_error}\n{fallback_error}"
            return pip_error

        cmd = f"{env_prefix_install}{shlex.quote(python_exe)} -m pip install py-spy"
        try:
            result = run_ssh(host, cmd, timeout=120)
        except subprocess.TimeoutExpired:
            return "pip install py-spy timeout"
        if result.returncode == 0:
            return None
        retry = run_ssh(
            host,
            f"{env_prefix_install}{shlex.quote(python_exe)} -m pip install --user py-spy",
            timeout=120,
        )
        if retry.returncode == 0:
            return None
        pip_error = (retry.stderr or retry.stdout or "").strip() or "pip install py-spy failed"
        if should_try_uv(pip_error):
            uv_error = uv_install_remote(host, python_exe, env_prefix_install)
            if uv_error is None:
                return None
            fallback_error = pip_user_install_remote(host, env_prefix_install)
            if fallback_error is None:
                return None
            return f"{uv_error}\n{fallback_error}"
        return pip_error

    try:
        result = run_dump()
    except FileNotFoundError as exc:
        result = None
        error_exc = exc
    except subprocess.TimeoutExpired:
        return None, f"py-spy timeout on {host}:{proc.pid}"
    else:
        error_exc = None

    if result is not None and result.returncode == 0:
        return result.stdout, None

    if missing_py_spy(error_exc, result):
        key = f"{host}|{proc.python_exe or sys.executable}"
        if key not in install_attempted:
            install_attempted.add(key)
            install_error = ensure_installed()
            if install_error:
                return None, f"py-spy install failed on {host}: {install_error}"
            try:
                retry = run_dump()
            except FileNotFoundError:
                return None, f"py-spy still missing on {host}:{proc.pid}"
            except subprocess.TimeoutExpired:
                return None, f"py-spy timeout on {host}:{proc.pid}"
            if retry.returncode == 0:
                return retry.stdout, None
            stderr = (retry.stderr or retry.stdout or "").strip()
            return None, stderr or f"py-spy failed on {host}:{proc.pid}"

    stderr = (result.stderr or result.stdout or "").strip() if result is not None else str(error_exc)
    return None, stderr or f"py-spy failed on {host}:{proc.pid}"


def build_env_prefix(env: Dict[str, str]) -> str:
    if not env:
        return ""
    parts = []
    for key, value in env.items():
        if not value:
            continue
        parts.append(f"{key}={shlex.quote(value)}")
    return " ".join(parts) + " " if parts else ""


def merge_env(
    proc: RankProcess,
    pythonpath: str,
    base: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    env = dict(base or {})
    for key, value in (proc.env or {}).items():
        if value:
            env[key] = value
    venv = env.get("VIRTUAL_ENV")
    if venv:
        venv_bin = os.path.join(venv, "bin")
        path = env.get("PATH", "")
        path_parts = path.split(":") if path else []
        if not path_parts or path_parts[0] != venv_bin:
            env["PATH"] = ":".join([venv_bin] + [p for p in path_parts if p != venv_bin])
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return env


def should_try_uv(error_text: str) -> bool:
    lowered = error_text.lower()
    return "externally-managed-environment" in lowered or "managed by uv" in lowered


def find_uv_binary(env: Dict[str, str]) -> Optional[str]:
    uv_path = shutil.which("uv", path=env.get("PATH")) if env else shutil.which("uv")
    if uv_path:
        return uv_path
    home = env.get("HOME") if env else None
    candidates = []
    if home:
        candidates.extend(
            [
                os.path.join(home, ".local", "bin", "uv"),
                os.path.join(home, ".cargo", "bin", "uv"),
            ]
        )
    candidates.extend(["/usr/local/bin/uv", "/opt/homebrew/bin/uv"])
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def uv_install_local(python_exe: str, env: Dict[str, str], pip_error: str) -> Optional[str]:
    uv_path = find_uv_binary(env)
    if not uv_path:
        return f"{pip_error}\nuv not found on PATH"
    try:
        result = subprocess.run(
            [uv_path, "pip", "install", "--python", python_exe, "py-spy"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except Exception as exc:
        return f"{pip_error}\nuv install failed: {exc}"
    if result.returncode == 0:
        return None
    uv_error = (result.stderr or result.stdout or "").strip()
    return f"{pip_error}\nuv install failed: {uv_error or 'unknown error'}"


def uv_install_remote(host: str, python_exe: str, env_prefix: str) -> Optional[str]:
    script = textwrap.dedent(
        f"""
        uv_cmd=""
        if command -v uv >/dev/null 2>&1; then
            uv_cmd="uv"
        else
            for cand in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "/usr/local/bin/uv" "/opt/homebrew/bin/uv"; do
                if [ -x "$cand" ]; then
                    uv_cmd="$cand"
                    break
                fi
            done
        fi
        if [ -z "$uv_cmd" ]; then
            echo "uv not found on PATH" 1>&2
            exit 127
        fi
        "$uv_cmd" pip install --python {shlex.quote(python_exe)} py-spy
        """
    ).strip()
    cmd = f"{env_prefix}sh -lc {shlex.quote(script)}"
    try:
        result = run_ssh(host, cmd, timeout=120)
    except subprocess.TimeoutExpired:
        return "uv install timeout"
    if result.returncode == 0:
        return None
    uv_error = (result.stderr or result.stdout or "").strip() or "uv install failed"
    return f"uv install failed: {uv_error}"


def pip_user_install_local(env: Dict[str, str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["pip", "install", "--user", "py-spy"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except FileNotFoundError:
        return "pip not found on PATH"
    except Exception as exc:
        return f"pip install failed: {exc}"
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "pip install --user failed"


def pip_user_install_remote(host: str, env_prefix: str) -> Optional[str]:
    cmd = f"{env_prefix}pip install --user py-spy"
    try:
        result = run_ssh(host, cmd, timeout=120)
    except subprocess.TimeoutExpired:
        return "pip install timeout"
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "pip install --user failed"


STACK_LINE_RE = re.compile(r"^(\s*)(.*?)\s+\((.*):(\d+)\)\s*$")


def pastel_color(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    saturation = 0.35
    value = 0.92
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def highlight_substring(text: str, substring: str, color: str) -> Text:
    idx = text.find(substring)
    if idx == -1:
        return Text(text)
    output = Text()
    output.append(text[:idx])
    output.append(substring, style=color)
    output.append(text[idx + len(substring):])
    return output


def highlight_py_path(text: str) -> Text:
    match = re.search(r"\S+\.py\b", text)
    if not match:
        return Text(text)
    path = match.group(0)
    color = pastel_color(path)
    output = Text()
    output.append(text[:match.start()])
    output.append(path, style=color)
    output.append(text[match.end():])
    return output


def style_program_display(selector: ProgramSelector) -> Text:
    if not selector.display:
        return Text("python")
    if selector.script and selector.script in selector.display:
        return highlight_substring(selector.display, selector.script, pastel_color(selector.script))
    if selector.module and selector.module in selector.display:
        return highlight_substring(selector.display, selector.module, pastel_color(selector.module))
    return highlight_py_path(selector.display)


def style_program_line(line: str, selector: ProgramSelector) -> Text:
    if selector.script and selector.script in line:
        return highlight_substring(line, selector.script, pastel_color(selector.script))
    if selector.module and selector.module in line:
        return highlight_substring(line, selector.module, pastel_color(selector.module))
    return highlight_py_path(line)


def wrap_program_lines(selector: ProgramSelector, width: int) -> List[Text]:
    display = selector.display or "python"
    if width <= 0:
        return [style_program_line(display, selector)]
    try:
        parts = shlex.split(display)
    except ValueError:
        parts = display.split()
    if not parts:
        return [style_program_line(display, selector)]

    prefix_tokens = [parts[0]]
    arg_tokens = parts[1:]

    if selector.module and "-m" in parts:
        idx = parts.index("-m")
        if idx + 1 < len(parts):
            prefix_tokens = parts[: idx + 2]
            arg_tokens = parts[idx + 2 :]
    elif selector.script:
        base = os.path.basename(selector.script)
        script_idx = None
        for i, tok in enumerate(parts[1:], start=1):
            if tok == selector.script or tok.endswith("/" + selector.script) or os.path.basename(tok) == base:
                script_idx = i
                break
        if script_idx is not None:
            prefix_tokens = parts[: script_idx + 1]
            arg_tokens = parts[script_idx + 1 :]

    prefix = " ".join(prefix_tokens)
    args_str = " ".join(arg_tokens)
    indent = len(prefix) + 1

    if not args_str:
        return [style_program_line(prefix, selector)]

    if indent >= width:
        wrapped = textwrap.wrap(display, width=width, break_long_words=False, break_on_hyphens=False)
        if not wrapped:
            wrapped = [display]
        return [style_program_line(line, selector) for line in wrapped]

    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    wrapped_args = wrapper.wrap(args_str) or [args_str]
    first = wrapped_args[0]
    if first.startswith(" " * indent):
        first = first[indent:]
    lines = [f"{prefix} {first}"]
    lines.extend(wrapped_args[1:])
    return [style_program_line(line, selector) for line in lines]


def style_detail_line(line: str) -> Text:
    lower = line.lower()
    if lower.startswith("program:"):
        return highlight_py_path(line)
    return Text(line)


def style_stack_line(line: str) -> Text:
    marker = ""
    if line.startswith("➤ "):
        marker = "➤ "
        line = line[2:]
    match = STACK_LINE_RE.match(line)
    if not match:
        output = Text()
        if marker:
            output.append(marker, style=KEY_STYLE)
        output.append(line)
        return output
    indent, func, file_path, line_no = match.groups()
    color = pastel_color(file_path)
    output = Text()
    if marker:
        output.append(marker, style=KEY_STYLE)
    output.append(indent)
    output.append(func, style=color)
    output.append(" ")
    output.append("(", style=PUNCT_STYLE)
    output.append(file_path, style=color)
    output.append(":", style=PUNCT_STYLE)
    output.append(line_no, style=PUNCT_STYLE)
    output.append(")", style=PUNCT_STYLE)
    return output


def style_lines(lines: List[str]) -> Text:
    output = Text()
    for idx, line in enumerate(lines):
        if idx:
            output.append("\n")
        output.append_text(style_stack_line(line))
    output.no_wrap = True
    output.overflow = "crop"
    return output


def format_rss(rss_kb: Optional[int]) -> str:
    if rss_kb is None:
        return "unknown"
    value = float(rss_kb) * 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1000.0 and idx < len(units) - 1:
        value /= 1000.0
        idx += 1
    if units[idx] == "B":
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def parse_pyspy_output(output: str) -> ParsedPySpy:
    details: List[str] = []
    threads: List[ThreadBlock] = []
    current_header: Optional[str] = None
    current_stack: List[str] = []
    in_threads = False
    for line in output.splitlines():
        if line.startswith("Thread "):
            if current_header is not None:
                threads.append(ThreadBlock(header=current_header, stack=current_stack))
            current_header = line
            current_stack = []
            in_threads = True
            continue
        if not in_threads:
            details.append(line)
        else:
            current_stack.append(line)
    if current_header is not None:
        threads.append(ThreadBlock(header=current_header, stack=current_stack))
    return ParsedPySpy(details=details, threads=threads)


def invert_stack_lines(lines: List[str]) -> List[str]:
    output: List[str] = []
    stack_block: List[str] = []
    for line in lines:
        if line.startswith("  "):
            stack_block.append(line)
            continue
        if stack_block:
            output.extend(reversed(stack_block))
            stack_block = []
        output.append(line)
    if stack_block:
        output.extend(reversed(stack_block))
    return output


def filter_detail_lines(lines: List[str]) -> List[str]:
    kept: List[str] = []
    for line in lines:
        lower = line.lower()
        if lower.startswith("program:") or lower.startswith("python version:"):
            kept.append(line)
    return kept


def select_threads(threads: List[ThreadBlock], show_threads: bool) -> Tuple[List[ThreadBlock], int]:
    if show_threads:
        return threads, 0
    if not threads:
        return [], 0
    main_thread = None
    for thread in threads:
        if "MainThread" in thread.header:
            main_thread = thread
            break
    if main_thread is None:
        main_thread = threads[0]
    return [main_thread], len(threads) - 1


def render_pyspy_output(output: str, show_threads: bool) -> Tuple[List[str], List[str]]:
    parsed = parse_pyspy_output(output)
    details = filter_detail_lines(parsed.details)
    inverted_threads = [
        ThreadBlock(header=thread.header, stack=invert_stack_lines(thread.stack))
        for thread in parsed.threads
    ]
    display_threads, other_count = select_threads(inverted_threads, show_threads)
    lines: List[str] = []
    if not display_threads:
        lines.append("no thread data")
    else:
        for idx, thread in enumerate(display_threads):
            if thread.header:
                lines.append(thread.header)
            lines.extend(thread.stack)
            if show_threads and idx < len(display_threads) - 1:
                lines.append("")
        if not show_threads and other_count > 0:
            lines.append(f"(+{other_count} other threads)")
    return lines, details


def extract_stack_lines(lines: List[str]) -> List[str]:
    if not lines:
        return []
    start = 1 if lines[0].startswith("Thread ") else 0
    stack: List[str] = []
    for line in lines[start:]:
        if line.startswith("  "):
            stack.append(line)
        else:
            break
    return stack


def common_prefix_length(stacks_by_rank: Dict[int, List[str]]) -> int:
    if not stacks_by_rank:
        return 0
    stack_lists = list(stacks_by_rank.values())
    min_len = min(len(stack) for stack in stack_lists)
    prefix_len = 0
    for idx in range(min_len):
        values = [stack[idx] for stack in stack_lists]
        if all(value == values[0] for value in values):
            prefix_len += 1
        else:
            break
    return prefix_len


def mark_diff_line(lines: List[str], diff_index: int) -> List[str]:
    if diff_index is None:
        return lines
    marked = list(lines)
    stack_pos = 0
    for idx, line in enumerate(marked):
        if line.startswith("  "):
            if stack_pos == diff_index:
                if line.startswith("  "):
                    marked[idx] = "➤ " + line[2:]
                else:
                    marked[idx] = "➤ " + line
                break
            stack_pos += 1
    return marked


def is_local_host(host: str) -> bool:
    host = host.split(".")[0]
    local = socket.gethostname().split(".")[0]
    return host == local or host in {"localhost", "127.0.0.1"}


def shorten(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def build_header(
    state: State, last_update: str, errors: List[str], refresh: int, width: int
) -> Tuple[Text, int]:
    program_lines = wrap_program_lines(state.selector, width)
    if not program_lines:
        program_lines = [Text("python")]
    for line in program_lines:
        line.no_wrap = True
        line.overflow = "crop"

    controls_plain = "q quit | space refresh | t threads | d details"
    padding = max(0, width - len(controls_plain))
    line2 = Text(" " * padding)
    line2.append("q", style=KEY_STYLE)
    line2.append(" quit | ")
    line2.append("space", style=KEY_STYLE)
    line2.append(" refresh | ")
    line2.append("t", style=KEY_STYLE)
    line2.append(" threads | ")
    line2.append("d", style=KEY_STYLE)
    line2.append(" details")
    line2.truncate(width)

    text = Text()
    for idx, line in enumerate(program_lines):
        if idx:
            text.append("\n")
        text.append_text(line)
    text.append("\n")
    text.append_text(line2)
    text.no_wrap = True
    text.overflow = "crop"
    return text, len(program_lines) + 1


def render_columns(
    ranks: List[RankInfo],
    stacks: Dict[int, Text],
    details: Optional[Text],
    body_height: int,
    rank_to_proc: Dict[int, RankProcess],
) -> Table:
    panels = []
    for entry in ranks:
        title = f"rank {entry.rank} @ {entry.host}"
        proc = rank_to_proc.get(entry.rank)
        if proc and proc.rss_kb is not None:
            title = f"{title} | {format_rss(proc.rss_kb)}"
        stack_text = stacks.get(entry.rank, Text("No process"))
        stack_text.no_wrap = True
        stack_text.overflow = "crop"
        panels.append(
            Panel(
                stack_text,
                title=title,
                height=body_height,
                padding=(0, 1),
                border_style=BORDER_STYLE,
            )
        )
    if details is not None:
        details.no_wrap = True
        details.overflow = "crop"
        panels.append(
            Panel(
                details,
                title="details",
                height=body_height,
                padding=(0, 1),
                border_style=BORDER_STYLE,
            )
        )
    grid = Table.grid(expand=True)
    for _ in panels:
        grid.add_column(ratio=1)
    grid.add_row(*panels)
    return grid


def wrap_cmdline(cmdline: str, width: int) -> List[str]:
    prefix = "cmd: "
    if width <= len(prefix) + 2:
        return [f"{prefix}{cmdline}"]
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapper.wrap(cmdline) or [f"{prefix}{cmdline}"]


def build_details_text(
    ranks: List[RankInfo],
    rank_to_proc: Dict[int, RankProcess],
    details_by_rank: Dict[int, List[str]],
    cmd_width: int,
) -> Text:
    output = Text()
    for idx, entry in enumerate(ranks):
        if idx:
            output.append("\n\n")
        output.append(f"rank {entry.rank} @ {entry.host}", style="bold")
        proc = rank_to_proc.get(entry.rank)
        lines = details_by_rank.get(entry.rank, [])
        if proc is None:
            output.append("\n")
            output.append("No process")
            continue
        output.append("\n")
        output.append(f"pid: {proc.pid}")
        output.append("\n")
        output.append(f"rss: {format_rss(proc.rss_kb)}")
        for cmd_line in wrap_cmdline(proc.cmdline, cmd_width):
            output.append("\n")
            output.append_text(highlight_py_path(cmd_line))
        for line in lines:
            output.append("\n")
            output.append_text(style_detail_line(line))
    return output


def format_elapsed(start: Optional[float]) -> str:
    if start is None:
        return "0:00"
    elapsed = max(0, int(time.time() - start))
    return format_duration(elapsed)


def format_duration(elapsed: int) -> str:
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def build_live_header(
    state: State,
    last_update: str,
    refresh: int,
    record_line: Optional[str],
    width: int,
) -> Tuple[Text, int]:
    program_lines = wrap_program_lines(state.selector, width)
    if not program_lines:
        program_lines = [Text("python")]
    for line in program_lines:
        line.no_wrap = True
        line.overflow = "crop"

    record_text = None
    if record_line:
        record_text = Text()
        record_text.append("REC", style="bold red")
        record_text.append(" recording: ")
        record_text.append(record_line)
        record_text.truncate(width)
        record_text.no_wrap = True
        record_text.overflow = "crop"

    controls_plain = "q quit | space refresh | t threads | d details | r record"
    padding = max(0, width - len(controls_plain))
    controls_line = Text(" " * padding + controls_plain)
    for token in ["q", "space", "t", "d", "r"]:
        start = controls_plain.find(token)
        if start != -1:
            controls_line.stylize(KEY_STYLE, padding + start, padding + start + len(token))
    controls_line.truncate(width)
    controls_line.no_wrap = True
    controls_line.overflow = "crop"

    text = Text()
    for idx, line in enumerate(program_lines):
        if idx:
            text.append("\n")
        text.append_text(line)
    text.append("\n")
    if record_text is not None:
        text.append_text(record_text)
        text.append("\n")
    text.append_text(controls_line)
    text.no_wrap = True
    text.overflow = "crop"
    extra_lines = 2 if record_text is not None else 1
    return text, len(program_lines) + extra_lines


def build_review_header(
    state: State,
    event_index: int,
    event_total: int,
    event_time: str,
    timeline_lines: List[Text],
    width: int,
) -> Tuple[Text, int]:
    program_lines = wrap_program_lines(state.selector, width)
    if not program_lines:
        program_lines = [Text("python")]
    status_line = Text(
        f"review {event_index + 1}/{event_total} | {event_time}"
    )
    status_line.truncate(width)

    controls_plain = "q quit | left/right move | down zoom | up zoom out | t threads | d details"
    padding = max(0, width - len(controls_plain))
    controls_line = Text(" " * padding + controls_plain)
    for token in ["q", "left/right", "down", "up", "t", "d"]:
        start = controls_plain.find(token)
        if start != -1:
            controls_line.stylize(KEY_STYLE, padding + start, padding + start + len(token))
    controls_line.truncate(width)
    controls_line.no_wrap = True
    controls_line.overflow = "crop"

    text = Text()
    for idx, line in enumerate(program_lines):
        if idx:
            text.append("\n")
        text.append_text(line)
    text.append("\n")
    text.append_text(status_line)
    for line in timeline_lines:
        text.append("\n")
        text.append_text(line)
    text.append("\n")
    text.append_text(controls_line)
    text.no_wrap = True
    text.overflow = "crop"
    return text, len(program_lines) + 1 + len(timeline_lines) + 1


def build_buckets(start: int, end: int, width: int) -> List[Tuple[int, int]]:
    count = max(0, end - start)
    if count == 0:
        return []
    bucket_count = max(1, min(width, count))
    base = count // bucket_count
    remainder = count % bucket_count
    buckets: List[Tuple[int, int]] = []
    current = start
    for idx in range(bucket_count):
        size = base + (1 if idx < remainder else 0)
        buckets.append((current, current + size))
        current += size
    return buckets


def divergence_color(ratio: float) -> str:
    clamped = min(1.0, max(0.0, ratio))
    intensity = clamped ** 0.7
    base = (170, 170, 170)
    hot = (255, 122, 0)
    r = int(base[0] + (hot[0] - base[0]) * intensity)
    g = int(base[1] + (hot[1] - base[1]) * intensity)
    b = int(base[2] + (hot[2] - base[2]) * intensity)
    return f"#{r:02x}{g:02x}{b:02x}"


def compute_event_metrics(
    events: List[SessionEvent],
    ranks: List[RankInfo],
    show_threads: bool,
) -> Tuple[List[int], List[float], List[int]]:
    max_stack_lens: List[int] = []
    divergence_ratios: List[float] = []
    common_prefixes: List[int] = []
    for event in events:
        stacks_by_rank: Dict[int, List[str]] = {}
        for info in ranks:
            payload = event.ranks.get(info.rank, {})
            if payload.get("error"):
                stacks_by_rank[info.rank] = []
                continue
            output = payload.get("py_spy")
            if not output:
                stacks_by_rank[info.rank] = []
                continue
            lines, _details = render_pyspy_output(str(output), show_threads)
            stacks_by_rank[info.rank] = extract_stack_lines(lines)
        max_len = max((len(stack) for stack in stacks_by_rank.values()), default=0)
        common_len = common_prefix_length(stacks_by_rank)
        similarity = float(common_len) / float(max_len) if max_len else 0.0
        ratio = 1.0 - similarity if max_len else 0.0
        max_stack_lens.append(max_len)
        divergence_ratios.append(ratio)
        common_prefixes.append(common_len)
    return max_stack_lens, divergence_ratios, common_prefixes


def render_timeline_lines(
    levels: List[TimelineLevel],
    max_stack_lens: List[int],
    divergence_ratios: List[float],
    width: int,
) -> List[Text]:
    lines: List[Text] = []
    for level_index, level in enumerate(levels):
        level.buckets = build_buckets(level.start, level.end, width)
        if level.buckets:
            level.selected = max(0, min(level.selected, len(level.buckets) - 1))
        stats: List[Tuple[int, float]] = []
        for start, end in level.buckets:
            bucket_heights = max_stack_lens[start:end]
            bucket_ratios = divergence_ratios[start:end]
            height = max(bucket_heights) if bucket_heights else 0
            ratio = max(bucket_ratios) if bucket_ratios else 0.0
            stats.append((height, ratio))
        max_height = max((height for height, _ in stats), default=1)
        if max_height <= 0:
            max_height = 1
        text = Text()
        for idx, (height, ratio) in enumerate(stats):
            normalized = float(height) / float(max_height) if max_height else 0.0
            level_idx = int(round(normalized * (len(SPARKLINE_CHARS) - 1)))
            level_idx = max(0, min(level_idx, len(SPARKLINE_CHARS) - 1))
            char = SPARKLINE_CHARS[level_idx]
            style = divergence_color(ratio)
            if idx == level.selected:
                if level_index == len(levels) - 1:
                    style = f"{style} bold underline"
                else:
                    style = f"{style} underline"
            text.append(char, style=style)
        text.no_wrap = True
        text.overflow = "crop"
        lines.append(text)
    return lines


def event_snapshots_from_event(
    event: SessionEvent,
    ranks: List[RankInfo],
    show_threads: bool,
) -> Dict[int, RankSnapshot]:
    snapshots: Dict[int, RankSnapshot] = {}
    for info in ranks:
        payload = event.ranks.get(info.rank)
        if not payload:
            snapshots[info.rank] = RankSnapshot(
                output=None,
                error="No data",
                stack_lines=["No data"],
                details=[],
            )
            continue
        if payload.get("error"):
            snapshots[info.rank] = RankSnapshot(
                output=None,
                error=str(payload.get("error")),
                stack_lines=[str(payload.get("error"))],
                details=[],
            )
            continue
        output = payload.get("py_spy")
        if not output:
            snapshots[info.rank] = RankSnapshot(
                output=None,
                error="No data",
                stack_lines=["No data"],
                details=[],
            )
            continue
        lines, details = render_pyspy_output(str(output), show_threads)
        snapshots[info.rank] = RankSnapshot(
            output=str(output),
            error=None,
            stack_lines=lines,
            details=details,
        )
    return snapshots


def rank_to_proc_from_event(
    event: SessionEvent,
    ranks: List[RankInfo],
) -> Dict[int, RankProcess]:
    rank_to_proc: Dict[int, RankProcess] = {}
    for info in ranks:
        payload = event.ranks.get(info.rank)
        if not payload:
            continue
        pid = payload.get("pid")
        cmdline = payload.get("cmdline")
        rss_kb = payload.get("rss_kb")
        if pid is None or cmdline is None:
            continue
        try:
            pid_value = int(pid)
        except (TypeError, ValueError):
            continue
        rss_value = None
        if rss_kb is not None:
            try:
                rss_value = int(rss_kb)
            except (TypeError, ValueError):
                rss_value = None
        rank_to_proc[info.rank] = RankProcess(
            pid=pid_value,
            cmdline=str(cmdline),
            rss_kb=rss_value,
            python_exe=None,
            env={},
        )
    return rank_to_proc


def compute_divergence_from_snapshots(
    ranks: List[RankInfo], snapshots: Dict[int, RankSnapshot]
) -> Tuple[float, int, int]:
    stack_lines_by_rank = {
        info.rank: extract_stack_lines(snapshots.get(info.rank, RankSnapshot(None, "No data", [], [])).stack_lines)
        for info in ranks
    }
    max_len = max((len(stack) for stack in stack_lines_by_rank.values()), default=0)
    common_len = common_prefix_length(stack_lines_by_rank)
    similarity = float(common_len) / float(max_len) if max_len else 0.0
    divergence = 1.0 - similarity if max_len else 0.0
    return divergence, common_len, max_len


def read_key(timeout: float) -> Optional[str]:
    if sys.stdin not in select_with_timeout(timeout):
        return None
    key = sys.stdin.read(1)
    if key != "\x1b":
        return key
    seq = key
    for _ in range(2):
        if sys.stdin in select_with_timeout(0.01):
            seq += sys.stdin.read(1)
    if seq == "\x1b[A":
        return "up"
    if seq == "\x1b[B":
        return "down"
    if seq == "\x1b[C":
        return "right"
    if seq == "\x1b[D":
        return "left"
    return None


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

def detect_state(args: argparse.Namespace) -> State:
    procs = read_ps()
    prte = find_prterun(procs, args.prterun_pid)
    rankfile = args.rankfile or find_rankfile_path(prte.args)
    if not rankfile:
        raise SystemExit("rankfile not found in prterun/mpirun args")
    ranks = parse_rankfile(rankfile)
    children = build_children_map(procs)
    descendants = find_descendants(children, prte.pid)
    program_proc = select_program(procs, descendants)
    selector = parse_python_selector(program_proc.args if program_proc else "")
    return State(prte_pid=prte.pid, rankfile=rankfile, ranks=ranks, selector=selector)


def collect_rank_pids(state: State) -> Tuple[Dict[int, RankProcess], List[str]]:
    errors: List[str] = []
    rank_to_proc: Dict[int, RankProcess] = {}
    hosts = sorted({entry.host for entry in state.ranks})
    rank_set = {entry.rank for entry in state.ranks}

    for host in hosts:
        if is_local_host(host):
            entries = find_rank_pids_local(state.selector)
            host_error = None
        else:
            entries, host_error = find_rank_pids_remote(host, state.selector)
        if host_error:
            errors.append(host_error)
        for rank, pid, cmd, rss_kb, python_exe, env_subset in entries:
            if rank not in rank_set:
                continue
            existing = rank_to_proc.get(rank)
            if existing is None or pid > existing.pid:
                venv = env_subset.get("VIRTUAL_ENV") if env_subset else None
                venv_python = os.path.join(venv, "bin", "python") if venv else None
                rank_to_proc[rank] = RankProcess(
                    pid=pid,
                    cmdline=cmd,
                    rss_kb=rss_kb,
                    python_exe=python_exe or venv_python or extract_python_exe(cmd),
                    env=env_subset or {},
                )
    return rank_to_proc, errors


def collect_stacks(
    state: State,
    rank_to_proc: Dict[int, RankProcess],
    pythonpath: str,
    show_threads: bool,
    install_attempted: set,
) -> Tuple[Dict[int, RankSnapshot], List[str]]:
    snapshots: Dict[int, RankSnapshot] = {}
    errors: List[str] = []
    for entry in state.ranks:
        proc = rank_to_proc.get(entry.rank)
        if proc is None:
            snapshots[entry.rank] = RankSnapshot(
                output=None,
                error="No process",
                stack_lines=["No process"],
                details=[],
            )
            continue
        output, error = run_py_spy(entry.host, proc, pythonpath, install_attempted)
        if error:
            errors.append(error)
            snapshots[entry.rank] = RankSnapshot(
                output=None,
                error=error,
                stack_lines=[error],
                details=[],
            )
            continue
        lines, details = render_pyspy_output(output or "", show_threads)
        snapshots[entry.rank] = RankSnapshot(
            output=output,
            error=None,
            stack_lines=lines,
            details=details,
        )
    return snapshots, errors


def parse_live_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show MPI Python stacks across hosts.")
    parser.add_argument("--rankfile", help="Override rankfile path")
    parser.add_argument("--prterun-pid", type=int, help="PID of prterun/mpirun")
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval (seconds)")
    parser.add_argument(
        "--pythonpath",
        help="PYTHONPATH to export remotely (defaults to local PYTHONPATH)",
    )
    parser.add_argument(
        "--out",
        help="Output path for recordings (.jsonl file or directory)",
    )
    return parser.parse_args(argv)


def parse_review_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review a recorded mpiptop session.")
    parser.add_argument("path", help="Path to a recorded session (.jsonl file or directory)")
    return parser.parse_args(argv)


def parse_summarize_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a recorded mpiptop session.")
    parser.add_argument("path", help="Path to a recorded session (.jsonl file or directory)")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Top signatures to report",
    )
    return parser.parse_args(argv)


def parse_record_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an mpiptop session.")
    parser.add_argument("--rankfile", help="Override rankfile path")
    parser.add_argument("--prterun-pid", type=int, help="PID of prterun/mpirun")
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval (seconds)")
    parser.add_argument(
        "--pythonpath",
        help="PYTHONPATH to export remotely (defaults to local PYTHONPATH)",
    )
    parser.add_argument(
        "--out",
        help="Output path for recordings (.jsonl file or directory)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print start/stop lines",
    )
    args = parser.parse_args(argv)
    args.record = True
    return args


def run_live(args: argparse.Namespace) -> int:
    pythonpath = args.pythonpath if args.pythonpath is not None else os.environ.get("PYTHONPATH", "")

    state = detect_state(args)
    console = Console()
    refresh = max(1, args.refresh)
    show_threads = False
    show_details = False
    install_attempted: set = set()
    record_session: Optional[RecordSession] = None
    recording_enabled = bool(getattr(args, "record", False))
    record_started_at: Optional[float] = None
    record_path = args.out

    def handle_sigint(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    layout = Layout()
    layout.split_column(Layout(name="header", size=HEADER_HEIGHT), Layout(name="body"))

    last_update = "never"
    next_refresh = 0.0

    def start_recording() -> None:
        nonlocal record_session, recording_enabled, record_started_at, record_path
        if record_session is None:
            record_path = record_path or default_session_path()
            record_session = RecordSession(record_path, state, refresh, pythonpath)
        recording_enabled = True
        if record_started_at is None:
            record_started_at = time.time()

    def stop_recording() -> None:
        nonlocal recording_enabled, record_started_at
        recording_enabled = False
        record_started_at = None

    if recording_enabled:
        start_recording()

    def refresh_view() -> None:
        nonlocal last_update, state, record_session
        rank_to_proc, _pid_errors = collect_rank_pids(state)
        candidate = best_selector_from_procs(rank_to_proc.values())
        if candidate and selector_score(candidate) > selector_score(state.selector):
            state = dataclasses.replace(state, selector=candidate)
        snapshots, _stack_errors = collect_stacks(
            state, rank_to_proc, pythonpath, show_threads, install_attempted
        )
        if recording_enabled and record_session is not None:
            record_session.record_if_changed(state, rank_to_proc, snapshots)
        stacks_text: Dict[int, Text] = {}
        stack_lines_by_rank = {
            rank: extract_stack_lines(snapshot.stack_lines)
            for rank, snapshot in snapshots.items()
        }
        prefix_len = common_prefix_length(stack_lines_by_rank)
        diff_index = None
        if any(stack_lines_by_rank.values()):
            diff_index = max(0, prefix_len - 1) if prefix_len > 0 else 0
        for rank, snapshot in snapshots.items():
            lines = snapshot.stack_lines
            marked = mark_diff_line(lines, diff_index) if diff_index is not None else lines
            stacks_text[rank] = style_lines(marked)
        details_by_rank = {
            rank: snapshot.details for rank, snapshot in snapshots.items()
        }
        last_update = time.strftime("%H:%M:%S")
        width, height = shutil.get_terminal_size((120, 40))
        content_width = max(0, width - 4)
        record_line = None
        if record_session is not None and recording_enabled:
            record_line = f"{record_session.log_path} | events {record_session.event_count} | {format_elapsed(record_started_at)}"
            record_line = shorten(record_line, max(10, content_width - 12))
        header, header_lines = build_live_header(
            state, last_update, refresh, record_line, content_width
        )
        header_height = header_lines + 2
        header_height = max(3, min(header_height, max(3, height - 1)))
        layout["header"].size = header_height
        body_height = max(1, height - header_height)
        total_columns = len(state.ranks) + (1 if show_details else 0)
        column_width = max(1, content_width // max(1, total_columns))
        inner_width = max(1, column_width - 4)
        details_text = (
            build_details_text(state.ranks, rank_to_proc, details_by_rank, inner_width)
            if show_details
            else None
        )
        layout["header"].update(
            Panel(header, padding=(0, 1), border_style=BORDER_STYLE)
        )
        layout["body"].update(
            render_columns(state.ranks, stacks_text, details_text, body_height, rank_to_proc)
        )

    try:
        refresh_view()
        next_refresh = time.time() + refresh
        with Live(layout, console=console, refresh_per_second=10, screen=True):
            while True:
                now = time.time()
                if now >= next_refresh:
                    refresh_view()
                    next_refresh = now + refresh

                key = read_key(0.1)
                if key is None:
                    continue
                if key == "q":
                    return 0
                if key == " ":
                    next_refresh = 0.0
                if key == "t":
                    show_threads = not show_threads
                    next_refresh = 0.0
                if key == "d":
                    show_details = not show_details
                    next_refresh = 0.0
                if key == "r":
                    if recording_enabled:
                        stop_recording()
                    else:
                        start_recording()
                    next_refresh = 0.0
    except KeyboardInterrupt:
        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if record_session is not None:
            record_session.close()
            if record_session.event_count > 0:
                print(f"Recording saved to: {record_session.log_path}")

    return 0


def run_record_batch(args: argparse.Namespace) -> int:
    pythonpath = args.pythonpath if args.pythonpath is not None else os.environ.get("PYTHONPATH", "")
    state = detect_state(args)
    refresh = max(1, args.refresh)
    record_path = args.out or default_session_path()
    record_session = RecordSession(record_path, state, refresh, pythonpath)
    quiet = bool(args.quiet)
    install_attempted: set = set()
    start_time = time.time()
    last_change: Optional[float] = None
    last_heartbeat = start_time
    last_divergence_time = 0.0
    stop_reason = "completed"

    target = state.selector.display or "python"
    target = shorten(target, 120)
    print(
        f"recording start | path={record_session.log_path} | ranks={len(state.ranks)} | "
        f"refresh={refresh}s | target={target}"
    )

    try:
        while True:
            loop_start = time.time()
            if not is_pid_alive(state.prte_pid):
                stop_reason = "prterun-exited"
                break
            rank_to_proc, _pid_errors = collect_rank_pids(state)
            snapshots, _stack_errors = collect_stacks(
                state, rank_to_proc, pythonpath, False, install_attempted
            )
            if record_session.record_if_changed(state, rank_to_proc, snapshots):
                last_change = time.time()
            divergence, common_len, max_len = compute_divergence_from_snapshots(state.ranks, snapshots)
            now = time.time()
            if not quiet and now - last_heartbeat >= HEARTBEAT_INTERVAL:
                last_change_age = "never"
                if last_change is not None:
                    last_change_age = format_duration(int(now - last_change))
                elapsed = format_duration(int(now - start_time))
                print(
                    f"heartbeat | events={record_session.event_count} | "
                    f"last_change={last_change_age} | elapsed={elapsed}"
                )
                last_heartbeat = now
            if (
                not quiet
                and divergence >= DIVERGENCE_THRESHOLD
                and now - last_divergence_time >= DIVERGENCE_INTERVAL
            ):
                print(
                    f"divergence | ratio={divergence:.2f} | common={common_len} | max={max_len}"
                )
                last_divergence_time = now
            elapsed = time.time() - loop_start
            sleep_for = refresh - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        record_session.close()
        elapsed = format_duration(int(time.time() - start_time))
        print(
            f"recording stop | reason={stop_reason} | events={record_session.event_count} | "
            f"elapsed={elapsed} | path={record_session.log_path}"
        )

    return 0


def run_review(args: argparse.Namespace) -> int:
    metadata = load_session_metadata(args.path)
    ranks = [
        RankInfo(rank=int(item["rank"]), host=str(item["host"]))
        for item in metadata.get("ranks", [])
        if "rank" in item and "host" in item
    ]
    if not ranks:
        raise SystemExit("no ranks found in metadata")
    selector_payload = metadata.get("selector", {}) if isinstance(metadata.get("selector"), dict) else {}
    selector = ProgramSelector(
        module=selector_payload.get("module"),
        script=selector_payload.get("script"),
        display=selector_payload.get("display", ""),
    )
    state = State(
        prte_pid=int(metadata.get("prte_pid", 0) or 0),
        rankfile=str(metadata.get("rankfile", "")),
        ranks=ranks,
        selector=selector,
    )
    events = load_session_events(args.path)
    if not events:
        raise SystemExit("no events recorded")

    console = Console()
    show_threads = False
    show_details = False
    levels = [TimelineLevel(0, len(events), selected=0)]
    max_stack_lens, divergence_ratios, _ = compute_event_metrics(
        events, ranks, show_threads
    )

    def handle_sigint(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    layout = Layout()
    layout.split_column(Layout(name="header", size=HEADER_HEIGHT), Layout(name="body"))

    def refresh_view() -> None:
        width, height = shutil.get_terminal_size((120, 40))
        content_width = max(0, width - 4)
        timeline_lines = render_timeline_lines(levels, max_stack_lens, divergence_ratios, content_width)
        active_level = levels[-1]
        if not active_level.buckets:
            return
        current_index = active_level.buckets[active_level.selected][0]
        current_index = max(0, min(current_index, len(events) - 1))
        event = events[current_index]
        snapshots = event_snapshots_from_event(event, ranks, show_threads)
        rank_to_proc = rank_to_proc_from_event(event, ranks)
        stack_lines_by_rank = {
            rank: extract_stack_lines(snapshot.stack_lines)
            for rank, snapshot in snapshots.items()
        }
        prefix_len = common_prefix_length(stack_lines_by_rank)
        diff_index = None
        if any(stack_lines_by_rank.values()):
            diff_index = max(0, prefix_len - 1) if prefix_len > 0 else 0
        stacks_text: Dict[int, Text] = {}
        for rank, snapshot in snapshots.items():
            lines = snapshot.stack_lines
            marked = mark_diff_line(lines, diff_index) if diff_index is not None else lines
            stacks_text[rank] = style_lines(marked)
        details_by_rank = {
            rank: snapshot.details for rank, snapshot in snapshots.items()
        }
        event_time = iso_timestamp(event.timestamp)
        header, header_lines = build_review_header(
            state,
            current_index,
            len(events),
            event_time,
            timeline_lines,
            content_width,
        )
        header_height = header_lines + 2
        header_height = max(3, min(header_height, max(3, height - 1)))
        layout["header"].size = header_height
        body_height = max(1, height - header_height)
        total_columns = len(ranks) + (1 if show_details else 0)
        column_width = max(1, content_width // max(1, total_columns))
        inner_width = max(1, column_width - 4)
        details_text = (
            build_details_text(ranks, rank_to_proc, details_by_rank, inner_width)
            if show_details
            else None
        )
        layout["header"].update(
            Panel(header, padding=(0, 1), border_style=BORDER_STYLE)
        )
        layout["body"].update(
            render_columns(ranks, stacks_text, details_text, body_height, rank_to_proc)
        )

    try:
        refresh_view()
        with Live(layout, console=console, refresh_per_second=10, screen=True):
            while True:
                key = read_key(0.1)
                if key is None:
                    continue
                if key == "q":
                    return 0
                if key == "t":
                    show_threads = not show_threads
                    max_stack_lens, divergence_ratios, _ = compute_event_metrics(
                        events, ranks, show_threads
                    )
                    refresh_view()
                if key == "d":
                    show_details = not show_details
                    refresh_view()
                if key == "left":
                    level = levels[-1]
                    level.selected = max(0, level.selected - 1)
                    refresh_view()
                if key == "right":
                    level = levels[-1]
                    level.selected = min(max(0, len(level.buckets) - 1), level.selected + 1)
                    refresh_view()
                if key == "down":
                    level = levels[-1]
                    if not level.buckets:
                        continue
                    bucket = level.buckets[level.selected]
                    if bucket[1] - bucket[0] <= 1:
                        continue
                    levels.append(TimelineLevel(bucket[0], bucket[1], selected=0))
                    refresh_view()
                if key == "up":
                    if len(levels) > 1:
                        levels.pop()
                        refresh_view()
    except KeyboardInterrupt:
        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return 0


def run_summarize(args: argparse.Namespace) -> int:
    metadata = load_session_metadata(args.path)
    events = load_session_events(args.path)
    ranks = [
        RankInfo(rank=int(item["rank"]), host=str(item["host"]))
        for item in metadata.get("ranks", [])
        if "rank" in item and "host" in item
    ]
    if not ranks:
        raise SystemExit("no ranks found in metadata")
    if not events:
        raise SystemExit("no events recorded")

    rank_order = [info.rank for info in ranks]
    signature_counts: Dict[Tuple[str, ...], int] = {}
    signature_examples: Dict[Tuple[str, ...], Dict[int, str]] = {}
    rank_change_counts: Dict[int, int] = {rank: 0 for rank in rank_order}
    previous_rank_signature: Dict[int, str] = {rank: "" for rank in rank_order}
    max_stack_lens, divergence_ratios, common_prefixes = compute_event_metrics(
        events, ranks, show_threads=False
    )

    for event in events:
        per_rank_signature: Dict[int, str] = {}
        per_rank_top_frame: Dict[int, str] = {}
        for info in ranks:
            payload = event.ranks.get(info.rank, {})
            if payload.get("error"):
                signature = f"error:{payload.get('error')}"
                top_frame = signature
            else:
                output = payload.get("py_spy")
                if output:
                    lines, _details = render_pyspy_output(str(output), show_threads=False)
                    stack_lines = extract_stack_lines(lines)
                    signature = hashlib.sha1(
                        "\n".join(stack_lines).encode("utf-8", errors="ignore")
                    ).hexdigest()
                    top_frame = stack_lines[0].strip() if stack_lines else "empty"
                else:
                    signature = "empty"
                    top_frame = "empty"
            per_rank_signature[info.rank] = signature
            per_rank_top_frame[info.rank] = top_frame

        for rank, signature in per_rank_signature.items():
            if previous_rank_signature.get(rank) != signature:
                rank_change_counts[rank] = rank_change_counts.get(rank, 0) + 1
            previous_rank_signature[rank] = signature

        signature_key = tuple(per_rank_signature[rank] for rank in rank_order)
        signature_counts[signature_key] = signature_counts.get(signature_key, 0) + 1
        if signature_key not in signature_examples:
            signature_examples[signature_key] = per_rank_top_frame

    sorted_signatures = sorted(
        signature_counts.items(), key=lambda item: item[1], reverse=True
    )
    top_signatures = sorted_signatures[: max(1, args.top)]
    total_events = len(events)
    start_time = iso_timestamp(events[0].timestamp)
    end_time = iso_timestamp(events[-1].timestamp)

    if args.format == "json":
        payload = {
            "metadata": metadata,
            "event_count": total_events,
            "time_range": {"start": start_time, "end": end_time},
            "rank_change_counts": rank_change_counts,
            "top_signatures": [
                {
                    "count": count,
                    "ratio": count / float(total_events),
                    "example_top_frames": signature_examples.get(signature_key, {}),
                }
                for signature_key, count in top_signatures
            ],
            "most_divergent": sorted(
                [
                    {
                        "index": idx,
                        "timestamp": iso_timestamp(events[idx].timestamp),
                        "divergence_ratio": divergence_ratios[idx],
                        "common_prefix_len": common_prefixes[idx],
                        "max_stack_len": max_stack_lens[idx],
                    }
                    for idx in range(total_events)
                ],
                key=lambda item: item["divergence_ratio"],
                reverse=True,
            )[:5],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Session: {args.path}")
    print(f"Events: {total_events} ({start_time} -> {end_time})")
    print(f"Ranks: {', '.join(str(rank) for rank in rank_order)}")
    print("")
    print("Top stack signatures:")
    for idx, (signature_key, count) in enumerate(top_signatures, start=1):
        ratio = count / float(total_events)
        print(f"{idx}. {count} events ({ratio:.1%})")
        example = signature_examples.get(signature_key, {})
        for rank in rank_order:
            frame = example.get(rank, "")
            frame = shorten(frame, 120)
            print(f"   rank {rank}: {frame}")
    print("")
    print("Rank change counts:")
    for rank in rank_order:
        print(f"  rank {rank}: {rank_change_counts.get(rank, 0)}")
    print("")
    print("Most divergent events:")
    divergent = sorted(
        range(total_events),
        key=lambda idx: divergence_ratios[idx],
        reverse=True,
    )[:5]
    for idx in divergent:
        print(
            f"  #{idx + 1} @ {iso_timestamp(events[idx].timestamp)} | "
            f"ratio {divergence_ratios[idx]:.2f} | "
            f"common {common_prefixes[idx]} | "
            f"max {max_stack_lens[idx]}"
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] in {"review", "summarize", "record"}:
        command = argv[0]
        sub_args = argv[1:]
        if command == "review":
            return run_review(parse_review_args(sub_args))
        if command == "record":
            return run_record_batch(parse_record_args(sub_args))
        return run_summarize(parse_summarize_args(sub_args))
    return run_live(parse_live_args(argv))


def select_with_timeout(timeout: float):
    import select

    try:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
    except ValueError:
        return []
    return readable


if __name__ == "__main__":
    raise SystemExit(main())
