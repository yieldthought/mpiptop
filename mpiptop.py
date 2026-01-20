#!/usr/bin/env python3
"""mpiptop: visualize MPI python stacks across hosts using py-spy."""

from __future__ import annotations

import argparse
import colorsys
import dataclasses
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


PUNCT_STYLE = "grey62"
BORDER_STYLE = "grey62"
KEY_STYLE = "#7ad7ff"
HEADER_HEIGHT = 3
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
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], List[str]]:
    stacks: Dict[int, List[str]] = {}
    details_by_rank: Dict[int, List[str]] = {}
    errors: List[str] = []
    for entry in state.ranks:
        proc = rank_to_proc.get(entry.rank)
        if proc is None:
            stacks[entry.rank] = ["No process"]
            details_by_rank[entry.rank] = []
            continue
        output, error = run_py_spy(entry.host, proc, pythonpath, install_attempted)
        if error:
            errors.append(error)
            stacks[entry.rank] = [error]
            details_by_rank[entry.rank] = []
            continue
        lines, details = render_pyspy_output(output or "", show_threads)
        stacks[entry.rank] = lines
        details_by_rank[entry.rank] = details
    return stacks, details_by_rank, errors


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show MPI Python stacks across hosts.")
    parser.add_argument("--rankfile", help="Override rankfile path")
    parser.add_argument("--prterun-pid", type=int, help="PID of prterun/mpirun")
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval (seconds)")
    parser.add_argument(
        "--pythonpath",
        help="PYTHONPATH to export remotely (defaults to local PYTHONPATH)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pythonpath = args.pythonpath if args.pythonpath is not None else os.environ.get("PYTHONPATH", "")

    state = detect_state(args)
    console = Console()
    refresh = max(1, args.refresh)
    show_threads = False
    show_details = False
    install_attempted: set = set()

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

    def refresh_view() -> None:
        nonlocal last_update, state
        rank_to_proc, pid_errors = collect_rank_pids(state)
        candidate = best_selector_from_procs(rank_to_proc.values())
        if candidate and selector_score(candidate) > selector_score(state.selector):
            state = dataclasses.replace(state, selector=candidate)
        stacks, details_by_rank, stack_errors = collect_stacks(
            state, rank_to_proc, pythonpath, show_threads, install_attempted
        )
        stacks_text: Dict[int, Text] = {}
        stack_lines_by_rank = {rank: extract_stack_lines(lines) for rank, lines in stacks.items()}
        prefix_len = common_prefix_length(stack_lines_by_rank)
        diff_index = None
        if any(stack_lines_by_rank.values()):
            if prefix_len > 0:
                diff_index = prefix_len - 1
            else:
                diff_index = 0
        for rank, lines in stacks.items():
            marked = mark_diff_line(lines, diff_index) if diff_index is not None else lines
            stacks_text[rank] = style_lines(marked)
        errors = pid_errors + stack_errors
        last_update = time.strftime("%H:%M:%S")
        width, height = shutil.get_terminal_size((120, 40))
        content_width = max(0, width - 4)
        header, header_lines = build_header(state, last_update, errors, refresh, content_width)
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

                if sys.stdin in select_with_timeout(0.1):
                    key = sys.stdin.read(1)
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
    except KeyboardInterrupt:
        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return 0


def select_with_timeout(timeout: float):
    import select

    try:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
    except ValueError:
        return []
    return readable


if __name__ == "__main__":
    raise SystemExit(main())
