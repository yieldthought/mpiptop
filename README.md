<p align="center">
  <img src="https://raw.githubusercontent.com/yieldthought/mpiptop/main/screenshot.png" alt="mpiptop screenshot" width="920">
</p>
<p align="center">
  <strong>mpiptop</strong> - live, side-by-side Python stack traces across MPI ranks.
</p>
<p align="center">
  A focused TUI that makes distributed debugging feel calm and fast.
</p>

## Why it helps
- Auto-detects active `mpirun/prterun` and its rankfile
- One column per rank with subtle diff highlights
- Toggle main thread vs all threads, and a details panel
- Works over passwordless SSH; propagates venv env vars

## Install
```bash
pip install mpiptop
```

## Usage
```bash
mpiptop
```

Common options:
```bash
mpiptop --rankfile /etc/mpirun/rankfile_01_02
mpiptop --prterun-pid 12345
mpiptop --refresh 5
mpiptop --pythonpath /path/to/your/code
```

Controls: `q` quit | `space` refresh | `t` threads | `d` details
