"""驗證 dev/ 底下所有腳本都能執行（清理後的迴歸測試）。

用法（WSL 內，因為 cudaq 只在 WSL 有）：
    cd <repo>
    /root/qbn/.venv/bin/python dev/verify_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

DEV = Path(__file__).resolve().parent
PROJECT = DEV.parent
EXCLUDE = {"verify_all.py"}

SCRIPTS = sorted(p for p in DEV.rglob("*.py") if p.name not in EXCLUDE)
TIMEOUT = 900

rows: list[tuple[str, str, float, str]] = []
for path in SCRIPTS:
    rel = path.relative_to(PROJECT).as_posix()
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(PROJECT), capture_output=True, text=True,
            timeout=TIMEOUT, encoding="utf-8", errors="replace",
        )
        dt = time.perf_counter() - t0
        status = "OK" if proc.returncode == 0 else f"EXIT{proc.returncode}"
        out = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        dt = time.perf_counter() - t0
        status, out = "TIMEOUT", ""
    except Exception as exc:  # noqa: BLE001
        dt = time.perf_counter() - t0
        status, out = "ERROR", str(exc)

    tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or [""]
    rows.append((rel, status, dt, tail[0][:110]))

print("=" * 100)
print("dev/ 迴歸測試")
print("=" * 100)
for rel, status, dt, tail in rows:
    flag = "  OK  " if status == "OK" else " FAIL "
    print(f"[{flag}] {rel:44s} {dt:6.1f}s")
    print(f"         {tail}")

ok = sum(1 for _, s, _, _ in rows if s == "OK")
print(f"\n  {ok}/{len(rows)} 可執行")
bad = [(r, s) for r, s, _, _ in rows if s != "OK"]
if bad:
    print("\n  有問題：")
    for r, s in bad:
        print(f"    [{s}] {r}")
sys.exit(1 if bad else 0)
