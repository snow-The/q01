#!/usr/bin/env python3
"""CPU vs GPU（RTX 3060 Laptop）在 CUDA-Q 上的對照：時間 + 精度。

- qpp-cpu        ：CPU 狀態向量（fp64）
- nvidia         ：GPU 狀態向量（cuStateVec，預設 fp32）
- nvidia-fp64    ：GPU 狀態向量（fp64）
同一顆 H^n + 環形 CX 電路；精度對 qpp-cpu 的 fp64 結果比對。
"""
import sys
import time

import numpy as np
import cudaq

TMP = tempfile.gettempdir()
sys.path.insert(0, TMP)
print("CUDA-Q", cudaq.__version__.split("(")[0].strip(), "| gpus =", cudaq.num_available_gpus())
print()


def make_kernel(n):
    src = ("import cudaq\n\n"
           "@cudaq.kernel\ndef uni_{n}():\n"
           "    q = cudaq.qvector({n})\n"
           "    for i in range({n}):\n        h(q[i])\n"
           "    for i in range({n}):\n        cx(q[i], q[(i + 1) % {n}])\n").format(n=n)
    path = f"{TMP}/_g{n}.py"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    mod = __import__(f"_g{n}")
    return getattr(mod, f"uni_{n}")


def measure(kern, reps=5, warm=2):
    first = None
    for i in range(warm):
        t0 = time.perf_counter(); np.array(cudaq.get_state(kern)); dt = time.perf_counter() - t0
        if i == 0:
            first = dt
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); np.array(cudaq.get_state(kern)); ts.append(time.perf_counter() - t0)
    return first * 1e3, min(ts) * 1e3


NS_CPU = (5, 10, 14, 16, 18, 20, 22)
NS_GPU = (5, 10, 14, 16, 18, 20, 22, 24)
results = {}

for target, ns in (("qpp-cpu", NS_CPU), ("nvidia", NS_GPU), ("nvidia-fp64", NS_GPU)):
    try:
        cudaq.set_target(target)
        print(f"=== target = {target} ===")
    except Exception as exc:
        print(f"=== target = {target} 不可用：{type(exc).__name__}: {str(exc)[:80]}")
        continue
    for n in ns:
        kern = make_kernel(n)
        try:
            first, steady = measure(kern, reps=3 if n >= 20 else 5)
            p = np.abs(np.array(cudaq.get_state(kern))) ** 2
            results[(target, n)] = (first, steady, p)
            print(f"  n={n:2d}  首呼 {first:9.2f} ms  穩態 {steady:9.3f} ms  記憶體 {2**n*16/1e6:8.1f} MB(f64)")
        except Exception as exc:
            print(f"  n={n:2d}  失敗：{type(exc).__name__}: {str(exc)[:70]}")
    print()

# ---------------- 精度：GPU 對 CPU fp64 ----------------
print("=" * 78)
print("精度：GPU 結果對 qpp-cpu（fp64）的 max|ΔP|")
print("=" * 78)
for n in NS_CPU:
    ref = results.get(("qpp-cpu", n))
    if ref is None:
        continue
    line = f"  n={n:2d}"
    for target in ("nvidia", "nvidia-fp64"):
        got = results.get((target, n))
        if got is None:
            line += f"   {target}: -"
            continue
        d = float(np.max(np.abs(got[2] - ref[2])))
        line += f"   {target}: {d:.3e}"
    print(line)

# ---------------- 加速比 ----------------
print()
print("=" * 78)
print("穩態加速比（CPU 時間 / GPU 時間；>1 表示 GPU 較快）")
print("=" * 78)
for n in NS_CPU:
    ref = results.get(("qpp-cpu", n))
    if ref is None:
        continue
    line = f"  n={n:2d}  CPU {ref[1]:9.3f} ms"
    for target in ("nvidia", "nvidia-fp64"):
        got = results.get((target, n))
        if got:
            line += f"   |  {target}: {got[1]:8.3f} ms  x{ref[1]/got[1]:6.2f}"
    print(line)
