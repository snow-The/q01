#!/usr/bin/env python3
"""FP32 vs FP64 的真實代價：SIMD、超越函數、以及我們的電路到底卡在哪裡。"""
import time

import numpy as np

print("=" * 78)
print("A. 這台機器與這個 numpy build 的 SIMD 能力")
print("=" * 78)
print("numpy", np.__version__)
try:
    from numpy._core import _multiarray_umath as mu
    feats = {k: v for k, v in mu.__cpu_features__.items() if v}
    print("  CPU features (True 者):", ", ".join(sorted(feats)))
    disp = getattr(mu, "__cpu_dispatch__", None)
    print("  dispatch 啟用的實作 :", disp)
except Exception as exc:
    print("  cpu features 讀不到:", type(exc).__name__, exc)
import subprocess
print("  CPU:", subprocess.run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
                               capture_output=True, text=True).stdout.strip())
print()
print(np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {}) if hasattr(np, "show_config") else "")


def best(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


print()
print("=" * 78)
print("B. 超越函數 vs 純乘法：FP32/FP64 的 SIMD 差距（1M 元素）")
print("=" * 78)
N = 1 << 20
rng = np.random.default_rng(0)
print(f"  {'dtype':9s} {'sin (ms)':>10s} {'sin GB/s':>10s} {'mul (ms)':>10s} {'mul GB/s':>10s} {'sin/mul':>8s}")
for dt in (np.float32, np.float64):
    a = rng.random(N).astype(dt)
    ts = best(lambda: np.sin(a))
    tm = best(lambda: a * dt(1.000001))
    bytes_moved = N * np.dtype(dt).itemsize
    print(f"  {np.dtype(dt).name:9s} {ts*1e3:10.3f} {bytes_moved/ts/1e9:10.2f} "
          f"{tm*1e3:10.3f} {bytes_moved/tm/1e9:10.2f} {ts/tm:8.2f}x")

print()
print("=" * 78)
print("C. 我們的電路：complex64 vs complex128 隨規模的差距")
print("=" * 78)
RING = lambda n, s: [(i, (i + s) % n) for i in range(n)]


def circuit(n, dtype, depth=2):
    psi = np.zeros(1 << n, dtype=dtype)
    psi[0] = 1
    def apply1(mat, k):
        nonlocal psi
        t = psi.reshape((2,) * n)
        t = np.moveaxis(t, k, 0)
        t = np.tensordot(mat, t, axes=([1], [0]))
        t = np.moveaxis(t, 0, k)
        psi = np.ascontiguousarray(t).reshape(-1)
    idx = np.arange(1 << n)
    for d in range(depth):
        for i in range(n):
            th = 0.3 + 0.1 * i
            c, s = np.cos(th / 2), np.sin(th / 2)
            apply1(np.array([[c, -s], [s, c]], dtype=dtype), i)
            ph = np.exp(1j * th / 2)
            apply1(np.array([[np.conj(ph), 0], [0, ph]], dtype=dtype), i)
        for c_, t_ in RING(n, 1 if d == 0 else 2):
            ctrl = (idx >> (n - 1 - c_)) & 1
            tgt = (idx >> (n - 1 - t_)) & 1
            lo = np.nonzero((ctrl == 1) & (tgt == 0))[0]
            hi = np.nonzero((ctrl == 1) & (tgt == 1))[0]
            psi[lo], psi[hi] = psi[hi].copy(), psi[lo].copy()
    return np.abs(psi) ** 2


print(f"  {'n':>3} {'complex64 (ms)':>15s} {'complex128 (ms)':>16s} {'比值 f64/f32':>13s} {'記憶體 f64':>11s}")
for n in (5, 10, 12, 14, 16, 18):
    t32 = best(lambda: circuit(n, np.complex64, depth=2), reps=10, warm=2)
    t64 = best(lambda: circuit(n, np.complex128, depth=2), reps=10, warm=2)
    print(f"  {n:3d} {t32*1e3:15.3f} {t64*1e3:16.3f} {t64/t32:13.2f}x {(1<<n)*16/1e6:10.2f} MB")

print()
print("=" * 78)
print("D. 5 qubit：0.5 ms 到底花在哪？（開銷 vs 算數）")
print("=" * 78)
n = 5
ops = 2 * n * 2 + 2 * n          # depth2：(RY+RZ)*n*2 + CX*2n = 40 個閘
t_full = best(lambda: circuit(n, np.complex128, depth=2), reps=20)


def only_overhead():
    psi = np.zeros(1 << n, dtype=np.complex128); psi[0] = 1
    for _ in range(ops):
        t = psi.reshape((2,) * n)
        t = np.moveaxis(t, 0, 0)
        t = np.tensordot(np.eye(2, dtype=complex), t, axes=([1], [0]))
        psi = np.ascontiguousarray(t).reshape(-1)
    return psi


t_ov = best(only_overhead, reps=20)


def only_cos():
    tot = 0.0
    for i in range(ops):
        tot += np.cos(0.3 + 0.01 * i)
    return tot


t_cos = best(only_cos, reps=20)
print(f"  完整電路（{ops} 個閘）        : {t_full*1e3:8.3f} ms")
print(f"  同樣次數的 tensordot（空閘）  : {t_ov*1e3:8.3f} ms  -> 佔 {t_ov/t_full*100:5.1f}%")
print(f"  同樣次數的純量 np.cos        : {t_cos*1e3:8.3f} ms  -> 佔 {t_cos/t_full*100:5.1f}%")
print("  註：n=5 只有 32 個振幅，SIMD 寬度（4 vs 8 lane）在這規模沒有意義。")
