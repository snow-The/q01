#!/usr/bin/env python3
"""PyTorch GPU 原型 v2：能不能複刻 CUDA-Q 的 GPU 速度？

設計（與我們 numpy 核心的關鍵差別）：
  1. **little-endian 內部佈局**：qubit t 的 stride 恰為 2^t
     ⇒ 單閘 = view(-1, 2, 2^t) + matmul（零拷貝、零 moveaxis）
  2. 受控閘用 **permute 視圖 + flip/copy_**，不建任何索引張量（記憶體 O(狀態)）
  3. 全程留在 GPU，只在量測時同步一次
"""
import sys
import time

import numpy as np
import torch

DEV = "cuda"
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print()


def build_ops(n, dtype):
    ops = []
    for i in range(n):
        th = 0.3 + 0.1 * i
        c, s = np.cos(th / 2), np.sin(th / 2)
        ops.append(("1q", i, torch.tensor([[c, -s], [s, c]], dtype=dtype, device=DEV)))
    for d in range(2):
        step = 1 if d == 0 else 2
        for i in range(n):
            ph = np.exp(1j * (0.2 + 0.05 * i) / 2)
            ops.append(("1q", i, torch.tensor([[np.conj(ph), 0], [0, ph]], dtype=dtype, device=DEV)))
        for i in range(n):
            ops.append(("cx", i, (i + step) % n))
    return ops


def apply_1q(psi, t, u):
    v = psi.view(-1, 2, 1 << t)
    v[:, 0, :], v[:, 1, :] = (u[0, 0] * v[:, 0, :] + u[0, 1] * v[:, 1, :],
                              u[1, 0] * v[:, 0, :] + u[1, 1] * v[:, 1, :])


def apply_cx(psi, control, target, n):
    """little-endian：把 control 與 target 兩個軸排到最前面，取 control=1 後對 target 做 flip。"""
    others = [k for k in range(n) if k not in (control, target)]
    v = psi.view((2,) * n).permute([control, target] + others)
    sub = v[1]                       # 視圖：control = 1
    sub.copy_(sub.flip(0))           # 對 target 軸翻轉（基本索引 + 就地寫回）


def run(n, dtype, reps=5, warm=2):
    ops = build_ops(n, dtype)
    psi = torch.zeros(1 << n, dtype=dtype, device=DEV)
    psi[0] = 1.0
    torch.cuda.synchronize()

    def once():
        for kind, a, b in ops:
            if kind == "1q":
                apply_1q(psi, a, b)
            else:
                apply_cx(psi, a, b, n)
        return psi

    t0 = time.perf_counter(); once(); torch.cuda.synchronize()
    first = (time.perf_counter() - t0) * 1e3
    for _ in range(warm):
        once()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); once(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    p = (psi.abs() ** 2).cpu().numpy()
    return first, min(ts) * 1e3, p


CQ_GPU64 = {14: 3.686, 16: 3.536, 18: 6.108, 20: 14.294, 22: 45.489, 24: 144.602}
CQ_GPU32 = {14: 3.821, 16: 4.018, 18: 4.392, 20: 7.065, 22: 20.515, 24: 64.442}
CQ_CPU64 = {14: 57.357, 16: 190.825, 18: 493.625, 20: 2048.131, 22: 8481.835}

print("=" * 100)
print("PyTorch GPU（complex128，view 佈局）vs CUDA-Q GPU / CPU")
print("=" * 100)
print("   n   torch 首呼   torch 穩態   CUDA-Q fp64  CUDA-Q fp32   CUDA-Q CPU   torch/CQ64")
ref = {}
for n in (14, 16, 18, 20, 22, 24):
    try:
        first, steady, p = run(n, torch.complex128, reps=3 if n >= 22 else 5)
        ref[n] = p
        print("  %3d %11.2f %11.3f %12.3f %12.3f %12.1f   x%5.2f"
              % (n, first, steady, CQ_GPU64[n], CQ_GPU32[n], CQ_CPU64.get(n, float("nan")),
                 steady / CQ_GPU64[n]))
    except Exception as exc:
        print("  %3d  失敗：%s: %s" % (n, type(exc).__name__, str(exc)[:70]))

print()
print("=" * 100)
print("精度與記憶體：complex64 vs complex128（同一顆電路）")
print("=" * 100)
for n in (14, 20, 22):
    try:
        _, s32, p32 = run(n, torch.complex64, reps=2)
        d = float(np.max(np.abs(p32 - ref[n]))) if n in ref else float("nan")
        print("  n=%2d  complex64 %8.3f ms   max|ΔP| vs complex128 = %.3e" % (n, s32, d))
    except Exception as exc:
        print("  n=%2d  失敗：%s" % (n, str(exc)[:60]))

print()
print("=" * 100)
print("同步底線：D2H + sync（解釋 CUDA-Q 的 ~3 ms 地板）")
print("=" * 100)
for n in (5, 14, 22):
    psi = torch.zeros(1 << n, dtype=torch.complex128, device=DEV); psi[0] = 1
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = (psi.abs() ** 2).cpu().numpy()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    print("  n=%2d  D2H + sync = %.3f ms（狀態 %.2f MB）" % (n, min(ts) * 1e3, (1 << n) * 16 / 1e6))