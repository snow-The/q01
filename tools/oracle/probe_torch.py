#!/usr/bin/env python3
"""PyTorch GPU／dtype 能力探針：通用 GPU 路線可行嗎？精度代價多少？"""
import sys
import time

import numpy as np
import torch

print("=" * 78)
print("A. 這個 torch build 能碰到什麼")
print("=" * 78)
print("torch", torch.__version__, "| python", sys.version.split()[0])
print("  cuda.is_available      :", torch.cuda.is_available())
print("  cuda.device_count      :", torch.cuda.device_count())
for attr, label in (("mps", "Apple Metal (MPS)"), ("xpu", "Intel XPU"), ("mkldnn", "oneDNN/MKL-DNN CPU")):
    try:
        mod = getattr(torch.backends, attr, None)
        print(f"  backends.{attr:8s} ({label}):", getattr(mod, "is_available", lambda: "n/a")() if mod else "無此後端")
    except Exception as e:
        print(f"  backends.{attr}: ERR {type(e).__name__}")
try:
    print("  可用裝置清單           :", torch._C._get_available_devices())
except Exception:
    pass
print("  編譯時是否含 CUDA       :", torch.version.cuda)
print("  是否為 ROCm build       :", getattr(torch.version, "hip", None))

print()
print("=" * 78)
print("B. dtype 支援（在 CPU 上測；GPU 上 MPS/XPU 的 complex 支援較窄）")
print("=" * 78)
for name, dt in (("complex64", torch.complex64), ("complex128", torch.complex128),
                 ("float32", torch.float32), ("float64", torch.float64)):
    try:
        t = torch.zeros(4, dtype=dt)
        r = torch.einsum("ij,jk->ik", t.reshape(2, 2).to(dt), t.reshape(2, 2).to(dt))
        print(f"  {name:10s} 可用 ✓   einsum ok, dtype={r.dtype}")
    except Exception as e:
        print(f"  {name:10s} ✗ {type(e).__name__}: {str(e)[:60]}")

# ---------------------------------------------------------- 精度：真實代價
print()
print("=" * 78)
print("C. 精度代價：同一顆 5 qubit QBN 電路，complex64 vs complex128")
print("=" * 78)
N, DEPTH = 5, 2
RING1 = [(i, (i + 1) % N) for i in range(N)]
RING2 = [(i, (i + 2) % N) for i in range(N)]
rng = np.random.default_rng(7)
x = rng.random(N)
w = rng.normal(0.0, 0.9, size=(DEPTH, N, 2))


def np_reference() -> np.ndarray:
    psi = np.zeros(2 ** N, dtype=np.complex128)
    psi[0] = 1.0
    def apply_1q(mat, k):
        nonlocal psi
        t = psi.reshape((2,) * N)
        t = np.moveaxis(t, k, 0)
        t = np.tensordot(mat, t, axes=([1], [0]))
        t = np.moveaxis(t, 0, k)
        psi = np.ascontiguousarray(t).reshape(-1)
    def ry(th, k):
        c, s = np.cos(th / 2), np.sin(th / 2)
        apply_1q(np.array([[c, -s], [s, c]], dtype=complex), k)
    def rz(th, k):
        apply_1q(np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)]), k)
    for i in range(N):
        ry(x[i], i)
    for d in range(DEPTH):
        for i in range(N):
            ry(w[d, i, 0], i)
            rz(w[d, i, 1], i)
        for c, t in (RING1 if d == 0 else RING2):
            # CX：半片交換（big-endian：控制位 c 的 bit）
            t3 = psi.reshape((2,) * N)
            ctrl = (np.arange(2 ** N) >> (N - 1 - c)) & 1
            tgt = (np.arange(2 ** N) >> (N - 1 - t)) & 1
            lo = np.nonzero((ctrl == 1) & (tgt == 0))[0]
            hi = np.nonzero((ctrl == 1) & (tgt == 1))[0]
            psi[lo], psi[hi] = psi[hi].copy(), psi[lo].copy()
    return np.abs(psi) ** 2


def torch_probs(dtype):
    psi = torch.zeros(2 ** N, dtype=dtype)
    psi[0] = 1.0
    def apply_1q(mat, k):
        nonlocal psi
        t = psi.reshape((2,) * N)
        t = torch.movedim(t, k, 0)
        t = torch.tensordot(mat, t, dims=([1], [0]))
        t = torch.movedim(t, 0, k)
        psi = t.reshape(-1).contiguous()
    def ry(th, k):
        c, s = np.cos(th / 2), np.sin(th / 2)
        apply_1q(torch.tensor([[c, -s], [s, c]], dtype=dtype), k)
    def rz(th, k):
        apply_1q(torch.tensor(np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)]), dtype=dtype), k)
    for i in range(N):
        ry(x[i], i)
    for d in range(DEPTH):
        for i in range(N):
            ry(w[d, i, 0], i)
            rz(w[d, i, 1], i)
        for c, t in (RING1 if d == 0 else RING2):
            idx = torch.arange(2 ** N)
            ctrl = (idx >> (N - 1 - c)) & 1
            tgt = (idx >> (N - 1 - t)) & 1
            lo = torch.nonzero((ctrl == 1) & (tgt == 0)).flatten()
            hi = torch.nonzero((ctrl == 1) & (tgt == 1)).flatten()
            tmp = psi[lo].clone()
            psi[lo] = psi[hi]
            psi[hi] = tmp
    return (psi.abs() ** 2).numpy()


ref = np_reference()
for dt_name, dt in (("complex64", torch.complex64), ("complex128", torch.complex128)):
    p = torch_probs(dt)
    print(f"  {dt_name:10s} max|ΔP| vs numpy complex128 = {np.max(np.abs(p - ref)):.3e}")
print(f"  （判準 1e-10；complex64 是 ~1e-7 相對精度，累積後通常在 1e-6~1e-5 級）")

print()
print("=" * 78)
print("D. autograd：量子層參數的梯度（QML 訓練要用的）")
print("=" * 78)


def loss(theta_t):
    psi = torch.zeros(2, dtype=torch.complex128)
    psi[0] = 1.0
    c, s = torch.cos(theta_t / 2), torch.sin(theta_t / 2)
    u = torch.stack([torch.stack([c + 0j, -s + 0j]), torch.stack([s + 0j, c + 0j])])
    psi2 = u @ psi
    z = torch.tensor([1.0 + 0j, -1.0 + 0j], dtype=torch.complex128)
    # <Z> = <psi|Z|psi>
    val = torch.vdot(psi2, z * psi2)
    return val.real


theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
L = loss(theta)
L.backward()
print(f"  loss(0.7) = {L.item():.15f}   解析 cos(0.7) = {np.cos(0.7):.15f}")
print(f"  梯度      = {theta.grad.item():.15f}   解析 -sin(0.7) = {-np.sin(0.7):.15f}")
print(f"  誤差      = {abs(theta.grad.item() + np.sin(0.7)):.3e}")
print("  （PyTorch 對 complex 的 autograd 採 Wirtinger 導數；實數參數 + 實數損失這條路可用）")

print()
print("=" * 78)
print("E. CPU 速度：torch vs numpy（同一顆電路，暖機後 best-of-100）")
print("=" * 78)


def timeit(fn, reps=100):
    for _ in range(10):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


print(f"  numpy  complex128 : {timeit(np_reference):8.3f} ms")
print(f"  torch  complex128 : {timeit(lambda: torch_probs(torch.complex128)):8.3f} ms")
print(f"  torch  complex64  : {timeit(lambda: torch_probs(torch.complex64)):8.3f} ms")
print("  （CPU 上 torch 通常沒有優勢；它的價值在 (a) 真 autograd (b) 換裝置就能吃 GPU）")
