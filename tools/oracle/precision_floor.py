#!/usr/bin/env python3
"""精度底線稽核：我們的 fp64 離「精確解」還有多遠？DD 在 numpy 裡划不划算？

參考解用 mpmath 60 位有效數字獨立重寫同一顆電路（也是獨立實作，順便再驗一次結構）。
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from q01.circuit import Op  # noqa: E402
from q01.simulator import Simulator  # noqa: E402

N = 5
X = [0.3, 0.4, 0.5, 0.6, 0.7]
P = [0.1, -0.2, 0.3, -0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def ops_for():
    ops = [Op("ry", (X[i],), (i,), ()) for i in range(N)]
    ops += [Op("x", (), ((i + 1) % N,), (i,)) for i in range(N)]
    ops += [Op("ry", (P[i],), (i,), ()) for i in range(N)]
    ops += [Op("rz", (P[5 + i],), (i,), ()) for i in range(N)]
    return ops


def fp64_probs():
    sim = Simulator(N)
    for op in ops_for():
        sim.apply(op.name, op.params, op.qubits, op.controls)
    return sim.probabilities()


def mp_reference(dps=60):
    """mpmath 獨立重寫（60 位有效數字）。"""
    from mpmath import mp, mpf, mpc, cos, sin, exp, sqrt
    mp.dps = dps
    dim = 1 << N
    psi = [mpc(0) for _ in range(dim)]
    psi[0] = mpc(1)

    def bit(idx, k):                      # big-endian：qubit k 的權重 2^(N-1-k)
        return (idx >> (N - 1 - k)) & 1

    def apply1(u, k):
        nonlocal psi
        w = 1 << (N - 1 - k)
        new = list(psi)
        for i in range(dim):
            if bit(i, k):
                continue
            j = i | w
            a, b = psi[i], psi[j]
            new[i] = u[0][0] * a + u[0][1] * b
            new[j] = u[1][0] * a + u[1][1] * b
        psi = new

    for op in ops_for():
        if op.name == "x":
            u = [[mpc(0), mpc(1)], [mpc(1), mpc(0)]]
        elif op.name == "ry":
            t = mpf(op.params[0]) / 2
            c, s = cos(t), sin(t)
            u = [[c, -s], [s, c]]
        elif op.name == "rz":
            t = mpf(op.params[0]) / 2
            u = [[exp(-1j * t), mpc(0)], [mpc(0), exp(1j * t)]]
        else:
            raise KeyError(op.name)
        if op.controls:
            ctrl, tgt = op.controls[0], op.qubits[0]
            wc = 1 << (N - 1 - ctrl)
            wt = 1 << (N - 1 - tgt)
            new = list(psi)
            for i in range(dim):
                if not (i & wc) or (i & wt):
                    continue
                j = i | wt
                a, b = psi[i], psi[j]
                new[i] = u[0][0] * a + u[0][1] * b
                new[j] = u[1][0] * a + u[1][1] * b
            psi = new
        else:
            apply1(u, op.qubits[0])
    return np.array([float(abs(z) ** 2) for z in psi], dtype=float)


print("numpy", np.__version__)
print("has np.fma        :", hasattr(np, "fma"), "（向量化 FMA 是 DD 的前提）")
try:
    import math
    print("has math.fma      :", hasattr(math, "fma"), "（純量；無法向量化）")
except Exception as e:
    print("math.fma ?", e)
try:
    import mpmath
    print("mpmath            :", mpmath.__version__)
except ImportError:
    print("mpmath            : 未安裝")
    raise SystemExit(1)

p64 = fp64_probs()
print()
print("計算 mpmath 60 位參考解（獨立重寫）…")
t0 = time.perf_counter()
pref = mp_reference(60)
tref = time.perf_counter() - t0
print("  完成，耗時 %.2f s" % tref)

eps = np.finfo(float).eps
print()
print("=" * 78)
print("A. 我們的 fp64 離精確解多遠？")
print("=" * 78)
d = np.abs(p64 - pref)
print("  max|P_fp64 - P_exact| = %.3e" % d.max())
print("  sum(P_fp64) = %.17f    (精確 = 1)" % p64.sum())
print("  雙精度 eps  = %.3e" % eps)
print("  => 我們的誤差 = %.1f x eps" % (d.max() / eps))
print("  電路規模：20 個閘、32 個振幅；每個 RY/RZ 要算 2 個超越函數")
print("  理論量級（√K 隨機遊走模型，K≈有效運算數）: %.1f–%.1f x eps" % (2, 20))

print()
print("=" * 78)
print("B. 「更準」能買到什麼？（本專案的實際敏感度）")
print("=" * 78)
print("  我們要報的數字是準確率／NLL／ECE，由**有限樣本**決定：")
print("  - 5 個種子、檢定力 0.275（專案已知弱點）-> 統計不確定性 ~1e-2 級")
print("  - 5 qubit 電路本身捨入 ~1e-16 級")
print("  => 兩者差 14 個數量級；把模擬做到 1e-32 對結論沒有任何影響")

print()
print("=" * 78)
print("C. 真正吃到精度的三個地方（值得用 EFT／補償技術）")
print("=" * 78)
print("  1. 去相位消融：要區分 1.4e-17（≈0）與 0.0502 —— 這是**相消敏感**的判定")
print("  2. ECE 恆等式：專案實測誤差 2.78e-17，要主張「恆等」需要誤差界")
print("  3. sum(P) = 1：補償求和可讓它精確到 eps²")
print("  這三處該做的是 Kahan/Neumaier 補償求和 + DD 參考解，不是把整個模擬器換成 DD。")