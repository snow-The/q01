#!/usr/bin/env python3
"""CPU 優化原型 v2：修正正確性檢查（bit-reversal）＋交錯量測降低雜訊。"""
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from q01.circuit import Op  # noqa: E402
from q01.simulator import Simulator  # noqa: E402


def ops_for(n, depth=2):
    ops = [Op("ry", (0.3 + 0.1 * i,), (i,), ()) for i in range(n)]
    for d in range(depth):
        ops += [Op("rz", (0.2 + 0.05 * i,), (i,), ()) for i in range(n)]
        step = 1 if d == 0 else 2
        ops += [Op("x", (), ((i + step) % n,), (i,)) for i in range(n)]
    return ops


def ry_mat(th):
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_mat(th):
    return np.array([[np.exp(-1j * th / 2), 0], [0, np.exp(1j * th / 2)]], dtype=np.complex128)


def perm_le_to_be(n):
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(1 << n)])


def run_A(n):
    sim = Simulator(n)
    for op in ops_for(n):
        sim.apply(op.name, op.params, op.qubits, op.controls)
    return sim.probabilities()


def apply1_le(psi, u, t):
    v = psi.reshape(-1, 2, 1 << t)
    a = v[:, 0, :].copy(); b = v[:, 1, :].copy()
    v[:, 0, :] = u[0, 0] * a + u[0, 1] * b
    v[:, 1, :] = u[1, 0] * a + u[1, 1] * b


def apply_cx_le(psi, c, t, n):
    mv = np.moveaxis(psi.reshape((2,) * n), (c, t), (0, 1))
    sub = mv[1]
    sub[...] = sub[::-1].copy()


def run_B(n):
    psi = np.zeros(1 << n, dtype=np.complex128); psi[0] = 1.0
    for op in ops_for(n):
        if op.controls:
            apply_cx_le(psi, op.controls[0], op.qubits[0], n)
        else:
            u = ry_mat(op.params[0]) if op.name == "ry" else rz_mat(op.params[0])
            apply1_le(psi, u, op.qubits[0])
    return np.abs(psi) ** 2


def run_C(n, pool, nthreads=8):
    psi = np.zeros(1 << n, dtype=np.complex128); psi[0] = 1.0
    for op in ops_for(n):
        if op.controls:
            apply_cx_le(psi, op.controls[0], op.qubits[0], n)
        else:
            u = ry_mat(op.params[0]) if op.name == "ry" else rz_mat(op.params[0])
            t = op.qubits[0]
            v = psi.reshape(-1, 2, 1 << t)
            outer = v.shape[0]
            step = max(1, outer // nthreads)
            chunks = [(i, min(i + step, outer)) for i in range(0, outer, step)]

            def work(ab, v=v, u=u):
                lo, hi = ab
                blk = v[lo:hi]
                a = blk[:, 0, :].copy(); b = blk[:, 1, :].copy()
                blk[:, 0, :] = u[0, 0] * a + u[0, 1] * b
                blk[:, 1, :] = u[1, 0] * a + u[1, 1] * b

            list(pool.map(work, chunks))
    return np.abs(psi) ** 2


# ---- 正確性（先驗，否則時間沒有意義）----
print("== 正確性：B/C 的 little-endian 結果經 bit-reversal 後與 A 比 ==")
ok = True
for n in (5, 10, 12):
    pa = run_A(n)
    rev = perm_le_to_big = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(1 << n)])
    pb = run_B(n)[rev]
    pc = run_C(n, ThreadPoolExecutor(max_workers=8))[rev]
    d1 = float(np.max(np.abs(pa - pb))); d2 = float(np.max(np.abs(pa - pc)))
    ok = ok and d1 < 1e-12 and d2 < 1e-12
    print(f"  n={n:2d}  |A-B| = {d1:.3e}   |A-C| = {d2:.3e}")
print("  =>", "一致" if ok else "★ 不一致，時間數據不可用")

pool = ThreadPoolExecutor(max_workers=8)
print()
print("== 效能（每個 n 交錯量測，取 min-of-5）==")
print("   n        A 現行    B little-e   C 8執行緒    B/A    C/A")
for n in (5, 10, 14, 16, 18, 20):
    ta, tb, tc = [], [], []
    for _ in range(5):
        t0 = time.perf_counter(); run_A(n); ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); run_B(n); tb.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); run_C(n, pool); tc.append(time.perf_counter() - t0)
    a, b, c = min(ta) * 1e3, min(tb) * 1e3, min(tc) * 1e3
    print("  %3d %11.3f %11.3f %12.3f %6.2fx %6.2fx" % (n, a, b, c, a / b, a / c))