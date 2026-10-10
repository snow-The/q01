"""SPEC §6.1／§6.2／§6.3 的延遲基準（可重跑）。

為什麼要有這一支：原本那三張表是由**臨時腳本**產生的，腳本沒有進 repo，
所以數字無法重現——閘融合（§6.7(8)）之後也沒辦法重測「同一條電路」。
這裡把電路定義寫成程式碼，讓數字與腳本一起進版控。

兩條電路（都走 kernel API，即包含追蹤與執行的完整使用者路徑）：

  qbn5   5 qubit、depth 2 環形：RY 編碼層，之後 2 層「每個 qubit 的 RY,RZ +
         環形 CX」。§6.1 與 §6.2 用的就是這條。

  h_ring n qubit 的 H^n + 一層環形 CX。§6.3 的規模對照用這條。

用法：
  python latency.py            # 跑全部
  python latency.py --quick    # 減少重複次數（煙霧測試用）
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import q01  # noqa: E402

NQ = 5
DEPTH = 2
_RNG = np.random.default_rng(20260920)
_ENC = [float(v) for v in _RNG.normal(0.0, 0.7, size=NQ)]
_W = [float(v) for v in _RNG.normal(0.0, 0.5, size=(DEPTH, NQ, 2)).reshape(-1)]


@q01.kernel
def qbn5(enc: list[float], w: list[float]):
    q = q01.qvector(5)
    for i in range(5):
        q01.ry(enc[i], q[i])
    for d in range(2):
        for i in range(5):
            q01.ry(w[d * 10 + i * 2 + 0], q[i])
            q01.rz(w[d * 10 + i * 2 + 1], q[i])
        for i in range(5):
            q01.cx(q[i], q[(i + 1) % 5])


@q01.kernel
def h_ring(n: int):
    q = q01.qvector(n)
    for i in range(n):
        q01.h(q[i])
    for i in range(n):
        q01.cx(q[i], q[(i + 1) % n])


# 三個層次，因為它們量的東西不同，而舊表的腳本沒進 repo、無法還原：
#   sim  = 直接驅動 Simulator 套閘（裸電路，不含 kernel 追蹤與取樣）
#   st   = 走完整 kernel 路徑取狀態向量（含追蹤）
#   smp  = 走完整 kernel 路徑 + 取樣 100 shots
def one_qbn5_sim():
    sim = q01.Simulator(NQ)
    for i in range(NQ):
        sim.apply("ry", (_ENC[i],), (i,))
    for d in range(DEPTH):
        for i in range(NQ):
            sim.apply("ry", (_W[d * 10 + i * 2 + 0],), (i,))
            sim.apply("rz", (_W[d * 10 + i * 2 + 1],), (i,))
        for i in range(NQ):
            sim.apply("x", (), ((i + 1) % NQ,), (i,))
    return sim.probabilities()


def one_qbn5_state():
    return q01.get_state(qbn5, _ENC, _W)


def one_qbn5():
    return q01.sample(qbn5, _ENC, _W, shots_count=100)


def one_h_ring(n):
    return q01.sample(h_ring, n, shots_count=100)


def time_once(fn, reps):
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    return xs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_lat, n_stab, n_scale = (30, 40, 6) if a.quick else (300, 200, 8)

    print("q01", q01.__version__, "| numpy", np.__version__)
    print()

    # ---- §6.1 首次呼叫 / 穩態 ----
    t0 = time.perf_counter()
    one_qbn5()
    first = (time.perf_counter() - t0) * 1e3          # 首次（含 q01 的暖機）
    print("=== §6.1 延遲（qbn5, %d 次）===" % n_lat)
    print("  首次呼叫（含暖機）        %8.3f ms" % first)
    for label, fn in (("核心 Simulator（裸電路）", one_qbn5_sim),
                      ("kernel 取狀態向量", one_qbn5_state),
                      ("kernel + 取樣 100 shots", one_qbn5)):
        fn()                                          # 暖機
        xs = time_once(fn, n_lat)
        print("  %-24s best %7.3f   median %7.3f ms"
              % (label, min(xs), statistics.median(xs)))
    print()

    # ---- §6.2 延遲穩定性 ----
    print("=== §6.2 延遲穩定性（%d 次連續呼叫）===" % n_stab)
    for label, fn in (("sim n=5", one_qbn5_sim), ("smp n=5", one_qbn5)):
        xs = time_once(fn, n_stab)
        xs_sorted = sorted(xs)
        p = lambda q: xs_sorted[min(len(xs_sorted) - 1, int(q * len(xs_sorted)))]
        over = sum(1 for v in xs if v > 5.0)
        print("  %-5s min %.3f  p50 %.3f  p90 %.3f  p99 %.3f  max %.3f  >5ms %d/%d"
              % (label, min(xs), statistics.median(xs), p(0.90), p(0.99), max(xs),
                 over, n_stab))
    print()

    # ---- §6.3 規模對照 ----
    print("=== §6.3 規模對照（H^n + 環形 CX）===")
    print("  %4s %12s %12s" % ("n", "min ms", "狀態向量 MB"))
    for n in (5, 8, 10, 12, 14, 16, 18, 20):
        if n > 18 and a.quick:
            continue
        try:
            one_h_ring(n)                       # 暖機
            xs = time_once(lambda: one_h_ring(n), n_scale)
            print("  %4d %12.3f %12.2f" % (n, min(xs), (1 << n) * 16 / 1e6), flush=True)
        except Exception as exc:                # pragma: no cover
            print("  %4d  失敗：%s" % (n, type(exc).__name__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
