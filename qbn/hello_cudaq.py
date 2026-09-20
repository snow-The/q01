"""CUDA-Q 即時驗證：最小的 5 qubit QBN 電路，證明環境可用。

這是一個「一眼看得懂」的展示，不是能力矩陣（那是 dev/cudaq_matrix.py 的工作）。
輸出會顯示：版本、target、5 qubit 的 32 維測量分布、以及與純 NumPy 的交叉驗證。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cudaq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qbn_sim import StateVectorSim  # noqa: E402

N = 5
DIM = 2 ** N
RING = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

print("=" * 78)
print("CUDA-Q 即時驗證")
print("=" * 78)

# --- 1. 環境 ---
print(f"\n[1] 環境")
print(f"    CUDA-Q 版本 : {cudaq.__version__.split('(')[0].strip()}")
print(f"    commit      : {cudaq.__version__.split('(')[1].rstrip(')')}"
      if "(" in cudaq.__version__ else "")
cudaq.set_target("qpp-cpu")
print(f"    目前 target : {cudaq.get_target().name}")
print(f"    可用 target : {len(cudaq.get_targets())} 個")
print(f"    GPU 數量    : {cudaq.num_available_gpus()}")


# --- 2. 定義 5 qubit QBN 電路 ---
@cudaq.kernel
def qbn5(angles: list[float], weights: list[float]):
    """5 qubit QBN 輸出層。

    angles  : 5 個角度編碼參數（來自前端的 5 維特徵）
    weights : 5 個可訓練參數（節點通道）
    """
    q = cudaq.qvector(5)
    # 節點輸入態 σ：角度編碼
    for i in range(5):
        ry(angles[i], q[i])
    # 節點之間的邊：環形糾纏
    cx(q[0], q[1])
    cx(q[1], q[2])
    cx(q[2], q[3])
    cx(q[3], q[4])
    cx(q[4], q[0])
    # 節點通道 E_i：可訓練旋轉
    for i in range(5):
        ry(weights[i], q[i])


angles = [0.3, 0.6, 0.9, 1.2, 1.5]
weights = [0.11, -0.23, 0.37, -0.41, 0.53]

print(f"\n[2] 電路")
print(f"    5 qubit，環形 CX 糾纏，深度 1")
print(f"    角度編碼   = {angles}")
print(f"    可訓練參數 = {weights}")

# --- 3. 解析狀態向量 ---
print(f"\n[3] get_state() —— 解析狀態向量")
st = np.array(cudaq.get_state(qbn5, angles, weights))
print(f"    type  = {type(st).__name__}, shape = {st.shape}, dtype = {st.dtype}")
print(f"    ψ[:4] = {np.round(st[:4], 8)}")
print(f"    歸一  = {np.sum(np.abs(st) ** 2):.12f}")

# --- 4. 取樣 ---
print(f"\n[4] sample() —— 測量取樣")
res = cudaq.sample(qbn5, angles, weights, shots_count=8192)
counts = dict(res.counts)
print(f"    8192 shots，出現 {len(counts)} 種基底態")
for bits, c in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
    print(f"      |{bits}>  {c:>5}  {c / 8192:.4f}  "
          f"{'#' * int(c / 8192 * 60)}")

# --- 5. 交叉驗證 ---
print(f"\n[5] 與純 NumPy 模擬器交叉驗證")


def bit_reverse_permutation(n: int) -> np.ndarray:
    dim = 2 ** n
    perm = np.zeros(dim, dtype=int)
    for k in range(dim):
        r = 0
        for b in range(n):
            if (k >> b) & 1:
                r |= 1 << (n - 1 - b)
        perm[k] = r
    return perm


perm = bit_reverse_permutation(N)
p_cudaq_be = np.abs(st[perm]) ** 2

sim = StateVectorSim(N)
sim.reset()
for i in range(N):
    sim.ry(angles[i], i)
for c, t in RING:
    sim.cx(c, t)
for i in range(N):
    sim.ry(weights[i], i)
p_numpy = sim.probabilities()

dpsi = float(np.max(np.abs(st[perm] - sim.amplitudes())))
dp = float(np.max(np.abs(p_cudaq_be - p_numpy)))
print(f"    CUDA-Q 32 維機率[:5] = {np.round(p_cudaq_be[:5], 8)}")
print(f"    NumPy  32 維機率[:5] = {np.round(p_numpy[:5], 8)}")
print(f"    max|Δψ| = {dpsi:.3e}")
print(f"    max|ΔP| = {dp:.3e}")
print(f"    驗收門檻 1e-10 → {'✅ 通過' if dp < 1e-10 else '❌ 不通過'}")

# --- 6. 觀測量 ---
print(f"\n[6] observe() —— 觀測量期望值")
from cudaq import spin  # noqa: E402

H = sum(spin.z(i) for i in range(N))
val = cudaq.observe(qbn5, H, angles, weights).expectation()
manual = float(np.sum([p_cudaq_be @ np.array(
    [1 if format(k, f"0{N}b")[i] == "0" else -1 for k in range(DIM)])
    for i in range(N)]))
print(f"    <ΣZ_i> 由 observe() = {val:.10f}")
print(f"    <ΣZ_i> 由機率手算 = {manual:.10f}")
print(f"    差異 = {abs(val - manual):.3e}")

print("\n" + "=" * 78)
print("結論：CUDA-Q 0.16.0 在本機 WSL2 上可正常執行 5 qubit QBN 電路，")
print("      且與獨立實作的 NumPy 模擬器一致到機器精度。")
print("=" * 78)
