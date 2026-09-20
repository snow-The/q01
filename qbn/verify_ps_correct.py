"""參數平移規則的正確用法：只對量子層輸出，古典連鎖律穿透損失。

★ 這是一個真實踩過的坑，也是本書必須講清楚的一件事。

錯誤用法（本專案初版）：
    把參數平移直接套在「量子輸出 + 古典損失」的整體函數上
        ∂L/∂θ ≈ [L(θ+π/2) − L(θ−π/2)] / 2
    結果不精確。原因：參數平移規則的推導前提是「函數是量子期望值」，
    而期望值是 sin/cos 的**線性組合**，平移才恰好等於導數。
    一旦中間夾了非線性函數（例如 -log(p) 或 softmax），規則就不再精確。

正確用法（PennyLane / Qiskit / 所有主流量子 ML 框架的做法）：
    1. 用參數平移算**量子層輸出**對參數的導數：∂p_k/∂θ_i
       這裡 p_k = |⟨k|ψ(θ)⟩|² 是量子期望值（√p 是振幅，p 是二次式），規則精確。
    2. 用古典連鎖律把損失的導數接上去：∂L/∂θ_i = Σ_k (∂L/∂p_k)(∂p_k/∂θ_i)

驗證方式：與高精度中央差分比較（步長要選得夠小、且檢查收斂）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qbn_sim import StateVectorSim  # noqa: E402

N = 5
RING = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
DIM = 2 ** N


def run_circuit(thetas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """回傳 32 維機率（big-endian 索引）。"""
    sim = StateVectorSim(N)
    sim.reset()
    for i in range(N):
        sim.ry(float(thetas[i]), i)
    for c, t in RING:
        sim.cx(c, t)
    for i in range(N):
        sim.ry(float(weights[i]), i)
    return sim.probabilities()


# --- 損失：交叉熵（對某個目標基底態）----------------------------------------
TARGET = 0b10100


def loss(params: np.ndarray) -> float:
    thetas, weights = params[:N], params[N:]
    p = run_circuit(thetas, weights)
    return float(-np.log(max(p[TARGET], 1e-300)))


def dloss_dp(params: np.ndarray) -> np.ndarray:
    """∂L/∂p_k，古典部分（解析）。L = -log(p_T) ⇒ ∂L/∂p_T = -1/p_T，其餘為 0。"""
    thetas, weights = params[:N], params[N:]
    p = run_circuit(thetas, weights)
    g = np.zeros(DIM)
    g[TARGET] = -1.0 / max(p[TARGET], 1e-300)
    return g


def dp_dtheta(params: np.ndarray, i: int) -> np.ndarray:
    """∂p_k/∂θ_i：用參數平移（對 RY 精確，因為 p 是振幅的二次式）。"""
    thetas, weights = params[:N], params[N:]
    tp, tm = thetas.copy(), thetas.copy()
    tp[i] += np.pi / 2
    tm[i] -= np.pi / 2
    return (run_circuit(tp, weights) - run_circuit(tm, weights)) / 2


def dp_dweight(params: np.ndarray, i: int) -> np.ndarray:
    """∂p_k/∂w_i：同上，只是對第二層的參數。"""
    thetas, weights = params[:N], params[N:]
    tp, tm = weights.copy(), weights.copy()
    tp[i] += np.pi / 2
    tm[i] -= np.pi / 2
    return (run_circuit(thetas, tp) - run_circuit(thetas, tm)) / 2


print("=" * 78)
print("驗證：正確用法（量子層參數平移 + 古典連鎖律）")
print("=" * 78)

params = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 0.11, -0.23, 0.37, -0.41, 0.53])

# --- 正確梯度 ---
grad_correct = np.zeros(2 * N)
dl = dloss_dp(params)
for i in range(N):
    grad_correct[i] = float(dl @ dp_dtheta(params, i))
for i in range(N):
    grad_correct[N + i] = float(dl @ dp_dweight(params, i))

# --- 錯誤梯度（直接對損失做參數平移）---
grad_wrong = np.zeros(2 * N)
for i in range(2 * N):
    tp, tm = params.copy(), params.copy()
    tp[i] += np.pi / 2
    tm[i] -= np.pi / 2
    grad_wrong[i] = (loss(tp) - loss(tm)) / 2

# --- 參考：理查森外推的中央差分（高精度）---
def central_diff(params: np.ndarray, i: int, h: float) -> float:
    tp, tm = params.copy(), params.copy()
    tp[i] += h
    tm[i] -= h
    return (loss(tp) - loss(tm)) / (2 * h)


def richardson(params: np.ndarray, i: int) -> float:
    """D(h) 與 D(h/2) 外推消去 h² 項： (4·D(h/2) − D(h)) / 3"""
    h = 1e-4
    d1 = central_diff(params, i, h)
    d2 = central_diff(params, i, h / 2)
    return (4 * d2 - d1) / 3


ref = np.array([richardson(params, i) for i in range(2 * N)])

print(f"\n  正確用法（平移量子層 + 連鎖律） = {np.round(grad_correct, 8)}")
print(f"  錯誤用法（直接平移損失）        = {np.round(grad_wrong, 8)}")
print(f"  參考（理查森外推中央差分）      = {np.round(ref, 8)}")
print(f"\n  |正確 − 參考| = {np.max(np.abs(grad_correct - ref)):.3e}")
print(f"  |錯誤 − 參考| = {np.max(np.abs(grad_wrong - ref)):.3e}")
print(f"  兩者的差      = {np.max(np.abs(grad_correct - grad_wrong)):.3e}")

print("\n" + "=" * 78)
print("用單一參數展示誤差的來源（1 qubit、L = -log P(1)）")
print("=" * 78)


def p1(x: float) -> float:
    sim = StateVectorSim(1)
    sim.reset()
    sim.ry(x, 0)
    return float(sim.probabilities()[1])


theta = 0.7
exact = -1.0 / np.tan(theta / 2)          # d/dθ [-log sin²(θ/2)]
shift_wrong = (-np.log(p1(theta + np.pi / 2)) + np.log(p1(theta - np.pi / 2))) / 2
# 正確：先算 dp/dθ（參數平移，精確），再用連鎖律
dpdtheta = (p1(theta + np.pi / 2) - p1(theta - np.pi / 2)) / 2
shift_correct = (-1.0 / p1(theta)) * dpdtheta

print(f"  精確值              = {exact:.12f}")
print(f"  直接平移損失（錯誤） = {shift_wrong:.12f}   誤差 {abs(shift_wrong-exact):.3e}")
print(f"  平移 p + 連鎖律（正確）= {shift_correct:.12f}   誤差 {abs(shift_correct-exact):.3e}")

print("\n  錯誤的來源：對非線性損失 f，")
print("    [f(θ+h) − f(θ−h)]/2 ≈ f'(θ) + (h²/6)f'''(θ) + ...")
print("  它不是 f'(θ) 的估計，而是「f' 在 θ 附近的平均」，誤差是 O(h²)。")
print("  這就是為什麼必須把參數平移限制在量子層（期望值），古典部分用連鎖律。")
