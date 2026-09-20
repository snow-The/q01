"""W11 驗證腳本：docs/03-實作/混合式訓練迴圈.md 的所有數字。

本檔只用 NumPy 實作 5-qubit 狀態向量模擬器 + 參數平移規則 + 兩階段訓練，
不需要 CUDA-Q（本機 Windows 無 win_amd64 wheel，見 docs/_summary/00-專案規格常數.md 第六節）。
繪圖用 matplotlib；若未安裝則自動跳過繪圖，其餘結果照常輸出。

執行：
    .\\.venv\\Scripts\\python.exe dev\\train_qbn.py
或：
    uv run --with numpy --with matplotlib python dev/train_qbn.py

輸出分段：
    [0] 環境
    [1] 參數平移規則：單一 RY 閘的解析驗證
    [2] 參數平移 vs 有限差分：電路機率（線性可觀測量）
    [3] 反例：直接對「損失」用參數平移會錯
    [4] 正確的鏈式法則梯度 vs 有限差分
    [5] 為什麼不能在量子輸出後再加 softmax
    [6] 失敗模式 1：梯度範數 vs 電路深度（barren plateau）
    [7] 失敗模式 2：未標準化 → loss 卡在 log K
    [8] 失敗模式 3：類別不平衡 → loss 降但準確率不升
    [9] 失敗模式 4：電路參數過多 → 過擬合
    [10] 失敗模式 5：種子與取樣雜訊 → 結果不穩定
    [11] 主訓練：200 步兩階段訓練 + loss 曲線圖 + 最終準確率
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(precision=6, suppress=True, linewidth=140)

BAR = "=" * 74
SEEDS = [7, 21, 42, 84, 168]          # 專案規格常數：五個固定種子
N_QUBIT = 5
DIM = 2 ** N_QUBIT                     # 32
K_CLASS = 4
EMB_DIM = 256                          # potion-multilingual-128M 的嵌入維度
ABTT_DROP = 1                          # 丟棄前 r 個主成分（ABTT）


def head(title: str) -> None:
    print("\n" + BAR)
    print(title)
    print(BAR)


# ===========================================================================
# 0. 環境
# ===========================================================================
head("[0] 環境")

print(f"Python      : {sys.version.split()[0]}")
print(f"NumPy       : {np.__version__}")
print(f"qubit 數    : {N_QUBIT}（希爾伯特空間維度 {DIM}）")
print(f"類別數 K    : {K_CLASS}")
print(f"嵌入維度    : {EMB_DIM}（potion-multilingual-128M）")
print(f"固定種子    : {SEEDS}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
    print(f"matplotlib  : {matplotlib.__version__}（可繪圖）")
except ImportError:
    HAS_MPL = False
    print("matplotlib  : 未安裝 → 跳過繪圖（其餘結果不受影響）")

print("CUDA-Q      : 未安裝（本機 win_amd64 無 wheel）→ 本章梯度全部以 NumPy 狀態向量自算")


# ===========================================================================
# 5-qubit 狀態向量模擬器（純 NumPy，批次）
# ---------------------------------------------------------------------------
# 位元序約定：big-endian。基底索引 i 的二進位表示 b0 b1 b2 b3 b4 中，
# b0 是 qubit 0（最高位），b4 是 qubit 4（最低位）。
# 因此 qubit q 的 stride = 2^(N_QUBIT-1-q)。這個約定與 CUDA-Q 的
# 字串輸出 '01011' 一致（最左邊是 q[0]）。
# ===========================================================================
def ry_matrix(theta: float) -> np.ndarray:
    """R_Y(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]。"""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def apply_ry_batch(state: np.ndarray, thetas: np.ndarray, q: int) -> np.ndarray:
    """對批次中每個樣本施加不同角度的 R_Y(thetas[b]) 於 qubit q。

    state: (B, 32) complex128；thetas: (B,) float
    """
    b = state.shape[0]
    hi, lo = 2 ** q, 2 ** (N_QUBIT - 1 - q)
    s = state.reshape(b, hi, 2, lo)
    c = np.cos(thetas / 2.0)[:, None, None]
    sn = np.sin(thetas / 2.0)[:, None, None]
    a0 = s[:, :, 0, :].copy()
    a1 = s[:, :, 1, :].copy()
    s[:, :, 0, :] = c * a0 - sn * a1
    s[:, :, 1, :] = sn * a0 + c * a1
    return s.reshape(b, DIM)


def apply_cx(state: np.ndarray, ctrl: int, targ: int) -> np.ndarray:
    """受控 X（CNOT）：控制位為 1 時翻轉目標位。"""
    b = state.shape[0]
    s = state.reshape(b, DIM)
    idx = np.arange(DIM)
    cb = (idx >> (N_QUBIT - 1 - ctrl)) & 1
    tb = (idx >> (N_QUBIT - 1 - targ)) & 1
    i0 = idx[(cb == 1) & (tb == 0)]
    i1 = i0 | (1 << (N_QUBIT - 1 - targ))
    tmp = s[:, i0].copy()
    s[:, i0] = s[:, i1]
    s[:, i1] = tmp
    return s


RING_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]   # 環形 CX 糾纏


def circuit_state(angles: np.ndarray, thetas: np.ndarray,
                  n_layer: int = 1, extra_thetas: np.ndarray | None = None) -> np.ndarray:
    """5-qubit 電路：角度編碼 → 環形 CX → 可訓練 R_Y 層（可多層）。

    angles : (B, 5)  編碼角（凍結，來自 PCA）
    thetas : (5,)    第一層可訓練角 θ_circ
    n_layer: 可訓練層數（每層 = 5 個 R_Y + 環形 CX）
    """
    b = angles.shape[0]
    psi = np.zeros((b, DIM), dtype=complex)
    psi[:, 0] = 1.0
    for q in range(N_QUBIT):                      # 角度編碼（凍結）
        psi = apply_ry_batch(psi, angles[:, q], q)
    for edge in RING_EDGES:                       # 編碼後的糾纏
        psi = apply_cx(psi, *edge)

    all_theta = [thetas]
    if extra_thetas is not None:
        all_theta.extend(list(extra_thetas))
    for layer in range(n_layer):
        th = all_theta[layer]
        for q in range(N_QUBIT):
            psi = apply_ry_batch(psi, np.full(b, th[q]), q)
        if layer < n_layer - 1:
            for edge in RING_EDGES:
                psi = apply_cx(psi, *edge)
    return psi


def probs32(angles: np.ndarray, thetas: np.ndarray, n_layer: int = 1,
            extra_thetas=None) -> np.ndarray:
    """32 維計算基底機率 p_i = |A_i|²（已由 |A|² 保證非負且總和為 1）。"""
    psi = circuit_state(angles, thetas, n_layer, extra_thetas)
    return np.abs(psi) ** 2


def z_expectations(p32: np.ndarray) -> np.ndarray:
    """5 個 ⟨Z_q⟩ 期望值（規格常數檔的讀出第一版用的特徵）。"""
    idx = np.arange(DIM)
    signs = np.array([1 - 2 * ((idx >> (N_QUBIT - 1 - q)) & 1) for q in range(N_QUBIT)])
    return p32 @ signs.T


# ===========================================================================
# 讀出層與損失
# ---------------------------------------------------------------------------
# W_read 是 K×32 的「行隨機矩陣」（column-stochastic）：每一行非負、每列和為 1。
# 它是古典通道（把 32 維對角分布壓成 K 類），因此
#     P(y) = W_read @ p32  ，Σ_y P(y) = 1 自動成立，不需要 softmax。
# ===========================================================================
def adjacent_grouping() -> np.ndarray:
    """初始化：相鄰分組 —— 32 個基底態每 32/K = 8 個一類。"""
    w = np.zeros((K_CLASS, DIM))
    per = DIM // K_CLASS
    for k in range(K_CLASS):
        w[k, k * per:(k + 1) * per] = 1.0 / per
    return w


def readout(p32: np.ndarray, w_read: np.ndarray) -> np.ndarray:
    """P(y) = W_read @ p32；輸出保證是合法機率分布（每一列和為 1）。"""
    return p32 @ w_read.T


EPS = 1e-12


def cross_entropy(py: np.ndarray, y: np.ndarray, soft_targets: np.ndarray | None = None) -> float:
    """交叉熵 −(1/B)Σ_b Σ_k ỹ_bk log P_bk。ỹ 為 one-hot 時等同 NLL。"""
    if soft_targets is None:
        tgt = np.zeros_like(py)
        tgt[np.arange(len(y)), y] = 1.0
    else:
        tgt = soft_targets
    return float(-np.mean(np.sum(tgt * np.log(py + EPS), axis=1)))


def nll(py: np.ndarray, y: np.ndarray) -> float:
    """負對數似然 −(1/B)Σ_b log P_b(y_b)：one-hot 標籤下與交叉熵逐位相同。"""
    return float(-np.mean(np.log(py[np.arange(len(y)), y] + EPS)))


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def simplex_project(v: np.ndarray) -> np.ndarray:
    """把向量投影到機率單體 {x ≥ 0, Σx = 1}（Duchi 2008 的 O(n log n) 演算法）。"""
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# ===========================================================================
# 1. 參數平移規則：單一 RY 閘的解析驗證
# ===========================================================================
head("[1] 參數平移規則：單一 RY 閘的解析驗證")

print("""設定：|ψ(θ)⟩ = R_Y(θ)|0⟩，可觀測量 f(θ) = P(1) = sin²(θ/2)。
      解析導數 f'(θ) = sin(θ)/2。
      參數平移估計 (f(θ+π/2) − f(θ−π/2))/2 應該逐點完全相等。""")

rows = []
for t in [0.0, 0.3, np.pi / 4, 1.0, np.pi / 2, 2.0, np.pi]:
    f = lambda x: np.sin(x / 2.0) ** 2                       # noqa: E731
    analytic = np.sin(t) / 2.0
    shift = (f(t + np.pi / 2) - f(t - np.pi / 2)) / 2.0
    rows.append((t, analytic, shift, abs(analytic - shift)))

print(f"\n{'θ':>8} {'解析 f′(θ)':>16} {'參數平移':>16} {'絕對誤差':>14}")
for t, a, s, e in rows:
    print(f"{t:>8.4f} {a:>16.10f} {s:>16.10f} {e:>14.2e}")
print(f"\n最大絕對誤差 = {max(r[3] for r in rows):.3e}  → 參數平移在浮點精度內完全精確")


# ===========================================================================
# 2. 參數平移 vs 有限差分：電路機率（線性可觀測量）
# ===========================================================================
head("[2] 參數平移 vs 有限差分：電路機率（線性可觀測量）")

rng = np.random.default_rng(42)
angles_demo = rng.uniform(-np.pi, np.pi, size=(1, N_QUBIT))
theta0 = rng.uniform(-np.pi, np.pi, size=N_QUBIT)
K_STATE = 11                                     # 觀測基底態 |01011⟩

f_state = lambda th: probs32(angles_demo, th)[0, K_STATE]     # noqa: E731

shift_grad = np.array([
    (f_state(theta0 + np.eye(N_QUBIT)[i] * np.pi / 2)
     - f_state(theta0 - np.eye(N_QUBIT)[i] * np.pi / 2)) / 2.0
    for i in range(N_QUBIT)
])

H_FD = 1e-6
fd_grad = np.array([
    (f_state(theta0 + np.eye(N_QUBIT)[i] * H_FD)
     - f_state(theta0 - np.eye(N_QUBIT)[i] * H_FD)) / (2 * H_FD)
    for i in range(N_QUBIT)
])

print(f"觀測量 f(θ) = P(|{K_STATE:05b}⟩)（線性於密度矩陣，參數平移精確成立）")
print(f"中央差分步長 h = {H_FD:g}\n")
print(f"{'參數 i':>8} {'參數平移':>18} {'有限差分':>18} {'絕對差':>14}")
for i in range(N_QUBIT):
    print(f"{i:>8} {shift_grad[i]:>18.12f} {fd_grad[i]:>18.12f} "
          f"{abs(shift_grad[i] - fd_grad[i]):>14.3e}")
err2 = np.max(np.abs(shift_grad - fd_grad))
print(f"\n最大絕對差 = {err2:.3e}  （有限差分的截斷誤差 O(h²) 與捨入誤差 O(ε/h) 造成）")
print(f"是否 < 1e-6 ？ {err2 < 1e-6}")
print(f"電路呼叫次數：參數平移 {2 * N_QUBIT} 次，中央差分 {2 * N_QUBIT} 次（打平）")


# ===========================================================================
# 3. 反例：直接對「損失」用參數平移會錯
# ===========================================================================
head("[3] 反例：直接對「損失」用參數平移會錯（本節最重要的陷阱）")

w_demo = adjacent_grouping()
y_demo = np.array([2])                       # 真實類別 k = 2
x_demo = rng.uniform(0, 1, size=(3, N_QUBIT))

def loss_of_theta(th: np.ndarray) -> float:
    py = readout(probs32(angles_demo, th), w_demo)
    return cross_entropy(py, y_demo)

naive_shift = np.array([
    (loss_of_theta(theta0 + np.eye(N_QUBIT)[i] * np.pi / 2)
     - loss_of_theta(theta0 - np.eye(N_QUBIT)[i] * np.pi / 2)) / 2.0
    for i in range(N_QUBIT)
])
fd_loss = np.array([
    (loss_of_theta(theta0 + np.eye(N_QUBIT)[i] * H_FD)
     - loss_of_theta(theta0 - np.eye(N_QUBIT)[i] * H_FD)) / (2 * H_FD)
    for i in range(N_QUBIT)
])

print("""損失含 log(P)，是機率的非線性函數；參數平移只對「線性於密度矩陣」的
量（機率、期望值）精確成立。若把參數平移直接套在損失上，梯度會系統性偏誤。""")
print(f"\n{'參數 i':>8} {'平移(套在 loss 上)':>22} {'中央差分(真值)':>20} {'絕對差':>12}")
for i in range(N_QUBIT):
    print(f"{i:>8} {naive_shift[i]:>22.8f} {fd_loss[i]:>20.8f} "
          f"{abs(naive_shift[i] - fd_loss[i]):>12.3e}")
err3 = np.max(np.abs(naive_shift - fd_loss))
print(f"\n最大絕對差 = {err3:.3e}  → 比第 [2] 節的 {err2:.1e} 大了約 "
      f"{err3 / max(err2, 1e-16):.0f} 倍，這是系統性錯誤，不是浮點誤差")


# ===========================================================================
# 4. 正確的鏈式法則梯度 vs 有限差分
# ===========================================================================
head("[4] 正確的鏈式法則梯度 vs 有限差分（本章採用的做法）")

def full_loss(th: np.ndarray, ang: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return cross_entropy(readout(probs32(ang, th), w), y)

# 造一批小型資料
rng4 = np.random.default_rng(3)
ang4 = rng4.uniform(-np.pi, np.pi, size=(20, N_QUBIT))
y4 = rng4.integers(0, K_CLASS, size=20)
w4 = adjacent_grouping()
th4 = rng4.uniform(-np.pi, np.pi, size=N_QUBIT)

def chain_rule_grad(th: np.ndarray) -> np.ndarray:
    """三步：(a) 參數平移算 ∂p32/∂θ_i（精確）；(b) 解析算 ∂L/∂p32；(c) 相乘。"""
    py = readout(probs32(ang4, th), w4)
    dp32 = np.zeros((len(y4), DIM))
    dpy = np.zeros_like(py)
    dpy[np.arange(len(y4)), y4] = -1.0 / (py[np.arange(len(y4)), y4] + EPS)
    dpy /= len(y4)
    dp32 = dpy @ w4                                     # ∂L/∂p32
    g = np.zeros(N_QUBIT)
    for i in range(N_QUBIT):
        e = np.eye(N_QUBIT)[i]
        p_plus = probs32(ang4, th + e * np.pi / 2)
        p_minus = probs32(ang4, th - e * np.pi / 2)
        dp_dtheta = (p_plus - p_minus) / 2.0            # ∂p32/∂θ_i（精確）
        g[i] = np.sum(dp32 * dp_dtheta)
    return g

g_chain = chain_rule_grad(th4)
g_fd = np.array([
    (full_loss(th4 + np.eye(N_QUBIT)[i] * H_FD, ang4, y4, w4)
     - full_loss(th4 - np.eye(N_QUBIT)[i] * H_FD, ang4, y4, w4)) / (2 * H_FD)
    for i in range(N_QUBIT)
])

print(f"損失：交叉熵（20 筆樣本、K={K_CLASS}）")
print(f"\n{'參數 i':>8} {'鏈式法則(平移)':>20} {'中央差分':>20} {'絕對差':>14}")
for i in range(N_QUBIT):
    print(f"{i:>8} {g_chain[i]:>20.12f} {g_fd[i]:>20.12f} "
          f"{abs(g_chain[i] - g_fd[i]):>14.3e}")
err4 = np.max(np.abs(g_chain - g_fd))
print(f"\n最大絕對差 = {err4:.3e}   是否 < 1e-6 ？ {err4 < 1e-6}")
print(f"梯度範數 ‖∇L‖ = {np.linalg.norm(g_chain):.8f}")
print(f"\n成本：每步 {2 * N_QUBIT} 次電路前向（機率向量），古典端反向傳播免費。")
print(f"      有限差分同樣 {2 * N_QUBIT} 次，但精度差 {err4 / max(np.finfo(float).eps, 1):.1e} 個數量級。")


# ===========================================================================
# 5. 為什麼不能在量子輸出後再加 softmax
# ===========================================================================
head("[5] 為什麼不能在量子輸出後再加 softmax（重複正規化）")

p_peaked = np.zeros(DIM)
p_peaked[7] = 1.0
p_mid = np.array([0.5, 0.3] + [0.2 / 30] * 30)
p_flat = np.full(DIM, 1.0 / DIM)

print("""量子輸出 p32 = |A|² 已經是合法機率分布（非負、Σ=1，由 Tr[ρ]=1 承擔正規化）。
若再套一次 softmax，會發生什麼事？""")
print(f"\n{'輸入 p32':>34} {'argmax':>8} {'max softmax(p)':>16} {'最大/最小比':>14}")
for name, p in [("完全確定（one-hot）", p_peaked),
                ("中度確定", p_mid),
                ("完全均勻", p_flat)]:
    sm = softmax(p)
    ratio = sm.max() / sm.min()
    print(f"{name:>34} {int(np.argmax(p)):>8} {sm.max():>16.6f} {ratio:>14.4f}")

print(f"""
關鍵：p_i ∈ [0, 1]，所以 exp(p_i) ∈ [1, e]，任意兩態的比值最多 e ≈ {np.e:.4f}。
      也就是說 softmax(p) 幾乎一定是接近均勻的分布，
      不管 p 本身有多確定 —— 這正是「信心被壓平、校準被破壞」的機制。

反過來說：softmax(log p) 恰好等於 p（恆等），所以「需要 logits」時取 log 是安全的，
但把機率當 logits 餵進 softmax 是毀滅性的。""")

sm_log = softmax(np.log(p_mid + EPS))
print(f"驗證 softmax(log p) == p ：最大絕對差 = {np.max(np.abs(sm_log - p_mid)):.3e}")


# ===========================================================================
# 6. 失敗模式 1：梯度範數 vs 電路深度（barren plateau）
# ===========================================================================
head("[6] 失敗模式 1：loss 完全不動 —— 梯度範數 vs 電路深度")

print("""診斷動作：印出梯度範數。若 ‖∇L‖ ≈ 0 且與初始化無關地小，
就是貧瘠高原（barren plateau）：梯度隨 qubit 數與深度指數衰減。""")

def grad_norm_at(seed: int, n_layer: int, n_sample: int = 12) -> float:
    r = np.random.default_rng(seed)
    ang = r.uniform(-np.pi, np.pi, size=(n_sample, N_QUBIT))
    yy = r.integers(0, K_CLASS, size=n_sample)
    ww = adjacent_grouping()
    base = r.uniform(-np.pi, np.pi, size=N_QUBIT)
    extra = [r.uniform(-np.pi, np.pi, size=N_QUBIT) for _ in range(n_layer - 1)]

    def loss(th_list):
        th0 = th_list[0]
        ex = th_list[1:] if len(th_list) > 1 else None
        return cross_entropy(readout(probs32(ang, th0, n_layer, ex), ww), yy)

    th_list = [base] + extra
    g = np.zeros((n_layer, N_QUBIT))
    for layer in range(n_layer):
        for i in range(N_QUBIT):
            plus = [t.copy() for t in th_list]
            minus = [t.copy() for t in th_list]
            plus[layer][i] += np.pi / 2
            minus[layer][i] -= np.pi / 2
            p32p = probs32(ang, plus[0], n_layer, plus[1:] if n_layer > 1 else None)
            p32m = probs32(ang, minus[0], n_layer, minus[1:] if n_layer > 1 else None)
            py = readout((p32p + p32m) / 2.0, ww)       # 只借路徑，梯度用差分形式
            dpy = np.zeros_like(py)
            dpy[np.arange(n_sample), yy] = -1.0 / (readout(p32p, ww)[np.arange(n_sample), yy] + EPS)
            dpy /= n_sample
            dp = dpy @ ww
            g[layer, i] = np.sum(dp * (p32p - p32m) / 2.0)
    return float(np.linalg.norm(g))

print(f"\n{'可訓練層數':>12} {'參數量':>8} {'‖∇L‖ (5 seed 平均)':>22} {'標準差':>12}")
layer_stats = []
for nl in [1, 2, 3, 4, 6]:
    norms = [grad_norm_at(s, nl) for s in SEEDS]
    m, sd = float(np.mean(norms)), float(np.std(norms))
    layer_stats.append((nl, m, sd))
    print(f"{nl:>12} {nl * N_QUBIT:>8} {m:>22.8f} {sd:>12.2e}")
decay = layer_stats[0][1] / max(layer_stats[-1][1], 1e-16)
print(f"\n1 層 → 6 層的梯度範數衰減倍率：{decay:.2f}×")
print("結論：深度增加會讓梯度變小（本專案 1–2 層仍在可用範圍）。")
print("      這正是「先用淺電路 + 兩階段訓練」的理由（見第五節）。")


# ===========================================================================
# 7. 失敗模式 2：未標準化 → loss 卡在 log K
# ===========================================================================
head("[7] 失敗模式 2：loss 卡在 log K（隨機水準）")

logK = float(np.log(K_CLASS))
print(f"K = {K_CLASS} 的隨機水準：−log(1/K) = log K = {logK:.6f}")

def make_synth(seed: int, n: int, scale: float = 1.0):
    """合成「句向量」：K 個類別中心 + 高斯雜訊，維度 = 256（模擬 potion 輸出）。"""
    r = np.random.default_rng(seed)
    centers = r.normal(0, 1, size=(K_CLASS, EMB_DIM))
    y = r.integers(0, K_CLASS, size=n)
    x = centers[y] * scale + r.normal(0, 1, size=(n, EMB_DIM))
    return x, y

def pca_fit(x: np.ndarray, n_comp: int):
    mu = x.mean(axis=0)
    xc = x - mu
    cov = xc.T @ xc / (len(xc) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return mu, evecs[:, order[:n_comp]], evals[order[:n_comp]]

def encode(x: np.ndarray, mu, comps, drop: int = ABTT_DROP, standardize: bool = True):
    """ABTT：丟棄前 drop 個主成分 → 逐維標準化 → 取 5 維 → 當角度。"""
    z = (x - mu) @ comps
    z = z[:, drop:drop + N_QUBIT]
    if standardize:
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-12)
    else:
        z = z / (np.abs(z).max(axis=0) + 1e-12) * 3.0
    return np.clip(z, -np.pi, np.pi)

x_tr, y_tr = make_synth(7, 400)
mu7, comps7, ev7 = pca_fit(x_tr, ABTT_DROP + N_QUBIT)
ang_good = encode(x_tr, mu7, comps7, standardize=True)
ang_bad = encode(x_tr, mu7, comps7, standardize=False)

print(f"\nPCA 前 6 個主成分的變異數：{np.round(ev7, 4)}")
print(f"丟棄前 {ABTT_DROP} 個後，剩下 5 維的原始標準差："
      f"{np.round(((x_tr - mu7) @ comps7)[:, ABTT_DROP:].std(axis=0), 4)}")

def eval_random_level(ang: np.ndarray, tag: str) -> None:
    th = np.zeros(N_QUBIT)
    w = adjacent_grouping()
    py = readout(probs32(ang, th), w)
    ce = cross_entropy(py, y_tr)
    print(f"{tag:<32} 編碼角範圍 [{ang.min():+.3f}, {ang.max():+.3f}]  "
          f"標準差 {ang.std():.4f}   loss = {ce:.6f}（隨機水準 {logK:.4f}）")

print()
eval_random_level(ang_good, "有標準化（正確）")
eval_random_level(ang_bad, "未標準化（錯誤）")

print(f"""
診斷方法：
  1. 印出編碼後 5 維的 mean / std —— 若 std 遠小於 1，編碼幾乎沒有資訊。
  2. 檢查 loss 是否 ≈ log K。是的話模型等於在猜。
  3. 檢查 PCA 是否只用「訓練集」fit（用測試集 fit 是資料洩漏，另一種病）。
""")
print(f"未標準化時 5 維標準差僅 {ang_bad.std():.4f}，角度擠在 0 附近，")
print(f"R_Y(θ) ≈ I，電路輸出幾乎與輸入無關 → loss 貼近 log K。")


# ===========================================================================
# 8. 失敗模式 3：類別不平衡 → loss 降但準確率不升
# ===========================================================================
head("[8] 失敗模式 3：loss 下降但準確率不升（類別不平衡）")

def train_simple(ang_tr, y_tr, ang_te, y_te, steps=120, lr_w=0.7, lr_th=0.0,
                 class_weight=None, seed=42):
    r = np.random.default_rng(seed)
    w = adjacent_grouping()
    th = r.uniform(-0.3, 0.3, size=N_QUBIT)
    hist = []
    for t in range(steps):
        p32 = probs32(ang_tr, th)
        py = readout(p32, w)
        cw = np.ones(K_CLASS) if class_weight is None else class_weight
        sw = cw[y_tr] / np.mean(cw[y_tr])
        loss = float(-np.mean(sw * np.log(py[np.arange(len(y_tr)), y_tr] + EPS)))
        # 讀出層梯度
        b = len(y_tr)
        dpy = np.zeros_like(py)
        dpy[np.arange(b), y_tr] = -sw / (py[np.arange(b), y_tr] + EPS)
        dpy /= b
        dp32 = dpy @ w
        gw = dpy.T @ p32
        w = w - lr_w * gw
        w = np.stack([simplex_project(w[k]) for k in range(K_CLASS)])
        if lr_th > 0:
            g = np.zeros(N_QUBIT)
            for i in range(N_QUBIT):
                e = np.eye(N_QUBIT)[i]
                g[i] = np.sum(dp32 * (probs32(ang_tr, th + e * np.pi / 2)
                                      - probs32(ang_tr, th - e * np.pi / 2)) / 2.0)
            th = th - lr_th * g
        hist.append((loss, acc(readout(probs32(ang_tr, th), w), y_tr),
                     acc(readout(probs32(ang_te, th), w), y_te)))
    return w, th, hist


def acc(py: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(py, axis=1) == y))


def confusion(py: np.ndarray, y: np.ndarray) -> np.ndarray:
    cm = np.zeros((K_CLASS, K_CLASS), dtype=int)
    for t, p in zip(y, np.argmax(py, axis=1)):
        cm[t, p] += 1
    return cm


# 造不平衡資料：類別 0 佔 85%
rng8 = np.random.default_rng(11)
x8, y8 = make_synth(11, 600)
y8 = np.where(rng8.random(600) < 0.85, 0, rng8.integers(1, K_CLASS, size=600))
mu8, comps8, _ = pca_fit(x8, ABTT_DROP + N_QUBIT)
ang8 = encode(x8, mu8, comps8)
print(f"訓練集類別分布：{np.bincount(y8, minlength=K_CLASS).tolist()}"
      f"  多數類別佔比 = {np.bincount(y8).max() / len(y8):.3f}")

w8, th8, hist8 = train_simple(ang8, y8, ang8, y8, steps=120, lr_w=0.7, class_weight=None)
py8 = readout(probs32(ang8, th8), w8)
print(f"\n未加權：loss {hist8[0][0]:.4f} → {hist8[-1][0]:.4f}，"
      f"準確率 {hist8[0][1]:.4f} → {hist8[-1][1]:.4f}（多數類別基準 "
      f"{np.bincount(y8).max() / len(y8):.4f}）")
print("混淆矩陣（列=真類別，行=預測類別）：")
print(confusion(py8, y8))

cw = len(y8) / (K_CLASS * np.bincount(y8, minlength=K_CLASS))
w8b, th8b, hist8b = train_simple(ang8, y8, ang8, y8, steps=120, lr_w=0.7, class_weight=cw)
py8b = readout(probs32(ang8, th8b), w8b)
print(f"\n加權後：loss {hist8b[0][0]:.4f} → {hist8b[-1][0]:.4f}，"
      f"準確率 {hist8b[0][1]:.4f} → {hist8b[-1][1]:.4f}")
print(f"類別權重 = {np.round(cw, 4)}")
print("混淆矩陣（列=真類別，行=預測類別）：")
print(confusion(py8b, y8))
print("\n診斷：loss 下降只代表「預測的機率更貼近標籤」，不代表 argmax 變對。"
      "\n      不平衡時模型只要全部猜多數類就能把 loss 壓低。")


# ===========================================================================
# 9. 失敗模式 4：電路參數過多 → 過擬合
# ===========================================================================
head("[9] 失敗模式 4：訓練集好、測試集差（過擬合）")

x_tr9, y_tr9 = make_synth(7, 400)
x_te9, y_te9 = make_synth(999, 200)
mu9, comps9, _ = pca_fit(x_tr9, ABTT_DROP + N_QUBIT)
ang_tr9 = encode(x_tr9, mu9, comps9)
ang_te9 = encode(x_te9, mu9, comps9)      # 注意：測試集用「訓練集」的 PCA 參數

print(f"{'可訓練層數':>12} {'參數量':>8} {'訓練準確率':>12} {'測試準確率':>12} {'差距':>10}")
over_rows = []
for nl in [1, 2, 4]:
    r = np.random.default_rng(42)
    w = adjacent_grouping()
    th = r.uniform(-0.3, 0.3, size=N_QUBIT)
    extra = [r.uniform(-0.3, 0.3, size=N_QUBIT) for _ in range(nl - 1)]
    for step in range(160):
        p32 = probs32(ang_tr9, th, nl, extra if nl > 1 else None)
        py = readout(p32, w)
        b = len(y_tr9)
        dpy = np.zeros_like(py)
        dpy[np.arange(b), y_tr9] = -1.0 / (py[np.arange(b), y_tr9] + EPS)
        dpy /= b
        dp32 = dpy @ w
        w = np.stack([simplex_project(w[k] - 0.7 * (dpy.T @ p32)[k]) for k in range(K_CLASS)])
        for layer in range(nl):
            tgt = th if layer == 0 else extra[layer - 1]
            g = np.zeros(N_QUBIT)
            for i in range(N_QUBIT):
                e = np.eye(N_QUBIT)[i]
                def f_at(sign):
                    t0 = th + (e * sign * np.pi / 2 if layer == 0 else 0)
                    ex = list(extra)
                    if layer > 0:
                        ex[layer - 1] = ex[layer - 1] + e * sign * np.pi / 2
                    return probs32(ang_tr9, t0, nl, ex if nl > 1 else None)
                g[i] = np.sum(dp32 * (f_at(+1) - f_at(-1)) / 2.0)
            tgt -= 0.15 * g
    tr_acc = acc(readout(probs32(ang_tr9, th, nl, extra if nl > 1 else None), w), y_tr9)
    te_acc = acc(readout(probs32(ang_te9, th, nl, extra if nl > 1 else None), w), y_te9)
    over_rows.append((nl, tr_acc, te_acc))
    print(f"{nl:>12} {nl * N_QUBIT:>8} {tr_acc:>12.4f} {te_acc:>12.4f} {tr_acc - te_acc:>10.4f}")
print("\n診斷：畫學習曲線（訓練 vs 驗證 loss）。兩條線分岔 = 過擬合。")
print("對策：減層數（回到 1–2 層）、減小學習率、早停、加權重衰減。")


# ===========================================================================
# 10. 失敗模式 5：種子與取樣雜訊
# ===========================================================================
head("[10] 失敗模式 5：每次跑結果差異很大")

def run_once(seed: int, shots: int | None = None, steps: int = 200):
    r = np.random.default_rng(seed)
    x_tr, y_tr = make_synth(seed, 400)
    x_te, y_te = make_synth(seed + 5000, 200)
    mu, comps, _ = pca_fit(x_tr, ABTT_DROP + N_QUBIT)
    a_tr, a_te = encode(x_tr, mu, comps), encode(x_te, mu, comps)
    w = adjacent_grouping()
    th = r.uniform(-0.3, 0.3, size=N_QUBIT)
    for step in range(steps):
        p32 = probs32(a_tr, th)
        if shots is not None:
            cnt = r.multinomial(shots, p32[0] / p32[0].sum())
            p32 = (cnt / shots)[None, :].repeat(len(y_tr), axis=0)
        py = readout(p32, w)
        b = len(y_tr)
        dpy = np.zeros_like(py)
        dpy[np.arange(b), y_tr] = -1.0 / (py[np.arange(b), y_tr] + EPS)
        dpy /= b
        dp32 = dpy @ w
        w = np.stack([simplex_project(w[k] - 0.7 * (dpy.T @ p32)[k]) for k in range(K_CLASS)])
        g = np.zeros(N_QUBIT)
        for i in range(N_QUBIT):
            e = np.eye(N_QUBIT)[i]
            g[i] = np.sum(dp32 * (probs32(a_tr, th + e * np.pi / 2)
                                  - probs32(a_tr, th - e * np.pi / 2)) / 2.0)
        th = th - 0.15 * g
    return acc(readout(probs32(a_te, th), w), y_te)

accs = [run_once(s) for s in SEEDS]
print(f"解析機率（無取樣雜訊）：accuracy = "
      f"{np.mean(accs):.4f} ± {np.std(accs):.4f}   逐種子 {[f'{a:.4f}' for a in accs]}")

accs_shots = [run_once(s, shots=1024) for s in SEEDS]
print(f"shots = 1024 取樣    ：accuracy = "
      f"{np.mean(accs_shots):.4f} ± {np.std(accs_shots):.4f}   "
      f"逐種子 {[f'{a:.4f}' for a in accs_shots]}")

r10 = np.random.default_rng(0)
_x10, _ = make_synth(7, 4)
p_true = probs32(encode(_x10, mu7, comps7), np.zeros(N_QUBIT))[0]
print(f"\n取樣雜訊的量級（同一個電路，重複用 shots 估計 P(00000)）：")
for shots in [256, 1024, 4096]:
    est = [r10.multinomial(shots, p_true)[0] / shots for _ in range(200)]
    print(f"  shots = {shots:>5}：真值 {p_true[0]:.6f}，"
          f"估計標準差 {np.std(est):.6f}，相對誤差 {np.std(est) / p_true[0] * 100:.3f}%")

print("""
診斷與對策：
  - 固定種子重跑 5 次（本專案規格：7, 21, 42, 84, 168），報告 mean ± std。
  - 梯度用「解析機率」（statevector），不要用 shots 估計 —— 後者會讓梯度有變異數。
  - 若必須用 shots，shots 需 ≥ 4096（規格常數檔的正式實驗設定）。
""")


# ===========================================================================
# 11. 主訓練：200 步兩階段訓練
# ===========================================================================
head("[11] 主訓練：200 步兩階段訓練（第一階段 100 步只訓練讀出層）")

SEED = 42
STEPS = 200
STAGE1 = 100
LR_W1, LR_W2, LR_TH = 0.8, 0.25, 0.06
BETA1, BETA2, EPS_ADAM = 0.9, 0.999, 1e-8

t0 = time.time()
x_train, y_train = make_synth(SEED, 400)
x_test, y_test = make_synth(SEED + 5000, 200)
mu, comps, evals = pca_fit(x_train, ABTT_DROP + N_QUBIT)
ang_train = encode(x_train, mu, comps)
ang_test = encode(x_test, mu, comps)

rng11 = np.random.default_rng(SEED)
theta = rng11.uniform(-0.3, 0.3, size=N_QUBIT)
w_read = adjacent_grouping()

m_th = np.zeros(N_QUBIT)
v_th = np.zeros(N_QUBIT)
m_w = np.zeros_like(w_read)
v_w = np.zeros_like(w_read)

history = {"step": [], "loss": [], "train_acc": [], "test_acc": [],
           "grad_norm": [], "stage": []}

print(f"資料：訓練 {len(y_train)} 筆、測試 {len(y_test)} 筆、維度 {EMB_DIM} → 5 角度")
print(f"ABTT 丟棄前 {ABTT_DROP} 個主成分，保留 5 維；"
      f"保留維度的變異數 = {np.round(evals[ABTT_DROP:], 5).tolist()}")
print(f"編碼角範圍 = [{ang_train.min():+.3f}, {ang_train.max():+.3f}]，"
      f"標準差 = {ang_train.std():.4f}")
print(f"初始 θ = {np.round(theta, 4)}")
print(f"初始 W_read = 相鄰分組（32 態每 8 個一類）\n")

for step in range(1, STEPS + 1):
    stage = 1 if step <= STAGE1 else 2
    p32 = probs32(ang_train, theta)
    py = readout(p32, w_read)
    loss = cross_entropy(py, y_train)

    b = len(y_train)
    dpy = np.zeros_like(py)
    dpy[np.arange(b), y_train] = -1.0 / (py[np.arange(b), y_train] + EPS)
    dpy /= b
    dp32 = dpy @ w_read
    gw = dpy.T @ p32

    lr_w = LR_W1 if stage == 1 else LR_W2
    m_w = BETA1 * m_w + (1 - BETA1) * gw
    v_w = BETA2 * v_w + (1 - BETA2) * gw ** 2
    mw_hat = m_w / (1 - BETA1 ** step)
    vw_hat = v_w / (1 - BETA2 ** step)
    w_read = w_read - lr_w * mw_hat / (np.sqrt(vw_hat) + EPS_ADAM)
    w_read = np.stack([simplex_project(w_read[k]) for k in range(K_CLASS)])

    gnorm = 0.0
    if stage == 2:
        g = np.zeros(N_QUBIT)
        for i in range(N_QUBIT):
            e = np.eye(N_QUBIT)[i]
            g[i] = np.sum(dp32 * (probs32(ang_train, theta + e * np.pi / 2)
                                  - probs32(ang_train, theta - e * np.pi / 2)) / 2.0)
        gnorm = float(np.linalg.norm(g))
        m_th = BETA1 * m_th + (1 - BETA1) * g
        v_th = BETA2 * v_th + (1 - BETA2) * g ** 2
        mt_hat = m_th / (1 - BETA1 ** step)
        vt_hat = v_th / (1 - BETA2 ** step)
        theta = theta - LR_TH * mt_hat / (np.sqrt(vt_hat) + EPS_ADAM)

    if step % 10 == 0 or step == 1:
        tr = acc(readout(probs32(ang_train, theta), w_read), y_train)
        te = acc(readout(probs32(ang_test, theta), w_read), y_test)
        history["step"].append(step)
        history["loss"].append(loss)
        history["train_acc"].append(tr)
        history["test_acc"].append(te)
        history["grad_norm"].append(gnorm)
        history["stage"].append(stage)
        print(f"step {step:>4}  stage {stage}  loss {loss:.6f}  "
              f"train_acc {tr:.4f}  test_acc {te:.4f}  ‖∇_θL‖ {gnorm:.6f}")

elapsed = time.time() - t0
py_tr = readout(probs32(ang_train, theta), w_read)
py_te = readout(probs32(ang_test, theta), w_read)
final_loss = cross_entropy(py_te, y_test)
final_tr = acc(py_tr, y_train)
final_te = acc(py_te, y_test)

print(f"\n訓練耗時 {elapsed:.2f} 秒（{STEPS} 步，每步 {2 * N_QUBIT} 次電路前向）")
print(f"初始 loss（step 1）  : {history['loss'][0]:.6f}")
print(f"最終訓練 loss        : {cross_entropy(py_tr, y_train):.6f}")
print(f"最終測試 loss        : {final_loss:.6f}   （隨機水準 log K = {logK:.6f}）")
print(f"最終訓練準確率       : {final_tr:.4f}")
print(f"最終測試準確率       : {final_te:.4f}")
print(f"訓練後 θ             = {np.round(theta, 4)}")
print(f"W_read 每列最大值索引 = {np.argmax(w_read, axis=1).tolist()}")

cm = confusion(py_te, y_test)
print(f"\n測試集混淆矩陣（列=真，行=預測）：\n{cm}")
print(f"各類別召回率 = {np.round(np.diag(cm) / np.maximum(cm.sum(axis=1), 1), 4).tolist()}")

# ---- 繪圖 ----
fig_path = os.path.join("docs", "assets", "figs", "training_curve.png")
if HAS_MPL:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "MingLiU", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(history["step"], history["loss"], color="#4338ca", lw=2, label="訓練損失")
    ax.axhline(logK, ls="--", c="#b91c1c", lw=1.2, label=f"隨機水準 log K = {logK:.3f}")
    ax.axvline(STAGE1 + 0.5, ls=":", c="#047857", lw=1.5, label="第二階段起點")
    ax.set_xlabel("訓練步數")
    ax.set_ylabel("交叉熵損失")
    ax.set_title("(a) 損失曲線：200 步兩階段訓練")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["step"], history["train_acc"], color="#047857", lw=2, label="訓練準確率")
    ax.plot(history["step"], history["test_acc"], color="#b45309", lw=2, label="測試準確率")
    ax.axhline(1.0 / K_CLASS, ls="--", c="#b91c1c", lw=1.2, label="隨機水準 1/K")
    ax.axvline(STAGE1 + 0.5, ls=":", c="#047857", lw=1.5)
    ax.set_xlabel("訓練步數")
    ax.set_ylabel("準確率")
    ax.set_title("(b) 準確率：訓練 vs 測試")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    gs = [(s, g) for s, g in zip(history["step"], history["grad_norm"]) if g > 0]
    ax.plot([s for s, _ in gs], [g for _, g in gs], color="#7c3aed", lw=2)
    ax.set_xlabel("訓練步數")
    ax.set_ylabel(r"$\|\nabla_\theta L\|$")
    ax.set_title("(c) 電路參數梯度範數（第二階段）")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"\n圖已存檔：{fig_path}")
else:
    print("\n未安裝 matplotlib → 未產生圖檔（請見章節說明）")

print("\n" + BAR)
print("全部實驗完成")
print(BAR)
