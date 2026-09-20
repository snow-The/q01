# -*- coding: utf-8 -*-
"""dev/verify_5q_readout.py — 5 qubit QBN 輸出層：純 NumPy 參考實作與驗證腳本

用途
----
本書〈03-實作/五量子位輸出層實作.md〉的所有數字都由本腳本產生。
不需要任何量子套件（純 NumPy），任何人可重跑。

執行：
    uv run python dev/verify_5q_readout.py

位元順序約定（全書一致）
------------------------
**big-endian**：基底態 |q0 q1 q2 q3 q4>，q0 是最高位元（most significant bit）。
    索引 i = q0*16 + q1*8 + q2*4 + q3*2 + q4
    "00000" -> 0        "00001" -> 1        "10000" -> 16       "11111" -> 31

作者：W9（03-實作 教材撰寫）
"""
from __future__ import annotations

import numpy as np

# =============================================================================
# 0. 常數
# =============================================================================
N_QUBITS = 5
DIM = 1 << N_QUBITS          # 32
SEED = 42

# 糾纏拓撲（有向邊清單；CX 的 (control, target)）
RING = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
CHAIN = [(0, 1), (1, 2), (2, 3), (3, 4)]
ALL2ALL = [(0, 1), (0, 2), (0, 3), (0, 4),
           (1, 0), (1, 2), (1, 3), (1, 4),
           (2, 0), (2, 1), (2, 3), (2, 4),
           (3, 0), (3, 1), (3, 2), (3, 4),
           (4, 0), (4, 1), (4, 2), (4, 3)]


# =============================================================================
# 1. 基底態 ↔ 位元字串 ↔ 索引
# =============================================================================
def index_to_bitstring(i: int, n: int = N_QUBITS) -> str:
    """索引 -> 位元字串。0 -> '00000'（big-endian）。"""
    return format(i, f"0{n}b")


def bitstring_to_index(s: str) -> int:
    """位元字串 -> 索引。'00000' -> 0。"""
    return int(s, 2)


def index_to_bits(i: int, n: int = N_QUBITS) -> tuple[int, ...]:
    """索引 -> 位元 tuple，(q0, q1, q2, q3, q4)。"""
    return tuple((i >> (n - 1 - q)) & 1 for q in range(n))


def bits_to_index(bits) -> int:
    """位元 tuple -> 索引。"""
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def table_of_basis_states(n: int = N_QUBITS) -> list[tuple[int, str]]:
    """回傳 [(索引, 位元字串), ...]，共 2^n 列。"""
    return [(i, index_to_bitstring(i, n)) for i in range(1 << n)]


# =============================================================================
# 2. 量子閘（2x2 么正矩陣）
# =============================================================================
def ry_matrix(theta: float) -> np.ndarray:
    """R_Y(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]  —— 實數矩陣。"""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_matrix(phi: float) -> np.ndarray:
    """R_Z(phi) = diag(e^{-i phi/2}, e^{+i phi/2})  —— 複數，引入相對相位。"""
    return np.array([[np.exp(-0.5j * phi), 0.0],
                     [0.0, np.exp(0.5j * phi)]], dtype=complex)


def angle_from_prob(p: float) -> float:
    """反解角度：R_Y(theta)|0> 測得 0 的機率是 cos^2(theta/2)。

    P(0) = cos^2(theta/2)  =>  theta = 2 * arccos(sqrt(p))
    """
    p = float(np.clip(p, 0.0, 1.0))
    return 2.0 * np.arccos(np.sqrt(p))


def prob_from_angle(theta: float) -> float:
    """正向：cos^2(theta/2)。"""
    return float(np.cos(theta / 2.0) ** 2)


# =============================================================================
# 3. 狀態向量與閘的施加
# =============================================================================
def zero_state(n: int = N_QUBITS) -> np.ndarray:
    """|00...0>，長度 2^n 的複數向量。"""
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    return psi


def apply_1q(state: np.ndarray, u: np.ndarray, q: int, n: int = N_QUBITS) -> np.ndarray:
    """把單量子位閘 u 施加在第 q 個 qubit 上（big-endian 軸對應）。"""
    t = state.reshape([2] * n)
    t = np.moveaxis(t, q, 0)
    t = np.tensordot(u, t, axes=([1], [0]))
    t = np.moveaxis(t, 0, q)
    return t.reshape(-1)


def apply_cx(state: np.ndarray, control: int, target: int, n: int = N_QUBITS) -> np.ndarray:
    """受控 X：control 為 1 時翻轉 target。big-endian 索引運算。"""
    out = state.copy()
    cmask = 1 << (n - 1 - control)
    tmask = 1 << (n - 1 - target)
    for i in range(1 << n):
        if (i & cmask) and not (i & tmask):
            j = i | tmask
            out[i], out[j] = state[j], state[i]
    return out


# =============================================================================
# 4. 電路：編碼層 -> 糾纏層 -> 可訓練層（可堆疊）
# =============================================================================
def encode(angles, n: int = N_QUBITS) -> np.ndarray:
    """編碼層：每個 qubit 一個 R_Y(theta_i)。"""
    psi = zero_state(n)
    for q, th in enumerate(angles):
        psi = apply_1q(psi, ry_matrix(th), q, n)
    return psi


def entangle(state: np.ndarray, edges, n: int = N_QUBITS) -> np.ndarray:
    """依 edges 的順序施加 CX。"""
    for c, t in edges:
        state = apply_cx(state, c, t, n)
    return state


def trainable_layer(state: np.ndarray, ry_angles, rz_angles, n: int = N_QUBITS) -> np.ndarray:
    """可訓練層：每個 qubit 依序 R_Y(alpha_i) 然後 R_Z(beta_i)。

    注意：不同 qubit 的閘彼此交換，所以「逐 qubit 做完 RY+RZ」與
    「全部 RY 做完再做全部 RZ」在數學上等價。
    """
    for q in range(n):
        state = apply_1q(state, ry_matrix(ry_angles[q]), q, n)
        state = apply_1q(state, rz_matrix(rz_angles[q]), q, n)
    return state


def circuit_state(angles, params, edges=RING, n_layers: int = 1,
                  use_rz: bool = True, n: int = N_QUBITS) -> np.ndarray:
    """完整電路。

    angles : (n,)            編碼角（由前端決定，量子層不訓練）
    params : (n_layers,2,n)  [層, 0=RY / 1=RZ, qubit]
    """
    params = np.asarray(params, dtype=float).reshape(n_layers, 2, n)
    psi = encode(angles, n)
    for layer in range(n_layers):
        psi = entangle(psi, edges, n)
        ry_a = params[layer, 0]
        rz_a = params[layer, 1] if use_rz else np.zeros(n)
        psi = trainable_layer(psi, ry_a, rz_a, n)
    return psi


# =============================================================================
# 5. 測量：解析機率、取樣、期望值、約化態
# =============================================================================
def probabilities(state: np.ndarray) -> np.ndarray:
    """計算基底測量的解析機率：p_i = |<i|psi>|^2。"""
    return np.abs(state) ** 2


def sample_counts(probs: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """多項式取樣：模擬真機的 shots 次測量。"""
    return rng.multinomial(shots, probs)


def marginal_zero_prob(probs: np.ndarray, q: int, n: int = N_QUBITS) -> float:
    """P(q_q = 0)：邊際機率。"""
    idx = np.arange(1 << n)
    bit = (idx >> (n - 1 - q)) & 1
    return float(probs[bit == 0].sum())


def z_expectations(probs: np.ndarray, n: int = N_QUBITS) -> np.ndarray:
    """5 個 <Z_i> = P(q_i=0) - P(q_i=1) = 2 P(q_i=0) - 1。"""
    idx = np.arange(1 << n)
    out = np.empty(n)
    for q in range(n):
        bit = (idx >> (n - 1 - q)) & 1
        out[q] = probs[bit == 0].sum() - probs[bit == 1].sum()
    return out


def zz_correlations(probs: np.ndarray, n: int = N_QUBITS) -> np.ndarray:
    """<Z_i Z_j>，回傳 (n,n) 矩陣。對角線為 1。"""
    idx = np.arange(1 << n)
    sign = np.empty((n, 1 << n))
    for q in range(n):
        sign[q] = 1.0 - 2.0 * ((idx >> (n - 1 - q)) & 1)
    return sign @ (sign * probs).T


def reduced_density_1q(state: np.ndarray, q: int, n: int = N_QUBITS) -> np.ndarray:
    """單一 qubit 的約化密度矩陣（partial trace 掉其餘 n-1 個 qubit）。"""
    a = np.moveaxis(state.reshape([2] * n), q, 0).reshape(2, -1)
    return a @ a.conj().T


def von_neumann_entropy(rho: np.ndarray) -> float:
    """馮紐曼熵，以 bit 為單位。"""
    ev = np.linalg.eigvalsh(rho).real
    ev = ev[ev > 1e-15]
    return float(-(ev * np.log2(ev)).sum())


def output_entropy_bits(probs: np.ndarray) -> float:
    """輸出分布自身的夏農熵（bit）。"""
    p = probs[probs > 1e-15]
    return float(-(p * np.log2(p)).sum())


# =============================================================================
# 6. 三種讀出策略
# =============================================================================
def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def readout_expectation(probs: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a) 5 個 <Z_i> -> 線性層 (5 -> K)。

    注意：這裡的 softmax 屬於 **古典分類頭**（pince-nez 之後的古典節點），
    不是量子節點內部的歸一化。量子節點的歸一化由跡承擔（Pejic L1283）。
    """
    return softmax(z_expectations(probs) @ w + b)


def readout_full(probs: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(b) 32 維機率向量 -> 線性層 (32 -> K)。"""
    return softmax(probs @ w + b)


def grouped_bounds(k_class: int, dim: int = DIM) -> list[int]:
    """(c) 相鄰分組的切點。用 floor(k*DIM/K) 保證 K 不整除 32 時仍完整覆蓋。"""
    return [int(np.floor(k * dim / k_class)) for k in range(k_class)] + [dim]


def readout_grouped(probs: np.ndarray, k_class: int, dim: int = DIM) -> np.ndarray:
    """(c) 相鄰分組：零參數，每 32/K 個基底態一類。"""
    b = grouped_bounds(k_class, dim)
    return np.array([probs[b[k]:b[k + 1]].sum() for k in range(k_class)])


def n_params_expectation(k_class: int, n: int = N_QUBITS) -> int:
    """(a) 權重 n*K + 偏置 K。"""
    return n * k_class + k_class


def n_params_full(k_class: int, dim: int = DIM) -> int:
    """(b) 權重 32*K + 偏置 K。"""
    return dim * k_class + k_class


def n_params_grouped(k_class: int) -> int:
    """(c) 零參數。"""
    return 0


# =============================================================================
# 7. 驗證：四項正確性檢查
# =============================================================================
def check_1_probabilities_sum_to_one(rng) -> tuple[bool, float]:
    """[1] 32 個機率的和為 1。"""
    angles = rng.uniform(-np.pi, np.pi, N_QUBITS)
    params = rng.uniform(-np.pi, np.pi, (2, 2, N_QUBITS))
    probs = probabilities(circuit_state(angles, params, RING, 2))
    err = abs(probs.sum() - 1.0)
    return err < 1e-12, err


def check_2_all_zero_gives_ket_zero() -> tuple[bool, float]:
    """[2] 所有參數為 0 -> |00000> 機率 1。"""
    angles = np.zeros(N_QUBITS)
    params = np.zeros((2, 2, N_QUBITS))
    probs = probabilities(circuit_state(angles, params, RING, 2))
    return abs(probs[0] - 1.0) < 1e-12, float(probs[0])


def check_3_encoding_only_marginals(rng) -> tuple[bool, float, np.ndarray, np.ndarray]:
    """[3] 只開編碼層：5 個 qubit 的邊際 = 輸入的 5 個值。"""
    target_p = rng.uniform(0.05, 0.95, N_QUBITS)
    angles = np.array([angle_from_prob(p) for p in target_p])
    probs = probabilities(encode(angles))       # 沒有糾纏層
    got = np.array([marginal_zero_prob(probs, q) for q in range(N_QUBITS)])
    err = float(np.max(np.abs(got - target_p)))
    return err < 1e-12, err, target_p, got


def check_4_ring_entanglement_entropy() -> tuple[bool, np.ndarray, float]:
    """[4] 環形 CX 層之後，單一 qubit 的約化態熵 > 0（糾纏真的產生了）。

    注意：整條電路是么正的，**整體態仍是純態，整體熵恆為 0**。
    有意義的是「單一 qubit 的約化態熵」（糾纏熵）。
    """
    rng = np.random.default_rng(SEED)
    angles = rng.uniform(0.3, np.pi - 0.3, N_QUBITS)
    psi_before = encode(angles)
    psi_after = entangle(psi_before, RING)
    ent_before = np.array([von_neumann_entropy(reduced_density_1q(psi_before, q))
                           for q in range(N_QUBITS)])
    ent_after = np.array([von_neumann_entropy(reduced_density_1q(psi_after, q))
                          for q in range(N_QUBITS)])
    return bool(ent_after.min() > 1e-6), ent_before, float(ent_after.min())


# =============================================================================
# 8. 實驗：取樣誤差（100 個種子 x 4096 shots）
# =============================================================================
def experiment_sampling_error(n_trials: int = 100, shots: int = 4096):
    rng = np.random.default_rng(SEED)
    angles = np.array([0.9, 1.7, 2.4, 1.1, 2.9])
    params = rng.uniform(-np.pi, np.pi, (1, 2, N_QUBITS))
    probs = probabilities(circuit_state(angles, params, RING, 1))

    counts = np.array([sample_counts(probs, shots, rng) for _ in range(n_trials)])
    counts_std = counts.std(axis=0, ddof=1)
    theory_counts_std = np.sqrt(shots * probs * (1.0 - probs))

    phat = counts / shots
    phat_std = phat.std(axis=0, ddof=1)
    theory_p_std = np.sqrt(probs * (1.0 - probs) / shots)

    z_runs = np.array([z_expectations(c / shots) for c in counts])
    z_std = z_runs.std(axis=0, ddof=1)
    z_true = z_expectations(probs)
    theory_z_std = np.sqrt(np.maximum(1.0 - z_true ** 2, 0.0) / shots)

    return dict(probs=probs, shots=shots, n_trials=n_trials,
                counts=counts, counts_std=counts_std, theory_counts_std=theory_counts_std,
                phat_std=phat_std, theory_p_std=theory_p_std,
                z_std=z_std, theory_z_std=theory_z_std, z_true=z_true)


# =============================================================================
# 9. 實驗：拓撲比較、RZ 必要性、深度
# =============================================================================
def experiment_topology():
    rng = np.random.default_rng(SEED)
    angles = rng.uniform(0.3, np.pi - 0.3, N_QUBITS)
    rows = []
    for name, edges in (("線性鏈 chain", CHAIN), ("環形 ring", RING), ("全連接 all-to-all", ALL2ALL)):
        psi = entangle(encode(angles), edges)
        probs = probabilities(psi)
        ents = np.array([von_neumann_entropy(reduced_density_1q(psi, q)) for q in range(N_QUBITS)])
        zz = zz_correlations(probs)
        z1 = z_expectations(probs)
        # 二體關聯強度：| <ZiZj> - <Zi><Zj> | 的最大值
        corr = np.abs(zz - np.outer(z1, z1))
        np.fill_diagonal(corr, 0.0)
        rows.append(dict(name=name, n_cx=len(edges), mean_entropy=float(ents.mean()),
                         min_entropy=float(ents.min()), max_corr=float(corr.max())))
    return rows


def experiment_rz_necessity():
    """1 qubit 上 RY-RZ-RY 的可達 P(0) 範圍 vs 沒有 RZ 的 RY-RY。"""
    theta1 = theta2 = np.pi / 2.0
    without = prob_from_angle(theta1 + theta2)          # 兩個 RY 合成一個 RY
    phis = np.linspace(0.0, 2.0 * np.pi, 9)
    with_rz = []
    for phi in phis:
        psi = apply_1q(zero_state(1), ry_matrix(theta1), 0, 1)
        psi = apply_1q(psi, rz_matrix(phi), 0, 1)
        psi = apply_1q(psi, ry_matrix(theta2), 0, 1)
        with_rz.append(probabilities(psi)[0])
    return theta1, theta2, without, phis, np.array(with_rz)


def experiment_depth(n_layers_max: int = 4, n_draws: int = 200):
    """深度 vs 表達力（輸出分布的可動範圍）與梯度量級（貧瘠高原的經驗徵兆）。"""
    rng = np.random.default_rng(SEED)
    angles = np.array([0.8, 1.9, 2.2, 1.3, 2.7])
    rows = []
    for depth in range(1, n_layers_max + 1):
        draws = np.array([probabilities(circuit_state(
            angles, rng.uniform(-np.pi, np.pi, (depth, 2, N_QUBITS)), RING, depth))
            for _ in range(n_draws)])
        # 輸出分布之間的平均 TV 距離（表達力代理量）
        sub = draws[:60]
        tv = np.mean([0.5 * np.abs(sub[a] - sub[b]).sum()
                      for a in range(len(sub)) for b in range(a + 1, len(sub))])
        ent = np.array([output_entropy_bits(p) for p in draws])

        # 梯度量級：對 2*n*depth 個參數做中央差分，取 |dp/dtheta| 的平均
        grads = []
        for _ in range(20):
            th = rng.uniform(-np.pi, np.pi, (depth, 2, N_QUBITS))
            eps = 1e-5
            for layer in range(depth):
                for kind in range(2):
                    for q in range(N_QUBITS):
                        tp = th.copy(); tp[layer, kind, q] += eps
                        tm = th.copy(); tm[layer, kind, q] -= eps
                        pp = probabilities(circuit_state(angles, tp, RING, depth))
                        pm = probabilities(circuit_state(angles, tm, RING, depth))
                        grads.append(np.abs((pp - pm) / (2 * eps)).max())
        rows.append(dict(depth=depth, n_params_quantum=2 * N_QUBITS * depth,
                         mean_tv=float(tv), mean_entropy=float(ent.mean()),
                         min_entropy=float(ent.min()),
                         mean_grad=float(np.mean(grads)), max_grad=float(np.max(grads))))
    return rows


# =============================================================================
# 10. 端到端驗證
# =============================================================================
def end_to_end(seed: int = 7, shots: int = 4096, depths=(1, 2)):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, N_QUBITS)
    out = {}
    for depth in depths:
        params = rng.uniform(-np.pi, np.pi, (depth, 2, N_QUBITS))
        psi = circuit_state(angles, params, RING, depth)
        probs = probabilities(psi)
        sum_err = abs(probs.sum() - 1.0)
        counts = sample_counts(probs, shots, rng)
        z = z_expectations(probs)
        out[depth] = dict(angles=angles, params=params, probs=probs, sum_err=sum_err,
                          counts=counts, z=z, entropy=output_entropy_bits(probs))
    return out


def main() -> None:
    line = "=" * 78
    print(line)
    print("dev/verify_5q_readout.py — 5 qubit QBN 輸出層驗證")
    print(f"numpy {np.__version__} | qubits={N_QUBITS} | dim={DIM} | seed={SEED}")
    print(line)

    # ---- 0. 基底態表 ----
    print("\n[0] 32 個基底態（big-endian，q0 為最高位元）")
    tbl = table_of_basis_states()
    for row in range(0, DIM, 4):
        cells = "  ".join(f"{i:>2}:'{s}'" for i, s in tbl[row:row + 4])
        print("    " + cells)

    # ---- 1..4 正確性檢查 ----
    rng = np.random.default_rng(SEED)
    print("\n[1] 32 個機率的和為 1")
    ok1, err1 = check_1_probabilities_sum_to_one(rng)
    print(f"    |sum(p) - 1| = {err1:.3e}   -> {'PASS' if ok1 else 'FAIL'}")

    print("\n[2] 所有參數為 0 -> |00000> 機率 1")
    ok2, p0 = check_2_all_zero_gives_ket_zero()
    print(f"    P(|00000>) = {p0!r}   -> {'PASS' if ok2 else 'FAIL'}")

    print("\n[3] 只開編碼層：邊際機率 = 輸入值")
    ok3, err3, want, got = check_3_encoding_only_marginals(rng)
    for q in range(N_QUBITS):
        print(f"    q{q}: 輸入 {want[q]:.12f}  得到 {got[q]:.12f}  差 {abs(want[q]-got[q]):.2e}")
    print(f"    最大絕對誤差 = {err3:.3e}   -> {'PASS' if ok3 else 'FAIL'}")

    print("\n[4] 環形 CX 層的糾纏熵")
    ok4, ent_before, ent_after_min = check_4_ring_entanglement_entropy()
    print(f"    糾纏前單一 qubit 熵 = {np.array2string(ent_before, precision=10)}")
    rng4 = np.random.default_rng(SEED)
    a4 = rng4.uniform(0.3, np.pi - 0.3, N_QUBITS)
    psi4 = entangle(encode(a4), RING)
    ent_after = np.array([von_neumann_entropy(reduced_density_1q(psi4, q)) for q in range(N_QUBITS)])
    print(f"    糾纏後單一 qubit 熵 = {np.array2string(ent_after, precision=6)}")
    print(f"    整體態熵 = {von_neumann_entropy(np.outer(psi4, psi4.conj())):.3e} (么正演化 -> 恆為 0)")
    print(f"    最小糾纏熵 = {ent_after_min:.6f} > 0   -> {'PASS' if ok4 else 'FAIL'}")

    # ---- 5. 取樣誤差 ----
    print("\n[5] 取樣誤差：100 個種子 x 4096 shots")
    se = experiment_sampling_error()
    print(f"    解析機率前 6 個 = {np.array2string(se['probs'][:6], precision=6)}")
    print(f"    最大解析機率 = {se['probs'].max():.6f} (索引 {int(se['probs'].argmax())}, "
          f"'{index_to_bitstring(int(se['probs'].argmax()))}')")
    print(f"    機率估計標準差：實測最大 {se['phat_std'].max():.6f} | "
          f"理論最大 {se['theory_p_std'].max():.6f} | 1/(2*sqrt(N)) = {0.5/np.sqrt(se['shots']):.6f}")
    print(f"    計數標準差：實測最大 {se['counts_std'].max():.4f} | "
          f"理論最大 {se['theory_counts_std'].max():.4f}")
    print(f"    <Z> 估計標準差：實測最大 {se['z_std'].max():.6f} | "
          f"理論最大 {se['theory_z_std'].max():.6f} | 1/sqrt(N) = {1.0/np.sqrt(se['shots']):.6f}")
    print(f"    100 次取樣中，單一基底態計數的最大絕對偏差 = "
          f"{np.abs(se['counts'] - se['probs']*se['shots']).max():.1f} 個計數")

    # ---- 6. 拓撲比較 ----
    print("\n[6] 糾纏拓撲比較（同一組編碼角，只開編碼層 + 糾纏層）")
    print(f"    {'拓撲':<18}{'CX 數':>6}{'平均糾纏熵':>14}{'最小糾纏熵':>14}{'最大 |<ZiZj>-<Zi><Zj>|':>26}")
    for r in experiment_topology():
        print(f"    {r['name']:<18}{r['n_cx']:>6}{r['mean_entropy']:>14.6f}"
              f"{r['min_entropy']:>14.6f}{r['max_corr']:>26.6f}")

    # ---- 7. RZ 必要性 ----
    print("\n[7] R_Z 必要性：單一 qubit 的 R_Y(pi/2) -> R_Z(phi) -> R_Y(pi/2)")
    t1, t2, without, phis, with_rz = experiment_rz_necessity()
    print(f"    沒有 R_Z（兩個 R_Y 合成 R_Y(pi)）: P(0) = {without:.12f}")
    for phi, val in zip(phis, with_rz):
        print(f"    phi = {phi:.4f} rad -> P(0) = {val:.12f}")
    print(f"    有 R_Z 時 P(0) 的範圍 = [{with_rz.min():.6f}, {with_rz.max():.6f}]")

    # ---- 8. 深度 ----
    print("\n[8] 電路深度 vs 表達力 / 梯度量級（200 次隨機抽樣，梯度用中央差分）")
    print(f"    {'深度':>4}{'量子參數數':>12}{'平均 TV 距離':>16}{'平均輸出熵':>14}"
          f"{'最小輸出熵':>14}{'平均|dp/dθ|':>16}")
    for r in experiment_depth():
        print(f"    {r['depth']:>4}{r['n_params_quantum']:>12}{r['mean_tv']:>16.6f}"
              f"{r['mean_entropy']:>14.6f}{r['min_entropy']:>14.6f}{r['mean_grad']:>16.6f}")

    # ---- 9. 端到端 ----
    print("\n[9] 端到端驗證（seed=7, shots=4096）")
    res = end_to_end()
    for depth, r in res.items():
        print(f"\n    --- 深度 {depth}（量子參數 {2*N_QUBITS*depth} 個）---")
        print(f"    編碼角 θ = {np.array2string(r['angles'], precision=6)}")
        print(f"    |sum(p) - 1| = {r['sum_err']:.3e}")
        print(f"    輸出熵 = {r['entropy']:.6f} bit （上限 log2(32) = 5）")
        print(f"    <Z_i> = {np.array2string(r['z'], precision=6)}")
        top = np.argsort(r['probs'])[::-1][:5]
        print("    前 5 高機率的基底態：")
        for i in top:
            print(f"        |{index_to_bitstring(int(i))}>  解析 {r['probs'][i]:.6f}  "
                  f"取樣 {r['counts'][i]:>5}  ({r['counts'][i]/4096:.6f})")
        print(f"    最大機率的基底態 = |{index_to_bitstring(int(r['probs'].argmax()))}> "
              f"({r['probs'].max():.6f})")

    # ---- 10. 三種讀出策略 ----
    print("\n[10] 三種讀出策略（K=2 與 K=10）")
    r0 = res[1]
    for K in (2, 10):
        rngr = np.random.default_rng(SEED + K)
        w_a = rngr.normal(0, 1, (N_QUBITS, K)); b_a = rngr.normal(0, 1, K)
        w_b = rngr.normal(0, 1, (DIM, K));      b_b = rngr.normal(0, 1, K)
        pa = readout_expectation(r0['probs'], w_a, b_a)
        pb = readout_full(r0['probs'], w_b, b_b)
        pc = readout_grouped(r0['probs'], K)
        print(f"\n    === K = {K} ===")
        print(f"    (a) 5 個 <Z_i> -> 線性 (5->{K})   參數 {n_params_expectation(K):>4}  "
              f"分布 = {np.array2string(pa, precision=6)}  和 = {pa.sum():.12f}")
        print(f"    (b) 32 維機率 -> 線性 (32->{K})   參數 {n_params_full(K):>4}  "
              f"分布 = {np.array2string(pb, precision=6)}  和 = {pb.sum():.12f}")
        print(f"    (c) 相鄰分組（零參數）            參數 {n_params_grouped(K):>4}  "
              f"分布 = {np.array2string(pc, precision=6)}  和 = {pc.sum():.12f}")
        print(f"    參數量差 (b)-(a) = {n_params_full(K) - n_params_expectation(K)}")
        if K in (2, 4, 8, 16, 32):
            print(f"    分組切點 = {grouped_bounds(K)}")
        else:
            print(f"    分組切點 = {grouped_bounds(K)}  （32 不被 {K} 整除，用 floor 保證完整覆蓋）")
            sizes = np.diff(grouped_bounds(K))
            print(f"    各組大小 = {sizes.tolist()}  總和 = {sizes.sum()}")

    # ---- 11. K=2 時 32 個基底態如何被三種策略歸類 ----
    print("\n[11] K=2：32 個基底態在三種策略下的歸類")
    print("    (a)/(b) 是學習式的，沒有先驗歸類（標籤由權重決定）")
    print(f"    (c) 相鄰分組的固定歸類：類 0 = 索引 0..15，類 1 = 索引 16..31")
    for cls, rng_ in ((0, range(0, 16)), (1, range(16, 32))):
        s = " ".join(index_to_bitstring(i) for i in rng_)
        print(f"        類 {cls}: {s}")

    print("\n" + line)
    print("全部檢查完成。")
    print(line)


if __name__ == "__main__":
    main()
