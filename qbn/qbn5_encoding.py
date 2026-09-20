# -*- coding: utf-8 -*-
"""qbn5_encoding.py — 5 qubit 輸出層的完整參考實作（純 NumPy 古典模擬版）。

本檔是 docs/03-實作/無CUDA-Q的參考實作.md 的可執行程式，也是
docs/02-QBN理論/五量子位輸出層設計.md 的實驗依據。它負責四件事：

1. **角度編碼**：把 5 維古典特徵映射成 5 個 :math:`R_Y` 角度（`encode_angles`）。
2. **變分電路**：5 個 :math:`R_Y` 編碼 + 環形 CX 糾纏 + 可訓練 :math:`R_Y/R_Z`
   （`qbn_layer`），深度可參數化。
3. **讀出**：32 維機率向量（`readout`），以及把 32 維聚合成 :math:`K` 類的
   四種策略（`collapse_to_classes`）。
4. **去相位消融**：對應 Tucci 的古典化算子 ``cl``（`dephase`）。

執行方式::

    python dev/qbn5_encoding.py

需要 numpy，**不需要 CUDA-Q**。

去相位的理論依據
----------------
Tucci 的 ``cl`` 算子定義為

.. math::
    \\mathrm{cl}_{\\underline{b}}(\\rho_{\\underline{b},\\underline{a}})
      = \\sum_b \\big[|b\\rangle_{\\underline{b}}\\langle b|_{\\underline{b}}\\;
        \\rho_{\\underline{b},\\underline{a}}\\big]\\,[\\mathrm{h.c.}]

其 Kraus 算子為 :math:`K_a=|a\\rangle\\langle a|`，效果就是**丟掉所有非對角項**。
出處：docs/_extract/F1-tucci-qbn-mixed-states.md
【L358】（定義）、【L410】（「這個操作有時也稱為去相位，因為我們丟掉了一些
非對角項」）、【L526】（Kraus 算子）、【L53】（對每個節點套 ``cl``，
QB net 就退化成古典貝氏網路）。
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qbn_sim import StateVectorSim

__all__ = [
    "N_QUBITS",
    "DIM",
    "encode_angles",
    "angles_for_prob_zero",
    "ring_pairs",
    "qbn_layer",
    "qbn_layer_ablated",
    "readout",
    "adjacent_grouping",
    "hamming_grouping",
    "gray_code_order",
    "POVMReadout",
    "collapse_to_classes",
    "dephase",
    "dephased_probabilities",
    "classical_joint_from_rho",
]

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
N_QUBITS: int = 5
DIM: int = 2 ** N_QUBITS          # 32 —— 即希爾伯特空間的維度
N_UNITARY_PARAMS: int = DIM ** 2 - 1   # 1023 —— su(32) 的維度

# 兩條不同的 5-cycle（環形糾纏拓撲）。
# 出處：Wang et al. (2026) 的 ring topology，
# docs/_extract/F3-bayesian-frontend-hybrid-qc.md【L220】【L261】。
RING_STEP1: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
RING_STEP2: List[Tuple[int, int]] = [(0, 2), (2, 4), (4, 1), (1, 3), (3, 0)]


# ===========================================================================
# 1. 角度編碼
# ===========================================================================
def encode_angles(x: np.ndarray) -> np.ndarray:
    r"""把 :math:`n` 維特徵 :math:`x\in[0,1]^n` 映射成 :math:`n` 個 :math:`R_Y` 角度。

    .. math::
        \theta_i = 2\arcsin\!\big(\sqrt{x_i}\big)

    推導：:math:`R_Y(\theta)|0\rangle=\cos\frac\theta2|0\rangle+\sin\frac\theta2|1\rangle`，
    所以 :math:`P(|1\rangle)=\sin^2\frac\theta2`。代入
    :math:`\frac\theta2=\arcsin(\sqrt{x_i})` 得 :math:`P(|1\rangle)=x_i`。

    **注意公約**：這裡讓「**測到 1 的機率**」等於特徵值 :math:`x_i`。
    若你要的是「測到 0 的機率」等於 :math:`x_i`，請改用 `angles_for_prob_zero`。
    （〈QBN 形式定義〉的 3 節點範例用的是後者，見 docs/02-QBN理論/QBN形式定義.md:253。）
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x 必須是 1 維向量，收到形狀 {x.shape}")
    if np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
        raise ValueError(
            f"x 必須在 [0,1]（或先正規化），實際範圍 [{x.min():.6g}, {x.max():.6g}]"
        )
    return 2.0 * np.arcsin(np.sqrt(np.clip(x, 0.0, 1.0)))


def angles_for_prob_zero(p: np.ndarray) -> np.ndarray:
    r"""使 :math:`P(|0\rangle)=p_i` 的角度：:math:`\theta_i=2\arccos(\sqrt{p_i})`。"""
    p = np.asarray(p, dtype=float)
    if np.any(p < -1e-12) or np.any(p > 1.0 + 1e-12):
        raise ValueError(f"p 必須在 [0,1]，實際範圍 [{p.min():.6g}, {p.max():.6g}]")
    return 2.0 * np.arccos(np.sqrt(np.clip(p, 0.0, 1.0)))


def ring_pairs(n_qubits: int = N_QUBITS, step: int = 1) -> List[Tuple[int, int]]:
    """回傳 :math:`n` 個 qubit 上的環形糾纏配對 ``(control, target)``。

    ``step`` 與 :math:`n` 必須互質，這樣 ``i -> i+step`` 才是一條完整的環。
    對 :math:`n=5`：``step=1`` 與 ``step=2`` 都合法（5 是質數）。
    """
    if np.gcd(step, n_qubits) != 1:
        raise ValueError(f"step={step} 與 n_qubits={n_qubits} 不互質，無法構成單一環")
    return [(i, (i + step) % n_qubits) for i in range(n_qubits)]


# ===========================================================================
# 2. 變分電路
# ===========================================================================
def _layer_pairs(d: int, n_qubits: int, entanglement: str) -> List[Tuple[int, int]]:
    """第 ``d`` 層糾纏層的配對。環形時兩層用不同的 :math:`n`-cycle 交替。"""
    if entanglement == "ring":
        return ring_pairs(n_qubits, step=1 if d % 2 == 0 else 2)
    return [(i, i + 1) for i in range(n_qubits - 1)]


def qbn_layer(
    angles: np.ndarray,
    weights: np.ndarray,
    depth: int = 2,
    entanglement: str = "ring",
    n_qubits: int = N_QUBITS,
) -> StateVectorSim:
    r"""建構 QBN 的 5 qubit 輸出層電路。

    電路結構（依 Wang et al. 2026 的 Circuit 2 骨架，
    docs/_extract/F3-bayesian-frontend-hybrid-qc.md【L222】【L261】）::

        |0>^5 --[ RY(θ_i) 角度編碼 ]--+--[ 可訓練 RY(φ) RZ(λ) ]--[ 環形 CX ]--+-- ...
                                      |                                      |
                                      +-------- 重複 depth 次 --------------+

    Args:
        angles: 長度 :math:`n` 的編碼角度（由 `encode_angles` 產生）。
        weights: 形狀 ``(depth, n, 2)`` 的可訓練參數，最後一軸是 ``[RY, RZ]``。
        depth: 可訓練層數。預設 2，理由見〈五量子位輸出層設計〉的深度取捨一節。
        entanglement: ``"ring"``（兩條不同的 5-cycle 交替）或 ``"chain"``（線性鏈）。
        n_qubits: qubit 數，預設 5。

    Returns:
        已演化完畢的 `StateVectorSim`。

    參數量：``depth * n * 2``。``depth=2, n=5`` 時是 20 個實數。
    對照：:math:`\mathfrak{su}(32)` 的維度是 :data:`N_UNITARY_PARAMS` = 1023，
    所以我們的 ansatz 是**高度受限的么正子集**——這是為了可訓練性刻意付出的代價。

    若要在電路**中間**插入去相位（古典化某個節點）以做消融，請用
    `qbn_layer_ablated`。
    """
    angles = np.asarray(angles, dtype=float).ravel()
    if angles.size != n_qubits:
        raise ValueError(f"angles 長度必須是 {n_qubits}，收到 {angles.size}")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (depth, n_qubits, 2):
        raise ValueError(
            f"weights 形狀必須是 {(depth, n_qubits, 2)}，收到 {weights.shape}"
        )
    if entanglement not in ("ring", "chain"):
        raise ValueError(f"entanglement 必須是 'ring' 或 'chain'，收到 {entanglement!r}")

    sim = StateVectorSim(n_qubits)

    # --- 編碼層：angle encoding ---
    for i in range(n_qubits):
        sim.ry(float(angles[i]), i)

    # --- 可訓練層 + 糾纏層 ---
    for d in range(depth):
        for i in range(n_qubits):
            sim.ry(float(weights[d, i, 0]), i)
            sim.rz(float(weights[d, i, 1]), i)
        for c, t in _layer_pairs(d, n_qubits, entanglement):
            sim.cx(c, t)

    return sim


def qbn_layer_ablated(
    angles: np.ndarray,
    weights: np.ndarray,
    depth: int = 2,
    dephase_at: Optional[int] = None,
    targets: Optional[Sequence[int]] = None,
    entanglement: str = "ring",
    n_qubits: int = N_QUBITS,
):
    r"""與 `qbn_layer` 相同，但可在第 ``dephase_at`` 層之後插入去相位（``cl``）。

    這是本書最重要的消融實驗的實作。三個必須分清楚的設定：

    **設定一：去相位擺在電路最末端（緊接量測之前）。**
    此時 32 維量測機率**完全不變**，差異恆為 0。理由：:math:`P(i)=\rho_{ii}`，
    而去相位只殺非對角項。**這個設定下消融是觀察不到的**——書中必須誠實說明，
    否則會誤導讀者以為「加了去相位就會掉準確率」。

    **設定二：去相位擺在中間，之後只接 CX。**
    差異**仍然是 0**。理由更微妙：CX 在計算基底上是一個**置換矩陣**
    （:math:`|b\rangle\mapsto|b\oplus e_t\rangle`），而計算基底的去相位正是
    「投影到該基底的對角代數」。任何基底置換都保持對角代數不變，所以

    .. math::
        \mathrm{cl}\big(\mathrm{CX}\,\rho\,\mathrm{CX}^\dagger\big)
        = \mathrm{CX}\,\mathrm{cl}(\rho)\,\mathrm{CX}^\dagger .

    換言之，**只由 CX 與對角閘（RZ）組成的電路，去相位在量測上是不可觀測的**。

    **設定三：去相位擺在中間，之後接會混基底的閘（:math:`R_Y`、:math:`R_X`、:math:`H`）。**
    此時差異才出現。因為去相位把態變成古典混態，後續的 :math:`R_Y`
    對混態中每個分量作用後**不再互相干涉**，與全量子版本的結果分道揚鑣。

    這三點合起來才是「去相位消融」的正確敘述，`main` 會把三者一起印出來。

    Args:
        dephase_at: 在第幾層（0-based）之後去相位；``None`` 表示不去相位。
        targets: 要古典化的 qubit；``None`` 表示全部。
    """
    angles = np.asarray(angles, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (depth, n_qubits, 2):
        raise ValueError(
            f"weights 形狀必須是 {(depth, n_qubits, 2)}，收到 {weights.shape}"
        )
    if dephase_at is not None and not (0 <= dephase_at < depth):
        raise ValueError(f"dephase_at={dephase_at} 必須在 [0, {depth - 1}] 或 None")

    sim: object = StateVectorSim(n_qubits)
    for i in range(n_qubits):
        sim.ry(float(angles[i]), i)                     # type: ignore[attr-defined]

    for d in range(depth):
        for i in range(n_qubits):
            sim.ry(float(weights[d, i, 0]), i)          # type: ignore[attr-defined]
            sim.rz(float(weights[d, i, 1]), i)          # type: ignore[attr-defined]
        for c, t in _layer_pairs(d, n_qubits, entanglement):
            sim.cx(c, t)                                # type: ignore[attr-defined]

        if dephase_at is not None and d == dephase_at:
            tg = list(range(n_qubits)) if targets is None else list(targets)
            sim = _sim_from_rho(dephase(sim, tg), n_qubits)   # type: ignore[arg-type]

    return sim


# ===========================================================================
# 3. 讀出
# ===========================================================================
def readout(sim: StateVectorSim) -> np.ndarray:
    r"""回傳 32 維機率向量 :math:`P(i)=|\alpha_i|^2`（計算基底量測）。

    對應定義 2.5 的 Born 規則 :math:`P(x)=\Tr[M_x\rho]`，
    其中 :math:`M_x=|x\rangle\langle x|`（docs/02-QBN理論/QBN形式定義.md:131）。
    """
    return sim.probabilities()


def basis_labels(n_qubits: int = N_QUBITS) -> List[str]:
    """回傳 32 個基底態的位元字串標籤（最左字元是 qubit 0）。"""
    return [format(i, f"0{n_qubits}b") for i in range(2 ** n_qubits)]


# ===========================================================================
# 4. 把 32 維機率聚合成 K 類
# ===========================================================================
def _group_index(n_basis: int, n_classes: int, order: Optional[np.ndarray] = None) -> np.ndarray:
    """把 ``n_basis`` 個基底態指派到 ``n_classes`` 組（回傳長度 n_basis 的組別索引）。

    ``order`` 給定時，先依該排列重排基底再分組（Gray code 策略用）。
    """
    if n_classes < 1:
        raise ValueError(f"n_classes 必須 >= 1，收到 {n_classes}")
    if n_classes > n_basis:
        raise ValueError(f"n_classes={n_classes} 超過基底數 {n_basis}")
    idx = np.arange(n_basis) if order is None else np.asarray(order)
    # 以「等分邊界」分組：第 i 個基底（重排後的位置 p）屬於第 floor(p*K/N) 組
    pos = np.empty(n_basis, dtype=int)
    pos[idx] = np.arange(n_basis)
    return (pos * n_classes) // n_basis


def gray_code_order(n_bits: int) -> np.ndarray:
    """回傳長度 :math:`2^n` 的 Gray code 順序（相鄰元素只差一個位元）。"""
    idx = np.arange(2 ** n_bits)
    return idx ^ (idx >> 1)


def adjacent_grouping(
    probs: np.ndarray, n_classes: int, use_gray: bool = False
) -> np.ndarray:
    r"""策略 A/B：**相鄰分組**。每 :math:`32/K` 個相鄰基底態算一類。

    :math:`K\in\{2,4,8,16,32\}` 時每組大小完全相同；
    其他 :math:`K` 值（例如 10）會產生「前面幾組多一個」的不等分組。

    ``use_gray=True`` 時先按 Gray code 重排：相鄰的基底態只差一個位元，
    這讓同一類裡的狀態在漢明距離上彼此靠近，**避免「漢明斷崖」**
    （相鄰索引其實是 01111 與 10000，五個位元全變）。

    這是**常數讀出**：不需要任何可訓練參數，且它就是一個
    :math:`K\times 32` 的 0/1 行隨機矩陣（見 `POVMReadout` 的說明）。
    """
    probs = np.asarray(probs, dtype=float).ravel()
    other = 2 ** int(round(np.log2(probs.size)))
    if probs.size != other:
        raise ValueError(f"probs 長度必須是 2 的次方，收到 {probs.size}")
    order = gray_code_order(int(round(np.log2(probs.size)))) if use_gray else None
    gi = _group_index(probs.size, n_classes, order)
    return np.bincount(gi, weights=probs, minlength=n_classes)


def hamming_grouping(probs: np.ndarray, n_classes: int = 6) -> np.ndarray:
    r"""策略 C：**漢明權重分組**。把「1 的個數相同」的基底態歸為一類。

    5 個 qubit 的漢明權重是 :math:`0,1,2,3,4,5`，共 6 組，大小分別是
    :math:`1,5,10,10,5,1`（二項式係數）。這對應「有幾個 qubit 是激發態」
    這個**有物理意義**的量，可解釋性最好。

    :math:`K\ne 6` 時，把相鄰的權重組依比例合併。
    """
    probs = np.asarray(probs, dtype=float).ravel()
    n_qubits = int(round(np.log2(probs.size)))
    if 2 ** n_qubits != probs.size:
        raise ValueError(f"probs 長度必須是 2 的次方，收到 {probs.size}")
    idx = np.arange(probs.size)
    popcount = np.array([bin(int(i)).count("1") for i in idx])
    if n_classes == n_qubits + 1:
        gi = popcount
    else:
        gi = (popcount * n_classes) // (n_qubits + 1)
        gi = np.minimum(gi, n_classes - 1)
    return np.bincount(gi, weights=probs, minlength=n_classes)


class POVMReadout:
    r"""策略 D：**學習式線性讀出**——它就是一個 POVM。

    古典側的線性讀出是一個 :math:`K\times 32` 矩陣 :math:`W`：

    .. math::
        p^{\mathrm{class}}_k=\sum_{j=0}^{31}W_{kj}\,P(j)
        =\sum_j W_{kj}\,\langle j|\rho|j\rangle
        =\Tr\!\Big[\underbrace{\Big(\sum_j W_{kj}|j\rangle\langle j|\Big)}_{E_k}\rho\Big]

    所以只要

    * :math:`W_{kj}\ge 0`（則 :math:`E_k\succeq 0`，正性）
    * :math:`\sum_k W_{kj}=1` 對每個 :math:`j`（則 :math:`\sum_k E_k=I`，完備性）

    這個線性讀出就**恰好是一個 POVM**，機率歸一由 :math:`\sum_k E_k=I` 保證，
    **不需要任何 softmax**。這是 Pejic 論文約束 8（pince-nez 讀出用 POVM + Born
    規則）的直接落地，見 docs/_extract/F2-pejic-qbn-dissertation.md【L1295】【L1731】。

    參數量：:math:`K\times 32`，扣掉 :math:`K` 條「行和為 1」的約束
    ⇒ :math:`K(K\text{-}1)`... 精確地說自由度是 :math:`32(K-1)`。

    實作細節：用 softmax **沿著類別軸**參數化，只是為了自動滿足「行和為 1」
    這個約束。在推論路徑上 :math:`P\to p^{\mathrm{class}}` 仍然是**線性**映射，
    這點很重要——書中「資料路徑上不能有 softmax」的禁令針對的是後者。
    """

    def __init__(self, weight_matrix: np.ndarray) -> None:
        W = np.asarray(weight_matrix, dtype=float)
        if W.ndim != 2:
            raise ValueError(f"weight_matrix 必須是 2 維，收到形狀 {W.shape}")
        if np.any(W < -1e-12):
            raise ValueError("weight_matrix 必須非負（否則 POVM 元素不正）")
        if not np.allclose(W.sum(axis=0), 1.0, atol=1e-9):
            raise ValueError("weight_matrix 的行和必須為 1（否則 sum_k E_k != I）")
        self.W: np.ndarray = W

    # -- 建構子 ---------------------------------------------------------
    @classmethod
    def from_logits(cls, logits: np.ndarray) -> "POVMReadout":
        """由未正規化的 logits 建構：沿類別軸做 softmax，保證行和為 1 且非負。"""
        L = np.asarray(logits, dtype=float)
        if L.ndim != 2:
            raise ValueError(f"logits 必須是 2 維，收到形狀 {L.shape}")
        e = np.exp(L - L.max(axis=0, keepdims=True))
        return cls(e / e.sum(axis=0, keepdims=True))

    @classmethod
    def random(cls, n_classes: int, dim: int = DIM, seed: Optional[int] = None) -> "POVMReadout":
        """隨機初始化（僅供示範；實務上要用梯度下降學 ``W``）。"""
        rng = np.random.default_rng(seed)
        return cls.from_logits(rng.normal(0.0, 1.0, size=(n_classes, dim)))

    @classmethod
    def from_grouping(cls, group_index: np.ndarray) -> "POVMReadout":
        """由硬分組（如相鄰分組／漢明分組）建出對應的 0/1 讀出矩陣。

        這說明策略 A/B/C 都只是策略 D 的特例——它們的 ``W`` 是 0/1 矩陣。
        """
        gi = np.asarray(group_index, dtype=int).ravel()
        K = int(gi.max()) + 1
        W = np.zeros((K, gi.size), dtype=float)
        W[gi, np.arange(gi.size)] = 1.0
        return cls(W)

    # -- 使用 -----------------------------------------------------------
    def __call__(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float).ravel()
        if probs.size != self.W.shape[1]:
            raise ValueError(
                f"probs 長度必須是 {self.W.shape[1]}，收到 {probs.size}"
            )
        return self.W @ probs

    @property
    def n_classes(self) -> int:
        return self.W.shape[0]

    def as_povm_elements(self) -> np.ndarray:
        r"""回傳 :math:`K` 個 :math:`32\times32` 的對角 POVM 元素 :math:`E_k`。

        用來**驗證** :math:`E_k\succeq 0` 與 :math:`\sum_k E_k=I`。
        """
        K, d = self.W.shape
        out = np.zeros((K, d, d), dtype=complex)
        for k in range(K):
            out[k] = np.diag(self.W[k].astype(complex))
        return out

    def verify_povm(self) -> Tuple[bool, str]:
        """檢查 POVM 的兩條公理（正性、完備性）。回傳 ``(是否通過, 說明)``。"""
        E = self.as_povm_elements()
        psd = all(np.min(np.linalg.eigvalsh(E[k])) >= -1e-12 for k in range(len(E)))
        complete = np.allclose(E.sum(axis=0), np.eye(E.shape[1]), atol=1e-12)
        ok = bool(psd and complete)
        msg = (
            f"正性 E_k>=0: {psd}; 完備性 sum_k E_k = I: {complete}; "
            f"max|sum E_k - I| = {np.max(np.abs(E.sum(axis=0) - np.eye(E.shape[1]))):.3g}"
        )
        return ok, msg


def collapse_to_classes(
    probs: np.ndarray,
    n_classes: int,
    strategy: str = "adjacent",
    readout_matrix: Optional[POVMReadout] = None,
) -> np.ndarray:
    r"""把 32 維機率聚合成 :math:`K` 類。

    Args:
        probs: 32 維機率向量。
        n_classes: 類別數 :math:`K`。
        strategy: ``"adjacent"`` | ``"gray"`` | ``"hamming"`` | ``"learned"``。
        readout_matrix: ``strategy="learned"`` 時必須提供的 `POVMReadout`。

    Returns:
        長度 :math:`K` 的類別機率（**恆為非負且和為 1**）。

    四種策略的比較見檔尾 `main` 的輸出，以及
    docs/02-QBN理論/五量子位輸出層設計.md 的策略比較表。
    """
    probs = np.asarray(probs, dtype=float).ravel()
    if strategy == "adjacent":
        return adjacent_grouping(probs, n_classes)
    if strategy == "gray":
        return adjacent_grouping(probs, n_classes, use_gray=True)
    if strategy == "hamming":
        return hamming_grouping(probs, n_classes)
    if strategy == "learned":
        if readout_matrix is None:
            raise ValueError("strategy='learned' 需要提供 readout_matrix")
        if readout_matrix.n_classes != n_classes:
            raise ValueError(
                f"readout_matrix 有 {readout_matrix.n_classes} 類，與 n_classes={n_classes} 不符"
            )
        return readout_matrix(probs)
    raise ValueError(
        f"未知策略 {strategy!r}；可用：adjacent / gray / hamming / learned"
    )


# ===========================================================================
# 5. 去相位（Tucci 的 cl 算子）
# ===========================================================================
def _coherence_mask(n_qubits: int, targets: Sequence[int]) -> np.ndarray:
    """回傳 ``(2^n, 2^n)`` 布林遮罩：兩索引在 ``targets`` 上完全相同者為 True。

    這就是「保留對角項（含在被去相位 qubit 上同值的區塊）、殺掉其餘」的遮罩。
    """
    dim = 2 ** n_qubits
    idx = np.arange(dim)[:, None]
    shifts = (n_qubits - 1 - np.arange(n_qubits))[None, :]
    bits = (idx >> shifts) & 1                     # (dim, n)
    mask = np.ones((dim, dim), dtype=bool)
    for t in targets:
        if not (0 <= t < n_qubits):
            raise IndexError(f"target={t} 超出範圍 [0, {n_qubits - 1}]")
        col = bits[:, t]
        mask &= col[:, None] == col[None, :]
    return mask


def dephase(sim: StateVectorSim, targets: Sequence[int]) -> np.ndarray:
    r"""對 ``targets`` 這幾個 qubit 套用 Tucci 的 ``cl`` 算子（去相位）。

    .. math::
        \rho \;\longmapsto\; \sum_{b} \big(|b\rangle\langle b|\otimes I\big)\,
        \rho\,\big(|b\rangle\langle b|\otimes I\big)

    即**只保留在 ``targets`` 上對角的區塊，其餘非對角項全部歸零**。

    **一個必須誠實面對的數學事實**：去相位**不改變計算基底的機率**。
    因為 :math:`P(i)=\rho_{ii}`，而去相位只動非對角項。所以「全量子 vs 去相位」
    在直接量測的機率上**差異恰好是 0**。差異只會出現在：

    1. 同調性敏感的觀測量（:math:`\\langle X\\rangle`、:math:`\\langle Y\\rangle`）；
    2. 純度 :math:`\\Tr[\\rho^2]`（純態 → 混態）；
    3. **去相位之後若再施加任何量子閘**，因為後續干涉已失去同調性，
       此時機率才會分歧。

    第 3 點才是「去相位消融」在操作上真正的意義，`main` 會把這三項全部印出來。

    Returns:
        去相位後的密度矩陣（``2^n x 2^n`` 複數矩陣）。要取機率請用
        `dephased_probabilities`，或直接取 ``np.real(np.diag(...))``。
    """
    targets = list(targets)
    if not targets:
        return sim.density_matrix()
    rho = sim.density_matrix()
    return rho * _coherence_mask(sim.n_qubits, targets)


def dephased_probabilities(sim: StateVectorSim, targets: Sequence[int]) -> np.ndarray:
    """去相位後的計算基底機率（= ``np.real(np.diag(dephase(...)))``）。

    理論上與 `readout` 完全相同；本函式存在的目的是讓程式**明確示範**這件事。
    """
    return np.real(np.diag(dephase(sim, targets)))


def classical_joint_from_rho(rho: np.ndarray) -> np.ndarray:
    r"""由去相位後的密度矩陣取出古典聯合分布（對角線）。

    去相位後 :math:`\rho=\sum_i p_i|i\rangle\langle i|`，此時
    :math:`\Tr[\rho^2]=\sum_i p_i^2`——**這正是古典分布的不確定性度量**，
    也是「去相位後就沒有糾纏了」的判準（古典混態的純度只由對角線決定）。
    """
    return np.real(np.diag(rho))


# ===========================================================================
# 示範與輸出
# ===========================================================================
def _demo_inputs() -> Tuple[np.ndarray, np.ndarray]:
    """固定的示範輸入（固定種子，確保書中的「預期輸出」可重現）。"""
    x5 = np.array([0.90, 0.20, 0.75, 0.40, 0.60])
    rng = np.random.default_rng(42)
    weights = rng.normal(0.0, 0.8, size=(2, N_QUBITS, 2))
    return x5, weights


def main() -> int:
    print("=" * 74)
    print("qbn5_encoding.py — 5 qubit 輸出層參考實作（純 NumPy）")
    print(f"NumPy 版本：{np.__version__}")
    print("=" * 74)

    x5, weights = _demo_inputs()
    angles = encode_angles(x5)

    # ------------------------------------------------------------------
    print("\n【1】角度編碼：θ_i = 2·arcsin(√x_i)，使 P(|1>) = x_i")
    print("-" * 74)
    sim_enc = StateVectorSim(N_QUBITS)
    for i in range(N_QUBITS):
        sim_enc.ry(float(angles[i]), i)
    print(f"  輸入 x            = {np.array2string(x5, precision=4)}")
    print(f"  角度 θ (rad)      = {np.array2string(angles, precision=6)}")
    marg = []
    for i in range(N_QUBITS):
        bit = sim_enc._bit_of(i)
        marg.append(float(np.sum(bit * sim_enc.probabilities())))
    print(f"  邊際 P(q_i = 1)   = {np.array2string(np.array(marg), precision=10)}")
    print(f"  與 x 的最大誤差   = {np.max(np.abs(np.array(marg) - x5)):.3e}")

    # ------------------------------------------------------------------
    print("\n【2】5 qubit 電路：depth=2、環形糾纏（兩條 5-cycle 交替）")
    print("-" * 74)
    sim = qbn_layer(angles, weights, depth=2, entanglement="ring")
    probs = readout(sim)
    labels = basis_labels()
    print(f"  ansatz 參數量     = {weights.size}（depth·n·2 = 2·5·2）")
    print(f"  su(32) 維度上界   = {N_UNITARY_PARAMS}（= 32² − 1）")
    print(f"  32 維機率總和     = {probs.sum():.15f}")
    order = np.argsort(probs)[::-1]
    print("\n  32 維機率前 5 大：")
    for rank, i in enumerate(order[:5], start=1):
        print(f"    {rank}. |{labels[i]}>  P = {probs[i]:.8f}")

    # ------------------------------------------------------------------
    print("\n【3】讀出策略比較（K 類）")
    print("-" * 74)
    for K in (4, 6, 10):
        print(f"\n  K = {K}：")
        rows: List[Tuple[str, np.ndarray, str]] = []
        rows.append(("adjacent (相鄰分組)", adjacent_grouping(probs, K),
                     f"{probs.size // K} 個基底一類" if probs.size % K == 0
                     else "不等分組（32 不被 K 整除）"))
        rows.append(("gray (Gray code 相鄰)", adjacent_grouping(probs, K, use_gray=True),
                     "相鄰索引只差一個位元"))
        if K == N_QUBITS + 1:
            rows.append(("hamming (漢明權重)", hamming_grouping(probs, K),
                         "每類的 1 的個數相同"))
        rd = POVMReadout.random(K, seed=7)
        rows.append((f"learned (POVM {K}×32)", rd(probs),
                     f"可訓練參數 {32 * (K - 1)} 個"))
        for name, vec, note in rows:
            ok_povm, _ = (rd.verify_povm() if name.startswith("learned")
                          else (True, ""))
            print(f"    {name:<24} sum={vec.sum():.10f}  max={vec.max():.6f}  {note}")
        # 硬分組 = 0/1 讀出矩陣（策略 A/B/C 都是策略 D 的特例）
        W_hard = POVMReadout.from_grouping(_group_index(probs.size, K))
        ok_hard, msg_hard = W_hard.verify_povm()
        print(f"    → 相鄰分組的等價 0/1 POVM 讀出矩陣驗證：{msg_hard}")
        print(f"    → 0/1 讀出矩陣是否為合法 POVM：{ok_hard}")

    # ------------------------------------------------------------------
    print("\n【4】去相位消融（Tucci 的 cl 算子）：全量子 vs 古典化")
    print("-" * 74)
    all_q = list(range(N_QUBITS))
    rho_q = sim.density_matrix()
    rho_c = dephase(sim, all_q)
    probs_c = np.real(np.diag(rho_c))

    print("  (a) 直接量測的 32 維機率：")
    print(f"      全量子  vs 去相位 的最大絕對差 = "
          f"{np.max(np.abs(probs - probs_c)):.3e}   ← 理論上必須是 0")

    print("\n  (b) 同調性敏感的觀測量（去相位應該殺掉 X、Y）：")
    print(f"      {'qubit':<7}{'<X> 量子':>12}{'<X> 去相位':>13}"
          f"{'<Y> 量子':>12}{'<Y> 去相位':>13}{'<Z> 量子':>12}{'<Z> 去相位':>13}")
    max_xy = 0.0
    for i in range(N_QUBITS):
        xq, xc = sim.expectation_pauli("X", i), _expect_from_rho(rho_c, "X", i, N_QUBITS)
        yq, yc = sim.expectation_pauli("Y", i), _expect_from_rho(rho_c, "Y", i, N_QUBITS)
        zq, zc = sim.expectation_pauli("Z", i), _expect_from_rho(rho_c, "Z", i, N_QUBITS)
        max_xy = max(max_xy, abs(xc), abs(yc))
        print(f"      {i:<7}{xq:>12.6f}{xc:>13.6f}{yq:>12.6f}{yc:>13.6f}"
              f"{zq:>12.6f}{zc:>13.6f}")
    print(f"      → 去相位後 |<X>|,|<Y>| 的最大殘值 = {max_xy:.3e}")

    print("\n  (c) 純度與糾纏熵：")
    pur_q = float(np.real(np.trace(rho_q @ rho_q)))
    pur_c = float(np.real(np.trace(rho_c @ rho_c)))
    print(f"      純度 Tr[ρ²]：量子 {pur_q:.10f} → 去相位 {pur_c:.10f}")
    print(f"      去相位後 Σ p_i² = {float(np.sum(probs_c ** 2)):.10f}"
          f"   （與去相位純度相符：{np.isclose(pur_c, np.sum(probs_c ** 2))}）")
    for sub in ([0], [0, 1], [0, 1, 2]):
        sq = sim.entanglement_entropy(sub)
        sc = _entropy_from_rho(rho_c, sub, N_QUBITS)
        print(f"      糾纏熵 S(qubits {sub})：量子 {sq:.8f} bit → 去相位 {sc:.8f} bit")

    print("\n  (d) 『去相位擺在哪裡』決定看不看得出來——三個設定的完整比較：")
    all_cx = RING_STEP1 + RING_STEP2

    # 設定一：去相位在電路最末端（緊接量測）
    d_end = np.max(np.abs(probs - dephased_probabilities(sim, all_q)))
    print(f"      (1) 去相位在【最末端】：最大機率差 = {d_end:.3e}"
          f"   ← 必須是 0（量測只讀對角線）")

    # 設定二：去相位在中間，之後只接 CX（基底置換）
    sim_cx = _sim_from_rho(rho_c, N_QUBITS)
    for c, t in all_cx:
        sim_cx.cx(c, t)
    sim_q_cx = sim.copy()
    for c, t in all_cx:
        sim_q_cx.cx(c, t)
    d_cx = np.max(np.abs(sim_q_cx.probabilities() - sim_cx.probabilities()))
    print(f"      (2) 去相位在【中間】，之後只接 10 個 CX：最大機率差 = {d_cx:.3e}"
          f"   ← 仍是 0（CX 是基底置換，與去相位交換）")

    # 設定三：去相位在中間，之後接會混基底的閘（RY / RZ / CX）
    sim_mid_q = qbn_layer(angles, weights, depth=2, entanglement="ring")
    sim_mid_c = qbn_layer_ablated(
        angles, weights, depth=2, dephase_at=0, targets=all_q, entanglement="ring"
    )
    p_mid_q = sim_mid_q.probabilities()
    p_mid_c = sim_mid_c.probabilities()
    diff3 = np.abs(p_mid_q - p_mid_c)
    print(f"      (3) 去相位在【中間】，之後接完整第 2 層（RY+RZ+CX）：")
    print(f"          最大機率差 = {diff3.max():.8f}"
          f"   ← 這才是可觀測的消融效果")
    top = np.sort(diff3)[::-1][:5]
    print(f"          前 5 大差異 = {[f'{v:.6f}' for v in top]}")
    print(f"          全變分距離 (1/2)·Σ|Δp| = {0.5 * diff3.sum():.8f}")
    print(f"          受影響的基底態個數 (|Δp|>1e-9) = {int((diff3 > 1e-9).sum())} / 32")

    # ------------------------------------------------------------------
    print("\n【5】POVM 學習式讀出的公理驗證")
    print("-" * 74)
    rd10 = POVMReadout.random(10, seed=2026)
    ok, msg = rd10.verify_povm()
    print(f"  K=10 隨機 POVM 讀出：{msg}")
    print(f"  是否為合法 POVM：{ok}")
    print(f"  行和是否全為 1：{np.allclose(rd10.W.sum(axis=0), 1.0)}")
    print(f"  元素是否全非負：{bool(np.all(rd10.W >= 0))}")
    print(f"  類別機率和 = {rd10(probs).sum():.15f}（由 sum_k E_k = I 保證，無需 softmax）")

    print("\n" + "=" * 74)
    print("完成。")
    print("=" * 74)
    return 0


# --- 示範用的輔助函式（由密度矩陣直接算期望值／熵，不經過 StateVectorSim） ---


def _pauli_on(n: int, target: int, which: str) -> np.ndarray:
    """建構作用在 qubit ``target`` 上的 Pauli 算符（張量到 n 個 qubit）。"""
    I = np.eye(2, dtype=complex)
    P = {"X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]], complex),
         "Z": np.array([[1, 0], [0, -1]], complex)}[which.upper()]
    out = np.array([[1.0 + 0.0j]])
    for k in range(n):
        out = np.kron(out, P if k == target else I)
    return out


def _expect_from_rho(rho: np.ndarray, which: str, target: int, n: int) -> float:
    return float(np.real(np.trace(_pauli_on(n, target, which) @ rho)))


def _entropy_from_rho(rho: np.ndarray, keep: Sequence[int], n: int) -> float:
    """對密度矩陣取部分跡後算馮紐曼熵（bit）。"""
    keep = sorted(keep)
    traced = [i for i in range(n) if i not in keep]
    t = rho.reshape((2,) * (2 * n))
    perm = keep + traced + [n + k for k in keep] + [n + t_ for t_ in traced]
    t = np.transpose(t, perm).reshape(2 ** len(keep), 2 ** len(traced),
                                      2 ** len(keep), 2 ** len(traced))
    rho_a = np.einsum("itjt->ij", t)
    ev = np.clip(np.linalg.eigvalsh((rho_a + rho_a.conj().T) / 2).real, 0, None)
    nz = ev[ev > 1e-12]
    return float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0


def _sim_from_rho(rho: np.ndarray, n: int) -> "_MixtureSim":
    """把（去相位後的）密度矩陣轉成「純態混合物」，以便繼續施加閘。

    去相位後的 :math:`\\rho=\\sum_i\\lambda_i|v_i\\rangle\\langle v_i|` 是混態，
    無法用單一狀態向量表示。本函式做譜分解，保留每個本徵向量與其權重，
    回傳 `_MixtureSim`：施加閘時對每個本徵向量各自演化，取機率時再以
    :math:`\\lambda_i` 加權平均。這對任何輸入 :math:`\\rho` 都正確，
    且是**確定性**的（不涉及抽樣），因此適合做可比較的消融實驗。
    """
    ev, evec = np.linalg.eigh((rho + rho.conj().T) / 2)
    keep = ev > 1e-12
    return _MixtureSim(n, ev[keep], evec[:, keep])


class _MixtureSim:
    """純態混合物的輕量容器：對每個本徵向量分別施加閘後加權平均。"""

    def __init__(self, n: int, weights: np.ndarray, vecs: np.ndarray) -> None:
        self.n_qubits = n
        self.weights = np.asarray(weights, dtype=float)
        self.vecs = np.asarray(vecs, dtype=complex)     # (dim, m)
        self._sims = [StateVectorSim(n) for _ in range(len(self.weights))]
        for s, col in zip(self._sims, self.vecs.T):
            s.state = np.array(col, dtype=complex)
            nrm = np.linalg.norm(s.state)
            if nrm > 0:
                s.state /= nrm

    def cx(self, c: int, t: int) -> "_MixtureSim":
        for s in self._sims:
            s.cx(c, t)
        return self

    def ry(self, theta: float, t: int) -> "_MixtureSim":
        for s in self._sims:
            s.ry(theta, t)
        return self

    def rz(self, theta: float, t: int) -> "_MixtureSim":
        for s in self._sims:
            s.rz(theta, t)
        return self

    def probabilities(self) -> np.ndarray:
        acc = np.zeros(2 ** self.n_qubits)
        for w, s in zip(self.weights, self._sims):
            acc += w * s.probabilities()
        return acc

    def density_matrix(self) -> np.ndarray:
        """混合態的密度矩陣 :math:`\\sum_i\\lambda_i|\\psi_i\\rangle\\langle\\psi_i|`。"""
        acc = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits), dtype=complex)
        for w, s in zip(self.weights, self._sims):
            acc += w * s.density_matrix()
        return acc

    def copy(self) -> "_MixtureSim":
        new = _MixtureSim.__new__(_MixtureSim)
        new.n_qubits = self.n_qubits
        new.weights = self.weights.copy()
        new.vecs = self.vecs.copy()
        new._sims = [s.copy() for s in self._sims]
        return new


if __name__ == "__main__":
    sys.exit(main())
