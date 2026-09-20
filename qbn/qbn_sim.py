# -*- coding: utf-8 -*-
"""qbn_sim.py — 純 NumPy 狀態向量模擬器（QBN 專案的「保險線」參考實作）。

為什麼會有這支程式
------------------
CUDA-Q 官方**沒有 Windows wheel**（只有 Linux 與 macOS ARM），所以原生 Windows
跑不動 CUDA-Q，必須靠 WSL2。對一本教學書而言這是風險：學生可能卡在環境就放棄。
因此本書採雙軌策略：

* **主線**：CUDA-Q（在 WSL2 裡跑）。
* **保險線**：本檔，純 NumPy，任何有 Python + NumPy 的機器都能跑。

本檔的角色**不是玩具**，而是「正確性基準」：它用來驗證數學、教學，
以及**交叉驗證 CUDA-Q 的輸出**（見 docs/03-實作/無CUDA-Q的參考實作.md 的
「交叉驗證協定」一節）。因此本檔附帶一組逐一比對解析結果的自我測試，
執行方式::

    python dev/qbn_sim.py

基底順序公約（很重要）
----------------------
``n`` 個 qubit 的狀態向量有 :math:`2^n` 個複數振幅，索引 ``i`` 的二位元展開是

    ``i = b_0 b_1 ... b_{n-1}``（``b_0`` 是最高位元）

且我們規定 **qubit 0 是最高位元**（big-endian），與 CUDA-Q ``qvector(n)`` 中
``q[0]`` 為最左邊張量因子的慣例一致。例如 ``n=2``：索引 0 = ``|00>``、
1 = ``|01>``、2 = ``|10>``、3 = ``|11>``，其中第一個字元是 qubit 0。

> TODO(核實)：CUDA-Q `cudaq.sample` 回傳的位元字串，其最左字元是否對應 `q[0]`，
> 需在 WSL2 內用「只對 q[0] 施加 X」的最小電路實測確認。這正是交叉驗證協定
> 第一個要抓的錯誤——順序若相反，貝爾態與所有糾纏結構都會對不上。

與〈QBN 形式定義〉的對應
------------------------
本檔每個方法都對應 docs/02-QBN理論/QBN形式定義.md 裡的一個數學物件：

============================ ==================================================
本檔方法                        對應的 QBN 數學物件（出處）
============================ ==================================================
``density_matrix()``         定義 2.4 的聯合密度矩陣 :math:`\\rho_{\\mathrm{QBN}}`（L118）
``probabilities()``          定義 2.5 的 Born 規則 :math:`P(x)=\\Tr[M_x\\rho]`（L131）
``partial_trace()``          定義 2.5 的邊際化 :math:`\\Tr_{\\mathrm{env}}`（L139）
``cx`` / ``cz``              邊＝節點間的量子通道（糾纏來源）（L64）
``cry``                      單父節點的條件機率表 :math:`P(x_i\\mid \\mathrm{pa}(x_i))`（L93）
``mcry``                     **多父節點**的條件通道 :math:`P(x_i\\mid \\mathrm{pa}(x_i))`（L93）
``ry`` / ``rx`` / ``rz``     節點的單量子位通道 :math:`\\mathcal{E}_i`（L93）
``entanglement_entropy``     非對角元所承載的關聯（L37）
``purity``                   純態 vs 混態的判別（L20）
``expectation_z``            可觀測量的期望值（L35 的測量 superoperator）
============================ ==================================================

``dephase``（去相位）不在本類別裡，它需要密度矩陣，見 `qbn5_encoding.py`；
其理論依據是 Tucci 的古典化算子 ``cl``（docs/_extract/F1-tucci-qbn-mixed-states.md
【L358】定義、【L410】「有時也稱為去相位，因為我們丟掉了一些非對角項」、
【L526】其 Kraus 算子為 :math:`K_a=|a\\rangle\\langle a|`）。

授權：本專案內部使用。
"""

from __future__ import annotations

import sys
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

__all__ = ["StateVectorSim"]

# ---------------------------------------------------------------------------
# 單量子位閘矩陣（2x2）。角度單位一律是弧度。
# 公約：|0> = [1, 0]^T，|1> = [0, 1]^T
# ---------------------------------------------------------------------------


def _ry(theta: float) -> np.ndarray:
    r""":math:`R_Y(\theta)=\begin{pmatrix}\cos\frac\theta2&-\sin\frac\theta2\\ \sin\frac\theta2&\cos\frac\theta2\end{pmatrix}`"""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rx(theta: float) -> np.ndarray:
    r""":math:`R_X(\theta)=\begin{pmatrix}\cos\frac\theta2&-i\sin\frac\theta2\\ -i\sin\frac\theta2&\cos\frac\theta2\end{pmatrix}`"""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    r""":math:`R_Z(\theta)=\begin{pmatrix}e^{-i\theta/2}&0\\0&e^{i\theta/2}\end{pmatrix}`"""
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex
    )


_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


class StateVectorSim:
    """``n_qubits`` 個 qubit 的狀態向量模擬器（純 NumPy、全向量化）。

    設計要求：**不得**用 Python 迴圈逐一走 :math:`2^n` 個振幅。
    單量子位閘作用在 qubit ``k`` 上的做法是：

    1. 把長度 :math:`2^n` 的向量 ``reshape`` 成 ``(2,)*n``——此時**第 k 軸就是
       qubit k**，因為我們採 big-endian（qubit 0 是最外層軸）。
    2. 把第 ``k`` 軸 ``moveaxis`` 到最前面，形狀變成 ``(2, rest...)``。
    3. ``np.tensordot(U, t, axes=([1], [0]))``——把 ``U`` 的行指標和第 0 軸縮併，
       等價於對每個「其餘位元固定」的 2 維子空間套用 ``U``。
    4. 移回原本的軸位置，再 ``reshape`` 回長度 :math:`2^n`。

    這個寫法對 :math:`n=5`（32 個振幅）與 :math:`n=15`（32768 個振幅）是同一份
    程式碼，成本只隨 :math:`2^n` 線性成長。

    受控閘則更省：只有「所有控制位皆為 1」的那一半振幅會被改寫，其餘不動。
    """

    # ------------------------------------------------------------------
    # 建構／重置
    # ------------------------------------------------------------------
    def __init__(self, n_qubits: int) -> None:
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValueError(f"n_qubits 必須是 >= 1 的整數，收到 {n_qubits!r}")
        self.n_qubits: int = int(n_qubits)
        self.dim: int = 2 ** self.n_qubits
        self.reset()

    def reset(self) -> "StateVectorSim":
        """重置到 :math:`|0\\cdots0\\rangle`（振幅索引 0 為 1，其餘為 0）。"""
        self.state: np.ndarray = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0 + 0.0j
        return self

    def copy(self) -> "StateVectorSim":
        """深拷貝（自我測試與消融實驗需要保留原態）。"""
        other = StateVectorSim(self.n_qubits)
        other.state = self.state.copy()
        return other

    # ------------------------------------------------------------------
    # 內部：向量化閘作用
    # ------------------------------------------------------------------
    def _check_target(self, target: int) -> int:
        if not (0 <= target < self.n_qubits):
            raise IndexError(f"target={target} 超出範圍 [0, {self.n_qubits - 1}]")
        return int(target)

    def _apply_1q(self, U: np.ndarray, target: int) -> None:
        """把 2x2 的 ``U`` 作用在 ``target`` 上（無控制位）。"""
        target = self._check_target(target)
        n = self.n_qubits
        t = self.state.reshape((2,) * n)          # 檢視（view），不複製
        t = np.moveaxis(t, target, 0)             # target 軸移到最前
        t = np.tensordot(U, t, axes=([1], [0]))   # 縮併 U 的行指標
        t = np.moveaxis(t, 0, target)             # 移回原位
        self.state = np.ascontiguousarray(t).reshape(-1)

    def _apply_controlled_1q(
        self, U: np.ndarray, controls: Sequence[int], target: int
    ) -> None:
        """把 ``U`` 作用在 ``target`` 上，條件是 ``controls`` 全部為 1。

        作法：把 target 軸移到最前，攤平成 ``(2, 2^(n-1))``；對每個「其餘位元
        組合」算出控制位是否全為 1 的布林遮罩，只對遮罩內的行做 2x2 乘法。
        """
        target = self._check_target(target)
        controls = [self._check_target(c) for c in controls]
        if target in controls:
            raise ValueError(f"target={target} 不可同時是控制位 controls={controls}")
        if not controls:
            self._apply_1q(U, target)
            return

        n = self.n_qubits
        perm = [target] + [i for i in range(n) if i != target]
        t = np.transpose(self.state.reshape((2,) * n), perm)
        flat = np.ascontiguousarray(t).reshape(2, -1)   # (2, 2^(n-1))，可寫入
        rest = perm[1:]                                 # 攤平軸依序對應的 qubit

        # 對每個攤平後的 column j，判斷其控制位是否全為 1。
        # rest[p] 是第 p 個「其餘軸」，它是第 (n-2-p) 個位元（big-endian）。
        j = np.arange(flat.shape[1])
        mask = np.ones(flat.shape[1], dtype=bool)
        for c in controls:
            shift = n - 2 - rest.index(c)
            mask &= ((j >> shift) & 1).astype(bool)

        flat[:, mask] = U @ flat[:, mask]

        inv = np.argsort(perm)
        back = np.transpose(flat.reshape((2,) * n), inv)
        self.state = np.ascontiguousarray(back).reshape(-1)

    # ------------------------------------------------------------------
    # 單量子位閘
    # ------------------------------------------------------------------
    def ry(self, theta: float, target: int) -> "StateVectorSim":
        r""":math:`R_Y(\theta)`。這是 QBN **角度編碼的主力**：
        :math:`R_Y(\theta)|0\rangle=\cos\frac\theta2|0\rangle+\sin\frac\theta2|1\rangle`，
        故 :math:`P(|1\rangle)=\sin^2\frac\theta2`。
        """
        self._apply_1q(_ry(theta), target)
        return self

    def rx(self, theta: float, target: int) -> "StateVectorSim":
        r""":math:`R_X(\theta)`。用來把量子態轉到便於量測的基底。"""
        self._apply_1q(_rx(theta), target)
        return self

    def rz(self, theta: float, target: int) -> "StateVectorSim":
        r""":math:`R_Z(\theta)`。**對計算基底機率沒有影響**（它只改相位），
        但會改變後續干涉的結果——這正是「相位為什麼重要」的最小示範。
        """
        self._apply_1q(_rz(theta), target)
        return self

    def h(self, target: int) -> "StateVectorSim":
        r"""哈達瑪閘 :math:`H`，產生 :math:`(|0\rangle+|1\rangle)/\sqrt2`。"""
        self._apply_1q(_H, target)
        return self

    def x(self, target: int) -> "StateVectorSim":
        r"""位元翻轉 :math:`X`（量子 NOT）。"""
        self._apply_1q(_X, target)
        return self

    # ------------------------------------------------------------------
    # 雙量子位糾纏閘
    # ------------------------------------------------------------------
    def cx(self, control: int, target: int) -> "StateVectorSim":
        r"""受控 NOT（CNOT）。等於 :math:`|0\rangle\langle0|\otimes I+|1\rangle\langle1|\otimes X`。

        在 QBN 圖上是**一條邊**：它讓 target 的分布依賴 control，並製造糾纏。
        """
        self._apply_controlled_1q(_X, [control], target)
        return self

    def cz(self, control: int, target: int) -> "StateVectorSim":
        r"""受控 Z。與 CNOT 只差 target 上的兩個 :math:`H`：
        :math:`\mathrm{CZ}=(I\otimes H)\,\mathrm{CX}\,(I\otimes H)`（同一 target）。
        """
        self._apply_controlled_1q(_Z, [control], target)
        return self

    def cry(self, theta: float, control: int, target: int) -> "StateVectorSim":
        r"""受控 :math:`R_Y(\theta)`——**QBN 節點的核心**。

        它精確對應條件機率表 :math:`P(x_{\\mathrm{target}}\\mid x_{\\mathrm{control}})`：
        control 為 0 時 target 不動；control 為 1 時 target 才旋轉。
        """
        self._apply_controlled_1q(_ry(theta), [control], target)
        return self

    def mcry(
        self, theta: float, controls: List[int], target: int
    ) -> "StateVectorSim":
        r"""多控制位 :math:`R_Y(\theta)`——**一個有多個父節點的 QBN 節點**。

        對應條件通道 :math:`P(x_i \\mid \\mathrm{pa}(x_i))`：只有當
        **所有**父節點（``controls``）都是 1 時，子節點（``target``）才旋轉
        ``theta``。多父節點的完整條件機率表需要 :math:`2^{|\\mathrm{pa}|}`
        個分支，實務上以「對每個分支套一次 ``mcry``，並用 ``X`` 翻轉控制位來
        切換分支」組合而成（見 docs/02-QBN理論/QBN形式定義.md 的 3 節點範例）。
        """
        self._apply_controlled_1q(_ry(theta), list(controls), target)
        return self

    # ------------------------------------------------------------------
    # 觀測量
    # ------------------------------------------------------------------
    def amplitudes(self) -> np.ndarray:
        r"""回傳 :math:`2^n` 個複數振幅 :math:`\alpha_i`（副本，可安全修改）。"""
        return self.state.copy()

    def probabilities(self) -> np.ndarray:
        r"""Born 規則的計算基底機率 :math:`P(i)=|\alpha_i|^2`。

        對應 docs/02-QBN理論/QBN形式定義.md 定義 2.5（L131）的
        :math:`P(x)=\\Tr[M_x\\rho]`，其中 :math:`M_x=|x\rangle\langle x|`。
        """
        return np.abs(self.state) ** 2

    def density_matrix(self) -> np.ndarray:
        r"""由純態組出密度矩陣 :math:`\rho=|\psi\rangle\langle\psi|`
        （定義 2.4，L118），形狀 :math:`(2^n, 2^n)`。

        純態的 :math:`\\rho` 秩為 1，這是「為什麼純態不夠用」的起點。
        """
        return np.outer(self.state, self.state.conj())

    def sample(self, shots: int, seed: Optional[int] = None) -> Dict[str, int]:
        r"""對計算基底做 ``shots`` 次取樣，回傳 ``{"01011": 計數, ...}``。

        位元字串的最左字元是 qubit 0（big-endian，與本檔公約一致）。
        使用 ``np.random.default_rng(seed)``，因此**同一個 seed 可完全重現**。
        """
        if shots < 1:
            raise ValueError(f"shots 必須 >= 1，收到 {shots}")
        probs = self.probabilities()
        probs = probs / probs.sum()                     # 防禦數值誤差
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.dim, size=int(shots), p=probs)
        counts = np.bincount(idx, minlength=self.dim)
        fmt = f"0{self.n_qubits}b"
        return {
            format(i, fmt): int(c) for i, c in enumerate(counts) if c > 0
        }

    def expectation_z(self, target: int) -> float:
        r""":math:`\langle Z_{\\mathrm{target}}\rangle=\sum_i (-1)^{b_{\\mathrm{target}}(i)}P(i)`。

        注意它**只是機率向量的線性泛函**，因此只用得到 :math:`\rho` 的對角元。
        這個觀察是〈五量子位輸出層設計〉討論「讀出用 :math:`\langle Z\rangle`
        還是用 32 維機率」的關鍵。
        """
        target = self._check_target(target)
        bits = self._bit_of(target)
        return float(np.sum((1.0 - 2.0 * bits) * self.probabilities()))

    def expectation_pauli(self, pauli: str, target: int) -> float:
        r""":math:`\langle P_{\\mathrm{target}}\rangle`，``pauli`` ∈ {"X","Y","Z"}。

        這是去相位實驗的關鍵工具：:math:`\langle X\rangle` 與 :math:`\langle Y\rangle`
        **完全來自非對角元（同調性）**，所以去相位會把它們歸零，而
        :math:`\langle Z\rangle` 不受影響。
        """
        target = self._check_target(target)
        P = {"X": _X, "Y": _Y, "Z": _Z}[pauli.upper()]
        # <P> = Tr[P rho] = psi^dag (P_k psi)
        psi = self.state.reshape((2,) * self.n_qubits)
        psi = np.moveaxis(psi, target, 0)
        psi = np.tensordot(P, psi, axes=([1], [0]))
        psi = np.moveaxis(psi, 0, target).reshape(-1)
        return float(np.real(np.vdot(self.state, psi)))

    def _bit_of(self, target: int) -> np.ndarray:
        """回傳長度 :math:`2^n` 的 0/1 陣列，表示各基底索引中 qubit ``target`` 的位元。"""
        idx = np.arange(self.dim)
        return (idx >> (self.n_qubits - 1 - target)) & 1

    def bits_matrix(self) -> np.ndarray:
        """回傳 ``(2^n, n)`` 的 0/1 矩陣，第 ``i`` 列是基底 ``i`` 的各 qubit 位元。"""
        idx = np.arange(self.dim)[:, None]
        shifts = (self.n_qubits - 1 - np.arange(self.n_qubits))[None, :]
        return (idx >> shifts) & 1

    # ------------------------------------------------------------------
    # 混態工具（部分跡、熵、純度）
    # ------------------------------------------------------------------
    def partial_trace(self, keep: List[int]) -> np.ndarray:
        r"""對 ``keep`` 以外的子系統取部分跡：:math:`\rho_{\\mathrm{keep}}=\\Tr_{\\bar{k}}[\rho]`。

        這正是 QBN 定義 2.5（L139）的**邊際化**，對應古典貝氏網路的
        :math:`\sum`。作法：把 :math:`\rho` 看成 ``(2,)*2n``（前 n 軸是 ket 指標、
        後 n 軸是 bra 指標），把要保留的軸排到前面，再對被縮併的 ket/bra 配對求和。
        """
        n = self.n_qubits
        keep = sorted(self._check_target(k) for k in keep)
        if len(set(keep)) != len(keep):
            raise ValueError(f"keep 有重複的 qubit：{keep}")
        traced = [i for i in range(n) if i not in keep]

        rho = self.density_matrix().reshape((2,) * (2 * n))
        perm = keep + traced + [n + k for k in keep] + [n + t for t in traced]
        rho = np.transpose(rho, perm)
        dk, dt = 2 ** len(keep), 2 ** len(traced)
        rho = rho.reshape(dk, dt, dk, dt)
        return np.einsum("itjt->ij", rho)

    def reduced_state(self, keep: List[int]) -> np.ndarray:
        """``partial_trace`` 的別名（語意更貼近「子系統的態」）。"""
        return self.partial_trace(keep)

    def entanglement_entropy(self, subsystem: List[int], base: float = 2.0) -> float:
        r"""馮紐曼熵 :math:`S=-\\Tr[\\rho_A\\log\\rho_A]`（單位：bit，取 :math:`\\log_2`）。

        對純態整體而言，任一部份的熵就是它與其補集的**糾纏熵**；
        乘積態的熵為 0，貝爾態的單邊熵為 1 bit（最大）。
        """
        rho_a = self.partial_trace(subsystem)
        evals = np.linalg.eigvalsh((rho_a + rho_a.conj().T) / 2.0)
        evals = np.clip(evals.real, 0.0, None)
        nz = evals[evals > 1e-12]
        if nz.size == 0:
            return 0.0
        return float(-np.sum(nz * np.log(nz) / np.log(base)))

    def purity(self) -> float:
        r"""純度 :math:`\\Tr[\\rho^2]`。純態為 1，完全混合態為 :math:`1/2^n`。"""
        return float(np.real(np.trace(self.density_matrix() @ self.density_matrix())))

    # ------------------------------------------------------------------
    # 雜訊（可選）
    # ------------------------------------------------------------------
    def probabilities_with_noise(self, p_depol: float) -> np.ndarray:
        r"""每個 qubit 各套一次去極化通道後的計算基底機率。

        單量子位去極化通道（depolarizing channel）:math:`\mathcal{D}_p`：

        .. math::
            \mathcal{D}_p(\rho)=(1-p)\,\rho+p\,\Tr_k[\rho]\otimes\frac{I}{2}
            =\Big(1-\tfrac{3p}{4}\Big)\rho+\frac{p}{4}\big(\rho+X\rho X+Y\rho Y+Z\rho Z\big)

        等號右邊第二種寫法是 Pauli 形式。**這個正規化很重要**：Pauli 形式裡的
        :math:`\rho` 項權重必須是 :math:`p/4`（四項等權），不能寫成
        :math:`p/3` 只用三個非單位的 Pauli。

        用 Bloch 向量檢查：:math:`\mathcal{D}_p` 把 Bloch 向量縮為
        :math:`(1-p)\,\vec r`，所以在 :math:`p=1` 時該 qubit 變成完全混合態
        :math:`I/2`。若誤用 :math:`(1-p)\rho+\frac p3(X\rho X+Y\rho Y+Z\rho Z)`，
        縮放因子會變成 :math:`(1-2p)`，:math:`p=1` 時得到
        :math:`\frac{2I-\rho}{3}`——那是一個合法的 Pauli 通道，**但不是**
        去極化通道，也不會變成 :math:`I/2`。（本檔自我測試第 [8] 項就是在守這一條。）

        :math:`p=0` 時無雜訊；:math:`p=1` 時整個 :math:`n`-qubit 暫存器變成
        完全混合態 :math:`I/2^n`（均勻分布）。

        本方法需要完整的 :math:`4^n` 密度矩陣，故**僅支援小規模**
        （這裡限制 :math:`n\le 10`，:math:`n=10` 時 :math:`\rho` 是 :math:`1024\times1024`）。

        對應 Wang et al. (2026) 的雜訊強健性協定
        （docs/_extract/F3-bayesian-frontend-hybrid-qc.md【L450】，
        :math:`p\in\{0,0.005,0.01,0.05\}`）。
        """
        if not (0.0 <= p_depol <= 1.0):
            raise ValueError(f"p_depol 必須在 [0,1]，收到 {p_depol}")
        if self.n_qubits > 10:
            raise ValueError(
                f"probabilities_with_noise 需要 4^n 密度矩陣，n={self.n_qubits} 過大"
            )
        n = self.n_qubits
        rho = self.density_matrix()
        if p_depol > 0.0:
            for k in range(n):
                # 對 qubit k 取均勻 Pauli 旋積（twirl）：
                #   (1/4)(rho + X rho X + Y rho Y + Z rho Z) = Tr_k[rho] ⊗ I/2
                twirl = rho.copy()
                for P in (_X, _Y, _Z):
                    twirl += self._conjugate_local(rho, k, P)
                twirl *= 0.25
                rho = (1.0 - p_depol) * rho + p_depol * twirl
        return np.real(np.diag(rho))

    def _conjugate_local(self, rho: np.ndarray, target: int, P: np.ndarray) -> np.ndarray:
        """計算 :math:`P_k\\,\\rho\\,P_k^\\dagger`（``P_k`` 只作用在 qubit ``target``）。"""
        n = self.n_qubits
        t = rho.reshape((2,) * (2 * n))
        # 左乘：收縮 ket 側的 target 軸
        t = np.tensordot(P, t, axes=([1], [target]))
        t = np.moveaxis(t, 0, target)
        # 右乘：收縮 bra 側的 target 軸（位置不變，仍是 n+target）
        t = np.tensordot(t, P.conj().T, axes=([n + target], [0]))
        t = np.moveaxis(t, -1, n + target)
        return t.reshape(2 ** n, 2 ** n)


# ===========================================================================
# 自我測試：逐一比對已知解析結果
# ===========================================================================


class _Checker:
    """極簡測試框架：記錄每項通過與否，最後印出摘要。"""

    def __init__(self) -> None:
        self.results: List[tuple] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
        return bool(ok)

    def close(
        self,
        name: str,
        got: float,
        want: float,
        tol: float = 1e-12,
        note: str = "",
    ) -> bool:
        ok = abs(float(got) - float(want)) <= tol
        detail = f"got={got:.12g}, want={want:.12g}, |diff|={abs(got - want):.3g} (tol={tol:g})"
        if note:
            detail += f"; {note}"
        return self.check(name, ok, detail)

    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    def summary(self) -> None:
        total = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        print()
        print("=" * 72)
        print(f"自我測試結果：{passed}/{total} 項通過")
        if passed != total:
            print("失敗項目：")
            for name, ok, _ in self.results:
                if not ok:
                    print(f"  - {name}")
        print("=" * 72)


def _reference_state_via_kron(n: int, ops: List[tuple]) -> np.ndarray:
    """獨立的參考實作：用 Kronecker 乘積直接建構么正矩陣（只適用小 n）。

    ``ops`` 是 ``(kind, ...)`` 序列，``kind`` ∈ {"1q", "ctrl"}。
    這條路徑**完全不共用** `StateVectorSim` 的向量化邏輯，因此可以抓出
    軸順序（big-endian 與否）之類的錯誤。
    """
    ident = np.eye(2, dtype=complex)
    U = np.eye(2 ** n, dtype=complex)

    def embed(single: np.ndarray, target: int) -> np.ndarray:
        """把單量子位矩陣放到第 ``target`` 個張量因子上（qubit 0 最左）。"""
        mats = [ident] * n
        mats[target] = single
        out = np.array([[1.0 + 0.0j]])
        for m in mats:
            out = np.kron(out, m)
        return out

    for op in ops:
        if op[0] == "1q":
            _, mat, target = op
            U = embed(mat, target) @ U
        elif op[0] == "ctrl":
            _, mat, control, target = op
            P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
            P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
            U = (embed(P0, control) + embed(mat, target) @ embed(P1, control)) @ U
        else:
            raise ValueError(op[0])
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0
    return U @ psi0


def run_self_tests(verbose: bool = True) -> bool:
    """執行全部自我測試。回傳 True 代表全部通過。"""
    c = _Checker()
    rng_seed = 20260601

    # ------------------------------------------------------------------
    print("\n[1] 基本閘與解析機率")
    # ------------------------------------------------------------------
    # RY(pi/2)|0> => P(0) = cos^2(pi/4) = 0.5, P(1) = sin^2(pi/4) = 0.5
    s = StateVectorSim(1).ry(np.pi / 2, 0)
    p = s.probabilities()
    c.close("ry(pi/2,0) 的 P(0) = cos^2(pi/4)", p[0], 0.5)
    c.close("ry(pi/2,0) 的 P(1) = sin^2(pi/4)", p[1], 0.5)

    # 一般角度：P(1) = sin^2(theta/2)
    theta = 0.7
    s = StateVectorSim(1).ry(theta, 0)
    c.close(f"ry({theta},0) 的 P(1) = sin^2(theta/2)", s.probabilities()[1],
            np.sin(theta / 2) ** 2)

    # H|0> => 均勻
    s = StateVectorSim(1).h(0)
    c.close("h(0) 的 P(0)", s.probabilities()[0], 0.5)
    c.close("h(0) 的 P(1)", s.probabilities()[1], 0.5)

    # RZ 不改計算基底機率，但改相位（所以後續干涉會不同）
    s_a = StateVectorSim(1).h(0)
    s_b = StateVectorSim(1).h(0).rz(0.9, 0)
    c.check(
        "rz 不改變計算基底機率",
        np.allclose(s_a.probabilities(), s_b.probabilities(), atol=1e-15),
        f"max|dp|={np.max(np.abs(s_a.probabilities() - s_b.probabilities())):.3g}",
    )
    # RZ 之後再 H，機率就變了（相位影響干涉）
    pa = StateVectorSim(1).h(0).h(0).probabilities()
    pb = StateVectorSim(1).h(0).rz(0.9, 0).h(0).probabilities()
    c.check(
        "rz 改變後續干涉（H 之後）",
        not np.allclose(pa, pb, atol=1e-6),
        f"P(0): {pa[0]:.6f} -> {pb[0]:.6f}",
    )

    # X 閘
    s = StateVectorSim(2).x(1)
    c.close("x(1) 之後 P(|01>)", s.probabilities()[1], 1.0)

    # ------------------------------------------------------------------
    print("\n[2] 貝爾態：H(0); CX(0,1)")
    # ------------------------------------------------------------------
    bell = StateVectorSim(2).h(0).cx(0, 1)
    pb_ = bell.probabilities()
    c.close("貝爾態 P(00)", pb_[0], 0.5)
    c.close("貝爾態 P(11)", pb_[3], 0.5)
    c.close("貝爾態 P(01)", pb_[1], 0.0)
    c.close("貝爾態 P(10)", pb_[2], 0.0)

    # 單一 qubit 的熵 = 1 bit
    c.close("貝爾態對 qubit 0 的糾纏熵 = 1 bit",
            bell.entanglement_entropy([0]), 1.0)
    c.close("貝爾態對 qubit 1 的糾纏熵 = 1 bit",
            bell.entanglement_entropy([1]), 1.0)

    # ------------------------------------------------------------------
    print("\n[3] 貝爾態的部分跡 = I/2（純度 0.5）")
    # ------------------------------------------------------------------
    rho0 = bell.partial_trace([0])
    c.check(
        "partial_trace([0]) 等於 I/2",
        np.allclose(rho0, np.eye(2) / 2.0, atol=1e-12),
        "max|rho - I/2| = %.3g" % np.max(np.abs(rho0 - np.eye(2) / 2.0)),
    )
    c.close("貝爾態對 qubit 0 取部分跡後的純度", bell.purity() * 0.0 + float(
        np.real(np.trace(rho0 @ rho0))), 0.5)
    # 整體仍是純態
    c.close("貝爾態整體純度 Tr[rho^2] = 1", bell.purity(), 1.0)

    # 乘積態的部分跡應為純態，熵為 0
    prod = StateVectorSim(2).h(0)
    c.close("乘積態 |+>|0> 對 qubit 1 的熵 = 0",
            prod.entanglement_entropy([1]), 0.0)
    c.close("乘積態 |+>|0> 對 qubit 0 的熵 = 0",
            prod.entanglement_entropy([0]), 0.0)

    # 三 qubit GHZ：任一 qubit 的熵 = 1 bit
    ghz = StateVectorSim(3).h(0).cx(0, 1).cx(1, 2)
    c.close("GHZ 態 P(000)", ghz.probabilities()[0], 0.5)
    c.close("GHZ 態 P(111)", ghz.probabilities()[7], 0.5)
    c.close("GHZ 態對 qubit 0 的熵 = 1 bit", ghz.entanglement_entropy([0]), 1.0)
    c.close("GHZ 態對 qubit 0,1 的熵 = 1 bit", ghz.entanglement_entropy([0, 1]), 1.0)

    # ------------------------------------------------------------------
    print("\n[4] 受控 RY：control=0 不動，control=1 完全翻轉")
    # ------------------------------------------------------------------
    # control=0 分支
    s0 = StateVectorSim(2).cry(np.pi, 0, 1)          # |00> -> |00>
    c.close("cry(pi,0,1) 於 |00>：P(00)", s0.probabilities()[0], 1.0)
    # control=1 分支：先把 control 設為 |1>
    s1 = StateVectorSim(2).x(0).cry(np.pi, 0, 1)     # |10> -> |11>
    c.close("cry(pi,0,1) 於 |10>：P(11)", s1.probabilities()[3], 1.0)
    c.close("cry(pi,0,1) 於 |10>：P(10)", s1.probabilities()[2], 0.0)
    # 疊加：貝爾態（RY(pi) 與 X 只差全域相位）
    sb = StateVectorSim(2).h(0).cry(np.pi, 0, 1)
    c.close("h(0);cry(pi,0,1) 的 P(00)", sb.probabilities()[0], 0.5)
    c.close("h(0);cry(pi,0,1) 的 P(11)", sb.probabilities()[3], 0.5)
    c.close("h(0);cry(pi,0,1) 的糾纏熵 = 1 bit", sb.entanglement_entropy([0]), 1.0)

    # 一般角度的條件機率：target 的 P(1) = sin^2(theta/2) 只在 control=1 時
    th = 1.1
    s = StateVectorSim(2).x(0).cry(th, 0, 1)
    c.close(f"cry({th},0,1) 於 |10>：P(11) = sin^2(theta/2)",
            s.probabilities()[3], np.sin(th / 2) ** 2)

    # ------------------------------------------------------------------
    print("\n[5] 多控制位 RY（mcry）")
    # ------------------------------------------------------------------
    s = StateVectorSim(3).x(0).x(1).mcry(np.pi, [0, 1], 2)   # |110> -> |111>
    c.close("mcry(pi,[0,1],2) 於 |110>：P(111)", s.probabilities()[7], 1.0)
    s = StateVectorSim(3).x(0).mcry(np.pi, [0, 1], 2)         # |100> 不該旋轉
    c.close("mcry(pi,[0,1],2) 於 |100>：P(100) 不變", s.probabilities()[4], 1.0)
    s = StateVectorSim(3).mcry(np.pi, [0, 1], 2)              # |000> 不該旋轉
    c.close("mcry(pi,[0,1],2) 於 |000>：P(000) 不變", s.probabilities()[0], 1.0)

    # ------------------------------------------------------------------
    print("\n[6] 與獨立 Kron 參考實作逐振幅比對（抓軸順序錯誤）")
    # ------------------------------------------------------------------
    n_ref = 3
    sim = StateVectorSim(n_ref)
    ops: List[tuple] = []
    for kind, mat, *rest in [
        ("1q", _ry(0.31), 0),
        ("1q", _rx(-0.77), 1),
        ("1q", _H, 2),
        ("ctrl", _X, 0, 1),
        ("ctrl", _ry(1.23), 1, 2),
        ("ctrl", _Z, 2, 0),
        ("1q", _rz(0.5), 1),
    ]:
        ops.append((kind, mat, *rest))
        if kind == "1q":
            sim._apply_1q(mat, rest[0])
        else:
            sim._apply_controlled_1q(mat, [rest[0]], rest[1])
    ref = _reference_state_via_kron(n_ref, ops)
    got = sim.amplitudes()
    c.check(
        "7 閘隨機電路 vs Kron 參考實作（振幅）",
        np.allclose(got, ref, atol=1e-13),
        f"max|dpsi| = {np.max(np.abs(got - ref)):.3g}",
    )
    c.check("態向量維持歸一", abs(np.linalg.norm(got) - 1.0) < 1e-13,
            f"||psi|| = {np.linalg.norm(got):.15f}")

    # 位元順序的獨立檢查：只對 qubit 0 施加 X，機率應落在索引 2^(n-1)
    s = StateVectorSim(4).x(0)
    c.close("x(0) 於 4 qubit：P(|1000>) 位於索引 8",
            s.probabilities()[8], 1.0, note="驗證 qubit 0 是最高位元")

    # ------------------------------------------------------------------
    print("\n[7] 取樣與解析機率的統計一致性（100000 shots, 3 sigma）")
    # ------------------------------------------------------------------
    shots = 100_000
    bell = StateVectorSim(2).h(0).cx(0, 1)
    counts = bell.sample(shots, seed=rng_seed)
    analytic = bell.probabilities()
    sigma_max = 0.0
    worst = ""
    for i in (0, 3):                                  # 只有 |00> 與 |11> 會出現
        fmt = format(i, "02b")
        got_freq = counts.get(fmt, 0) / shots
        want = analytic[i]
        sigma = np.sqrt(want * (1.0 - want) / shots)   # 二項分布的標準差
        z = abs(got_freq - want) / sigma
        if z > sigma_max:
            sigma_max, worst = z, f"|{fmt}>: freq={got_freq:.5f}, p={want}, z={z:.2f}"
        c.check(
            f"取樣頻率落在 3 sigma 內：|{fmt}>",
            z <= 3.0,
            f"freq={got_freq:.5f}, p={want:.5f}, z={z:.2f}",
        )
    c.check("|01> 與 |10> 未被取樣到",
            counts.get("01", 0) == 0 and counts.get("10", 0) == 0)
    c.check("取樣總數正確", sum(counts.values()) == shots,
            f"sum={sum(counts.values())}")
    # 可重現性
    c.check("相同 seed 可完全重現",
            bell.sample(1000, seed=7) == bell.sample(1000, seed=7))
    # 5 qubit 均勻疊加的卡方式檢查
    s5 = StateVectorSim(5)
    for k in range(5):
        s5.h(k)
    counts5 = s5.sample(shots, seed=rng_seed)
    freqs = np.array([counts5.get(format(i, "05b"), 0) / shots for i in range(32)])
    zs = np.abs(freqs - 1 / 32) / np.sqrt((1 / 32) * (1 - 1 / 32) / shots)
    c.check("5 qubit 均勻疊加：32 個結果全部落在 3 sigma 內",
            float(zs.max()) <= 3.0, f"max z = {zs.max():.2f}")

    # ------------------------------------------------------------------
    print("\n[8] 期望值與去極化雜訊")
    # ------------------------------------------------------------------
    s = StateVectorSim(1).ry(np.pi / 2, 0)      # <X> = 1, <Z> = 0
    c.close("<X> of RY(pi/2)|0>", s.expectation_pauli("X", 0), 1.0)
    c.close("<Z> of RY(pi/2)|0>", s.expectation_z(0), 0.0)
    c.close("<Z> of |0>", StateVectorSim(1).expectation_z(0), 1.0)
    c.close("<Z> of |1>", StateVectorSim(1).x(0).expectation_z(0), -1.0)
    bell = StateVectorSim(2).h(0).cx(0, 1)
    c.close("貝爾態 <Z_0> = 0", bell.expectation_z(0), 0.0)
    c.close("貝爾態 <Z_0 Z_1> = 1（完全相關）",
            float(bell.expectation_z(0) * 0 + np.sum(
                (1 - 2 * bell.bits_matrix()[:, 0])
                * (1 - 2 * bell.bits_matrix()[:, 1]) * bell.probabilities())), 1.0)

    # 去極化：p=0 不變；p=1 全均勻
    s = StateVectorSim(3).h(0).cx(0, 1).cx(1, 2)
    c.check("p_depol=0 時機率不變",
            np.allclose(s.probabilities_with_noise(0.0), s.probabilities(), atol=1e-14))
    pn = s.probabilities_with_noise(1.0)
    c.check("p_depol=1 時每個 qubit 完全混合 => 均勻分布",
            np.allclose(pn, np.full(8, 1 / 8), atol=1e-12),
            f"max|p - 1/8| = {np.max(np.abs(pn - 1 / 8)):.3g}")
    pn = s.probabilities_with_noise(0.1)
    c.close("去極化後機率仍歸一", float(pn.sum()), 1.0, tol=1e-12)
    c.check("去極化後 |000> 機率下降",
            pn[0] < s.probabilities()[0],
            f"{s.probabilities()[0]:.6f} -> {pn[0]:.6f}")

    # 解析釘死通道的精確作用：對單 qubit 的 rho=diag(p0, 1-p0)，
    # 去極化應給出 P(0) = (1-p)*p0 + p*(1/2)。
    # 這條式子可以區分 (1-p)rho + p*I/2（正確）與
    # (1-p)rho + (p/3)(X rho X + Y rho Y + Z rho Z)（錯誤，p=1 時給 0.4 而非 0.5）。
    p0 = 0.8
    theta0 = 2.0 * np.arccos(np.sqrt(p0))          # 使 P(0) = cos^2(theta/2) = p0
    s1q = StateVectorSim(1).ry(theta0, 0)
    c.close("單 qubit 起點 P(0)", s1q.probabilities()[0], p0, tol=1e-12)
    for pp in (0.0, 0.25, 0.5, 1.0):
        want = (1.0 - pp) * p0 + pp * 0.5
        c.close(f"去極化 p={pp} 的 P(0) = (1-p)*{p0} + p*1/2",
                s1q.probabilities_with_noise(pp)[0], want, tol=1e-12)
    # Bloch 向量收縮：<Z> 由 (2*p0-1) 縮為 (1-p)*(2*p0-1)
    for pp in (0.0, 0.3, 1.0):
        got = 2.0 * s1q.probabilities_with_noise(pp)[0] - 1.0
        c.close(f"去極化 p={pp} 的 <Z> = (1-p)*{2 * p0 - 1:.1f}",
                got, (1.0 - pp) * (2.0 * p0 - 1.0), tol=1e-12)
    # 貝爾態去極化：相關性消失但邊際仍是均勻
    bell_n = StateVectorSim(2).h(0).cx(0, 1).probabilities_with_noise(1.0)
    c.check("貝爾態 p=1 去極化 => 均勻分布",
            np.allclose(bell_n, np.full(4, 0.25), atol=1e-12),
            f"max|p - 1/4| = {np.max(np.abs(bell_n - 0.25)):.3g}")

    # ------------------------------------------------------------------
    print("\n[9] 效能：n=15 的向量化 gate（不得用 Python 迴圈走振幅）")
    # ------------------------------------------------------------------
    import time

    t0 = time.perf_counter()
    big = StateVectorSim(15)
    for k in range(15):
        big.h(k)
    for k in range(15):
        big.cx(k, (k + 1) % 15)
    big.mcry(0.3, [0, 1, 2], 7)
    dt = time.perf_counter() - t0
    pbig = big.probabilities()
    c.check(f"n=15（{2 ** 15} 振幅）完成 31 個閘", True, f"耗時 {dt * 1000:.1f} ms")
    c.check("n=15 機率歸一", abs(pbig.sum() - 1.0) < 1e-12,
            f"sum = {pbig.sum():.15f}")

    return c.all_passed(), c


def main() -> int:
    print("=" * 72)
    print("qbn_sim.py 自我測試：純 NumPy 狀態向量模擬器")
    print(f"NumPy 版本：{np.__version__}")
    print("=" * 72)
    ok, checker = run_self_tests()
    checker.summary()
    if not ok:
        print("有測試未通過——請修程式，不要改測試。")
        return 1
    print("全部通過。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
