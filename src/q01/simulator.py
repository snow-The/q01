"""狀態向量模擬器（NumPy）—— q01 的數值核心。

公約：**內部狀態是 big-endian**（q[0] = 最高位），與 `sample` 的字串鍵同構。
對外的 `get_state` 才是 little-endian，轉換只在 `bitorder.py` 發生（SPEC §4.1）。

最佳化策略（每一項都有 SPEC §6 的實測依據）：
- 單閘**全程就地切片、零配置**（§6.7(7)）：舊的 einsum／tensordot 路徑已移除，
  它在 n ≤ 12 反而慢一倍以上。
- **閘融合**（§6.9 可行清單第 4 項，2026-09-20 實測）：同一個 qubit 上連續的
  無控制 1q 閘先合成一個 2×2、再作用一次，省掉每次作用的固定開銷。
  實測 1.19–1.66×（n = 5–16）；數值差僅 ~1e-16（矩陣乘法的結合律 + 浮點捨入）。
  落地時機見 `flush()`。
- 對角閘（z/s/sdg/t/tdg/rz）走半片乘法；X 走 flip；這兩類不必做通用 2×2。
- 受控閘：對「控制位全為 1」的那一半做 2x2，不建全矩陣。
"""

from __future__ import annotations

import numpy as np

from . import bitorder
from .gates import matrix_of

EINSUM_MAX_QUBITS = 13          # 對齊 PennyLane EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD
_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

__all__ = ["Simulator"]


def _is_diagonal(u: np.ndarray) -> bool:
    return u[0, 1] == 0 and u[1, 0] == 0


def _is_x(u: np.ndarray) -> bool:
    return u[0, 0] == 0 and u[1, 1] == 0 and u[0, 1] == 1 and u[1, 0] == 1


class Simulator:
    def __init__(self, n_qubits: int, rng: np.random.Generator | None = None) -> None:
        if n_qubits < 0:
            raise ValueError("n_qubits 必須 >= 0（0 表示尚未配置 qubit，之後用 grow() 擴充）")
        self.n = int(n_qubits)
        self.dim = 1 << self.n
        self._psi = np.zeros(self.dim, dtype=np.complex128)
        self._psi[0] = 1.0
        self._flushing = False
        self.rng = rng if rng is not None else np.random.default_rng()
        # 單閘熱路徑的可重用暫存（SPEC §6.9：避免每閘配置 → n ≤ 12 快 2.0–2.5×）
        self._scratch_size = -1
        self._s1: np.ndarray | None = None
        self._s2: np.ndarray | None = None
        # 閘融合：qubit -> 待作用的合成 2×2（見 apply() 與 flush()）
        self._pending: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- 基本
    @property
    def psi(self) -> np.ndarray:
        """目前狀態。**讀取一定會先落地待作用的融合閘**，外部永遠看到最新狀態。

        這條不變式是閘融合能安全存在的關鍵：sampler、spin、測試與 oracle 工具
        都會直接讀 tr.live.psi，若這裡回傳未落地的舊陣列，它們會拿到錯的狀態
        （第一版就是這樣壞掉的：4 項測試失敗）。
        與其逐一去補 flush（下一個讀取點又會踩），不如讓「讀就一定是最新的」
        成為型別保證。
        """
        if self._pending and not self._flushing:
            self.flush()
        return self._psi

    @psi.setter
    def psi(self, value: np.ndarray) -> None:
        self._psi = value

    def _bit(self, k: int) -> int:
        """qubit k 在 big-endian 索引中的位元權重。"""
        return 1 << (self.n - 1 - k)

    def amplitudes(self) -> np.ndarray:
        return self.snapshot()

    def probabilities(self) -> np.ndarray:
        self.flush()
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        self.flush()
        return float(np.linalg.norm(self.psi))

    # ------------------------------------------------------------- 閘作用
    def apply(self, name: str, params: tuple[float, ...], qubits: tuple[int, ...], controls: tuple[int, ...] = ()) -> None:
        """套用一個 op。IR 的正規形式是「單 qubit 閘 + 控制位」（分解在 `gates.py` 完成）。

        **閘融合**：無控制的單 qubit 閘不急著作用，先與同一 qubit 上待作用的閘合成
        （`u @ prev`，即「先 prev 再 u」），等到必須落地時才作用一次。同一個 qubit 上
        連續的 1q 閘本來就可以合併，而每次作用都有固定的呼叫開銷
        （SPEC §6.2：0.205 ms 固定 + 8.5 ns/(振幅·閘)），所以合併直接省下那份開銷。

        必須落地的情況：任何受控閘（它的作用取決於控制位的值）、任何讀取狀態的動作、
        `grow()`、以及量測——都在各自的入口呼叫 `flush()`。
        """
        u = matrix_of(name, params)
        if controls:
            self.flush()
            self.apply_controlled_1q(u, controls, qubits[0])
            return
        k = qubits[0]
        prev = self._pending.get(k)
        self._pending[k] = u if prev is None else u @ prev

    def flush(self) -> None:
        """把所有待作用的融合閘落地。需要「看到目前狀態」的動作都必須先呼叫它。"""
        pending = self._pending
        if not pending:
            return
        self._flushing = True          # 期間不要再遞迴進來
        try:
            for k in sorted(pending):
                self.apply_1q(pending.pop(k), k)
        finally:
            self._flushing = False

    def snapshot(self) -> np.ndarray:
        """目前狀態的副本（先落地待作用的閘）。"""
        self.flush()
        return self.psi.copy()

    def grow(self, extra: int) -> None:
        """在**低位端**追加 qubit（big-endian 下即尾端），保持既有振幅。"""
        if extra <= 0:
            return
        self.flush()          # n 改變之後就無法再作用舊的融合閘
        new_n = self.n + extra
        new = np.zeros(1 << new_n, dtype=np.complex128)
        new[:: 1 << extra] = self.psi
        self.psi = new
        self.n = new_n
        self.dim = 1 << new_n

    def apply_1q(self, u: np.ndarray, k: int) -> None:
        """單閘作用：**全程就地、零配置**（SPEC §6.9 實測比 einsum 路徑快 2.0–2.5×，n ≤ 12）。

        big-endian 下 qubit k 的位元權重是 2^(n-1-k)，所以
        `psi.reshape(-1, 2, 1 << (n-1-k))` 的中間軸正好是 qubit k；
        對連續的 psi 而言這是**視圖**（不複製）。舊的 einsum／tensordot 路徑已移除
        （實測在 n=18–20 兩者持平，在 n ≤ 12 則慢一倍以上）。
        """
        if not self._flushing:
            self.flush()               # 被直接呼叫時，先讓待作用的閘落地以維持時序
        v = self.psi.reshape(-1, 2, 1 << (self.n - 1 - k))
        a = v[:, 0, :]
        b = v[:, 1, :]

        if _is_diagonal(u):
            # ★ 兩個半片各自乘自己的相位。**不可**先提出 u[0,0] 再只乘 u[1,1]——
            # 那會得到 diag(u00, u00*u11)，對 rz 這種 u00 != 1 的閘是錯的（相對相位差一級）。
            if u[0, 0] != 1:
                a *= u[0, 0]
            if u[1, 1] != 1:
                b *= u[1, 1]
            return

        if _is_x(u):
            s1, _ = self._scratch(a.shape)
            np.copyto(s1, a)
            np.copyto(a, b)
            np.copyto(b, s1)
            return

        s1, s2 = self._scratch(a.shape)
        np.multiply(a, u[0, 0], out=s1)      # s1 = u00·a
        np.multiply(b, u[0, 1], out=s2)
        np.add(s1, s2, out=s1)               # s1 = 新的 a
        np.multiply(a, u[1, 0], out=s2)      # a 還是舊的
        np.multiply(b, u[1, 1], out=b)
        np.add(s2, b, out=b)                 # b = 新的 b
        a[...] = s1                          # a ← 新的 a

    def _scratch(self, shape):
        """兩個可重用的暫存緩衝（大小不變就不重配；只有 grow() 之後才可能變）。"""
        need = 1
        for s in shape:
            need *= int(s)
        if self._scratch_size != need:
            self._s1 = np.empty(need, dtype=np.complex128)
            self._s2 = np.empty(need, dtype=np.complex128)
            self._scratch_size = need
        return self._s1.reshape(shape), self._s2.reshape(shape)

    def apply_controlled_1q(self, u: np.ndarray, controls: tuple[int, ...], target: int) -> None:
        self.flush()          # 受控閘的作用取決於控制位的值 -> 待作用的閘必須先落地
        controls = tuple(sorted(set(int(c) for c in controls)))
        for c in controls:
            if c == target:
                raise ValueError("控制位與目標位不能相同")
            if not 0 <= c < self.n:
                raise IndexError(f"控制位 {c} 超出 qubit 範圍 0..{self.n - 1}")
        if not 0 <= target < self.n:
            raise IndexError(f"目標位 {target} 超出 qubit 範圍 0..{self.n - 1}")

        if len(controls) == 1:
            self._apply_one_control(u, controls[0], target)
        else:
            self._apply_many_controls(u, controls, target)

    # ------------------------------------------------------------------
    # 兩條受控路徑（SPEC §6.7 的實測結論）：
    #   單一控制位 = 熱路徑（環形 CX、x.ctrl）→ 走**視圖**，不配置任何索引陣列。
    #   多控制位（ccx、多控制 RY）→ 走遮罩路徑；書中少用，正確性優先。
    # 為什麼要分：遮罩路徑每個閘都重建 np.arange(2^n) 與布林遮罩，
    # n=18 時每閘 4 MB，造成超線性（n=14→16→18 = 4.5 → 18.7 → 267 ms）。
    # ------------------------------------------------------------------
    def _apply_one_control(self, u: np.ndarray, control: int, target: int) -> None:
        n = self.n
        moved = np.moveaxis(self.psi.reshape((2,) * n), target, 0)   # 視圖；軸序 = [target, 其餘依原序]
        pos = control if control < target else control - 1            # 控制位在「其餘」中的位置
        idx = [slice(None)] * n
        idx[1 + pos] = 1                                              # 只取控制位 = 1 的那一片
        sub = moved[tuple(idx)]                                       # 視圖（基本索引，不複製）
        sub[...] = np.matmul(u, sub.reshape(2, -1)).reshape(sub.shape)

    def _apply_many_controls(self, u: np.ndarray, controls: tuple[int, ...], target: int) -> None:
        t = np.moveaxis(self.psi.reshape((2,) * self.n), target, 0).reshape(2, -1)
        mask = self._control_mask(controls, target)
        t[:, mask] = u @ t[:, mask]
        self.psi = np.ascontiguousarray(np.moveaxis(t.reshape((2,) * self.n), 0, target)).reshape(-1)

    def _control_mask(self, controls: tuple[int, ...], target: int) -> np.ndarray:
        """在 (2, 2^(n-1)) 視圖的**列索引**上，控制位全為 1 的布林遮罩。"""
        m = self.n - 1
        col = np.arange(1 << m, dtype=np.int64)
        mask = np.ones(1 << m, dtype=bool)
        for c in controls:
            pos = c if c < target else c - 1      # 移除 target 軸後的位置
            shift = m - 1 - pos
            mask &= ((col >> shift) & 1) == 1
        return mask

    # --------------------------------------------------------------- 量測
    def marginal(self, qubits: tuple[int, ...]) -> np.ndarray:
        """對給定的 qubit 取邊際分布；索引順序與 qubits 的順序一致（big-endian）。"""
        probs = self.probabilities().reshape((2,) * self.n)
        keep = tuple(qubits)
        drop = tuple(i for i in range(self.n) if i not in keep)
        reduced = probs.sum(axis=drop) if drop else probs
        # reduced 的軸順序是 keep 在原順序中的順序；重排成 qubits 給定的順序
        order = [keep.index(q) for q in qubits]
        if order != list(range(len(keep))):
            reduced = np.transpose(reduced, order)
        return np.ascontiguousarray(reduced).reshape(-1)

    def sample_counts(self, shots: int, qubits: tuple[int, ...] | None = None) -> dict[int, int]:
        """從精確機率抽樣（O(shots)）。回傳 {big-endian 局部索引: 次數}。"""
        if qubits is None:
            qubits = tuple(range(self.n))
        probs = self.marginal(tuple(qubits))
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        draws = self.rng.choice(len(probs), size=int(shots), p=probs)
        counts = np.bincount(draws, minlength=len(probs))
        return {int(i): int(c) for i, c in enumerate(counts) if c}

    def measure_collapse(self, qubits: tuple[int, ...]) -> tuple[int, ...]:
        """量測並塌縮；回傳每個 qubit 的位元（順序同 qubits）。"""
        qubits = tuple(int(q) for q in qubits)
        probs = self.marginal(qubits)
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:
            raise RuntimeError("狀態範數為零，無法量測")
        outcome = int(self.rng.choice(len(probs), p=probs / total))
        bits = tuple((outcome >> (len(qubits) - 1 - i)) & 1 for i in range(len(qubits)))

        keep = np.ones(self.dim, dtype=bool)
        idx = np.arange(self.dim, dtype=np.int64)
        for q, b in zip(qubits, bits):
            shift = self.n - 1 - q
            keep &= (((idx >> shift) & 1) == b)
        self.psi = np.where(keep, self.psi, 0.0)
        nrm = np.linalg.norm(self.psi)
        if nrm > 0:
            self.psi = self.psi / nrm
        return bits

    # --------------------------------------------------------- 期望值／相位
    def apply_pauli(self, qubit: int, which: str) -> None:
        self.apply(which, (), (qubit,))

    def expectation_pauli(self, terms: tuple[tuple[int, str], ...]) -> float:
        """⟨ψ|P|ψ⟩，P 是多個 (qubit, 'x'|'y'|'z') 的乘積。"""
        saved = self.snapshot()
        for q, which in terms:
            self.apply(which, (), (q,))
        self.flush()                      # 融合閘要落地才看得到
        val = float(np.real(np.vdot(saved, self.psi)))
        self.psi = saved
        return val

    def global_phase_of(self, other: np.ndarray) -> complex:
        """回傳 c 使得 psi ≈ c * other（最小平方）。"""
        self.flush()
        denom = np.vdot(other, other)
        return complex(np.vdot(other, self.psi) / denom) if denom != 0 else 0j

    @staticmethod
    def unitary_from(n_qubits: int, ops, gate_matrix=matrix_of) -> np.ndarray:
        """用給定的 op 序列建構 2^n x 2^n 么正矩陣（對每個基底態各跑一次）。

        這是 L1 算子級對帳的工具（SPEC §4.3：判準 `U†V ≈ cI`）。
        """
        dim = 1 << n_qubits
        u = np.zeros((dim, dim), dtype=np.complex128)
        for j in range(dim):
            sim = Simulator(n_qubits)
            sim.psi = np.zeros(dim, dtype=np.complex128)
            sim.psi[j] = 1.0
            for op in ops:
                sim.apply(op.name, op.params, op.qubits, op.controls)
            sim.flush()
            u[:, j] = sim.psi
        return u


def little_endian_view(psi: np.ndarray, n: int) -> np.ndarray:  # pragma: no cover - 便利函式
    return bitorder.state_big_to_little(psi, n)
