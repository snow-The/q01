"""PyTorch 後端：**可微分（autograd）** ＋ 可選 GPU（SPEC §5.4／§5.5／§6.8）。

設計（由 SPEC §6.8 的實測導出）：
- 內部佈局是 **little-endian**：qubit t 的 stride 恰為 2^t
  ⇒ 單閘 = `view(-1, 2, 2^t)`，零拷貝、零 moveaxis（這正是原型能達到 CUDA-Q GPU 1.35–4.45× 的原因）。
- 兩條路徑：
  - `grad=False`（量測用）：**就地**寫回（最快，無暫存配置）。
  - `grad=True`（訓練用）：**函數式**（out-of-place），讓 autograd 建圖。
- 精度：預設 `complex128`。`complex64` 必須明示選用，且**不得宣稱 1e-10**
  （實測 torch complex64 在 n=14 的誤差是 5.99e-04）。

GPU 不是萬靈丹（§6.8）：n ≤ 12 時 CPU 較快（GPU 有固定呼叫開銷），交叉點約 n ≈ 13–14。
"""

from __future__ import annotations

import numpy as np

from ..gates import FIXED as _FIXED_MATRICES

# sx 的矩陣（與 gates.py 同一定義；torch 端需要常數表，故在此鏡射一次）
_SX_MATRIX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)

name = "torch"
version = "?"

try:
    import torch

    version = torch.__version__
    _AVAILABLE = True
except Exception:                                 # pragma: no cover
    _AVAILABLE = False

_DTYPES = {"complex128": "complex128", "complex64": "complex64"}

# 可微分路徑用索引張量的規模上限：n=20 時每個索引張量是 4 MB（int64、半個狀態），
# 再大就改用 grad=False 或較小的 n（見檔頭的範圍說明）。
MAX_GRAD_QUBITS = 20

__all__ = ["available", "TorchBackend", "state_of", "probabilities", "unitary", "gradient_of_expectation"]


def available() -> bool:
    return _AVAILABLE


def devices() -> list[str]:
    if not _AVAILABLE:
        return []
    out = ["cpu"]
    try:
        if torch.cuda.is_available():
            out.append("cuda")
    except Exception:
        pass
    for attr in ("mps", "xpu"):
        try:
            mod = getattr(torch.backends, attr, None)
            if mod is not None and mod.is_available():
                out.append(attr)
        except Exception:
            pass
    return out


class TorchBackend:
    def __init__(self, device: str = "cpu", dtype: str = "complex128") -> None:
        if not _AVAILABLE:                        # pragma: no cover
            raise RuntimeError("這個環境沒有 PyTorch（q01 的 torch 後端是選配 extra）")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype 必須是 {sorted(_DTYPES)}，收到 {dtype!r}")
        self.device = device
        self.dtype = dtype
        self._t = getattr(torch, _DTYPES[dtype])
        self._mask_cache: dict = {}

    # ------------------------------------------------------------------
    def _tensor(self, arr):
        return torch.as_tensor(np.asarray(arr, dtype=np.complex128), dtype=self._t, device=self.device)

    def _matrix_of(self, name: str, params):
        """用 **torch 運算**建閘矩陣。

        ★ 不能走 \`gates.matrix_of\`：那裡用 \`float(params[0])\` 取值，會把 autograd 的圖切斷
        （實測症狀：\`backward()\` 報 "element 0 of tensors does not require grad"）。
        常數閘直接查表；參數閘用 torch 的 cos/sin/exp **函數式**組出矩陣（不可用 in-place 賦值，
        那會讓結果脫離計算圖）。
        """
        if name in _FIXED_MATRICES:
            return self._tensor(_FIXED_MATRICES[name])
        if name == "sx":
            return self._tensor(_SX_MATRIX)
        th = params[0]
        if not isinstance(th, torch.Tensor):
            th = torch.tensor(float(th), dtype=torch.float64, device=self.device)
        half = th / 2
        c, s = torch.cos(half), torch.sin(half)
        zero = torch.zeros((), dtype=self._t, device=self.device)
        one = torch.ones((), dtype=self._t, device=self.device)
        if name == "rx":
            mi = -1j * s
            return torch.stack([torch.stack([c.to(self._t), mi.to(self._t)]),
                                torch.stack([mi.to(self._t), c.to(self._t)])])
        if name == "ry":
            return torch.stack([torch.stack([c.to(self._t), (-s).to(self._t)]),
                                torch.stack([s.to(self._t), c.to(self._t)])])
        if name == "rz":
            return torch.stack([torch.stack([torch.exp(-1j * half).to(self._t), zero]),
                                torch.stack([zero, torch.exp(1j * half).to(self._t)])])
        raise KeyError(f"torch 後端不認得閘 {name!r}")

    def _indices(self, controls: tuple[int, ...], target: int, n: int):
        """(lo, hi)：**所有**控制位=1 且目標=0／1 的索引（little-endian 權重）。"""
        key = (tuple(controls), target, n)
        got = self._mask_cache.get(key)
        if got is None:
            idx = torch.arange(1 << n, dtype=torch.int64, device=self.device)
            cond = torch.ones_like(idx, dtype=torch.bool)
            for c in controls:
                cond &= ((idx >> c) & 1) == 1
            tbit = (idx >> target) & 1
            got = (torch.nonzero(cond & (tbit == 0)).flatten(),
                   torch.nonzero(cond & (tbit == 1)).flatten())
            self._mask_cache[key] = got
        return got

    def _apply_1q_inplace(self, psi, u, t: int):
        v = psi.view(-1, 2, 1 << t)
        a, b = v[:, 0, :], v[:, 1, :]
        if u[0, 1] == 0 and u[1, 0] == 0:          # 對角
            if u[0, 0] != 1:
                a.mul_(u[0, 0])
            if u[1, 1] != 1:
                b.mul_(u[1, 1])
            return
        na = u[0, 0] * a + u[0, 1] * b
        b.copy_(u[1, 0] * a + u[1, 1] * b)
        a.copy_(na)

    def _apply_1q_functional(self, psi, u, t: int):
        v = psi.view(-1, 2, 1 << t)
        a, b = v[:, 0, :], v[:, 1, :]
        na = u[0, 0] * a + u[0, 1] * b
        nb = u[1, 0] * a + u[1, 1] * b
        return torch.stack((na, nb), dim=1).reshape(-1)

    def _apply_controlled_view(self, psi, u, controls, target: int, n: int):
        """不建索引張量：把控制位與目標位排到最前面，取控制位全 1 的那一片（視圖）。

        任意控制位數都適用（little-endian：qubit q 的軸是 n-1-q）。
        """
        axes = [n - 1 - q for q in tuple(controls) + (target,)]
        others = [k for k in range(n) if k not in axes]
        v = psi.view((2,) * n).permute(axes + others)
        sub = v[(1,) * len(controls)]        # 視圖；軸 0 = 目標位
        a, b = sub[0], sub[1]
        na = u[0, 0] * a + u[0, 1] * b
        b.copy_(u[1, 0] * a + u[1, 1] * b)
        a.copy_(na)

    def _apply_controlled_grad(self, psi, u, controls, target: int, n: int):
        """可微分版：用遮罩 + torch.where（索引張量按 (controls, target, n) 快取）。"""
        lo, hi = self._indices(tuple(controls), target, n)
        sel = torch.zeros_like(psi)
        sel[lo] = u[0, 0] * psi[lo] + u[0, 1] * psi[hi]
        sel[hi] = u[1, 0] * psi[lo] + u[1, 1] * psi[hi]
        mask = torch.zeros(psi.shape, dtype=torch.bool, device=psi.device)
        mask[lo] = True
        mask[hi] = True
        return torch.where(mask, sel, psi)

    # ------------------------------------------------------------------
    def state_of(self, trace, grad: bool = False):
        """回傳 **little-endian** 的狀態張量（與 CUDA-Q 的 get_state 同序）。"""
        n = trace.n_qubits
        if grad and n > MAX_GRAD_QUBITS:
            raise ValueError(
                f"可微分路徑目前限制在 n ≤ {MAX_GRAD_QUBITS}（索引張量的記憶體考量）；"
                f"收到 n={n}。請改用 grad=False 取樣本，或縮小電路。"
            )
        psi = torch.zeros(1 << n, dtype=self._t, device=self.device)
        psi[0] = 1.0
        for op in trace.ops:
            u = self._matrix_of(op.name, op.params)
            if op.controls:
                if grad:
                    psi = self._apply_controlled_grad(psi, u, op.controls, op.qubits[0], n)
                else:
                    self._apply_controlled_view(psi, u, op.controls, op.qubits[0], n)
            elif grad:
                psi = self._apply_1q_functional(psi, u, op.qubits[0])
            else:
                self._apply_1q_inplace(psi, u, op.qubits[0])
        return psi

    def probabilities(self, trace) -> np.ndarray:
        psi = self.state_of(trace, grad=False)
        return (psi.abs() ** 2).detach().cpu().numpy()

    def expectation(self, trace, spin_op, grad: bool = False):
        psi = self.state_of(trace, grad=grad)
        n = trace.n_qubits
        total = None
        for term, coef in spin_op.terms.items():
            vec = psi
            for q, which in term:
                u = self._matrix_of(which, ())
                if grad:
                    vec = self._apply_1q_functional(vec, u, q)
                else:
                    vec = vec.clone()
                    self._apply_1q_inplace(vec, u, q)
            val = torch.vdot(psi, vec).real * float(np.real(coef))
            total = val if total is None else total + val
        return total if total is not None else torch.zeros((), dtype=torch.float64, device=self.device)


def state_of(trace, device: str = "cpu", dtype: str = "complex128"):
    return TorchBackend(device, dtype).state_of(trace)


def probabilities(trace) -> np.ndarray:
    """給 adapters 用的介面（預設 CPU；要 GPU 請直接建 TorchBackend）。"""
    return TorchBackend("cpu").probabilities(trace)


def gradient_of_expectation(kernel, spin_op, params, device: str = "cpu", dtype: str = "complex128"):
    """對 kernel 的參數向量（list[float]）算 <spin_op> 的精確梯度（autograd）。"""
    be = TorchBackend(device, dtype)
    ts = [torch.tensor(float(p), dtype=torch.float64, device=device, requires_grad=True) for p in params]
    trace = kernel.trace(*ts, live=False)
    value = be.expectation(trace, spin_op, grad=True)
    value.backward()
    return float(value.detach().cpu()), [float(t.grad.detach().cpu()) for t in ts]
