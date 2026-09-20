"""IR -> PennyLane QNode，用 PennyLane 的 `default.qubit`（純 Python）取機率。

公約（實測）：PennyLane 的 `qml.state()` 是 **big-endian**（q0 = MSB）⇒ 需要一次置換
（走 `bitorder`，全專案唯一置換處）。

⚠️ 踩過的坑：`qml.ctrl(<實例>, control=...)` **不會排進電路**（CNOT 靜默失效）。
必須傳**可呼叫物**：`qml.ctrl(qml.PauliX, control=[0])(wires=1)`。
這個 bug 在貝爾態上表現為「少了糾纏」，只看數字很像位元順序錯——所以 L0 公約探針要與
L2 電路比對分開跑。
"""

from __future__ import annotations

import numpy as np

from .. import bitorder

name = "pennylane"
version = "?"

try:
    import pennylane as qml

    version = qml.__version__
    _AVAILABLE = True
except Exception:                                 # pragma: no cover
    _AVAILABLE = False

# 名稱 -> (可呼叫物, 參數個數)；sdg/tdg 用 adjoint 包裝
_TABLE = {
    "h": (qml.Hadamard, 0) if _AVAILABLE else None,
    "x": (qml.PauliX, 0) if _AVAILABLE else None,
    "y": (qml.PauliY, 0) if _AVAILABLE else None,
    "z": (qml.PauliZ, 0) if _AVAILABLE else None,
    "s": (qml.S, 0) if _AVAILABLE else None,
    "t": (qml.T, 0) if _AVAILABLE else None,
    "sx": (qml.SX, 0) if _AVAILABLE else None,
    "rx": (qml.RX, 1) if _AVAILABLE else None,
    "ry": (qml.RY, 1) if _AVAILABLE else None,
    "rz": (qml.RZ, 1) if _AVAILABLE else None,
}
if _AVAILABLE:
    _TABLE["sdg"] = (qml.adjoint(qml.S), 0)
    _TABLE["tdg"] = (qml.adjoint(qml.T), 0)


def available() -> bool:
    return _AVAILABLE


def _perm(n: int) -> np.ndarray:
    return np.array([bitorder.little_to_big(i, n) for i in range(1 << n)])


def probabilities(trace) -> np.ndarray:
    dev = qml.device("default.qubit", wires=trace.n_qubits)

    @qml.qnode(dev)
    def circuit():
        for op in trace.ops:
            fn, n_par = _TABLE[op.name]
            args = tuple(float(p) for p in op.params[:n_par])
            target = op.qubits[0]
            if op.controls:
                qml.ctrl(fn, control=[int(c) for c in op.controls])(*args, wires=target)
            else:
                fn(*args, wires=target)
        return qml.state()

    p_big = np.abs(np.asarray(circuit())) ** 2
    return p_big[_perm(trace.n_qubits)]              # big-endian -> little-endian
