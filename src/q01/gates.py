"""閘集：矩陣定義 + 可呼叫物件（支援 CUDA-Q 的 `.ctrl()` 修飾語）。

矩陣公約與 CUDA-Q 實測逐元素相同（SPEC §5.2：最差 `max|U†V - cI|` = 2.47e-17）。
角度一律在前：`ry(theta, target)`、`ry.ctrl(theta, [c0, c1], target)`。
"""

from __future__ import annotations

import cmath

import numpy as np

__all__ = ["Gate", "GATES", "gate_names", "matrix_of"]

_S2 = 1.0 / np.sqrt(2.0)

CHAIN_CTRL_MESSAGE = (
    "{name}.ctrl(...) 的寫法不對。\n"
    "  真 CUDA-Q 0.16.0 對鏈式寫法 {name}.ctrl(c0).ctrl(c1)(theta, t) 的錯誤是：unknown function call\n"
    "  正確寫法是把**所有控制位放在同一個 list**：{name}.ctrl(theta, [q[0], q[1]], q[2])\n"
    "  （角度一律在前面；只有一個控制位時可以寫 {name}.ctrl(theta, q[0], q[1])。）"
)


class _CtrlCall:
    """`gate.ctrl(...)` 的回傳物；任何後續 `.ctrl()`／屬性存取都會給出教學式錯誤。"""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item):
        raise AttributeError(CHAIN_CTRL_MESSAGE.format(name=self._name))


def _rx(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t: float) -> np.ndarray:
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


FIXED: dict[str, np.ndarray] = {
    "h": np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, cmath.exp(-1j * np.pi / 4)]], dtype=complex),
}

PARAMETRIC = {"rx": _rx, "ry": _ry, "rz": _rz}


def _sx() -> np.ndarray:
    return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)


# 名稱 -> (參數個數, qubit 個數)
SPEC: dict[str, tuple[int, int]] = {
    "h": (0, 1), "x": (0, 1), "y": (0, 1), "z": (0, 1),
    "s": (0, 1), "sdg": (0, 1), "t": (0, 1), "tdg": (0, 1),
    "sx": (0, 1), "rx": (1, 1), "ry": (1, 1), "rz": (1, 1),
    "cx": (0, 2), "cz": (0, 2), "swap": (0, 2), "ccx": (0, 3),
}


def matrix_of(name: str, params: tuple[float, ...] = ()) -> np.ndarray:
    """回傳閘的 2x2 矩陣（僅單 qubit 閘；多 qubit 閘由模擬器以受控路徑處理）。"""
    if name in FIXED:
        return FIXED[name]
    if name == "sx":
        return _sx()
    if name in PARAMETRIC:
        return PARAMETRIC[name](float(params[0]))
    raise KeyError(f"{name} 沒有單一 2x2 矩陣（多 qubit 閘請走模擬器的受控／置換路徑）")


class Gate:
    """可呼叫的閘物件；`gate(...)` 只能在 `@kernel` 內使用，`gate.ctrl(...)` 額外帶控制位。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.n_params, self.n_qubits = SPEC[name]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<q01 gate {self.name}>"

    def _record(self, params: tuple[float, ...], qubits: tuple, controls: tuple) -> None:
        """把閘化成 IR 的正規形式：**單 qubit 閘 + 控制位元組**。

        這樣 adjoint／control 的組合、以及多控制位都能用同一條路徑處理
        （`swap` = 3 個受控 X；`ccx` = 2 控制的 X；`cz` = 受控 Z）。
        """
        from .circuit import current_trace
        tr = current_trace()
        name = self.name
        if name == "cx":
            tr.record("x", (), (qubits[1],), controls + (qubits[0],))
        elif name == "cz":
            tr.record("z", (), (qubits[1],), controls + (qubits[0],))
        elif name == "ccx":
            tr.record("x", (), (qubits[2],), controls + (qubits[0], qubits[1]))
        elif name == "swap":
            a, b = qubits
            for c, t in ((b, a), (a, b), (b, a)):
                tr.record("x", (), (t,), controls + (c,))
        else:
            tr.record(name, params, (qubits[0],), controls)

    def __call__(self, *args):
        # ★ 保留**原始參數物件**（不強制轉 float）：numpy 後端稍後自己 float()，
        # 而 torch 後端需要張量本體才能讓 autograd 的圖穿過 tracing（SPEC §5.4）。
        params = tuple(args[: self.n_params])
        qubits = tuple(split_qubits(a) for a in args[self.n_params:])
        flat = tuple(q for group in qubits for q in group)
        if len(flat) != self.n_qubits:
            raise TypeError(
                f"{self.name} 需要 {self.n_qubits} 個 qubit（收到 {len(flat)} 個）；"
                f"角度一律在參數前面：{self.name}(theta, target)"
            )
        self._record(params, flat, ())
        return None

    def ctrl(self, *args):
        """CUDA-Q 的修飾語：`ry.ctrl(theta, [c0, c1], t)`、`x.ctrl([c0, c1, c2], t)`。"""
        want = self.n_params + 2
        if len(args) != want:
            raise TypeError(CHAIN_CTRL_MESSAGE.format(name=self.name))
        params = tuple(args[: self.n_params])       # 同上：保留原始物件
        ctrls = split_qubits(args[self.n_params])
        target = split_qubits(args[self.n_params + 1])
        if len(target) != 1:
            raise TypeError(f"{self.name}.ctrl(...) 的目標必須剛好 1 個 qubit，收到 {target}")
        self._record(params, target, ctrls)
        return _CtrlCall(self.name)

    def __getattr__(self, item):  # pragma: no cover
        if item in ("ctrl", "adjoint"):
            raise AttributeError(item)
        raise AttributeError(
            f"q01 的閘物件沒有屬性 {item!r}。"
            f"（CUDA-Q 的鏈式寫法 ry.ctrl(c0).ctrl(c1)(...) 不支援，請寫 ry.ctrl(theta, [c0, c1], t)）"
        )


def split_qubits(obj) -> tuple[int, ...]:
    """把 qubit 或 qubit 序列攤平成索引元組。"""
    from .circuit import QubitRef
    if isinstance(obj, QubitRef):
        return (obj.index,)
    if isinstance(obj, (list, tuple)):
        out: list[int] = []
        for item in obj:
            out.extend(split_qubits(item))
        return tuple(out)
    raise TypeError(f"無法把 {obj!r} 當成 qubit（應為 q[i]、q.front(k) 或它們的序列）")


GATES: dict[str, Gate] = {name: Gate(name) for name in SPEC}


def gate_names() -> list[str]:
    return sorted(SPEC)
