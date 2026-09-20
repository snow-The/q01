"""trace 電路 IR 與 qubit 控制代碼。

設計要點（SPEC §4.4）：與 CUDA-Q 一樣採**追蹤式**執行；支援的 Python 構造以 CUDA-Q 的
AST 白名單為準。IR 必須有 `global_phase` 欄位（SPEC §8.2：tket `Circuit.hpp:1644` 同構設計），
否則受控閘分解的比對會永遠差一個相位。
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field

__all__ = ["Op", "QubitRef", "QVector", "Trace", "current_trace", "trace_context", "NoTraceError"]


class NoTraceError(RuntimeError):
    """在 `@kernel` 之外呼叫閘。CUDA-Q 的對應錯誤是編譯期錯誤（閘只存在於 kernel 內）。"""


@dataclass(frozen=True)
class Op:
    name: str
    params: tuple[float, ...]
    qubits: tuple[int, ...]
    controls: tuple[int, ...] = ()


@dataclass(frozen=True)
class QubitRef:
    index: int

    def __repr__(self) -> str:  # pragma: no cover
        return f"q[{self.index}]"


@dataclass
class QVector:
    """`cudaq.qvector(n)` 的回傳物；支援 `q[i]`、`q.front(k)`、`q.back()`。"""

    length: int
    offset: int = 0

    def __getitem__(self, i) -> QubitRef:
        if isinstance(i, slice):
            return QVector(len(range(*i.indices(self.length))), self.offset)
        idx = int(i)
        if not 0 <= idx < self.length:
            raise IndexError(f"qvector({self.length}) 的索引 {idx} 超出範圍")
        return QubitRef(self.offset + idx)

    def front(self, k: int = 1):
        refs = [QubitRef(self.offset + i) for i in range(k)]
        return refs[0] if k == 1 else refs

    def back(self, k: int = 1) -> QubitRef:
        return QubitRef(self.offset + self.length - k)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for i in range(self.length):
            yield QubitRef(self.offset + i)


@dataclass
class Trace:
    """一次 kernel 執行的記錄。`live` 不為 None 時是「軌跡模式」：每個閘即時作用在狀態上。"""

    n_qubits: int = 0
    ops: list[Op] = field(default_factory=list)
    global_phase: complex = 1.0 + 0.0j
    live: object | None = None          # simulator.Simulator 實例
    measured: bool = False              # 是否已出現量測（用於判斷線路中量測）
    mid_circuit: bool = False           # 量測之後還有閘 => 線路中量測
    measures: list = field(default_factory=list)     # 每次量測的 qubit 元組（依序）
    outcomes: list = field(default_factory=list)     # 每次量測的實際位元（依序；軌跡模式用）
    pre_measurement: object | None = None            # 第一次量測前的狀態快照

    def allocate(self, count: int) -> QVector:
        vec = QVector(count, self.n_qubits)
        self.n_qubits += count
        if self.live is not None:
            self.live.grow(count)
        return vec

    def record(self, name: str, params: tuple[float, ...], qubits: tuple[int, ...], controls: tuple[int, ...]) -> None:
        if self.measured:
            self.mid_circuit = True     # 量測之後還有閘 → 線路中量測
        normalized = tuple(sorted(controls))
        self.ops.append(Op(name, params, qubits, normalized))
        if self.live is not None:
            self.live.apply(name, params, qubits, normalized)

    def mark_measured(self, qubits: tuple[int, ...]) -> None:
        if not self.measured and self.live is not None:
            self.pre_measurement = self.live.psi.copy()   # 供 `sample` 在量測前抽樣
        self.measured = True
        self.measures.append(tuple(qubits))


_CURRENT: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("q01_trace", default=None)


def current_trace() -> Trace:
    tr = _CURRENT.get()
    if tr is None:
        raise NoTraceError(
            "閘只能在 @cudaq.kernel（q01 的 @kernel）內呼叫。"
            "CUDA-Q 的閘同樣只存在於 kernel 內——請把電路寫成函式並加上 @kernel。"
        )
    return tr


class trace_context:
    """`with trace_context(trace):` —— 在區塊內，閘會記錄到這個 trace。"""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self._token = None

    def __enter__(self) -> Trace:
        self._token = _CURRENT.set(self.trace)
        return self.trace

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _CURRENT.reset(self._token)
