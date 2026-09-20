"""IR -> Cirq 電路，用 Cirq 官方本地模擬器取機率。

公約（實測）：Cirq 的 `final_state_vector` 是 **big-endian**（q0 = MSB）⇒ 需要一次置換
（走 `bitorder`）。

⚠️ **Cirq 預設是 complex64**（實測：貝爾態機率 0.49999997），**達不到 1e-10 判準**。
本檔一律指定 `dtype=np.complex128`。
"""

from __future__ import annotations

import numpy as np

from .. import bitorder

name = "cirq"
version = "?"

try:
    import cirq

    version = cirq.__version__
    _AVAILABLE = True
except Exception:                                 # pragma: no cover
    _AVAILABLE = False


def available() -> bool:
    return _AVAILABLE


def _gate(name_: str, params: tuple[float, ...]):
    if name_ == "h":
        return cirq.H
    if name_ == "x":
        return cirq.X
    if name_ == "y":
        return cirq.Y
    if name_ == "z":
        return cirq.Z
    if name_ == "s":
        return cirq.S
    if name_ == "sdg":
        return cirq.S**-1
    if name_ == "t":
        return cirq.T
    if name_ == "tdg":
        return cirq.T**-1
    if name_ == "sx":
        return cirq.X**0.5
    if name_ in ("rx", "ry", "rz"):
        base = {"rx": cirq.rx, "ry": cirq.ry, "rz": cirq.rz}[name_]
        return base(float(params[0]))
    raise KeyError(name_)


def circuit_of(trace):
    qubits = cirq.LineQubit.range(trace.n_qubits)
    ops = []
    for op in trace.ops:
        gate = _gate(op.name, op.params)
        sub = gate.on(qubits[op.qubits[0]])
        if op.controls:
            sub = cirq.ControlledGate(sub.gate, num_controls=len(op.controls)).on(
                *[qubits[c] for c in op.controls], *[qubits[t] for t in op.qubits])
        ops.append(sub)
    return cirq.Circuit(ops), qubits


def probabilities(trace) -> np.ndarray:
    circuit, qubits = circuit_of(trace)
    sim = cirq.Simulator(dtype=np.complex128)      # ★ 不用預設的 complex64
    state = sim.simulate(circuit, qubit_order=qubits).final_state_vector
    p_big = np.abs(np.asarray(state)) ** 2
    n = trace.n_qubits
    perm = np.array([bitorder.little_to_big(i, n) for i in range(1 << n)])
    return p_big[perm]                             # big-endian -> little-endian
