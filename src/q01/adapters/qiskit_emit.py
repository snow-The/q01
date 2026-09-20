"""IR -> Qiskit 電路，用 Qiskit 官方（純 Python）模擬器取機率。

公約（實測）：Qiskit 的 `Statevector.data` 是 **little-endian**（q0 = LSB），與 CUDA-Q 的
`get_state` 相同 ⇒ **零置換**。
不需要 Aer：`qiskit.quantum_info.Statevector` 只依賴 quantum_info（SPEC §8.4）。
"""

from __future__ import annotations

import numpy as np

name = "qiskit"
version = "?"

try:
    import qiskit
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import (HGate, RXGate, RYGate, RZGate, SGate, SdgGate, SXGate,
                                        TGate, TdgGate, XGate, YGate, ZGate)
    from qiskit.quantum_info import Operator, Statevector

    version = qiskit.__version__
    _GATES = {"h": HGate, "x": XGate, "y": YGate, "z": ZGate,
              "s": SGate, "sdg": SdgGate, "t": TGate, "tdg": TdgGate, "sx": SXGate,
              "rx": RXGate, "ry": RYGate, "rz": RZGate}
except Exception:                                 # pragma: no cover
    _GATES = {}


def available() -> bool:
    return bool(_GATES)


def circuit_of(trace) -> "QuantumCircuit":
    qc = QuantumCircuit(trace.n_qubits)
    for op in trace.ops:
        base = _GATES[op.name](*op.params)
        if op.controls:
            qc.append(base.control(len(op.controls)), [*op.controls, *op.qubits])
        else:
            qc.append(base, list(op.qubits))
    return qc


def probabilities(trace) -> np.ndarray:
    return np.abs(Statevector(circuit_of(trace)).data) ** 2


def unitary(trace) -> np.ndarray:
    return np.asarray(Operator(circuit_of(trace)).data)
