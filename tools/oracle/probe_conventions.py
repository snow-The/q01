"""L0 公約探針：在每個框架上做『只對 qubit 0 施加 X』，看哪個索引是 1。

這是每個 adapter 的第一道關卡（SPEC §7.1 L0）：不先驗公約，後面所有比對都會失敗
而且你會花三天查一個不存在的 bug。
"""
import sys

import numpy as np

N = 3
print(f"python {sys.version.split()[0]} | numpy {np.__version__}")
print("判準：q0 只被 X 翻轉後的非零索引。little-endian -> 1；big-endian -> 4（2^(n-1)）")
print()

results = {}

# ---------------- qiskit ----------------
try:
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, Operator

    qc = QuantumCircuit(N)
    qc.x(0)
    sv = Statevector(qc).data
    idx = int(np.argmax(np.abs(sv)))
    results["qiskit"] = (qiskit.__version__, idx)

    qc2 = QuantumCircuit(2)
    qc2.h(0); qc2.cx(0, 1)
    p = np.abs(Statevector(qc2).data) ** 2
    bell = {i: float(v) for i, v in enumerate(p) if v > 1e-12}
    print(f"qiskit {qiskit.__version__}: x(q0) -> index {idx} | bell 非零 = {bell}")
except Exception as e:
    print(f"qiskit: ERR {type(e).__name__}: {e}")

# ---------------- cirq ----------------
try:
    import cirq
    q = cirq.LineQubit.range(N)
    c = cirq.Circuit(cirq.X(q[0]))
    sv = cirq.Simulator().simulate(c, qubit_order=q).final_state_vector
    idx = int(np.argmax(np.abs(sv)))
    results["cirq"] = (cirq.__version__, idx)
    c2 = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]))
    p = np.abs(cirq.Simulator().simulate(c2, qubit_order=q[:2]).final_state_vector) ** 2
    bell = {i: float(v) for i, v in enumerate(p) if v > 1e-12}
    print(f"cirq {cirq.__version__}: x(q0) -> index {idx} | bell 非零 = {bell}")
except Exception as e:
    print(f"cirq: ERR {type(e).__name__}: {e}")

# ---------------- pennylane ----------------
try:
    import pennylane as qml
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def circ():
        qml.PauliX(0)
        return qml.state()

    sv = np.asarray(circ())
    idx = int(np.argmax(np.abs(sv)))
    results["pennylane"] = (qml.__version__, idx)

    dev2 = qml.device("default.qubit", wires=2)

    @qml.qnode(dev2)
    def bell_circ():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        return qml.state()

    p = np.abs(np.asarray(bell_circ())) ** 2
    bell = {i: float(v) for i, v in enumerate(p) if v > 1e-12}
    print(f"pennylane {qml.__version__}: x(q0) -> index {idx} | bell 非零 = {bell}")
except Exception as e:
    print(f"pennylane: ERR {type(e).__name__}: {e}")

print()
print("=== 各自公約（q0 翻轉後的索引）===")
for k, (v, idx) in results.items():
    conv = "little-endian（q0=LSB）" if idx == 1 else ("big-endian（q0=MSB）" if idx == 2 ** (N - 1) else "??")
    print(f"  {k:10s} {v:10s} index={idx}  -> {conv}")
