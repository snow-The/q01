"""相容性探針：證明「默認零雲端，但可驗證相容」。

兩件事：
A. **後端清單** —— 每個框架的預設安裝到底帶了什麼、有沒有綁雲。
B. **硬體基底轉譯相容性** —— 把我們的電路轉譯成該框架的硬體基底閘集，用他們的模擬器
   重跑，仍然要對得上 CUDA-Q 黃金向量。這是「可驗證相容」最硬的證據：
   不是「我們說相容」，而是「轉譯之後跑出來還是同一個答案」。

判準：max|ΔP| ≤ 1e-10。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
sys.path.insert(0, str(ROOT / "src"))

import cases  # noqa: E402
from q01 import adapters, bitorder  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "golden" / "cudaq_golden.json").read_text(encoding="utf-8"))


def _gold(name: str) -> np.ndarray:
    return np.asarray(GOLDEN["cases"][name]["probabilities"], dtype=float)


def _perm(n: int) -> np.ndarray:
    return np.array([bitorder.little_to_big(i, n) for i in range(1 << n)])


# ---------------------------------------------------------------- A. 清單
print("=" * 78)
print("A. 後端清單：默認安裝有沒有綁雲？（我們關心的只有『本地可跑』與『需不需要帳號』）")
print("=" * 78)

try:
    import qiskit
    from qiskit.providers import BackendV2  # noqa: F401
    basic = []
    try:
        from qiskit.providers.basic_provider import BasicSimulator
        basic.append("BasicSimulator（純 numpy，24 qubit 上限）")
    except Exception:
        pass
    ibm = None
    try:
        import qiskit_ibm_runtime  # noqa: F401
        ibm = "已安裝（需要 IBM Quantum 帳號）"
    except Exception:
        ibm = "未安裝 → 預設安裝不含任何雲端 provider"
    print(f"  qiskit {qiskit.__version__}: 本地 = {basic or ['Statevector/Operator（純 Python）']}；雲端 = {ibm}")
except Exception as exc:
    print("  qiskit: 未安裝（" + str(exc)[:40] + "）")

try:
    import cirq
    google = None
    try:
        import cirq_google  # noqa: F401
        google = "已安裝（需要 Google Cloud 帳號）"
    except Exception:
        google = "未安裝 → 預設安裝不含任何雲端 provider"
    print(f"  cirq {cirq.__version__}: 本地 = Simulator（dtype 可選 complex128）；雲端 = {google}")
except Exception as exc:
    print("  cirq: 未安裝（" + str(exc)[:40] + "）")

try:
    import pennylane as qml
    devs = [d for d in qml.devices.__all__ if "qubit" in d] if hasattr(qml, "devices") else []
    print(f"  pennylane {qml.__version__}: 本地 = default.qubit（純 Python）；"
          f"雲端 = 各家 plugin 另裝（預設不含）")
except Exception as exc:
    print("  pennylane: 未安裝（" + str(exc)[:40] + "）")

# ------------------------------------------------- B. 硬體基底轉譯相容性
print()
print("=" * 78)
print("B. 硬體基底轉譯相容性：轉譯成硬體閘集之後，答案還對得上嗎？")
print("=" * 78)

rows: dict[str, list[tuple[str, float]]] = {}

# ---- Qiskit：轉譯成 IBM 風格的硬體基底
try:
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    qk = adapters.available_adapters().get("qiskit")
    if qk is not None:
        basis = ["rz", "sx", "x", "cx"]
        out = []
        for cname, kernel, args in cases.trace_cases():
            trace = kernel.trace(*args, live=False)
            qc = qk.circuit_of(trace)
            tqc = transpile(qc, basis_gates=basis, optimization_level=1, seed_transpiler=0)
            p = np.abs(Statevector(tqc).data) ** 2      # Qiskit 是 little-endian，零置換
            d = float(np.max(np.abs(p - _gold(cname))))
            out.append((cname, d))
            extra = f"{qc.count_ops()} -> {tqc.count_ops()}"
            print(f"  qiskit  {cname:22s} max|ΔP| = {d:.3e}   [{extra}]")
        rows["qiskit(basis=rz,sx,x,cx)"] = out
except Exception as exc:
    print("  qiskit 轉譯檢查失敗：", type(exc).__name__, str(exc)[:100])

# ---- Cirq：轉譯成 CZ 硬體閘集
try:
    import cirq
    cr = adapters.available_adapters().get("cirq")
    if cr is not None:
        out = []
        gateset = cirq.CZTargetGateset(allow_partial_czs=False) if hasattr(cirq, "CZTargetGateset") else None
        if gateset is None and hasattr(cirq, "google"):
            gateset = cirq.google.SycamoreTargetGateset()
        for cname, kernel, args in cases.trace_cases():
            trace = kernel.trace(*args, live=False)
            circuit, qubits = cr.circuit_of(trace)
            if gateset is not None:
                tcircuit = cirq.optimize_for_target_gateset(circuit, gateset=gateset)
            else:
                tcircuit = circuit
            n = trace.n_qubits
            state = cirq.Simulator(dtype=np.complex128).simulate(
                tcircuit, qubit_order=qubits).final_state_vector
            p = (np.abs(np.asarray(state)) ** 2)[_perm(n)]
            d = float(np.max(np.abs(p - _gold(cname))))
            out.append((cname, d))
            try:
                before = len(list(circuit.all_operations()))
                after = len(list(tcircuit.all_operations()))
                extra = f"{before} -> {after} ops"
            except Exception:
                extra = ""
            print(f"  cirq    {cname:22s} max|ΔP| = {d:.3e}   [{extra}]")
        rows["cirq(CZTargetGateset)"] = out
except Exception as exc:
    print("  cirq 轉譯檢查失敗：", type(exc).__name__, str(exc)[:120])

print()
worst = max((d for out in rows.values() for _, d in out), default=float("nan"))
print(f"  ==== 轉譯後最差 max|ΔP| = {worst:.3e}（判準 1e-10）====")
