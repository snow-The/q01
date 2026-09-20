"""相容性測試：「默認零雲端，但可驗證相容」。

證明：把 q01 的電路轉譯成**硬體基底閘集**（不是我們的實作）之後，用該框架自己的
模擬器重跑，仍然對得上 CUDA-Q 黃金向量。這不是「我們說相容」，而是「轉譯後答案不變」。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
GOLDEN = ROOT / "tests" / "golden" / "cudaq_golden.json"

from q01 import adapters, bitorder  # noqa: E402

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="尚無黃金向量")
TOL = 1e-10


def _gold(name):
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return np.asarray(data["cases"][name]["probabilities"], dtype=float)


def _cases():
    import cases
    return cases.trace_cases()


def test_qiskit_survives_transpile_to_hardware_basis():
    pytest.importorskip("qiskit")
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit.providers.basic_provider import BasicSimulator  # noqa: F401  （證明本地 provider 存在）
    mod = adapters.available_adapters().get("qiskit")
    if mod is None:
        pytest.skip("qiskit adapter 不可用")
    basis = ["rz", "sx", "x", "cx"]        # IBM 風格硬體基底
    worst = 0.0
    for cname, kernel, args in _cases():
        trace = kernel.trace(*args, live=False)
        qc = mod.circuit_of(trace)
        tqc = transpile(qc, basis_gates=basis, optimization_level=1, seed_transpiler=0)
        p = np.abs(Statevector(tqc).data) ** 2      # Qiskit little-endian，零置換
        worst = max(worst, float(np.max(np.abs(p - _gold(cname)))))
    assert worst < TOL, "qiskit 轉譯後最差 %.3e" % worst


def test_cirq_survives_transpile_to_cz_gateset():
    cirq = pytest.importorskip("cirq")
    mod = adapters.available_adapters().get("cirq")
    if mod is None:
        pytest.skip("cirq adapter 不可用")
    perm = lambda n: np.array([bitorder.little_to_big(i, n) for i in range(1 << n)])
    worst = 0.0
    for cname, kernel, args in _cases():
        trace = kernel.trace(*args, live=False)
        circuit, qubits = mod.circuit_of(trace)
        tcircuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
        state = cirq.Simulator(dtype=np.complex128).simulate(
            tcircuit, qubit_order=qubits).final_state_vector
        p = (np.abs(np.asarray(state)) ** 2)[perm(trace.n_qubits)]
        worst = max(worst, float(np.max(np.abs(p - _gold(cname)))))
    assert worst < TOL, "cirq 轉譯後最差 %.3e" % worst


def test_default_installs_carry_no_cloud_provider():
    """預設安裝不綁雲：這兩個 provider 套件不應該被自動裝上。"""
    import importlib.util
    for mod in ("qiskit_ibm_runtime", "cirq_google"):
        if importlib.util.find_spec("qiskit") is None and mod.startswith("qiskit"):
            continue
        if importlib.util.find_spec("cirq") is None and mod.startswith("cirq"):
            continue
        assert importlib.util.find_spec(mod) is None, mod + " 被裝上了——預設安裝應該零雲端"