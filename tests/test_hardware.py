"""`q01.hardware` 的離線測試 —— 不需要真機、不需要帳號、不會送出任何東西。

真機路徑本身無法在 CI 裡驗證（要排隊、要花錢），
但**最容易錯的兩個環節**可以在這裡釘住：
  1. 位元序反轉（Qiskit → q01）
  2. QASM → Qiskit 電路的轉換與量測補齊
"""

from __future__ import annotations

import pytest

from q01 import hardware


def test_import_does_not_need_hardware_deps():
    """沒有 qiskit-quantuminspire 也要能 import —— 學生端不裝它。"""
    assert hasattr(hardware, "submit")
    assert hasattr(hardware, "reverse_counts")


def test_reverse_counts_swaps_bit_order():
    # Qiskit："01" 的左邊是最高古典位元；q01：左邊是 qubit 0。兩者相反。
    assert hardware.reverse_counts({"01": 3}) == {"10": 3}
    assert hardware.reverse_counts({"10": 5}) == {"01": 5}


def test_reverse_counts_preserves_total_and_key_length():
    got = hardware.reverse_counts({"001": 7, "110": 4})
    assert sum(got.values()) == 11
    assert all(len(k) == 3 for k in got)


def test_reverse_counts_merges_after_reversal():
    # 反轉後若撞鍵必須相加，不能覆蓋
    got = hardware.reverse_counts({"01": 2, "10": 3})
    assert got == {"10": 2, "01": 3}


def test_qasm_to_circuit_keeps_measurement():
    qasm = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "measure q -> c;\n"
    )
    pytest.importorskip("qiskit")
    qc = hardware.qasm_to_circuit(qasm)
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2
    assert any(inst.operation.name == "measure" for inst in qc.data)


def test_qasm_without_measure_gets_one_added():
    qasm = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
    )
    pytest.importorskip("qiskit")
    qc = hardware.qasm_to_circuit(qasm)
    assert qc.num_clbits == 2
    assert any(inst.operation.name == "measure" for inst in qc.data)
