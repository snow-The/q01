"""驗證器：IR → QASM → Qiskit 解析 → 機率 → 比對 CUDA-Q 黃金向量。

這一步刻意**不碰 quafu**：先把「QASM 寫對了沒」與「真機 SDK 收不收」分開。
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "hardware"))

import cases  # noqa: E402
from qasm_export import QasmExportError, to_qasm  # noqa: E402
from qiskit import qasm2  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

GOLD = json.loads((ROOT / "tests" / "golden" / "cudaq_golden.json").read_text(encoding="utf-8"))

print(f"Qiskit {__import__('qiskit').__version__} 解析我們匯出的 QASM，再與黃金向量比對")
worst, rows = 0.0, []
for cname, kernel, args in cases.trace_cases():
    trace = kernel.trace(*args, live=False)
    try:
        text = to_qasm(trace, measure=False)  # 比機率不需要 measure
    except QasmExportError as exc:
        print(f"  {cname:24s} SKIP（{str(exc)[:60]}…）")
        continue
    qc = qasm2.loads(text)
    p = np.abs(Statevector(qc).data) ** 2      # Qiskit little-endian（與 CUDA-Q 同序）
    g = np.asarray(GOLD["cases"][cname]["probabilities"], dtype=float)
    d = float(np.max(np.abs(p - g)))
    worst = max(worst, d)
    rows.append((cname, len(text.splitlines()), d))
    print(f"  {cname:24s} qasm {len(text.splitlines()):3d} 行  max|dP| = {d:.3e}")
print()
print(f"  ==== QASM 往返最差 max|dP| = {worst:.3e}（判準 1e-10）====")
print("  ==>", "QASM 匯出正確" if worst < 1e-10 else "★ 有問題")