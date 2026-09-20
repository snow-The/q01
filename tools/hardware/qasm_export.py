"""IR → OpenQASM 2.0（真機路線的交換格式）。

為什麼用 QASM 當交換格式：
  1. q01 端**不必安裝** pyquafu（它釘 numpy<2，與 q01 衝突）。
  2. QASM 是標準，可以用**第三方的 parser 獨立驗證**我們匯出的東西對不對
     （qiskit.qasm2.loads 就是我們的驗證器）——先確定「QASM 寫對了」，再談「真機收不收」。

閘集範圍（刻意與硬體實驗所需一致）：
  h x y z s sdg t tdg sx rx ry rz、單控制 X（＝cx）、雙控制 X（＝ccx）。
  更多控制位不匯出（QASM 2.0 沒有對應閘），會明確報錯——不猜、不近似。
"""

from __future__ import annotations

import math

__all__ = ["to_qasm", "QasmExportError"]


class QasmExportError(ValueError):
    """IR 超出 QASM 2.0 可表達的閘集。"""


_FIXED = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"}
_PARAM = {"rx", "ry", "rz"}


def _fmt(x: float) -> str:
    """把角度寫成 QASM 可接受的十進位（保留 17 位有效數字，等於雙精度完整精度）。"""
    if x == 0:
        return "0"
    return repr(float(x))


def to_qasm(trace, *, measure: bool = True, creg_name: str = "c") -> str:
    """把 q01 的 IR（Trace）轉成 OpenQASM 2.0 字串。

    `measure=True` 會在最後加上 `measure q[i] -> c[i];`（全部 qubit，順序由 0 起）。
    """
    n = int(trace.n_qubits)
    if n <= 0:
        raise QasmExportError("trace 沒有配置任何 qubit")
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg q[{n}];"]
    if measure:
        lines.append(f"creg {creg_name}[{n}];")
    body: list[str] = []
    for op in trace.ops:
        name, params, qubits, controls = op.name, op.params, op.qubits, tuple(op.controls)
        t = int(qubits[0])
        if name in _FIXED:
            if controls:
                # 受控的固定閘：QASM 2.0 只直接支援受控 X（cx/ccx）
                if name == "x" and len(controls) == 1:
                    body.append(f"cx q[{controls[0]}],q[{t}];")
                    continue
                if name == "x" and len(controls) == 2:
                    body.append(f"ccx q[{controls[0]}],q[{controls[1]}],q[{t}];")
                    continue
                raise QasmExportError(
                    f"QASM 2.0 沒有「受控 {name}（{len(controls)} 個控制位）」的標準閘；"
                    f"請在電路層分解，或改用真機 SDK 的原生閘。"
                )
            body.append(f"{name} q[{t}];")
        elif name in _PARAM:
            theta = float(params[0])
            if controls:
                raise QasmExportError(
                    f"QASM 2.0 沒有「受控 {name}」的標準閘（{len(controls)} 個控制位）；"
                    f"多控制旋轉請先分解成 cx + 單閘。"
                )
            body.append(f"{name}({_fmt(theta)}) q[{t}];")
        else:
            raise QasmExportError(f"IR 出現未知閘 {name!r}")
    lines += body
    if measure:
        lines += [f"measure q[{i}] -> {creg_name}[{i}];" for i in range(n)]
    return "\n".join(lines) + "\n"


def counts_to_probs(counts: dict, n: int) -> dict:
    """把 counts（鍵是位元字串）正規化成機率；**不改鍵的順序**（呼叫端負責公約）。"""
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _cli() -> int:
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tools" / "oracle"))
    sys.path.insert(0, str(root / "src"))
    import cases  # noqa: E402

    out = root / "tools" / "hardware" / "qasm"
    out.mkdir(parents=True, exist_ok=True)
    for cname, kernel, args in cases.trace_cases():
        trace = kernel.trace(*args, live=False)
        try:
            text = to_qasm(trace)
        except QasmExportError as exc:
            print(f"  {cname:24s} SKIP（{exc}）")
            continue
        (out / f"{cname}.qasm").write_text(text, encoding="utf-8")
        print(f"  {cname:24s} {trace.n_qubits} qubit, {len(trace.ops)} op → qasm/{cname}.qasm")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
