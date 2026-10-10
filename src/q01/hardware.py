"""真機後端 —— 把 q01 的電路統一到 Quantum Inspire 上執行。

定位：q01 是**統一前端**。同一顆電路先用 q01 的模擬器驗證，
再原封不動送到真機；兩邊回傳**同一個型別**（`SampleResult`），所以可以直接對帳。

這個模組負責後半段：

    q01 kernel ──trace──> Qiskit 電路 ──transpile──> QI 後端 ──> counts
    或者 OpenQASM 2.0 ──qasm2.loads──> Qiskit 電路 ──（同上）

---

## 位元序：這一層最容易錯的一步

兩個公約不同：

===============  ==========================================
來源             字串的意義
===============  ==========================================
q01 `sample()`   第 i 個字元 = **qubit i**（q[0] 在最左）
Qiskit counts    最左邊的字元 = **最高**的古典位元
===============  ==========================================

也就是說兩者**相反**。本模組在放進 `SampleResult` 之前把 Qiskit 的字串反轉，
所以 `q01.sample(...)` 與 `q01.hardware.submit(...)` 的鍵可以直接比較。

（反轉只在本模組發生；`bitorder` 仍是振幅陣列唯一允許置換的地方。）

---

## 為什麼是 `job.result()` 而不是 `job.result(wait_for_results=False)`

2026-09-21 的實測：QI 的 `JobStatus.DONE` 只代表**任務**結束，
結果回傳管線尚未就緒；此時 `result(wait_for_results=False)` 拿到空結果並丟出
`QiskitError: Result failed , Experiment failed. ... No Results`。
那個 bug 讓 9 次硬體執行安靜地被跳過。本模組一律讓 `result()` 自己等，
並且**不把 `No Results` 當成可跳過的錯誤**。
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Sequence

import numpy as np

from .sampler import SampleResult

__all__ = [
    "DEFAULT_BACKEND", "HardwareUnavailable", "available_backends", "backend_status",
    "qasm_to_circuit", "trace_to_circuit", "submit_qasm", "submit", "reverse_counts",
]

DEFAULT_BACKEND = "Tuna-17"


class HardwareUnavailable(RuntimeError):
    """真機後端不可用（缺依賴、未登入、後端離線）。訊息一律附上怎麼修。"""


def _provider():
    try:
        from qiskit_quantuminspire.qi_provider import QIProvider
    except Exception as exc:                       # noqa: BLE001
        raise HardwareUnavailable(
            "找不到 qiskit-quantuminspire。真機功能需要它（模擬器不需要）："
            "  uv add qiskit-quantuminspire quantuminspire\n"
            "  並先完成裝置授權：qi login"
        ) from exc
    try:
        return QIProvider()
    except Exception as exc:                       # noqa: BLE001
        raise HardwareUnavailable(
            "Quantum Inspire 認證失敗。請先執行 `qi login` 完成裝置授權；"
            "憑證會放在 ~/.quantuminspire/config.json。"
            f"（原始錯誤：{type(exc).__name__}: {exc}）"
        ) from exc


def available_backends() -> list[str]:
    """列出這個帳號看得到的 QI 後端名稱（含模擬器與真機）。"""
    return [b.name for b in _provider().backends()]


def backend_status(name: str = DEFAULT_BACKEND) -> str:
    """回傳後端狀態字串（例如 `online`／`offline`／`calibrating`）。"""
    b = _provider().get_backend(name)
    return str(getattr(b, "status", "unknown"))


def reverse_counts(counts: dict) -> dict:
    """Qiskit 的 counts → q01 的 counts（把字串反轉）。

    Qiskit：最左邊是最高古典位元；q01：最左邊是 qubit 0。
    實際的反轉動作委派給 `bitorder.reverse_key` ——
    全專案只有那個模組可以做位元反轉（`test_core.py` 有守衛測試）。
    """
    from . import bitorder

    out: dict[str, int] = {}
    for key, n in counts.items():
        k = bitorder.reverse_key(str(key).replace(" ", ""))
        out[k] = out.get(k, 0) + int(n)
    return out


def qasm_to_circuit(qasm: str):
    """OpenQASM 2.0 字串 → Qiskit 電路（已含量測，直接可送）。"""
    from qiskit import qasm2
    from qiskit.circuit import ClassicalRegister

    qc = qasm2.loads(qasm)
    if qc.num_clbits == 0 and qc.num_qubits > 0:
        # 沒有 creg 的 QASM（例如只描述么正部分的檔案）要先補一個古典暫存器，
        # 否則 measure() 會因為「沒有 clbit 可放」而失敗。
        qc.add_register(ClassicalRegister(qc.num_qubits))
        qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    return qc


def trace_to_circuit(trace) -> "object":
    """q01 的 trace → 含量測的 Qiskit 電路。

    量測對映刻意選成 `measure(Q[i], i)`：
    這樣 Qiskit 的字串反轉之後，第 i 個字元正好是 qubit `Q[i]`，與 q01 同構。
    """
    from .adapters import qiskit_emit
    from .circuit import Trace

    if not qiskit_emit.available():
        raise HardwareUnavailable("qiskit 不可用；真機功能需要 pip 安裝 qiskit。")
    qc = qiskit_emit.circuit_of(trace)
    qubits = trace.measures[-1] if trace.measures else tuple(range(trace.n_qubits))
    qc.measure(list(qubits), list(range(len(qubits))))
    return qc


def _submit_circuits(circuits: Sequence, *, backend: str, shots_count: int,
                     timeout_s: float, on_status: Callable[[str, float], None] | None):
    """送出一批電路並取回 counts（list[dict]），內部共用送出／等待／取結果。"""
    from qiskit import transpile

    b = _provider().get_backend(backend)
    tqcs = [transpile(qc, b, optimization_level=1, seed_transpiler=0) for qc in circuits]
    job = b.run(tqcs, shots=shots_count)

    t0, last = time.time(), None
    while time.time() - t0 < timeout_s:
        st = str(job.status())
        if st != last:
            if on_status is not None:
                on_status(st, time.time() - t0)
            last = st
        if "DONE" in st or "ERROR" in st or "CANCELLED" in st:
            break
        time.sleep(5)

    # 一定要讓 result() 自己等；wait_for_results=False 會拿到空的結果。
    got = job.result().get_counts()
    return got if isinstance(got, list) else [got]


def submit_qasm(qasm: str, *, backend: str = DEFAULT_BACKEND, shots_count: int = 1024,
                timeout_s: float = 2400.0,
                on_status: Callable[[str, float], None] | None = None) -> SampleResult:
    """送一段 OpenQASM 2.0 到真機，回傳與 `q01.sample()` 同型的結果。"""
    qc = qasm_to_circuit(qasm)
    got = _submit_circuits([qc], backend=backend, shots_count=shots_count,
                           timeout_s=timeout_s, on_status=on_status)
    counts = reverse_counts(got[0])
    return SampleResult(counts, shots_count, len(next(iter(counts), "")))


def submit(kernel, *args, backend: str = DEFAULT_BACKEND, shots_count: int = 1024,
           timeout_s: float = 2400.0, seed: int | None = None,
           on_status: Callable[[str, float], None] | None = None, **kwargs) -> SampleResult:
    """把一個 q01 kernel 送到真機，回傳與 `q01.sample()` 同型的結果。

    用法：

        sim  = q01.sample(kernel, x, shots_count=1024)          # 模擬器驗證
        hw   = q01.hardware.submit(kernel, x, shots_count=1024) # 同一顆電路送真機
        # 兩者的 .counts 可以直接比較（同一個位元序公約）
    """
    from .sampler import _trace_once

    tr = _trace_once(kernel, args, kwargs, seed)
    if tr.mid_circuit or len(tr.measures) > 1:
        raise HardwareUnavailable(
            "真機後端不支援含『線路中量測』的 kernel（QI 的 QASM 2.0 沒有這個概念）。"
            "請改用終端量測的 kernel。"
        )
    qc = trace_to_circuit(tr)
    got = _submit_circuits([qc], backend=backend, shots_count=shots_count,
                           timeout_s=timeout_s, on_status=on_status)
    counts = reverse_counts(got[0])
    n = len(tr.measures[-1]) if tr.measures else tr.n_qubits
    return SampleResult(counts, shots_count, n)
