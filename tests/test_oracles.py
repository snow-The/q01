"""跨框架對帳：Qiskit / Cirq / PennyLane 的**本地**模擬器 vs CUDA-Q 黃金向量。

這是 P3 的核心驗收（SPEC §7.1 L2）：每個 adapter 都是**獨立的第三方實作**，
不是我們的程式碼換一層皮。沒安裝對應套件時自動跳過。
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

from q01 import adapters  # noqa: E402

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="尚無黃金向量")

ADAPTER_NAMES = ("qiskit", "cirq", "pennylane")


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _adapter(name: str):
    av = adapters.available_adapters()
    if name not in av:
        pytest.skip(name + " 未安裝（選配 extra，不影響 q01 本體）")
    return av[name]


def test_trace_cases_cover_all_golden_cases():
    import cases
    names = {n for n, _, _ in cases.trace_cases()}
    assert names == set(_golden()["cases"]), "cases.py 與黃金向量的案例必須一致"


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_adapter_probabilities_match_cudaq(name):
    import cases
    mod = _adapter(name)
    gold = _golden()
    worst_case, worst = None, 0.0
    for cname, kernel, args in cases.trace_cases():
        trace = kernel.trace(*args, live=False)
        p = np.asarray(mod.probabilities(trace), dtype=float)
        g = np.asarray(gold["cases"][cname]["probabilities"], dtype=float)
        assert p.shape == g.shape, name + "/" + cname + ": 形狀不符"
        d = float(np.max(np.abs(p - g)))
        if d > worst:
            worst_case, worst = cname, d
    assert worst < 1e-10, name + ": 最差 max|dP| = %.3e（%s）" % (worst, worst_case)


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_adapter_l0_bit_order(name):
    """L0 公約：對外一律 little-endian（與 CUDA-Q 的 get_state 相同）。

    對 qubit 0 施加 X 之後，非零振幅必須落在索引 1（不是 2^(n-1)）。
    """
    mod = _adapter(name)
    from q01.circuit import Op, Trace
    tr = Trace(n_qubits=3, ops=[Op("x", (), (0,), ())])
    p = np.asarray(mod.probabilities(tr), dtype=float)
    assert int(np.argmax(p)) == 1, name + " 的對外順序不是 little-endian"


def test_record_available_adapters():
    av = adapters.available_adapters()
    assert isinstance(av, dict)