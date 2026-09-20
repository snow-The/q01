"""對帳測試：q01 對 CUDA-Q 的黃金向量（SPEC §7）。

黃金向量由參考機（裝了真 CUDA-Q）產生：
    python tools/oracle/cases.py tests/golden/cudaq_golden.json
沒有黃金檔時整個檔案跳過——學生端不需要 CUDA-Q。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "cudaq_golden.json"
sys.path.insert(0, str(ROOT / "tools" / "oracle"))

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="尚無黃金向量（需在參考機產生）")

CASE_NAMES = ["bell", "mcry3", "qbn5_book", "qbn5_depth2", "entangle_then_rotate"]


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mine() -> dict:
    import cases  # noqa: E402  （tools/oracle/cases.py）
    assert cases.BACKEND == "q01", "本測試必須在 q01 軌上跑"
    return cases.collect()


@pytest.mark.parametrize("name", CASE_NAMES)
def test_probabilities_match_cudaq(golden, mine, name):
    g = golden["cases"][name]["probabilities"]
    m = mine["cases"][name]["probabilities"]
    assert len(g) == len(m), f"{name}: 長度 {len(g)} vs {len(m)}"
    diff = float(np.max(np.abs(np.asarray(g) - np.asarray(m))))
    assert diff < 1e-10, f"{name}: max|ΔP| = {diff:.3e}（CUDA-Q {golden['version']} vs q01 {mine['version']}）"


def test_expectations_match_cudaq(golden, mine):
    g = golden["cases"]["qbn5_book"]
    m = mine["cases"]["qbn5_book"]
    for key in ("expectation_z0", "expectation_z0z1"):
        if key in g and key in m:
            assert abs(g[key] - m[key]) < 1e-10, f"{key}: {g[key]} vs {m[key]}"


def test_golden_records_provenance(golden):
    assert golden["backend"] == "cudaq"
    for key in ("version", "numpy", "platform"):
        assert key in golden
