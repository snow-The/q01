"""P4：PyTorch 後端 —— 精度對帳、autograd 對帳、裝置選擇（SPEC §5.4／§5.5）。

判準：
- L2 電路：對 CUDA-Q 黃金向量 max|ΔP| ≤ 1e-10（complex128）
- 梯度：autograd vs 參數平移，絕對誤差 ≤ 1e-9（SPEC §5.4）
- complex64 必須明示選用，且不得宣稱 1e-10
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

torch = pytest.importorskip("torch")
import q01 as cudaq  # noqa: E402
from q01.adapters import torch_backend  # noqa: E402

pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="尚無黃金向量")
DEVICES = torch_backend.devices()


@cudaq.kernel
def ansatz(theta: float, phi: float):
    q = cudaq.qvector(2)
    ry(theta, q[0])
    cx(q[0], q[1])
    rz(phi, q[1])


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _cases():
    import cases
    return cases.trace_cases()


@pytest.mark.parametrize("device", DEVICES)
def test_probabilities_match_cudaq_golden(device):
    be = torch_backend.TorchBackend(device, "complex128")
    gold = _golden()
    worst, worst_case = 0.0, None
    for cname, kernel, args in _cases():
        trace = kernel.trace(*args, live=False)
        p = be.probabilities(trace)
        g = np.asarray(gold["cases"][cname]["probabilities"], dtype=float)
        d = float(np.max(np.abs(p - g)))
        if d > worst:
            worst, worst_case = d, cname
    assert worst < 1e-10, "%s: 最差 max|dP| = %.3e（%s）" % (device, worst, worst_case)


def test_l0_convention_is_little_endian():
    from q01.circuit import Op, Trace
    be = torch_backend.TorchBackend("cpu", "complex128")
    tr = Trace(n_qubits=3, ops=[Op("x", (), (0,), ())])
    assert int(np.argmax(be.probabilities(tr))) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_autograd_matches_parameter_shift(device):
    x0 = [0.7, -0.3]

    def loss(params):
        return float(np.real(cudaq.observe(
            ansatz, cudaq.spin.z(0), float(params[0]), float(params[1])).expectation()))

    g_ps = np.asarray(cudaq.gradients.ParameterShift().compute(x0, loss, loss(x0)), dtype=float)
    value, g_torch = torch_backend.gradient_of_expectation(
        ansatz, cudaq.spin.z(0), x0, device=device)
    assert value == pytest.approx(loss(x0), abs=1e-12), "autograd 的損失值與 numpy 路徑不一致"
    dev = float(np.max(np.abs(g_ps - np.asarray(g_torch, dtype=float))))
    assert dev <= 1e-9, "%s: autograd 與參數平移差 %.3e" % (device, dev)


def test_complex64_is_documented_to_miss_the_criterion():
    be64 = torch_backend.TorchBackend("cpu", "complex64")
    be128 = torch_backend.TorchBackend("cpu", "complex128")
    worst = 0.0
    for cname, kernel, args in _cases():
        trace = kernel.trace(*args, live=False)
        worst = max(worst, float(np.max(np.abs(be64.probabilities(trace) - be128.probabilities(trace)))))
    assert worst > 1e-10, "complex64 竟然達到 1e-10？前提要重新檢查"


def test_cpu_always_available():
    assert "cpu" in DEVICES


def test_grad_scope_limit_is_explicit():
    """可微分路徑對 n 有明確上限（索引張量的記憶體），超限要報錯而不是默默算錯。"""
    from q01.circuit import Op, Trace
    be = torch_backend.TorchBackend("cpu", "complex128")
    n = torch_backend.MAX_GRAD_QUBITS + 1
    tr = Trace(n_qubits=n, ops=[Op("x", (), (0,), ())])
    with pytest.raises(ValueError):
        be.state_of(tr, grad=True)