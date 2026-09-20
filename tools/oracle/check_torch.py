"""P4 驗收（獨立執行，不需要 pytest）：在裝了 torch 的環境跑。"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
sys.path.insert(0, str(ROOT / "src"))

import q01 as cudaq  # noqa: E402
from q01.adapters import torch_backend as tb  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "golden" / "cudaq_golden.json").read_text(encoding="utf-8"))
cases = __import__("cases")


@cudaq.kernel
def ansatz(theta: float, phi: float):
    q = cudaq.qvector(2)
    ry(theta, q[0])
    cx(q[0], q[1])
    rz(phi, q[1])


def main():
    print("torch", tb.version, "| devices:", tb.devices())
    ok = True
    for device in tb.devices():
        be = tb.TorchBackend(device, "complex128")
        worst = 0.0
        for cname, kernel, args in cases.trace_cases():
            tr = kernel.trace(*args, live=False)
            p = be.probabilities(tr)
            g = np.asarray(GOLDEN["cases"][cname]["probabilities"], dtype=float)
            worst = max(worst, float(np.max(np.abs(p - g))))
        good = worst < 1e-10
        ok = ok and good
        print("  L2 電路   %-5s 最差 max|dP| = %.3e  %s" % (device, worst, "PASS" if good else "FAIL"))

    x0 = [0.7, -0.3]

    def loss(params):
        return float(np.real(cudaq.observe(ansatz, cudaq.spin.z(0),
                                          float(params[0]), float(params[1])).expectation()))

    g_ps = np.asarray(cudaq.gradients.ParameterShift().compute(x0, loss, loss(x0)), dtype=float)
    for device in tb.devices():
        val, g_t = tb.gradient_of_expectation(ansatz, cudaq.spin.z(0), x0, device=device)
        dev = float(np.max(np.abs(g_ps - np.asarray(g_t, dtype=float))))
        good = dev <= 1e-9 and abs(val - loss(x0)) < 1e-12
        ok = ok and good
        print("  autograd  %-5s loss=%.12f  梯度差=%.3e  %s" % (device, val, dev, "PASS" if good else "FAIL"))
        print("            參數平移 =", np.round(g_ps, 12).tolist())
        print("            autograd =", np.round(np.asarray(g_t, dtype=float), 12).tolist())

    print("  ==>", "全部通過" if ok else "★ 有失敗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())