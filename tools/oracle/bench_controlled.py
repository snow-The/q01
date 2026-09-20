"""受控閘路徑的規模基準（SPEC §6.7 的修正驗證）。"""
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from q01.circuit import Op  # noqa: E402
from q01.simulator import Simulator  # noqa: E402


def ops_for(n, depth=2):
    ops = [Op("ry", (0.3 + 0.1 * i,), (i,), ()) for i in range(n)]
    for d in range(depth):
        ops += [Op("rz", (0.2 + 0.05 * i,), (i,), ()) for i in range(n)]
        step = 1 if d == 0 else 2
        ops += [Op("x", (), ((i + step) % n,), (i,)) for i in range(n)]
    return ops


def run(n):
    sim = Simulator(n)
    for op in ops_for(n):
        sim.apply(op.name, op.params, op.qubits, op.controls)
    return sim.probabilities()


def best(n, reps=5):
    run(n)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); run(n); ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


print("   n        ms     相對上一列   振幅倍數")
prev = None
for n in (5, 10, 12, 14, 16, 18, 20):
    t = best(n)
    g = "-" if prev is None else "x%5.2f" % (t / prev)
    print("  %3d %10.3f      %8s       x4" % (n, t, g))
    prev = t