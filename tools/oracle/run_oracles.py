"""對帳 runner：同一顆電路，跑 q01 與每個可用的 adapter，全部對 CUDA-Q 黃金向量比對。

判準（SPEC §5.1）：`max|ΔP| ≤ 1e-10`。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
sys.path.insert(0, str(ROOT / "src"))

import cases  # noqa: E402
from q01 import adapters  # noqa: E402

TOL = 1e-10
GOLDEN = ROOT / "tests" / "golden" / "cudaq_golden.json"


def main() -> int:
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    ad = adapters.available_adapters()
    names = list(ad)
    print(f"黃金向量：{gold['backend']} {gold['version']}（numpy {gold['numpy']}）")
    print(f"可用的 adapter：{', '.join(f'{n} {ad[n].version}' for n in names) if names else '（無）'}")
    print()
    header = f"  {'case':24s} {'dim':>4s} {'q01':>11s} " + " ".join(f"{n:>11s}" for n in names)
    print(header)
    print("  " + "-" * (len(header) - 2))

    worst = 0.0
    failures = []
    for cname, kernel, args in cases.trace_cases():
        trace = kernel.trace(*args, live=False)
        g = np.asarray(gold["cases"][cname]["probabilities"], dtype=float)
        row = []
        # q01 自己的核心：**走公開 API**（它會把內部的 big-endian 轉成對外的 little-endian）。
        # 手寫重放會拿到 big-endian 而與黃金向量對不上——這個坑踩過一次。
        import q01
        p_q01 = np.abs(np.asarray(q01.get_state(kernel, *args))) ** 2
        d = float(np.max(np.abs(p_q01 - g)))
        row.append(d); worst = max(worst, d)
        if d > TOL:
            failures.append((cname, "q01", d))
        for n in names:
            try:
                p = np.asarray(ad[n].probabilities(trace), dtype=float)
                dd = float(np.max(np.abs(p - g)))
            except Exception as exc:             # noqa: BLE001
                dd = float("nan")
                failures.append((cname, f"{n} EXC", str(exc)[:60]))
            row.append(dd)
            if dd == dd:
                worst = max(worst, dd)
                if dd > TOL:
                    failures.append((cname, n, dd))
        print(f"  {cname:24s} {len(g):4d} {row[0]:11.3e} " + " ".join(f"{v:11.3e}" for v in row[1:]))

    print()
    print(f"  ==== 最差 max|ΔP| = {worst:.3e}（判準 {TOL:g}）====")
    if failures:
        print("  失敗／例外：")
        for f in failures:
            print("   -", f)
        return 1
    print("  全部通過：獨立實作數 =", 1 + len(names), "（q01 核心", "＋ " + "、".join(names) if names else "", "）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
