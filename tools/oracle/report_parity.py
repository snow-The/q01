"""把 q01 與 CUDA-Q 黃金向量的差異印出來（報告／論文用）。"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "oracle"))
sys.path.insert(0, str(ROOT / "src"))

import cases  # noqa: E402

mine = cases.collect()
gold = json.loads((ROOT / "tests" / "golden" / "cudaq_golden.json").read_text(encoding="utf-8"))
print(f"golden: {gold['backend']} {gold['version']} | numpy {gold['numpy']} | python {gold['python']}")
print(f"mine  : {mine['backend']} {mine['version']} | numpy {mine['numpy']} | python {mine['python']}")
print()
worst = 0.0
for name in sorted(gold["cases"]):
    g = np.asarray(gold["cases"][name]["probabilities"], dtype=float)
    m = np.asarray(mine["cases"][name]["probabilities"], dtype=float)
    d = float(np.max(np.abs(g - m)))
    worst = max(worst, d)
    print(f"  {name:24s} dim={len(g):3d}   max|dP| = {d:.3e}")
gb, mb = gold["cases"]["qbn5_book"], mine["cases"]["qbn5_book"]
for key in ("expectation_z0", "expectation_z0z1"):
    if key in gb and key in mb:
        d = abs(gb[key] - mb[key]); worst = max(worst, d)
        print(f"  {key:24s}          |d|   = {d:.3e}   (CUDA-Q {gb[key]:+.15f} / q01 {mb[key]:+.15f})")
print()
print(f"  ==== 最差 = {worst:.3e}（判準 1e-10，SPEC §5.1）====")
