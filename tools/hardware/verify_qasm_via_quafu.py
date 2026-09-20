"""用 quafu 自己的模擬器驗證我們匯出的 QASM（本地、免費、不耗真機額度）。"""
import inspect
import json
import pathlib
import re
import sys

import numpy as np

from quafu import QuantumCircuit, simulate

print("== API 簽章 ==")
try:
    print("  QuantumCircuit.__init__:", inspect.signature(QuantumCircuit.__init__))
except Exception as e:
    print("  init sig 讀不到:", e)
for nm in ("from_openqasm", "from_openqasm_str"):
    f = getattr(QuantumCircuit, nm, None)
    if f is not None:
        try:
            print(f"  {nm}:", inspect.signature(f))
        except Exception as e:
            print(f"  {nm}: sig?", e)
try:
    print("  simulate:", inspect.signature(simulate))
except Exception as e:
    print("  simulate sig 讀不到:", e)

QASM = pathlib.Path(__file__).resolve().parents[2] / "tools/hardware/qasm"
GOLD = json.loads(pathlib.Path(__file__).resolve().parents[2] / "tests/golden/cudaq_golden.json".read_text(encoding="utf-8"))

print()
print("== 逐案例：QASM -> quafu -> 模擬 -> 比對黃金向量 ==")
worst = 0.0
for name in ("bell", "qbn5_book", "qbn5_depth2", "entangle_then_rotate"):
    path = QASM / f"{name}.qasm"
    if not path.exists():
        print(f"  {name:22s} 沒有 QASM 檔")
        continue
    text = path.read_text(encoding="utf-8")
    m_n = re.search(r"qreg q\[(\d+)\]", text)
    if not m_n:
        print(f"  {name:22s} QASM 裡找不到 qreg")
        continue
    try:
        qc = QuantumCircuit(int(m_n.group(1)))   # from_openqasm 是實例方法
        qc.from_openqasm(text)
    except Exception as exc:
        print(f"  {name:22s} from_openqasm 失敗：{type(exc).__name__}: {str(exc)[:90]}")
        continue
    n = qc.num
    try:
        res = simulate(qc, shots=0, simulator="statevector")
    except Exception as exc:
        print(f"  {name:22s} simulate 失敗：{type(exc).__name__}: {str(exc)[:90]}")
        continue
    # 盡量從 SimuResult 取出狀態或機率
    st = getattr(res, "state", None) or getattr(res, "amplitudes", None)
    probs = None
    if st is not None:
        st = np.asarray(st)
        if st.ndim == 1 and st.size == 1 << n:
            probs = np.abs(st) ** 2
    if probs is None:
        for attr in ("probabilities", "probs"):
            v = getattr(res, attr, None)
            if v is not None:
                probs = np.asarray(v, dtype=float)
                break
    if probs is None:
        print(f"  {name:22s} 拿到 {type(res).__name__}，但取不到狀態/機率；屬性：{ [a for a in dir(res) if not a.startswith(chr(95))][:12] }")
        continue
    g = np.asarray(GOLD["cases"][name]["probabilities"], dtype=float)
    if probs.size != g.size:
        print(f"  {name:22s} 大小不符：{probs.size} vs {g.size}（可能是位元順序或量測差異）")
        continue
    d = float(np.max(np.abs(probs - g)))
    # L0 公約判定：quafu 的狀態索引是不是 big-endian（q0 = MSB）？
    perm = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(1 << n)])
    d_rev = float(np.max(np.abs(probs[perm] - g)))
    worst = min(max(worst, d), max(worst, d_rev))
    tag = "little-endian（與 CUDA-Q 同）" if d < d_rev else "big-endian（需反轉）"
    print(f"  {name:22s} quafu {len(probs)} 維  直接 max|dP|={d:.3e}  反轉後={d_rev:.3e}  -> {tag}")
print()
print(f"  ==== quafu 端最差 max|dP| = {worst:.3e}（判準 1e-10）====")