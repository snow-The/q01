"""跨軌共用的電路定義與資料收集。

**同一個檔案在兩軌都能執行**（這就是驗收條件 G1）：

- 在裝了真 CUDA-Q 的機器上：`import cudaq` 成功 → 用 CUDA-Q 產生黃金向量。
- 在學生的 Windows 筆電上：只有 q01 → 產生同一份格式的結果，拿來與黃金向量比對。

用法：
    python tools/oracle/cases.py <輸出.json>

輸出格式（SPEC §7.2）：每個 case 含 `probabilities`（**little-endian**，與 CUDA-Q 的
get_state 同序——所以兩軌可以直接逐元素比，不需要任何置換）。
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys

import numpy as np

_FORCE_LOCAL = __import__("os").environ.get("Q01_FORCE_LOCAL") == "1"

try:
    if _FORCE_LOCAL:                              # 強制走 q01 軌（例如測 torch 後端時）
        raise ImportError("Q01_FORCE_LOCAL=1")
    import cudaq                                  # 正式軌（裝了真 CUDA-Q）
    BACKEND = "cudaq"
    _VERSION = cudaq.__version__.split("(")[0].strip()
except ImportError:                               # 練習軌（只有 q01）
    try:
        import q01
    except ImportError:                           # 從原始碼目錄直接執行時
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
        import q01
    cudaq = q01
    BACKEND = "q01"
    _VERSION = q01.__version__

cudaq.set_target("qpp-cpu")


# --------------------------------------------------------------------------
# 電路（全部取自本書，寫法與書中一致）
# --------------------------------------------------------------------------
@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])


@cudaq.kernel
def mcry3(theta: float):
    q = cudaq.qvector(4)
    x(q[0])
    x(q[1])
    x(q[2])
    ry.ctrl(theta, [q[0], q[1], q[2]], q[3])


@cudaq.kernel
def qbn5_book(x: list[float], params: list[float]):
    """〈QBN 形式定義〉的 5 qubit 輸出層：編碼 -> 環形 CX -> RY/RZ 可訓練層。"""
    q = cudaq.qvector(5)
    for i in range(5):
        ry(x[i], q[i])
    for i in range(5):
        cx(q[i], q[(i + 1) % 5])
    for i in range(5):
        ry(params[i], q[i])
    for i in range(5):
        rz(params[5 + i], q[i])


@cudaq.kernel
def qbn5_depth2(x: list[float], w: list[float]):
    """qbn5_encoding.qbn_layer(depth=2, entanglement='ring') 的同一顆電路。"""
    q = cudaq.qvector(5)
    for i in range(5):
        ry(x[i], q[i])
    for i in range(5):
        ry(w[2 * i], q[i])
        rz(w[2 * i + 1], q[i])
    for i in range(5):
        cx(q[i], q[(i + 1) % 5])
    for i in range(5):
        ry(w[10 + 2 * i], q[i])
        rz(w[10 + 2 * i + 1], q[i])
    for i in range(5):
        cx(q[i], q[(i + 2) % 5])


@cudaq.kernel
def entangle_then_rotate(theta: float):
    """去相位消融會用到的最小形狀：糾纏之後還有 RY（含 RY 的層之前去相位才可觀測）。"""
    q = cudaq.qvector(2)
    ry(theta, q[0])
    cx(q[0], q[1])
    ry(theta, q[1])


def trace_cases():
    """(name, kernel, args) —— 給 adapter 對帳用；只取 IR，不跑模擬。"""
    return [
        ("bell", bell, ()),
        ("mcry3", mcry3, (0.7,)),
        ("qbn5_book", qbn5_book, (list(X_BOOK), list(P_BOOK))),
        ("qbn5_depth2", qbn5_depth2, (list(X_D2), list(W_D2))),
        ("entangle_then_rotate", entangle_then_rotate, (0.83,)),
    ]


def _probs(kernel, *args, **kwargs) -> list[float]:
    st = np.array(cudaq.get_state(kernel, *args, **kwargs))
    return [float(v) for v in np.abs(st) ** 2]


X_BOOK = [0.3, 0.4, 0.5, 0.6, 0.7]
P_BOOK = [0.1, -0.2, 0.3, -0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
X_D2 = [0.9, 0.2, 0.75, 0.4, 0.6]
W_D2 = [0.31, -0.12, 0.44, 0.05, -0.9, 0.27, 0.13, -0.55, 0.62, -0.08,
        0.19, 0.71, -0.33, 0.24, 0.48, -0.66, 0.07, 0.52, -0.21, 0.38]


def collect() -> dict:
    cases: dict[str, dict] = {}

    cases["bell"] = {"qubits": 2, "probabilities": _probs(bell)}
    cases["mcry3"] = {"qubits": 4, "theta": 0.7, "probabilities": _probs(mcry3, 0.7)}
    cases["qbn5_book"] = {"qubits": 5, "x": X_BOOK, "params": P_BOOK,
                          "probabilities": _probs(qbn5_book, list(X_BOOK), list(P_BOOK))}
    cases["qbn5_depth2"] = {"qubits": 5, "x": X_D2, "weights": W_D2,
                            "probabilities": _probs(qbn5_depth2, list(X_D2), list(W_D2))}
    cases["entangle_then_rotate"] = {"qubits": 2, "theta": 0.83,
                                     "probabilities": _probs(entangle_then_rotate, 0.83)}

    try:
        obs = cudaq.observe(qbn5_book, cudaq.spin.z(0), list(X_BOOK), list(P_BOOK))
        cases["qbn5_book"]["expectation_z0"] = complex(obs.expectation()).real
        zz = cudaq.observe(qbn5_book, cudaq.spin.z(0) * cudaq.spin.z(1), list(X_BOOK), list(P_BOOK))
        cases["qbn5_book"]["expectation_z0z1"] = complex(zz.expectation()).real
    except Exception as exc:
        cases["qbn5_book"]["expectation_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "backend": BACKEND,
        "version": _VERSION,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "bit_order": "get_state little-endian（與 CUDA-Q 相同）",
        "cases": cases,
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "golden.json"
    payload = collect()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[{BACKEND} {_VERSION}] 已寫出 {out}；cases = {sorted(payload['cases'])}")
