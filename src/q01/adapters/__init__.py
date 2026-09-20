"""adapter 層：把 IR 發射成別的框架的電路，讓**他們自己的**本地模擬器算一遍。

定位（SPEC §8.3／§8.4 與 ARCH-REFACTOR §4）：adapter **不是** CUDA-Q 的替代品，
而是**獨立的驗證者**——「同一顆電路在 N 個框架上得到同一個答案」才是我們要的東西。

契約：`probabilities(trace) -> np.ndarray`，長度 2^n，順序一律 **little-endian**（與 CUDA-Q 的
`get_state` 相同）。各框架的公約不同（實測：Qiskit little-endian；Cirq／PennyLane big-endian），
轉換一律走 `bitorder`（**唯一置換處**）。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = ["Adapter", "registry", "available_adapters", "run_case"]


class Adapter(Protocol):
    name: str
    version: str

    @staticmethod
    def available() -> bool: ...

    @staticmethod
    def probabilities(trace) -> np.ndarray: ...


def registry() -> dict:
    """回傳 {名稱: 模組}；模組不存在時不報錯（學生端不需要這些依賴）。"""
    out = {}
    for name in ("qiskit_emit", "cirq_emit", "pennylane_emit"):
        try:
            out[name.replace("_emit", "")] = __import__(f"{__name__}.{name}", fromlist=["*"])
        except Exception:                        # noqa: BLE001 - 缺依賴是正常情況
            continue
    return out


def available_adapters() -> dict:
    return {k: v for k, v in registry().items() if v.available()}


def run_case(trace) -> dict:
    """對所有可用的 adapter 跑同一個 trace，回傳 {名稱: 機率向量}。"""
    return {name: mod.probabilities(trace) for name, mod in available_adapters().items()}
