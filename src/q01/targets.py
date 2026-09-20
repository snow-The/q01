"""target／隨機種子（與 CUDA-Q 同名同語意）。

q01 是**純 CPU** 模擬器，`set_target("qpp-cpu")` 收下並記錄；
另外接受 q01 自己的別名 `"numpy"`。torch／qiskit／cudaq 後端列在 P3／P4（尚未實作）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["Target", "set_target", "get_target", "get_targets", "set_random_seed",
           "make_rng", "num_available_gpus"]

_SUPPORTED = {"qpp-cpu", "numpy", "torch"}
_PLANNED = {"qiskit": "P3（Qiskit 交叉核對後端，走 emitter）",
            "cudaq": "由真 CUDA-Q 提供；q01 不代理"}
_current = "qpp-cpu"
_torch_device = "cpu"
_seed: int | None = None


class Target:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover
        return f"<q01.target {self.name}>"


def set_target(name: str, option: str | None = None, device: str | None = None) -> None:
    """`set_target("qpp-cpu")`／`set_target("numpy")`／`set_target("torch", device="cuda")`。

    `option` 是為了與真 CUDA-Q 的 `set_target("nvidia", option="fp64")` 同形；
    q01 目前只把 `device`（cpu／cuda／mps／xpu）用在 torch 後端（SPEC §5.5）。
    """
    global _current, _torch_device
    key = str(name)
    if key == "torch" and device:
        _torch_device = str(device)
    if key in _SUPPORTED:
        _current = key
        return
    if key in _PLANNED:
        raise NotImplementedError(
            f"q01 尚未提供 target={key!r}（規劃於 {_PLANNED[key]}）。"
            f"請改用 set_target(\"qpp-cpu\")；正式軌請用真 CUDA-Q 的 set_target({key!r})。"
        )
    if key in ("nvidia", "density-matrix-cpu", "tensornet"):
        raise NotImplementedError(
            f"q01 不支援 target={key!r}（GPU／密度矩陣／張量網路）。"
            f"這是 CUDA-Q 的專長：請在裝了 CUDA-Q 的機器上跑同一份程式碼。"
        )
    raise ValueError(f"未知的 target {key!r}；q01 支援 {sorted(_SUPPORTED)}")


def get_target() -> Target:
    return Target(_current)


def get_targets() -> list[Target]:
    return [Target("qpp-cpu"), Target("numpy")]


def set_random_seed(seed: int) -> None:
    global _seed
    _seed = int(seed)


def make_rng(seed: int | None = None) -> np.random.Generator:
    if seed is not None:
        return np.random.default_rng(int(seed))
    if _seed is not None:
        return np.random.default_rng(_seed)
    return np.random.default_rng()


def num_available_gpus() -> int:
    return 0
