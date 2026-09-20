"""參數平移梯度（與 CUDA-Q 同名同簽章）。

簽章取自本書〈CUDA-Q 核心 API〉§6.3 以 `__doc__` 取得的實測：

    ParameterShift.compute(parameter_vector: Sequence[float], function: Callable, funcAtX: float)
        -> list[float]

三個細節（書中已記錄）：參數向量永遠是 list（單參數也要 `[x]`）；第二個參數是**損失函式**
（收到整個參數向量）；第三個是該向量上的損失值。
"""

from __future__ import annotations

import numpy as np

__all__ = ["ParameterShift", "CentralDifference", "ForwardDifference"]


class ParameterShift:
    """`g_i = [f(x + π/2·e_i) − f(x − π/2·e_i)] / 2`（對量子期望值精確）。"""

    def compute(self, parameter_vector, function, funcAtX: float | None = None) -> list[float]:
        x = [float(v) for v in parameter_vector]
        shift = np.pi / 2
        grads: list[float] = []
        for i in range(len(x)):
            plus = list(x); plus[i] += shift
            minus = list(x); minus[i] -= shift
            grads.append(float((function(plus) - function(minus)) / 2.0))
        return grads


class CentralDifference:
    def __init__(self, *_, **__) -> None:
        raise NotImplementedError(
            "q01 v1 只實作 ParameterShift（量子電路的精確梯度）。"
            "CentralDifference 請用真 CUDA-Q，或自行用 numpy 做有限差分。"
        )


class ForwardDifference:
    def __init__(self, *_, **__) -> None:
        raise NotImplementedError(
            "q01 v1 只實作 ParameterShift。ForwardDifference 請用真 CUDA-Q。"
        )
