"""q01 —— CUDA-Q 的相容子集（純 NumPy 後端，原生 Windows 可跑）。

用法（與真 CUDA-Q 共用同一份程式碼）：

    try:
        import cudaq              # 正式軌
    except ImportError:
        import q01 as cudaq       # 練習軌

範圍與契約見 `SPEC.md`；超出子集的 API 一律丟 `NotImplementedError`，
錯誤訊息會附上真 CUDA-Q 的原文與改寫建議。
"""

from __future__ import annotations

from .circuit import Op, QubitRef, QVector, Trace, NoTraceError
from .gates import GATES, matrix_of
from .gradients import CentralDifference, ForwardDifference, ParameterShift
from .kernel import (Kernel, KernelCompileError, adjoint, check_kernel, control, kernel,
                     mx, my, mz, qubit, qvector)
from .sampler import SampleResult, get_state, run, sample
from .simulator import Simulator
from .spin import ObserveResult, SpinOperator, observe, spin
from .targets import get_target, get_targets, num_available_gpus, set_random_seed, set_target
from . import bitorder, gradients

__version__ = "0.0.1"

# 閘（kernel 內由 q01 注入，這裡也對外匯出以便測試與檢視）
h = GATES["h"]; x = GATES["x"]; y = GATES["y"]; z = GATES["z"]
s = GATES["s"]; sdg = GATES["sdg"]; t = GATES["t"]; tdg = GATES["tdg"]; sx = GATES["sx"]
rx = GATES["rx"]; ry = GATES["ry"]; rz = GATES["rz"]
cx = GATES["cx"]; cz = GATES["cz"]; swap = GATES["swap"]; ccx = GATES["ccx"]

__all__ = [
    "kernel", "Kernel", "KernelCompileError", "qvector", "qubit", "mz", "mx", "my",
    "adjoint", "control", "check_kernel",
    "sample", "run", "get_state", "SampleResult",
    "observe", "spin", "SpinOperator", "ObserveResult",
    "gradients", "ParameterShift", "CentralDifference", "ForwardDifference",
    "set_target", "get_target", "get_targets", "set_random_seed", "num_available_gpus",
    "Simulator", "Trace", "Op", "QubitRef", "QVector", "NoTraceError", "bitorder",
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "cx", "cz", "swap", "ccx",
    "__version__",
]
