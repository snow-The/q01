"""`sample` / `run` / `get_state` —— 取結果的三條路（SPEC §4.5）。

- 終端量測 → `sample`：從**量測前的精確機率**做多項式抽樣（O(shots)）。
- 線路中量測 → `sample` **必須報錯並導向 `run`**（真 CUDA-Q 自 0.14 起就是這個行為：
  `python/cudaq/runtime/sample.py:86-100`）；`run` 走逐 shot 軌跡。
- `get_state`：回傳 little-endian 振幅（與 CUDA-Q 相同）；kernel 內有量測時回傳塌縮態。
"""

from __future__ import annotations

import numpy as np

from . import bitorder
from .circuit import Trace
from .simulator import Simulator
from .targets import make_rng

__all__ = ["SampleResult", "sample", "run", "get_state", "MID_CIRCUIT_MESSAGE"]

MID_CIRCUIT_MESSAGE = (
    "q01／CUDA-Q 的 cudaq.sample 不支援含『線路中量測』的 kernel。\n"
    "  真 CUDA-Q 的錯誤（python/cudaq/runtime/sample.py:86-100）會把您導向 cudaq.run。\n"
    "  請改用： result = cudaq.run(kernel, *args, shots_count=N)\n"
    "  （run 會每個 shot 重跑一次電路，量測結果可參與後續條件控制。）"
)


class SampleResult:
    """`counts` 的鍵是位元字串，**第 i 個字元 = qubit i**（q[0] 在最左，與 CUDA-Q 一致）。"""

    def __init__(self, counts: dict[str, int], shots: int, n_qubits: int) -> None:
        self.counts = dict(counts)
        self.shots = int(shots)
        self.n_qubits = int(n_qubits)

    def probability(self, key: str) -> float:
        return self.counts.get(key, 0) / self.shots

    def __getitem__(self, key: str) -> int:
        return self.counts[key]

    def __len__(self) -> int:
        return len(self.counts)

    def __iter__(self):
        return iter(self.counts)

    def __repr__(self) -> str:
        body = " ".join(f"{k}:{v}" for k, v in sorted(self.counts.items(), key=lambda kv: -kv[1]))
        return "{ " + body + " }"

    def to_dict(self) -> dict[str, int]:
        return dict(self.counts)

    def most_probable(self) -> tuple[str, int]:
        return max(self.counts.items(), key=lambda kv: kv[1])


def _trace_once(kernel, args, kwargs, seed) -> Trace:
    return kernel.trace(*args, live=True, rng=make_rng(seed), **kwargs)


def _prepare_source(tr: Trace) -> tuple[np.ndarray, tuple[int, ...]]:
    qubits = tr.measures[-1] if tr.measures else tuple(range(tr.n_qubits))
    psi = tr.pre_measurement if tr.pre_measurement is not None else tr.live.psi
    return np.asarray(psi), qubits


def _counts(tr: Trace, shots: int, seed) -> SampleResult:
    psi, qubits = _prepare_source(tr)
    sim = Simulator(tr.n_qubits, rng=make_rng(seed))
    sim.psi = psi.copy()
    raw = sim.sample_counts(shots, qubits)
    n = len(qubits)
    counts = {bitorder.index_to_sample_key(i, n): c for i, c in raw.items()}
    return SampleResult(counts, shots, n)


def sample(kernel, *args, shots_count: int | None = None, seed: int | None = None, **kwargs) -> SampleResult:
    shots = 1000 if shots_count is None else int(shots_count)
    tr = _trace_once(kernel, args, kwargs, seed)
    if tr.mid_circuit or len(tr.measures) > 1:
        raise RuntimeError(MID_CIRCUIT_MESSAGE)
    return _counts(tr, shots, seed)


def run(kernel, *args, shots_count: int | None = None, seed: int | None = None, **kwargs) -> SampleResult:
    shots = 1000 if shots_count is None else int(shots_count)
    tr = _trace_once(kernel, args, kwargs, seed)
    if not (tr.mid_circuit or len(tr.measures) > 1):
        return _counts(tr, shots, seed)          # 快速路徑：無線路中量測

    counts: dict[str, int] = {}
    for s in range(shots):
        sub_seed = None if seed is None else int(seed) + s
        shot = _trace_once(kernel, args, kwargs, sub_seed)
        qubits = shot.measures[-1] if shot.measures else tuple(range(shot.n_qubits))
        bits = shot.outcomes[-1] if shot.outcomes else tuple(0 for _ in qubits)
        key = "".join(str(b) for b in bits)
        counts[key] = counts.get(key, 0) + 1
    n = len(qubits) if tr.measures else tr.n_qubits
    return SampleResult(counts, shots, n)


def get_state(kernel, *args, **kwargs) -> np.ndarray:
    """回傳 little-endian 振幅陣列（與 `np.array(cudaq.get_state(k, ...))` 同序）。"""
    tr = _trace_once(kernel, args, kwargs, None)
    return bitorder.state_big_to_little(tr.live.psi, tr.n_qubits)
