"""位元順序轉換 —— **全專案唯一允許做 bit-reversal 的地方**。

公約（實測 + 原始碼雙重確認，見 SPEC §4.1）：

===============  ==========================  ==============================
來源             順序                        證據
===============  ==========================  ==============================
cudaq.get_state  little-endian（q[0]=LSB）   QppCircuitSimulator.cpp:135-148
sample 的鍵      q[0] 在最左（字串）         CircuitSimulator.h:883-887
q01 內部狀態      big-endian（q[0]=MSB）      本檔的 STATE_IS_BIG_ENDIAN
===============  ==========================  ==============================

內部採 big-endian 的理由：它與 `sample` 的字串鍵同構（`format(i, "0nb")` 就是鍵），
而且本書〈無CUDA-Q的參考實作〉與 `dev/qbn_sim.py` 就是這個公約。
代價是 `get_state` 需要一次置換——**只在本檔發生**。
"""

from __future__ import annotations

import numpy as np

STATE_IS_BIG_ENDIAN = True

__all__ = [
    "STATE_IS_BIG_ENDIAN",
    "bits_of_index",
    "index_of_bits",
    "little_to_big",
    "big_to_little",
    "state_big_to_little",
    "state_little_to_big",
    "index_to_sample_key",
    "sample_key_to_index",
]


def bits_of_index(index: int, n: int) -> str:
    """內部（big-endian）索引 → 位元字串，第 i 個字元就是 qubit i。"""
    return format(index, f"0{n}b")


def index_of_bits(bits: str) -> int:
    return int(bits, 2)


def little_to_big(index: int, n: int) -> int:
    """little-endian 索引 → big-endian 索引（自身反函式）。"""
    return int(format(index, f"0{n}b")[::-1], 2)


big_to_little = little_to_big   # 同一個置換，語意由呼叫端決定


def _perm(n: int) -> np.ndarray:
    return np.array([little_to_big(i, n) for i in range(2 ** n)], dtype=np.intp)


def state_little_to_big(psi: np.ndarray, n: int) -> np.ndarray:
    """把 CUDA-Q `get_state` 的 little-endian 振幅陣列轉成內部 big-endian。"""
    psi = np.asarray(psi)
    return psi[_perm(n)]


def state_big_to_little(psi: np.ndarray, n: int) -> np.ndarray:
    """內部 big-endian 振幅陣列 → `get_state` 的 little-endian 順序。"""
    psi = np.asarray(psi)
    return psi[_perm(n)]


def index_to_sample_key(index: int, n: int) -> str:
    """內部索引 → `sample().counts` 的鍵（q[0] 在最左）。"""
    return bits_of_index(index, n)


def sample_key_to_index(key: str) -> int:
    return index_of_bits(key)
