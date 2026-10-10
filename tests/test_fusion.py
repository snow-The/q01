"""閘融合的不變式測試（SPEC §6.9 第 4 項）。

現有的黃金向量測試是**端到端**對帳（q01 vs CUDA-Q），它會抓到融合算錯，
但抓不到「融合的落地時機」這類問題——那需要在特定交界處放探針。

這裡測三個交界，每一個都是「如果沒落地就會算錯」的最小案例：
  1. 讀取狀態（外部會直接讀 tr.live.psi）
  2. 受控閘（它的作用取決於控制位的值）
  3. 量測

以及一個整體性質：同一串閘，中途讀不讀狀態，結果必須一致。
"""
from __future__ import annotations

import numpy as np
import pytest

from q01.simulator import Simulator


def _run(ops, n, read_between: bool):
    """跑一串閘；read_between=True 時每閘之後讀一次 psi（觸發落地）。

    兩者的數學結果相同，只差在浮點捨入（一次作用 vs 兩次作用），故容差 1e-14。
    """
    sim = Simulator(n)
    for name, params, qubits, controls in ops:
        sim.apply(name, params, qubits, controls)
        if read_between:
            _ = sim.psi                      # 讀取即落地
    return sim.probabilities()


OPS = [
    ("ry", (0.31,), (0,), ()),
    ("rz", (0.77,), (0,), ()),               # 與上一個同 qubit 且都無控制 -> 可融合
    ("ry", (0.12,), (1,), ()),
    ("x", (), (2,), (1,)),                   # 受控 -> 必須先落地
    ("rz", (-0.4,), (1,), ()),
    ("ry", (0.9,), (0,), ()),
    ("x", (), (0,), (2,)),
    ("rz", (0.2,), (2,), ()),
]


def test_read_between_does_not_change_result():
    """中途讀取狀態不改變結果——這正是 psi 做成 property 要保證的事。"""
    a = _run(OPS, 3, read_between=False)
    b = _run(OPS, 3, read_between=True)
    assert np.max(np.abs(a - b)) < 1e-14


def test_control_flushes_pending():
    """受控閘必須看到控制位的最新值。

    這是融合最尖的案例：若 CX 之前沒有把 qubit 0 的 X 落地，
    控制位會被讀成 0，CX 不作用，結果會是 |01> 而不是 |11>。
    """
    sim = Simulator(2)
    sim.apply("x", (), (0,))                 # qubit 0 -> |1>（待作用）
    sim.apply("x", (), (1,), (0,))           # CX(control=0, target=1)
    p = sim.probabilities()
    # big-endian：qubit 0 是最高位，|11> 的索引是 0b11 = 3
    assert np.allclose(p, [0.0, 0.0, 0.0, 1.0]), f"得到 {p}"


def test_measure_flushes_pending():
    """量測必須看到最新的狀態。"""
    sim = Simulator(1)
    sim.apply("x", (), (0,))                 # 待作用
    assert sim.measure_collapse((0,)) == (1,)


def test_grow_flushes_pending():
    """grow() 改變 n 之後，舊的融合閘就無法再作用，必須先落地。"""
    sim = Simulator(1)
    sim.apply("x", (), (0,))
    sim.grow(1)
    p = sim.probabilities()
    assert np.isclose(p.sum(), 1.0)
    # |q0 q1> = |10> -> 索引 2（big-endian）
    assert np.isclose(p[2], 1.0), f"得到 {p}"


def test_composition_order():
    """融合必須是「先前的閘先作用」：合成矩陣是 u_new @ u_prev，不是反過來。

    ★ 這裡**必須比對振幅，不能比對機率**。
      RZ 是對角閘，只改相位、不改任何測量機率；所以 RY·RZ 與 RZ·RY 的機率完全相同，
      用機率寫這個測試會**永遠通過**（第一版就是這樣寫的，結果它什麼都沒驗到——
      反過來說，它也沒有誤報，只是沒有鑑別力）。
      比對複數振幅才看得到那個相位差。
    """
    def run(first, second):
        s = Simulator(1)
        s.apply(first[0], first[1], (0,))
        s.apply(second[0], second[1], (0,))
        return s.amplitudes()

    a = run(("ry", (np.pi / 2,)), ("rz", (np.pi / 2,)))
    b = run(("rz", (np.pi / 2,)), ("ry", (np.pi / 2,)))
    assert np.max(np.abs(a - b)) > 1e-3, "順序被吃掉了——融合的方向可能反了"
