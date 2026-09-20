"""核心正確性：位元順序、閘的么正矩陣、取樣、API 契約。

判準依 SPEC §4.3：算子層用 `max|U†V - cI|`（除全域相位），不是單一輸入態的重疊。
"""
from __future__ import annotations

import numpy as np
import pytest

import q01 as cudaq
from q01 import bitorder
from q01.simulator import Simulator


# ------------------------------------------------------------------ kernels
@cudaq.kernel
def k_flip0():
    q = cudaq.qvector(5)
    x(q[0])


@cudaq.kernel
def k_bell():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])


def _kernels_1q():
    out = {}

    @cudaq.kernel
    def k_h():
        q = cudaq.qvector(1); h(q[0])

    @cudaq.kernel
    def k_x():
        q = cudaq.qvector(1); x(q[0])

    @cudaq.kernel
    def k_y():
        q = cudaq.qvector(1); y(q[0])

    @cudaq.kernel
    def k_z():
        q = cudaq.qvector(1); z(q[0])

    @cudaq.kernel
    def k_s():
        q = cudaq.qvector(1); s(q[0])

    @cudaq.kernel
    def k_sdg():
        q = cudaq.qvector(1); sdg(q[0])

    @cudaq.kernel
    def k_t():
        q = cudaq.qvector(1); t(q[0])

    @cudaq.kernel
    def k_tdg():
        q = cudaq.qvector(1); tdg(q[0])

    @cudaq.kernel
    def k_sx():
        q = cudaq.qvector(1); sx(q[0])

    @cudaq.kernel
    def k_rx():
        q = cudaq.qvector(1); rx(0.7231, q[0])

    @cudaq.kernel
    def k_ry():
        q = cudaq.qvector(1); ry(0.7231, q[0])

    @cudaq.kernel
    def k_rz():
        q = cudaq.qvector(1); rz(0.7231, q[0])

    for n, k in [("h", k_h), ("x", k_x), ("y", k_y), ("z", k_z), ("s", k_s), ("sdg", k_sdg),
                 ("t", k_t), ("tdg", k_tdg), ("sx", k_sx), ("rx", k_rx), ("ry", k_ry), ("rz", k_rz)]:
        out[n] = k
    return out


ANALYTIC = {
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
    "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
}


def _unitary_of_kernel(kern, n):
    tr = kern.trace()
    return Simulator.unitary_from(n, tr.ops)


def _phase_free_deviation(u, v):
    m = u.conj().T @ v
    c = np.trace(m) / m.shape[0]
    return float(np.max(np.abs(m - c * np.eye(m.shape[0], dtype=complex)))), abs(c)


# ------------------------------------------------------------------- tests
def test_bit_order_convention():
    """CUDA-Q: get_state 是 little-endian（x(q[0]) -> 索引 1）；q01 內部是 big-endian（索引 16）。"""
    st = cudaq.get_state(k_flip0)
    assert int(np.argmax(np.abs(st))) == 1
    tr = k_flip0.trace()
    assert int(np.argmax(np.abs(tr.live.psi))) == 16
    assert bitorder.STATE_IS_BIG_ENDIAN


@pytest.mark.parametrize("name", sorted(ANALYTIC))
def test_single_qubit_gates_match_analytic(name):
    u = _unitary_of_kernel(_kernels_1q()[name], 1)
    dev, c = _phase_free_deviation(u, ANALYTIC[name])
    assert dev < 1e-12, f"{name}: max|U†V - cI| = {dev:.3e}"
    assert c == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name,theta", [("rx", 0.7231), ("ry", 0.7231), ("rz", 0.7231)])
def test_parametric_gates_match_analytic(name, theta):
    kern = _kernels_1q()[name]
    u = _unitary_of_kernel(kern, 1)
    if name == "rx":
        c_, s_ = np.cos(theta / 2), np.sin(theta / 2)
        want = np.array([[c_, -1j * s_], [-1j * s_, c_]], dtype=complex)
    elif name == "ry":
        c_, s_ = np.cos(theta / 2), np.sin(theta / 2)
        want = np.array([[c_, -s_], [s_, c_]], dtype=complex)
    else:
        want = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
    dev, _ = _phase_free_deviation(u, want)
    assert dev < 1e-12


def test_multi_qubit_gates_match_analytic():
    @cudaq.kernel
    def k_cx():
        q = cudaq.qvector(2); cx(q[0], q[1])

    @cudaq.kernel
    def k_cz():
        q = cudaq.qvector(2); cz(q[0], q[1])

    @cudaq.kernel
    def k_swap():
        q = cudaq.qvector(2); swap(q[0], q[1])

    @cudaq.kernel
    def k_ccx():
        q = cudaq.qvector(3); ccx(q[0], q[1], q[2])

    eye = np.eye(4, dtype=complex)
    cx_m = eye.copy(); cx_m[2, 2] = 0; cx_m[2, 3] = 1; cx_m[3, 2] = 1; cx_m[3, 3] = 0
    cz_m = np.diag([1, 1, 1, -1]).astype(complex)
    swap_m = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    for kern, want, n in ((k_cx, cx_m, 2), (k_cz, cz_m, 2), (k_swap, swap_m, 2)):
        dev, c = _phase_free_deviation(_unitary_of_kernel(kern, n), want)
        assert dev < 1e-12, f"{kern.name}: {dev:.3e}"
    # ccx：控制位全 1 時翻轉目標（big-endian：|110> -> |111>）
    u = _unitary_of_kernel(k_ccx, 3)
    col = u[:, 0b110]
    assert int(np.argmax(np.abs(col))) == 0b111


def test_control_modifier_and_angle_first():
    @cudaq.kernel
    def k_mcry(theta: float):
        q = cudaq.qvector(3)
        x(q[0]); x(q[1])
        ry.ctrl(theta, [q[0], q[1]], q[2])

    st = cudaq.get_state(k_mcry, np.pi)
    # little-endian 索引：q0,q1,q2 全 1 -> 0b111 = 7
    assert np.abs(st[7]) == pytest.approx(1.0, abs=1e-12)


def test_cx_only_fires_when_control_is_one():
    @cudaq.kernel
    def k():
        q = cudaq.qvector(2)
        ry(0.5, q[1])
        cx(q[0], q[1])
    st = cudaq.get_state(k)
    p = np.abs(st) ** 2
    # 控制位 = 0 → 目標不受影響：P(|01>) = sin^2(0.25)（little-endian: q0=0,q1=1 → index 2）
    assert p[2] == pytest.approx(np.sin(0.25) ** 2, abs=1e-12)


def test_sampling_statistics_and_keys():
    r = cudaq.sample(k_bell, shots_count=20000, seed=1)
    assert set(r.counts) == {"00", "11"}
    assert r.probability("00") == pytest.approx(0.5, abs=0.02)
    assert r.shots == 20000
    assert sum(r.counts.values()) == 20000


def test_sampling_reproducible():
    a = cudaq.sample(k_bell, shots_count=500, seed=7)
    b = cudaq.sample(k_bell, shots_count=500, seed=7)
    assert a.counts == b.counts


def test_only_bitorder_module_does_reversal():
    """架構測試：bit-reversal 只能出現在 bitorder.py（SPEC §4.1）。"""
    import pathlib
    import q01
    root = pathlib.Path(q01.__file__).parent
    offenders = []
    for path in root.glob("*.py"):
        if path.name == "bitorder.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "[::-1]" in text:
            offenders.append(path.name)
    assert not offenders, f"這些檔案出現了 bit-reversal：{offenders}"


def test_unsupported_target_raises_with_guidance():
    with pytest.raises(NotImplementedError) as exc:
        cudaq.set_target("nvidia")
    assert "CUDA-Q" in str(exc.value)


def test_chain_ctrl_raises_helpful_error():
    @cudaq.kernel
    def k(theta: float):
        q = cudaq.qvector(3)
        ry.ctrl(q[0]).ctrl(q[1])(theta, q[2])

    with pytest.raises((TypeError, AttributeError)) as exc:
        k.trace(0.7)
    assert "ry.ctrl(theta, [q[0], q[1]], q[2])" in str(exc.value)
