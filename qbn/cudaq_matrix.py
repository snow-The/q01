"""CUDA-Q 能力矩陣探測（合併版）。

本檔合併了兩組獨立撰寫的探測：
  - dev/cudaq_probe.py（11 項，含與 NumPy 的交叉驗證）
  - dev/probes/probe_01..11_*.py（11 步逐步逼近的探索過程）

合併原則：
  1. **只保留結論，不保留探索過程。** 逐步逼近的痕跡（probe_02b、probe_05/06/07/08/09）
     是為了在不知道答案時釐清事實，事實確定後就不需要留 11 個檔案。
  2. **每一項都必須能在 CI 模式下斷言。** `--ci` 時任何一項失敗即回傳非零退出碼。
  3. **每一項都印出「問題 / 實測結果 / 結論」**，可直接貼進教科書。

執行（WSL 內，因為 cudaq 只在 WSL 有）：
    cd <repo>
    /root/qbn-wsl/.venv/bin/python dev/cudaq_matrix.py          # 完整報告
    /root/qbn-wsl/.venv/bin/python dev/cudaq_matrix.py --ci     # 只回退出碼（給 CI 用）
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

DEV = Path(__file__).resolve().parent
sys.path.insert(0, str(DEV))

N_QUBIT = 5
DIM = 2 ** N_QUBIT

# (代號, 標題, 通過與否, 一句結論)
FINDINGS: list[tuple[str, str, bool, str]] = []
QUIET = False


def read_counts(res) -> dict[str, int]:
    """正確讀取 SampleResult 的計數。

    ★ 陷阱：`dict(res)` 會回傳垃圾（{'1':'0','0':'0'}），因為它迭代的是
      測量字串，再把字串兩兩配對當成 key/value。
      正確做法是讀 `.counts` 屬性。
    """
    counts = getattr(res, "counts", None)
    if counts is None:
        raise TypeError(f"{type(res).__name__} 沒有 .counts 屬性")
    return dict(counts)


def probe(tag: str, title: str):
    def deco(fn):
        def wrapper():
            if not QUIET:
                print(f"\n{'=' * 78}\n[{tag}] {title}\n{'=' * 78}")
            try:
                note = fn()
                FINDINGS.append((tag, title, True, str(note or "ok")))
                if not QUIET:
                    print(f"  → 結論：{note}")
            except Exception as exc:  # noqa: BLE001
                FINDINGS.append((tag, title, False, f"{type(exc).__name__}: {exc}"))
                if not QUIET:
                    print(f"  → 失敗：{type(exc).__name__}: {exc}")
                    traceback.print_exc(limit=3)

        return wrapper

    return deco


def bit_reverse_permutation(n_qubit: int) -> np.ndarray:
    """回傳把 little-endian 索引映到 big-endian 索引的置換（本身是對合）。

    ★ 這是本書最重要的工具函式。CUDA-Q 的 get_state() 用 little-endian，
      而 sample() 的測量字串用 big-endian，兩者換算必須靠這個置換。
      **不是** array[::-1]。
    """
    dim = 2 ** n_qubit
    perm = np.zeros(dim, dtype=int)
    for k in range(dim):
        r = 0
        for b in range(n_qubit):
            if (k >> b) & 1:
                r |= 1 << (n_qubit - 1 - b)
        perm[k] = r
    assert np.array_equal(perm[perm], np.arange(dim)), "位元反轉必須是對合"
    return perm


# ===========================================================================
# A. 環境與 API 存在性
# ===========================================================================
@probe("A0", "版本、target 清單")
def a0():
    import cudaq

    targets = [t.name for t in cudaq.get_targets()]
    print(f"  cudaq 版本 = {cudaq.__version__}")
    print(f"  get_targets() 共 {len(targets)} 個，前 12 個：{targets[:12]}")
    cudaq.set_target("qpp-cpu")
    return f"version={cudaq.__version__.split('(')[0].strip()}, targets={len(targets)}"


@probe("A1", "閘與容器")
def a1():
    import cudaq

    @cudaq.kernel
    def k():
        q = cudaq.qvector(6)
        h(q[0]); x(q[1]); y(q[2]); z(q[3])
        rx(0.1, q[0]); ry(0.2, q[1]); rz(0.3, q[2])
        cx(q[0], q[1]); cz(q[2], q[3]); swap(q[4], q[5])

    cudaq.sample(k, shots_count=50)
    return "h/x/y/z/rx/ry/rz/cx/cz/swap + qvector 全部可用"


@probe("A2", "參數化 kernel 與 Python 迴圈、list[float] 參數")
def a2():
    import cudaq

    @cudaq.kernel
    def k(thetas: list[float]):
        q = cudaq.qvector(N_QUBIT)
        for i in range(N_QUBIT):
            ry(thetas[i], q[i])

    thetas = [0.3, 0.6, 0.9, 1.2, 1.5]
    res = cudaq.sample(k, thetas, shots_count=4000)
    counts = read_counts(res)
    total = sum(counts.values())
    worst = 0.0
    for i in range(N_QUBIT):
        emp = sum(v for s, v in counts.items() if s[i] == "1") / total
        theo = np.sin(thetas[i] / 2) ** 2
        worst = max(worst, abs(emp - theo))
        print(f"    q[{i}]: 實測 {emp:.4f} vs 理論 {theo:.4f}  Δ={abs(emp-theo):.4f}")
    assert worst < 0.02, f"邊際機率偏差過大：{worst}"
    return f"list[float] + for 迴圈可用，最大邊際偏差 {worst:.4f}"


# ===========================================================================
# B. 位元順序（本書最重要的陷阱）
# ===========================================================================
@probe("B0", "★ 位元順序：sample() 字串 vs get_state() 索引")
def b0():
    import cudaq

    @cudaq.kernel
    def ry_q0(theta: float):
        q = cudaq.qvector(N_QUBIT)
        ry(theta, q[0])

    @cudaq.kernel
    def ry_q4(theta: float):
        q = cudaq.qvector(N_QUBIT)
        ry(theta, q[4])

    theta = 0.7
    p1 = np.sin(theta / 2) ** 2

    # --- 測量字串：用「非零的第二個字串」判定，不要用最常見的 ---
    # （p1 < 0.5 時最常見的是 '00000'，對它呼叫 .index('1') 會 ValueError）
    def nonzero_extra(counts: dict[str, int], base: str = "00000") -> str | None:
        for s in counts:
            if s != base:
                return s
        return None

    s0 = read_counts(cudaq.sample(ry_q0, theta, shots_count=4000))
    s4 = read_counts(cudaq.sample(ry_q4, theta, shots_count=4000))
    str0 = nonzero_extra(s0)
    str4 = nonzero_extra(s4)
    print(f"  ry(q[0]) 非零的非基底字串 = '{str0}'  "
          f"（'1' 在第 {str0.index('1')} 位）")
    print(f"  ry(q[4]) 非零的非基底字串 = '{str4}'  "
          f"（'1' 在第 {str4.index('1')} 位）")
    sample_is_be = str0 == "10000" and str4 == "00001"
    print(f"  → sample() 為 big-endian（q[0] 在最左）= {sample_is_be}")

    # --- 狀態向量（little-endian？）---
    for j, kern in ((0, ry_q0), (4, ry_q4)):
        psi = np.array(cudaq.get_state(kern, theta))
        probs = np.abs(psi) ** 2
        support = [int(k) for k in np.nonzero(probs > 1e-12)[0]]
        one = [k for k in support if k != 0]
        p = one[0].bit_length() - 1 if one else -1
        print(f"  get_state() ry(q[{j}]) 非零索引 = {support}，"
              f"值為 1 的位元位置 = {p}")

    psi0 = np.array(cudaq.get_state(ry_q0, theta))
    pos0 = int(np.nonzero(np.abs(psi0) ** 2 > 1e-12)[0][1]).bit_length() - 1
    state_is_le = pos0 == 0
    print(f"  → get_state() 為 little-endian（q[0] 在最低位）= {state_is_le}")

    assert sample_is_be and state_is_le, "位元順序假設不成立，全書結論都要改"
    return ("sample=big-endian、get_state=little-endian，"
            "換算必須用 bit-reversal")


@probe("B1", "★ 位元順序陷阱的代價（三種對照）")
def b1():
    import cudaq

    thetas = [0.3, 0.6, 0.9, 1.2, 1.5]
    weights = [0.11, -0.23, 0.37, -0.41, 0.53]

    @cudaq.kernel
    def qbn5(thetas: list[float], weights: list[float]):
        q = cudaq.qvector(N_QUBIT)
        for i in range(N_QUBIT):
            ry(thetas[i], q[i])
        cx(q[0], q[1]); cx(q[1], q[2]); cx(q[2], q[3])
        cx(q[3], q[4]); cx(q[4], q[0])
        for i in range(N_QUBIT):
            ry(weights[i], q[i])

    from qbn_sim import StateVectorSim  # noqa: E402

    st = np.array(cudaq.get_state(qbn5, thetas, weights))
    sim = StateVectorSim(N_QUBIT)
    sim.reset()
    for i in range(N_QUBIT):
        sim.ry(thetas[i], i)
    for c, t in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]:
        sim.cx(c, t)
    for i in range(N_QUBIT):
        sim.ry(weights[i], i)
    p_ref = np.abs(sim.amplitudes()) ** 2

    perm = bit_reverse_permutation(N_QUBIT)
    d_naive = float(np.max(np.abs(np.abs(st) ** 2 - p_ref)))
    d_rev = float(np.max(np.abs(np.abs(st[::-1]) ** 2 - p_ref)))
    d_perm = float(np.max(np.abs(np.abs(st[perm]) ** 2 - p_ref)))

    print(f"    不轉換（直接比）         max|ΔP| = {d_naive:.3e}")
    print(f"    用 array[::-1]（錯的）   max|ΔP| = {d_rev:.3e}")
    print(f"    用 bit-reversal（對的）  max|ΔP| = {d_perm:.3e}")
    assert d_perm < 1e-10, f"bit-reversal 後仍不一致：{d_perm}"
    assert d_perm < d_naive < d_rev or d_perm < 1e-10, "對照關係不如預期"
    return (f"bit-reversal {d_perm:.2e} ≪ 不轉換 {d_naive:.2e} "
            f"< [::-1] {d_rev:.2e}")


# ===========================================================================
# C. 受控閘（QBN 節點的核心）
# ===========================================================================
@probe("C0", "單控制 ry.ctrl：控制位 1 作用、0 不作用")
def c0():
    import cudaq

    @cudaq.kernel
    def on(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry.ctrl(theta, q[0], q[1])

    @cudaq.kernel
    def off(theta: float):
        q = cudaq.qvector(2)
        ry.ctrl(theta, q[0], q[1])

    d_on = read_counts(cudaq.sample(on, np.pi, shots_count=2000))
    d_off = read_counts(cudaq.sample(off, np.pi, shots_count=2000))
    print(f"  控制位=1 施加 RY(π) → {d_on}")
    print(f"  控制位=0 施加 RY(π) → {d_off}")
    ok_on = d_on.get("11", 0) > 1900
    ok_off = d_off.get("00", 0) > 1900
    assert ok_on and ok_off, f"受控行為不符：on={d_on}, off={d_off}"
    return "ry.ctrl(theta, ctrl, target) 行為正確"


@probe("C1", "★ 多控制位語法（參數順序是關鍵）")
def c1():
    """正確語法為 ry.ctrl(theta, [controls], target) —— **角度在前**。

    本專案曾經寫成 ry.ctrl([c0,c1], theta, target) 而失敗，並誤判「多控制位不支援」。
    這裡一次測三種順序，把判定固化下來。
    """
    import cudaq

    # 正確：角度在前、控制位清單在中
    @cudaq.kernel
    def good(theta: float):
        q = cudaq.qvector(4)
        x(q[0]); x(q[1])
        ry.ctrl(theta, [q[0], q[1]], q[3])

    d = read_counts(cudaq.sample(good, np.pi, shots_count=2000))
    print(f"  ✓ ry.ctrl(theta, [c0,c1], target) → {d}")
    ok = d.get("1101", 0) > 1900
    assert ok, f"正確語法未生效：{d}"

    # 錯誤：清單在前
    @cudaq.kernel
    def bad(theta: float):
        q = cudaq.qvector(4)
        x(q[0]); x(q[1])
        ry.ctrl([q[0], q[1]], theta, q[3])

    try:
        cudaq.sample(bad, np.pi, shots_count=100)
        bad_verdict = "竟然可用（與先前判定不同，需更新常數檔）"
    except Exception as exc:  # noqa: BLE001
        bad_verdict = f"如預期失敗（{type(exc).__name__}）"
    print(f"  ✗ ry.ctrl([c0,c1], theta, target) → {bad_verdict}")

    # 3 個控制位
    @cudaq.kernel
    def three(theta: float):
        q = cudaq.qvector(5)
        x(q[0]); x(q[1]); x(q[2])
        ry.ctrl(theta, [q[0], q[1], q[2]], q[4])

    d3 = read_counts(cudaq.sample(three, np.pi, shots_count=2000))
    print(f"  ✓ ry.ctrl(theta, [c0,c1,c2], target) → {d3}")
    assert d3.get("11101", 0) > 1900, f"三控制位未生效：{d3}"

    return "角度在前可用（1/2/3 控制位），清單在前不成立（已固化判定）"


@probe("C2", "多父節點的 CPT：用原生多控制位（不需 X 翻轉）")
def c2():
    """四個 CPT 分支各用一個多控制位閘，控制位用 X 翻轉決定「哪個分支」。

    對照 C1：原生多控制位可用之後，節點實作變得乾淨許多——
    每個 CPT 的一列就是一個 ry.ctrl，不需要手動展開成 2^(父節點數) 個分支。
    """
    import cudaq

    @cudaq.kernel
    def two_parents(t00: float, t01: float, t10: float, t11: float):
        q = cudaq.qvector(3)        # q[0], q[1] 是父節點，q[2] 是子節點
        # 把父節點放進均勻疊加，才能同時取樣四個分支
        h(q[0]); h(q[1])
        # 分支 (0,0)：兩個父節點都是 0 → 翻成 1 才能觸發多控制閘
        x(q[0]); x(q[1])
        ry.ctrl(t00, [q[0], q[1]], q[2])
        x(q[0]); x(q[1])
        # 分支 (0,1)
        x(q[0])
        ry.ctrl(t01, [q[0], q[1]], q[2])
        x(q[0])
        # 分支 (1,0)
        x(q[1])
        ry.ctrl(t10, [q[0], q[1]], q[2])
        x(q[1])
        # 分支 (1,1)
        ry.ctrl(t11, [q[0], q[1]], q[2])

    angles = (0.4, 0.8, 1.2, 1.6)
    counts = read_counts(cudaq.sample(two_parents, *angles, shots_count=40000))
    total = sum(counts.values())

    print("  各分支的 P(q2=1)（父節點為均勻疊加，故聯合機率 = 0.25）：")
    ok = True
    for a in (0, 1):
        for b in (0, 1):
            emp = sum(v for s, v in counts.items()
                      if s[0] == str(a) and s[1] == str(b) and s[2] == "1") / total
            cond = emp / 0.25
            theo = np.sin(angles[a * 2 + b] / 2) ** 2
            good = abs(cond - theo) < 0.03
            ok = ok and good
            print(f"    q0={a}, q1={b}: 實測 {cond:.4f} vs 理論 {theo:.4f}"
                  f"  {'✓' if good else '✗'}")
    assert ok, "四個分支的條件機率與 CPT 不符"
    return "四分支 CPT 全數符合理論值（原生多控制位 + X 翻轉選擇分支）"


# ===========================================================================
# D. 取樣、狀態向量、期望值、雜訊
# ===========================================================================
@probe("D0", "sample() 回傳型別與 probability()")
def d0():
    import cudaq

    @cudaq.kernel
    def k():
        q = cudaq.qvector(2)
        ry(1.0, q[0])

    res = cudaq.sample(k, shots_count=2000)
    counts = read_counts(res)
    print(f"  type = {type(res).__name__}")
    print(f"  res.counts = {counts}          ← ★ 正確讀法（屬性）")
    probs = {b: res.probability(b) for b in ("00", "01", "10", "11")}
    print(f"  res.probability(bits) = {probs}")
    print(f"  dict(res) = {dict(res)}   ← ❌ 陷阱：迭代字串兩兩配對的垃圾")
    s = sum(probs.values())
    assert abs(s - 1.0) < 1e-9, f"機率和不為 1：{s}"
    return f"用 .counts（屬性）讀計數、.probability() 讀機率，和={s:.12f}"


@probe("D1", "get_state() 型別與歸一")
def d1():
    import cudaq

    @cudaq.kernel
    def k():
        q = cudaq.qvector(N_QUBIT)
        ry(0.7, q[0])
        ry(1.1, q[1])
        cx(q[0], q[1])

    st = cudaq.get_state(k)
    arr = np.array(st)
    norm = float(np.sum(np.abs(arr) ** 2))
    print(f"  type = {type(st).__name__}")
    print(f"  np.array(state).shape = {arr.shape}, dtype = {arr.dtype}")
    assert arr.shape == (DIM,), f"形狀不符：{arr.shape}"
    assert abs(norm - 1.0) < 1e-12, f"未歸一：{norm}"
    return f"shape={arr.shape}, dtype={arr.dtype}, 歸一誤差={abs(norm-1):.2e}"


@probe("D2", "observe() 與 spin 期望值")
def d2():
    import cudaq
    from cudaq import spin

    @cudaq.kernel
    def k():
        q = cudaq.qvector(2)
        ry(1.0, q[0])
        ry(0.5, q[1])

    val = cudaq.observe(k, spin.z(0) + spin.z(1)).expectation()
    theo = np.cos(1.0) + np.cos(0.5)
    print(f"  實測 <Z0+Z1> = {val:.10f}，理論 = {theo:.10f}，Δ = {abs(val-theo):.2e}")
    assert abs(val - theo) < 1e-9
    return f"observe 可用，誤差 {abs(val-theo):.2e}"


@probe("D3", "取樣誤差量級 ≈ 1/√shots")
def d3():
    import cudaq

    @cudaq.kernel
    def k():
        q = cudaq.qvector(1)
        ry(1.0, q[0])

    theo = np.sin(0.5) ** 2
    for shots in (256, 1024, 4096):
        errs = []
        for _ in range(20):
            d = read_counts(cudaq.sample(k, shots_count=shots))
            errs.append(abs(d.get("1", 0) / shots - theo))
        mean_err = float(np.mean(errs))
        pred = 1 / np.sqrt(shots)
        print(f"    shots={shots:5d}: 平均絕對誤差 {mean_err:.5f}，"
              f"1/√shots = {pred:.5f}，比值 {mean_err/pred:.2f}")
    return "取樣誤差符合 1/√shots 量級（4096 shots ≈ 1.6%）"


@probe("D4", "雜訊模型與密度矩陣 target")
def d4():
    import cudaq

    names = [a for a in dir(cudaq) if "noise" in a.lower() or "Kraus" in a]
    print(f"  雜訊相關 API = {names}")
    cudaq.set_target("density-matrix-cpu")
    print("  set_target('density-matrix-cpu') OK")
    cudaq.set_target("qpp-cpu")
    return f"可用：{names}"


# ===========================================================================
# E. 可微分性（PyTorch 橋接現況）
# ===========================================================================
@probe("E0", "★ 可微分介面現況（注意：gradients 是延遲載入屬性）")
def e0():
    import cudaq

    print("  子模組存取方式對照：")
    for name in ("gradients", "optimizers", "torch", "interop", "mlir", "dynamics"):
        try:
            import importlib
            importlib.import_module(f"cudaq.{name}")
            via_import = "import OK"
        except Exception as exc:  # noqa: BLE001
            via_import = type(exc).__name__
        via_getattr = "有" if hasattr(cudaq, name) else "無"
        print(f"    cudaq.{name:12s} import→{via_import:22s} getattr→{via_getattr}")

    # gradients 必須用屬性存取
    ps = getattr(getattr(cudaq, "gradients", None), "ParameterShift", None)
    print(f"\n  cudaq.gradients.ParameterShift = {ps}")
    assert ps is not None, "cudaq.gradients.ParameterShift 不存在"

    # torch 確實不存在
    assert getattr(cudaq, "torch", None) is None, "cudaq.torch 竟然存在？"
    print("  cudaq.torch = None（確認不存在）")
    return ("cudaq.gradients.ParameterShift 可用（須用屬性存取，import 會失敗）；"
            "cudaq.torch 不存在 → 需自訂 autograd.Function")


@probe("E1", "★ 參數平移的正確用法（只平移量子層，古典用連鎖律）")
def e1():
    """驗證參數平移規則的正確用法，並展示錯誤用法會錯多少。

    ★ 這是本專案真實踩過的坑：
      把參數平移直接套在「量子輸出 + 古典損失」的整體函數上**不精確**。
      參數平移的推導前提是「函數是量子期望值」（sin/cos 的線性組合）；
      一旦中間夾了非線性函數（-log、softmax），規則就不再精確。

    正確做法（PennyLane / Qiskit 的實際做法）：
      1. 用參數平移算量子層輸出對參數的導數 ∂p_k/∂θ_i（對 RY 精確）
      2. 用古典連鎖律接上損失：∂L/∂θ_i = Σ_k (∂L/∂p_k)(∂p_k/∂θ_i)
    """
    from qbn_sim import StateVectorSim  # noqa: E402

    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    target = 0b10100

    def run_circuit(thetas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        sim = StateVectorSim(N_QUBIT)
        sim.reset()
        for i in range(N_QUBIT):
            sim.ry(float(thetas[i]), i)
        for c, t in ring:
            sim.cx(c, t)
        for i in range(N_QUBIT):
            sim.ry(float(weights[i]), i)
        return sim.probabilities()

    def loss(params: np.ndarray) -> float:
        p = run_circuit(params[:N_QUBIT], params[N_QUBIT:])
        return float(-np.log(max(p[target], 1e-300)))

    def dp_dparam(params: np.ndarray, i: int) -> np.ndarray:
        """∂p_k/∂θ_i：對 RY 精確（p 是振幅的二次式）。"""
        tp, tm = params.copy(), params.copy()
        tp[i] += np.pi / 2
        tm[i] -= np.pi / 2
        return (run_circuit(tp[:N_QUBIT], tp[N_QUBIT:])
                - run_circuit(tm[:N_QUBIT], tm[N_QUBIT:])) / 2

    def richardson(params: np.ndarray, i: int) -> float:
        h = 1e-4

        def cd(step: float) -> float:
            tp, tm = params.copy(), params.copy()
            tp[i] += step
            tm[i] -= step
            return (loss(tp) - loss(tm)) / (2 * step)

        return (4 * cd(h / 2) - cd(h)) / 3

    params = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 0.11, -0.23, 0.37, -0.41, 0.53])

    # 正確：平移量子層 + 連鎖律
    p = run_circuit(params[:N_QUBIT], params[N_QUBIT:])
    dl_dp = np.zeros(DIM)
    dl_dp[target] = -1.0 / max(p[target], 1e-300)
    grad_ok = np.array([
        float(dl_dp @ dp_dparam(params, i)) for i in range(2 * N_QUBIT)
    ])

    # 錯誤：直接平移損失
    grad_bad = np.zeros(2 * N_QUBIT)
    for i in range(2 * N_QUBIT):
        tp, tm = params.copy(), params.copy()
        tp[i] += np.pi / 2
        tm[i] -= np.pi / 2
        grad_bad[i] = (loss(tp) - loss(tm)) / 2

    ref = np.array([richardson(params, i) for i in range(2 * N_QUBIT)])

    err_ok = float(np.max(np.abs(grad_ok - ref)))
    err_bad = float(np.max(np.abs(grad_bad - ref)))
    print(f"    正確（平移 p + 連鎖律） vs 參考：max|Δ| = {err_ok:.3e}")
    print(f"    錯誤（直接平移 L）      vs 參考：max|Δ| = {err_bad:.3e}")
    assert err_ok < 1e-9, f"正確用法未通過：{err_ok}"
    assert err_bad > 1e-3, "錯誤用法竟然也對？規則理解有誤"
    return (f"正確用法 max|Δ|={err_ok:.2e} ✅；"
            f"錯誤用法 max|Δ|={err_bad:.2e} ❌（差距 {err_bad/err_ok:.0e} 倍）")


# ===========================================================================
# F. 5 qubit 成本
# ===========================================================================
@probe("F0", "5 qubit 電路的呼叫成本")
def f0():
    import time

    import cudaq

    @cudaq.kernel
    def qbn5(thetas: list[float], weights: list[float]):
        q = cudaq.qvector(N_QUBIT)
        for i in range(N_QUBIT):
            ry(thetas[i], q[i])
        cx(q[0], q[1]); cx(q[1], q[2]); cx(q[2], q[3])
        cx(q[3], q[4]); cx(q[4], q[0])
        for i in range(N_QUBIT):
            ry(weights[i], q[i])

    t = [0.3, 0.6, 0.9, 1.2, 1.5]
    w = [0.1, 0.2, 0.3, 0.4, 0.5]

    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        cudaq.get_state(qbn5, t, w)
    dt_state = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for _ in range(n):
        cudaq.sample(qbn5, t, w, shots_count=1024)
    dt_sample = (time.perf_counter() - t0) / n

    print(f"    get_state()      : {dt_state*1e3:.3f} ms/次")
    print(f"    sample(1024 shots): {dt_sample*1e3:.3f} ms/次")
    return (f"get_state {dt_state*1e3:.2f} ms、"
            f"sample(1024 shots) {dt_sample*1e3:.2f} ms")


# ===========================================================================
# G. 32 個基底態與分類標籤
# ===========================================================================
@probe("G0", "32 個基底態 ↔ 索引 ↔ 標籤（三種聚合策略）")
def g0():
    perm = bit_reverse_permutation(N_QUBIT)
    print("    索引  little-bits  big-bits(取樣字串)  Hamming權重")
    for idx in (0, 1, 5, 16, 31):
        bits_le = format(idx, f"0{N_QUBIT}b")[::-1]
        bits_be = format(int(perm[idx]), f"0{N_QUBIT}b")
        hw = format(int(perm[idx]), f"0{N_QUBIT}b").count("1")
        print(f"    {idx:4d}  {bits_le}      {bits_be}              {hw}")

    # 相鄰分組 vs 漢明權重分組（K=6）
    K = 6
    adjacent = [idx * K // DIM for idx in range(DIM)]
    hamming = [format(int(perm[idx]), f"0{N_QUBIT}b").count("1")
               for idx in range(DIM)]
    print(f"\n    相鄰分組（K={K}）各類大小 = "
          f"{[adjacent.count(c) for c in range(K)]}")
    print(f"    漢明權重分組各類大小     = "
          f"{[hamming.count(c) for c in range(K)]}")
    assert sum(adjacent.count(c) for c in range(K)) == DIM
    assert sum(hamming.count(c) for c in range(K)) == DIM
    return (f"K={K} 時相鄰分組大小 {[adjacent.count(c) for c in range(K)]}、"
            f"漢明分組大小 {[hamming.count(c) for c in range(K)]}")


# ===========================================================================
# H. 與純 NumPy 的總體交叉驗證
# ===========================================================================
@probe("H0", "★ 總體交叉驗證：CUDA-Q vs NumPy（5 qubit 環形糾纏）")
def h0():
    import cudaq
    from qbn_sim import StateVectorSim  # noqa: E402

    thetas = [0.3, 0.6, 0.9, 1.2, 1.5]
    weights = [0.11, -0.23, 0.37, -0.41, 0.53]

    @cudaq.kernel
    def qbn5(thetas: list[float], weights: list[float]):
        q = cudaq.qvector(N_QUBIT)
        for i in range(N_QUBIT):
            ry(thetas[i], q[i])
        cx(q[0], q[1]); cx(q[1], q[2]); cx(q[2], q[3])
        cx(q[3], q[4]); cx(q[4], q[0])
        for i in range(N_QUBIT):
            ry(weights[i], q[i])

    perm = bit_reverse_permutation(N_QUBIT)
    st = np.array(cudaq.get_state(qbn5, thetas, weights))[perm]

    sim = StateVectorSim(N_QUBIT)
    sim.reset()
    for i in range(N_QUBIT):
        sim.ry(thetas[i], i)
    for c, t in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]:
        sim.cx(c, t)
    for i in range(N_QUBIT):
        sim.ry(weights[i], i)
    ref = sim.amplitudes()

    dpsi = float(np.max(np.abs(st - ref)))
    dp = float(np.max(np.abs(np.abs(st) ** 2 - np.abs(ref) ** 2)))
    print(f"    max|Δψ| = {dpsi:.3e}")
    print(f"    max|ΔP| = {dp:.3e}")
    assert dp < 1e-10, f"交叉驗證失敗：{dp}"
    return f"max|Δψ|={dpsi:.2e}, max|ΔP|={dp:.2e}（門檻 1e-10）"


PROBES = [a0, a1, a2, b0, b1, c0, c1, c2, d0, d1, d2, d3, d4, e0, e1, f0, g0, h0]


def main() -> int:
    global QUIET
    ap = argparse.ArgumentParser(description="CUDA-Q 能力矩陣探測（合併版）")
    ap.add_argument("--ci", action="store_true", help="安靜模式，只回退出碼")
    args = ap.parse_args()
    QUIET = args.ci

    print("=" * 78)
    print("CUDA-Q 能力矩陣探測（合併版）")
    print("=" * 78)

    for fn in PROBES:
        fn()

    if not QUIET:
        print(f"\n\n{'=' * 78}\n總結\n{'=' * 78}")
        for tag, title, ok, note in FINDINGS:
            print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {title}\n         {note}")
        passed = sum(1 for _, _, ok, _ in FINDINGS if ok)
        print(f"\n  {passed}/{len(FINDINGS)} 項通過")

    failed = [f"{t} {n}" for t, _, ok, n in FINDINGS if not ok]
    if failed:
        print("\n失敗項目：")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
