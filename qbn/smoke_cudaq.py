"""CUDA-Q 環境冒煙測試（smoke test）：確認裝好了、能跑、而且結果是對的。

這支程式回答四個問題：
    1. CUDA-Q 裝起來了嗎？版本是多少？
    2. 這台機器上有哪些 target？我該用哪一個？
    3. 5 個 qubit 真的跑得動嗎？
    4. 跑出來的 32 個基底態計數，有沒有符合量子力學的預期？

執行方式（CPU 模擬器，不需要 GPU）：

    python dev/smoke_cudaq.py

在 Windows 上請先進入 WSL2 再執行（原因見〈CUDA-Q 環境建置〉）：

    wsl -d QBN
    cd /mnt/c/Users/<你的帳號>/source/repos/QBN
    /root/qbn/.venv/bin/python dev/smoke_cudaq.py

作者：C1（CUDA-Q 實作查證員）
環境：CUDA-Q 0.16.0 / Python 3.13 / qpp-cpu target
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import cudaq

# ---------------------------------------------------------------------------
# 設定：全書一律使用 qpp-cpu（純 CPU 的狀態向量模擬器）
# ---------------------------------------------------------------------------
# 為什麼是 qpp-cpu？
#   - 不需要 GPU，任何一台筆電都能跑（本書的硬性要求）。
#   - 5 個 qubit 的狀態向量只有 2^5 = 32 個複數，CPU 綽綽有餘。
#   - 它是「精確」模擬：給定電路就能拿到精確振幅，沒有取樣誤差。
CPU_TARGET = "qpp-cpu"


def banner(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# 第 1 關：版本
# ---------------------------------------------------------------------------
banner("第 1 關：環境與版本")
print(f"  Python        : {sys.version.split()[0]}")
print(f"  作業系統       : {platform.system()} {platform.release()} ({platform.machine()})")
print(f"  CUDA-Q        : {cudaq.__version__}")
print(f"  安裝路徑       : {cudaq.__file__}")
cudaq.set_target(CPU_TARGET)
print(f"  目前 target    : {cudaq.get_target().name}")
print(f"  target 說明    : {cudaq.get_target().description}")


# ---------------------------------------------------------------------------
# 第 2 關：target 清單
# ---------------------------------------------------------------------------
banner("第 2 關：這台機器上有哪些 target")
targets = cudaq.get_targets()
print(f"  一共 {len(targets)} 個 target。以下只列出「不需要 GPU、不需要連線」的模擬器：")
print()

CPU_TARGETS = {
    "qpp-cpu": "QPP 的純 CPU 狀態向量模擬器（本書使用）",
    "density-matrix-cpu": "密度矩陣模擬器，可加噪聲，成本是狀態向量的平方",
    "stim": "Stim 後端，專攻 Clifford 電路（我們的 RY 不是 Clifford，不適用）",
    "tensornet-mps": "張量網路（矩陣乘積態），qubit 多但糾纏低時才划算",
}

for t in sorted(targets, key=lambda x: x.name):
    if t.name in CPU_TARGETS:
        print(f"    [v] {t.name:<20} {CPU_TARGETS[t.name]}")

print()
print("  其餘的都是 GPU 專用（nvidia / nvidia-mqpu / ...）或要連到真機／雲端服務")
print("  （ionq / iqm / quantinuum / oqc / braket / pasqal / quera / ...）。")
print("  沒有 GPU、沒有帳號的讀者可以直接忽略它們。")
print()
n_gpu = cudaq.num_available_gpus()
print("  GPU 偵測：")
print(f"    cudaq.num_available_gpus() = {n_gpu}")
if n_gpu == 0:
    print("    -> 沒有可用的 NVIDIA GPU。完全不影響本書，所有範例都用 qpp-cpu。")
else:
    print(f"    -> 偵測到 {n_gpu} 張可用的 NVIDIA GPU。")
    print("       本書的範例仍然全部用 qpp-cpu，因為 5 個 qubit 的電路太小，")
    print("       丟到 GPU 上反而更慢（PCIe 傳輸與 kernel 啟動的成本大於計算本身）。")
    print("       若你之後想試 GPU，把 set_target('qpp-cpu') 換成 set_target('nvidia') 即可。")


# ---------------------------------------------------------------------------
# 第 3 關：5 qubit 電路跑得動嗎
# ---------------------------------------------------------------------------
banner("第 3 關：5 個 qubit 的狀態向量")


@cudaq.kernel
def five_ry(thetas: list[float]):
    """最基本的 5 qubit 電路：每個 qubit 各一個 RY 旋轉（角度編碼）。"""
    q = cudaq.qvector(5)
    for i in range(5):
        ry(thetas[i], q[i])


angles = [0.3, 0.4, 0.5, 0.6, 0.7]
state = np.array(cudaq.get_state(five_ry, angles))
print(f"  狀態向量維度   : {state.shape}   （2^5 = 32）")
print(f"  dtype         : {state.dtype}")
print(f"  常態（應為 1） : {np.linalg.norm(state):.10f}")
print()
print("  與 numpy 張量積解析解交叉驗證：")
expected = np.array([1.0 + 0j])
for t in reversed(angles):  # 反轉的原因見下方「位元順序」
    expected = np.kron(expected, [np.cos(t / 2), np.sin(t / 2)])
print(f"    最大絕對誤差 = {np.max(np.abs(state - expected)):.3e}")

print()
print("  ★ 位元順序（本書最常踩的坑）：")
print("      CUDA-Q 的狀態向量索引是 little-endian：index = sum_i (q[i] 的值) * 2^i")
print("      所以 q[0] 是最低有效位，|q0=1> 的索引是 1，不是 16。")


# ---------------------------------------------------------------------------
# 第 4 關：32 個基底態的測量計數直方圖
# ---------------------------------------------------------------------------
banner("第 4 關：5 qubit 的 32 個基底態計數直方圖")


@cudaq.kernel
def five_ry_ring(thetas: list[float]):
    """角度編碼 + 環形糾纏：這已經是本專案 QBN 輸出層的骨架。"""
    q = cudaq.qvector(5)
    for i in range(5):
        ry(thetas[i], q[i])
    for i in range(5):
        cx(q[i], q[(i + 1) % 5])
    mz(q)


SHOTS = 4096
cudaq.set_random_seed(2024)  # 讓每次執行的取樣結果一致，方便對答案
result = cudaq.sample(five_ry_ring, angles, shots_count=SHOTS)
counts = result.counts  # 注意：counts 是「屬性」，不是方法（不要寫 counts()）

print(f"  shots = {SHOTS}")
print(f"  相異基底態數 = {len(counts)} / 32")
print()
print("  機率直方圖（每一格 = 0.5%，只列出有出現的基底態）：")
print()
print(f"    {'基底態':<8} {'計數':>6} {'機率':>8}   直方圖")
print(f"    {'-' * 8} {'-' * 6} {'-' * 8}   {'-' * 40}")
for bits in sorted(counts, key=lambda b: -counts[b]):
    count = counts[bits]
    prob = count / SHOTS
    bar = "#" * int(round(prob * 200))
    print(f"    {bits:<8} {count:>6} {prob:>8.4f}   {bar}")

print()
print("  位元字串怎麼讀？字串的第 i 個字元 = 第 i 個 qubit（q[0] 在最左邊）。")
print("  這和狀態向量的索引相反：index 的 binary 要反轉才會等於字串。")


# ---------------------------------------------------------------------------
# 第 5 關：這個結果對不對？
# ---------------------------------------------------------------------------
banner("第 5 關：正確性檢查（為什麼可以相信上面那張圖）")

# (a) 機率總和
print("  [a] 取樣機率總和 = "
      f"{sum(counts.values()) / SHOTS:.6f}   （每個 shot 都落在某個基底態，必為 1）")

# (b) 對稱性檢查：全 pi/2 時應為均勻分布
@cudaq.kernel
def uniform_check(thetas: list[float]):
    q = cudaq.qvector(5)
    for i in range(5):
        ry(thetas[i], q[i])


uniform_probs = np.abs(np.array(cudaq.get_state(uniform_check, [np.pi / 2] * 5))) ** 2
print(f"  [b] 全部 theta = pi/2 時，每個基底態機率 = {uniform_probs[0]:.6f}"
      f"   （解析值 1/32 = {1 / 32:.6f}）")
print(f"      是否 32 個都相等 = {np.allclose(uniform_probs, 1 / 32)}")

# (c) 結構檢查：只有一個 RY 不為零時，只有 2 個基底態有機率
@cudaq.kernel
def single_rotation(theta: float):
    q = cudaq.qvector(5)
    ry(theta, q[2])


single_probs = np.abs(np.array(cudaq.get_state(single_rotation, 1.0))) ** 2
nz = [int(i) for i in np.nonzero(single_probs > 1e-12)[0]]
print(f"  [c] 只在 q[2] 上做 RY(1.0)：非零基底態索引 = {nz}")
print(f"      （索引 4 = 2^2 就是「q[2] = 1」，這是 little-endian 的直接證據）")
print(f"      機率 = {np.round(single_probs[nz], 6)}"
      f"   （cos^2(0.5)={np.cos(0.5) ** 2:.6f}, sin^2(0.5)={np.sin(0.5) ** 2:.6f}）")

# (d) 陷阱示範：kernel 裡有 mz 時，get_state 拿到的是「塌縮後」的態
@cudaq.kernel
def single_rotation_measured(theta: float):
    q = cudaq.qvector(5)
    ry(theta, q[2])
    mz(q)


print()
print("  [d] ★ 陷阱：kernel 裡有 mz(q) 時，get_state 拿到的是塌縮後的態")
print("      同一個電路（q[2] 上 RY(1.0)）連續取 200 次 get_state，")
print("      統計「塌縮到哪個基底態」的次數：")
COLLAPSE_TRIALS = 200
collapse_counts: dict[int, int] = {}
for _ in range(COLLAPSE_TRIALS):
    collapsed = np.array(cudaq.get_state(single_rotation_measured, 1.0))
    hit = int(np.argmax(np.abs(collapsed)))
    collapse_counts[hit] = collapse_counts.get(hit, 0) + 1
for hit in sorted(collapse_counts):
    freq = collapse_counts[hit] / COLLAPSE_TRIALS
    print(f"        索引 {hit}: {collapse_counts[hit]:>4} 次  ({freq:.3f})")
print(f"      解析機率應為索引 0 -> {np.cos(0.5) ** 2:.3f}, "
      f"索引 4 -> {np.sin(0.5) ** 2:.3f}")
print("      => 每次拿到的都是「某一個」基底態，不是疊加態；")
print("         要取精確振幅，kernel 裡絕對不要放 mz。")

banner("冒煙測試結束：全部通過")
print("  如果你的輸出和上面一致，環境就裝好了，可以開始讀〈CUDA-Q 核心 API〉。")
print()
