"""本書的核心電路：5 qubit 的 QBN 輸出層。

===============================================================================
這個檔案裡的每一段，都對應〈QBN 的形式定義〉定義 2.6 的一個元素
===============================================================================

定義 2.6 說一個 QBN 是五元組 Q = (G, {H_i}, {E_i}, sigma, {M_x})。
下表說明本檔案的哪一行實作了哪一個元素：

| 定義 2.6 的元素          | 意義                       | 在本檔案哪裡實作                          |
|--------------------------|----------------------------|-------------------------------------------|
| G = (V, E)  圖結構        | 節點與邊的拓撲             | `ENTANGLE_EDGES`（環形：i -> i+1 mod 5）  |
| {H_i}  希爾伯特空間       | 每個節點的狀態空間         | `cudaq.qvector(N_QUBIT)`，H = (C^2)^5     |
| {E_i}  節點通道（CPTP）   | 條件機率通道               | `angle_encoding()` + `entangle()` +       |
|                          |                            | `trainable_layer()` 三個 @cudaq.kernel    |
| sigma  根節點初始態       | 先驗                       | `cudaq.qvector(5)` 的預設初態 |0...0>     |
| {M_x}  測量（POVM）       | 讀出分類結果               | `measure()` 的 `mz(q)`（計算基底測量）    |

電路分成四層，順序不能任意調換：

    |0>^5  --[角度編碼: 5 個 RY]-->  --[糾纏: 環形 5 個 CX]-->
          --[可訓練旋轉層: RY + RZ]-->  --[測量: mz(q)]-->  32 個基底態

    第 1 層 角度編碼（angle encoding）：
        把前端的 5 維實數向量（model2vec 嵌入經 PCA 降維後）編碼成 5 個 RY 角度。
        這對應 QBN 的「根節點先驗 sigma」——古典資料從這裡進入量子態。

    第 2 層 糾纏（entanglement）：
        環形 CX 讓相鄰 qubit 產生關聯。這對應圖 G 的邊。
        沒有這一層，輸出分布就只是 5 個獨立 RY 的乘積，等於一個沒有邊的圖。

    第 3 層 可訓練旋轉層（trainable layer）：
        每個 qubit 上的 RY + RZ，參數是我們要最佳化的對象。
        這對應 QBN 的「節點通道 E_i」——它是一個可調的么正（CPTP 的特例）。

    第 4 層 測量：
        計算基底測量，得到 32 個基底態的機率分布，再聚合成分類標籤的機率。
        這對應 {M_x}。

★ 位元順序（本書最常踩的坑，請務必讀完）
    CUDA-Q 有兩套完全相反的位元順序，混用會得到看起來「差不多但完全錯」的結果：

      (1) 狀態向量的**索引**是 little-endian：
              index = sum_i (q[i] 的值) * 2^i
          所以 q[0] 是最低有效位。x(q[0]) 之後索引是 1（不是 16）。
          由 `cudaq.get_state()` 拿到的 numpy 陣列就是這個順序。

      (2) 取樣結果的**位元字串**是 q[0] 在最左邊：
              bits[i] = 第 i 個 qubit 的測量值
          所以 x(q[0]) 之後取樣字串是 "10000"（不是 "00001"）。
          由 `cudaq.sample()` 拿到的 `counts` 的鍵就是這個順序。

      兩者互為反轉：bits = format(index, "05b")[::-1]。

    本檔案一律以 (2) 的字串當作對外的鍵，因為它和 `cudaq.sample()` 的輸出可以直接對照。
    轉換請用 `index_to_bits()`，不要自己手寫 format（很容易忘記反轉）。

執行方式：
    python dev/qbn_circuit.py

作者：C1（CUDA-Q 實作查證員）
環境：CUDA-Q 0.16.0 / Python 3.13 / qpp-cpu target（純 CPU）
"""

from __future__ import annotations

import numpy as np
import cudaq

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
N_QUBIT = 5
N_BASIS = 2**N_QUBIT  # 32

# 圖 G 的邊集合：環形拓撲 0-1-2-3-4-0
# 為什麼選環形？見〈五量子位輸出層設計〉：線性鏈會讓 q[0] 和 q[4] 永遠無法直接關聯，
# 環形用 5 條 CX 就把 5 個 qubit 串成一個沒有端點的環，是最便宜的連通圖。
ENTANGLE_EDGES: list[tuple[int, int]] = [(i, (i + 1) % N_QUBIT) for i in range(N_QUBIT)]

cudaq.set_target("qpp-cpu")


# ---------------------------------------------------------------------------
# 工具函式：位元順序轉換
# ---------------------------------------------------------------------------
def index_to_bits(index: int, n: int = N_QUBIT) -> str:
    """把狀態向量索引轉成 CUDA-Q 取樣字串。

    狀態索引是 little-endian（q[0] 是最低位），
    取樣字串是 q[0] 在最左邊，所以兩者互為反轉。

    >>> index_to_bits(1)     # q[0] = 1
    '10000'
    >>> index_to_bits(16)    # q[4] = 1
    '00001'
    """
    return format(index, f"0{n}b")[::-1]


def bits_to_index(bits: str) -> int:
    """`index_to_bits` 的反函式。"""
    return int(bits[::-1], 2)


def basis_labels(n: int = N_QUBIT) -> list[str]:
    """依 CUDA-Q 取樣字串的順序列出全部 2^n 個基底態。"""
    return [index_to_bits(i, n) for i in range(2**n)]


# ---------------------------------------------------------------------------
# 第 1 層：角度編碼 —— 對應 QBN 的根節點先驗 sigma
# ---------------------------------------------------------------------------
@cudaq.kernel
def angle_encoding(x: list[float]):
    """把 5 維實數向量編碼成 5 個 RY 角度。

    為什麼是 RY？
        因為 RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>，
        測量得到 0 的機率是 cos^2(theta/2)，是**單調且可逆**的對應。
        換句話說，RY 的角度可以直接解讀成一個機率，這正是貝氏網路要的東西。

    為什麼是 5 個角度？
        前端 model2vec 給 256 維靜態嵌入；我們要把它壓到 5 維才能餵進 5 個 qubit。
        降維的做法見〈資料編碼與降維〉。
    """
    q = cudaq.qvector(N_QUBIT)
    for i in range(N_QUBIT):
        ry(x[i], q[i])


# ---------------------------------------------------------------------------
# 第 2 層：糾纏 —— 對應圖 G 的邊
# ---------------------------------------------------------------------------
@cudaq.kernel
def entangle():
    """環形 CX 層：對每一條邊 (i, i+1 mod 5) 施加一個 CX。

    為什麼要糾纏？
        沒有糾纏，5 個 qubit 的聯合分布就是 5 個獨立分布的乘積，
        等價於圖上「沒有任何邊」的貝氏網路，表達力被鎖死在 5 個邊際分布。
        加上 CX 之後才會出現非對角元（相干項），
        這是量子版本唯一真正「比古典多出來」的東西。
    """
    q = cudaq.qvector(N_QUBIT)
    for i in range(N_QUBIT):
        cx(q[i], q[(i + 1) % N_QUBIT])


# ---------------------------------------------------------------------------
# 第 3 層：可訓練旋轉層 —— 對應 QBN 的節點通道 E_i
# ---------------------------------------------------------------------------
@cudaq.kernel
def trainable_layer(params: list[float]):
    """可訓練的旋轉層：每個 qubit 一個 RY 再一個 RZ，共 2 * 5 = 10 個參數。

    為什麼要 RY 加 RZ 兩個？
        只有 RY 的話，每個 qubit 的狀態被限制在實數球面上（Bloch 球的 XZ 平面），
        無法表達任意的單 qubit 么正。加上 RZ 才能覆蓋整個 Bloch 球。
        這是變分電路設計的標準做法（硬體效率 ansatz）。

    參數順序：params[0..4] 是 5 個 RY 角度，params[5..9] 是 5 個 RZ 角度。
    """
    q = cudaq.qvector(N_QUBIT)
    for i in range(N_QUBIT):
        ry(params[i], q[i])
    for i in range(N_QUBIT):
        rz(params[N_QUBIT + i], q[i])


# ---------------------------------------------------------------------------
# 完整電路（不含測量）：用來取狀態向量
# ---------------------------------------------------------------------------
@cudaq.kernel
def qbn_state(x: list[float], params: list[float]):
    """完整的 QBN 輸出層，**不含測量**，用來取精確的狀態向量。

    這支 kernel 是「可微分路徑」的基礎：
    `cudaq.get_state` 拿到的是精確振幅，沒有取樣誤差。
    """
    q = cudaq.qvector(N_QUBIT)

    # --- 第 1 層：角度編碼（根節點先驗 sigma）---
    for i in range(N_QUBIT):
        ry(x[i], q[i])

    # --- 第 2 層：環形糾纏（圖 G 的邊）---
    for i in range(N_QUBIT):
        cx(q[i], q[(i + 1) % N_QUBIT])

    # --- 第 3 層：可訓練旋轉層（節點通道 E_i）---
    for i in range(N_QUBIT):
        ry(params[i], q[i])
    for i in range(N_QUBIT):
        rz(params[N_QUBIT + i], q[i])


# ---------------------------------------------------------------------------
# 完整電路（含測量）：用來取樣
# ---------------------------------------------------------------------------
@cudaq.kernel
def qbn_measured(x: list[float], params: list[float]):
    """完整的 QBN 輸出層，**含測量**，用來取樣（{M_x}）。

    與 `qbn_state` 的差別只有最後的 `mz(q)`：
    那代表「把量子態讀成古典位元」這個不可逆動作。
    """
    q = cudaq.qvector(N_QUBIT)

    for i in range(N_QUBIT):
        ry(x[i], q[i])
    for i in range(N_QUBIT):
        cx(q[i], q[(i + 1) % N_QUBIT])
    for i in range(N_QUBIT):
        ry(params[i], q[i])
    for i in range(N_QUBIT):
        rz(params[N_QUBIT + i], q[i])

    mz(q)


# ---------------------------------------------------------------------------
# 對外介面
# ---------------------------------------------------------------------------
def run(
    x: list[float],
    params: list[float] | None = None,
    *,
    shots: int | None = None,
    seed: int = 2024,
) -> dict[str, float]:
    """執行 QBN 輸出層，回傳 32 個基底態的機率。

    參數
    ----
    x : list[float]，長度 5
        角度編碼的輸入（前端嵌入向量降維後的 5 個值）。
    params : list[float]，長度 10，可選
        可訓練參數。預設全 0（此時可訓練層等於單位矩陣，方便除錯）。
    shots : int，可選
        若給定，就用 `cudaq.sample` 取樣 `shots` 次並回傳經驗頻率；
        若不給（預設），就用 `cudaq.get_state` 取精確振幅。
        兩者的差別見〈CUDA-Q 核心 API〉的「取樣 vs 狀態向量」。
    seed : int
        取樣的隨機種子（只有 shots 給定時有作用）。

    回傳
    ----
    dict[str, float]
        鍵是 5 位元字串，第 i 個字元 = 第 i 個 qubit（q[0] 在最左邊），
        與 `cudaq.sample(...).counts` 的鍵完全一致。
        值的總和為 1。

    範例
    ----
    >>> p = run([0.3, 0.4, 0.5, 0.6, 0.7])
    >>> len(p)
    32
    >>> round(sum(p.values()), 10)
    1.0
    """
    if params is None:
        params = [0.0] * (2 * N_QUBIT)

    if len(x) != N_QUBIT:
        raise ValueError(f"x 的長度必須是 {N_QUBIT}，收到 {len(x)}")
    if len(params) != 2 * N_QUBIT:
        raise ValueError(f"params 的長度必須是 {2 * N_QUBIT}，收到 {len(params)}")

    probs = np.zeros(N_BASIS, dtype=float)

    if shots is None:
        # --- 解析路徑：精確振幅，無取樣誤差 ---
        sv = np.array(cudaq.get_state(qbn_state, list(x), list(params)))
        probs = np.abs(sv) ** 2
        return {index_to_bits(i): float(probs[i]) for i in range(N_BASIS)}

    # --- 取樣路徑：經驗頻率，有 1/sqrt(shots) 的統計誤差 ---
    cudaq.set_random_seed(seed)
    result = cudaq.sample(qbn_measured, list(x), list(params), shots_count=shots)
    counts = result.counts  # 屬性，不是方法
    return {bits: counts.get(bits, 0) / shots for bits in basis_labels()}


def label_probabilities(
    probs: dict[str, float], n_class: int = 4
) -> list[float]:
    """把 32 個基底態的機率聚合成 n_class 個分類標籤的機率。

    做法：取字串最左邊的 k = ceil(log2(n_class)) 位元當標籤，
    其餘位元視為「同一個類別內的不同微狀態」，把它們的機率加起來。

    為什麼不直接把 32 個基底態當成 32 類？
        因為 32 類的標籤需要 32 個訓練樣本類別，我們的資料集沒有那麼多類。
        用 2 位元當標籤（4 類）時，同一類有 8 個基底態可以分擔機率，
        等於模型在同一個類別內還有 8 種「內部狀態」可以表達，
        這是量子輸出層相對 softmax 多出來的表達空間。

    注意：這裡取的是**字串最左邊**的位元，也就是 q[0]、q[1]……
    因為位元字串的第 i 個字元就是 q[i]。若你要用「最高位 qubit」當標籤，
    應該取字串最右邊的位元，請確認你的設計與程式碼一致。
    """
    if n_class < 2:
        raise ValueError("n_class 至少要是 2")
    k = (n_class - 1).bit_length()  # ceil(log2(n_class))
    out = [0.0] * n_class
    for bits, prob in probs.items():
        out[int(bits[:k], 2)] += prob
    return out


# ---------------------------------------------------------------------------
# 主程式：跑幾個例子並自我檢查
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  QBN 輸出層（5 qubit）示範")
    print("=" * 70)

    x = [0.3, 0.4, 0.5, 0.6, 0.7]
    params = [0.1, -0.2, 0.3, -0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # --- (1) 解析路徑 ---
    print()
    print("[1] 解析路徑（cudaq.get_state，精確振幅）")
    probs = run(x, params)
    print(f"    基底態數量 = {len(probs)}")
    print(f"    機率總和   = {sum(probs.values()):.12f}")
    print()
    print("    機率最高的 6 個基底態：")
    for bits, prob in sorted(probs.items(), key=lambda kv: -kv[1])[:6]:
        print(f"      {bits}  ->  {prob:.6f}")

    # --- (2) 取樣路徑 ---
    print()
    print("[2] 取樣路徑（cudaq.sample，100,000 shots）")
    sampled = run(x, params, shots=100_000)
    worst = max(abs(sampled[b] - probs[b]) for b in probs)
    print(f"    與解析路徑的最大絕對誤差 = {worst:.5f}")
    print(f"    理論統計誤差量級 1/sqrt(N) = {1 / np.sqrt(100_000):.5f}")

    # --- (3) 分類標籤聚合 ---
    print()
    print("[3] 聚合成 4 類的機率")
    labels = label_probabilities(probs, n_class=4)
    for c, p in enumerate(labels):
        print(f"      類別 {c}: {p:.6f}")
    print(f"    總和 = {sum(labels):.12f}")

    # --- (4) 位元順序的自我檢查 ---
    print()
    print("[4] 位元順序自我檢查（這是本書最容易寫錯的地方）")

    @cudaq.kernel
    def flip_q0():
        q = cudaq.qvector(5)
        x(q[0])

    st = np.array(cudaq.get_state(flip_q0))
    idx = int(np.argmax(np.abs(st)))
    cudaq.set_random_seed(1)
    bits = next(iter(cudaq.sample(flip_q0, shots_count=10).counts))
    print(f"    x(q[0]) 之後：狀態索引 = {idx}，索引二進位 = {format(idx, '05b')}，"
          f"取樣字串 = {bits}")
    print(f"    index_to_bits({idx}) = {index_to_bits(idx)}"
          f"  ->  {'一致' if index_to_bits(idx) == bits else '不一致'}")

    # --- (5) 糾纏層的確有作用 ---
    print()
    print("[5] 糾纏層確實在做事（比較有無環形 CX）")

    @cudaq.kernel
    def no_entangle(x: list[float], params: list[float]):
        q = cudaq.qvector(N_QUBIT)
        for i in range(N_QUBIT):
            ry(x[i], q[i])
        for i in range(N_QUBIT):
            ry(params[i], q[i])
        for i in range(N_QUBIT):
            rz(params[N_QUBIT + i], q[i])

    sv_ent = np.array(cudaq.get_state(qbn_state, list(x), list(params)))
    sv_no = np.array(cudaq.get_state(no_entangle, list(x), list(params)))
    print(f"    有糾纏 vs 無糾纏的狀態向量 L2 距離 = "
          f"{np.linalg.norm(sv_ent - sv_no):.6f}")
    print("    （大於 0 表示糾纏層真的改變了狀態，不是裝飾品）")


if __name__ == "__main__":
    main()
