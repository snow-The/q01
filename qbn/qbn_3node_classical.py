"""3 節點古典貝氏網路：窮舉聯合分布（本書〈QBN 的形式定義〉的對照組）。

圖結構（與書中 3.1 節相同）：

    X1 ──> X2
     │      │
     └────> X3 <──┘

因式分解：

    P(x1, x2, x3) = P(x1) * P(x2 | x1) * P(x3 | x1, x2)

為什麼要有這支程式？
    量子貝葉斯網路（QBN）的定義是「把古典貝氏網路的每個元素換成量子版本」。
    要說清楚換掉了什麼，就得先有一個**完全算得出來**的古典版本當基準線。
    這支程式只用到 NumPy，不需要 CUDA-Q，任何一台電腦都能跑。

執行方式：
    uv run python dev/qbn_3node_classical.py

作者：C1（CUDA-Q 實作查證員）
"""

from __future__ import annotations

import itertools

import numpy as np

# ---------------------------------------------------------------------------
# 一、條件機率表（Conditional Probability Table, CPT）
# ---------------------------------------------------------------------------
# 每一列的總和必須是 1；這是貝氏網路的唯一參數來源。

# 根節點 X1 的先驗：P(X1=0) = 0.6, P(X1=1) = 0.4
P_X1 = np.array([0.6, 0.4])

# P(X2 | X1)：第 0 列是 X1=0，第 1 列是 X1=1
P_X2_GIVEN_X1 = np.array(
    [
        [0.9, 0.1],  # P(X2=0|X1=0), P(X2=1|X1=0)
        [0.3, 0.7],  # P(X2=0|X1=1), P(X2=1|X1=1)
    ]
)

# P(X3 | X1, X2)：索引順序是 [x1][x2][x3]
P_X3_GIVEN_X1_X2 = np.array(
    [
        [[0.8, 0.2], [0.5, 0.5]],  # X1=0
        [[0.4, 0.6], [0.1, 0.9]],  # X1=1
    ]
)


# ---------------------------------------------------------------------------
# 二、窮舉聯合分布
# ---------------------------------------------------------------------------
def joint_distribution() -> dict[tuple[int, int, int], float]:
    """用因式分解窮舉 2^3 = 8 種組合的聯合機率 P(x1, x2, x3)。

    迴圈裡的三行就是因式分解本身——這是整支程式最重要的一段：
    把三個小表格相乘，就得到整張圖的聯合分布。維度是指數成長的，
    所以 n 個二元節點需要 2^n 個條目（這正是量子版本想繞開的成長）。
    """
    joint: dict[tuple[int, int, int], float] = {}
    for x1, x2, x3 in itertools.product([0, 1], repeat=3):
        joint[(x1, x2, x3)] = float(
            P_X1[x1] * P_X2_GIVEN_X1[x1, x2] * P_X3_GIVEN_X1_X2[x1, x2, x3]
        )
    return joint


def marginal(joint: dict[tuple[int, int, int], float], keep: int) -> np.ndarray:
    """對聯合分布做邊際化，只留下第 keep 個變數（0= X1, 1= X2, 2= X3）。

    古典版本用「加總」把不要的變數消掉；量子版本的對應操作是「部分跡」。
    """
    out = np.zeros(2)
    for state, prob in joint.items():
        out[state[keep]] += prob
    return out


def conditional_from_joint(
    joint: dict[tuple[int, int, int], float], x1: int, x2: int
) -> np.ndarray:
    """從聯合分布反推 P(X3 | X1=x1, X2=x2)，用來驗證因式分解沒寫錯。"""
    numer = np.array([joint[(x1, x2, 0)], joint[(x1, x2, 1)]])
    denom = numer.sum()
    return numer / denom


def ancestral_sample(n: int, seed: int = 2024) -> np.ndarray:
    """祖先取樣：照拓撲順序 X1 → X2 → X3 逐節點抽樣。

    這是貝氏網路「可以從圖直接生成資料」的具體展現，也是 QBN 電路的
    古典對照：量子版本把每一步抽樣換成一個受控旋轉。
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((n, 3), dtype=int)

    samples[:, 0] = rng.choice([0, 1], size=n, p=P_X1)
    for i in range(n):
        x1 = samples[i, 0]
        samples[i, 1] = rng.choice([0, 1], p=P_X2_GIVEN_X1[x1])
        x2 = samples[i, 1]
        samples[i, 2] = rng.choice([0, 1], p=P_X3_GIVEN_X1_X2[x1, x2])
    return samples


# ---------------------------------------------------------------------------
# 三、主程式
# ---------------------------------------------------------------------------
def main() -> None:
    joint = joint_distribution()

    print("聯合分布（因式分解）:")
    for key in sorted(joint):
        print(f"  P{key} = {joint[key]:.4f}")
    total = sum(joint.values())
    print(f"總和 = {total:.10f}  （必須為 1）")
    print()

    # --- 邊際化 ---
    print("邊際分布（把其他變數加總掉）:")
    for keep, name in enumerate("X1 X2 X3".split()):
        m = marginal(joint, keep)
        print(f"  P({name}=0) = {m[0]:.4f}, P({name}=1) = {m[1]:.4f}")
    print()

    # --- 自我檢查：從聯合分布反推的條件機率，必須等於原本的 CPT ---
    print("自我檢查：由聯合分布反推的 P(X3|X1,X2) 是否等於原始 CPT？")
    ok = True
    for x1, x2 in itertools.product([0, 1], repeat=2):
        recovered = conditional_from_joint(joint, x1, x2)
        original = P_X3_GIVEN_X1_X2[x1, x2]
        same = np.allclose(recovered, original)
        ok &= bool(same)
        print(
            f"  X1={x1}, X2={x2}: 反推 = [{recovered[0]:.4f}, {recovered[1]:.4f}]  "
            f"原始 = [{original[0]:.4f}, {original[1]:.4f}]  {'一致' if same else '不一致'}"
        )
    print(f"  全部一致：{ok}")
    print()

    # --- 祖先取樣 vs 精確分布 ---
    n = 200_000
    samples = ancestral_sample(n)
    print(f"祖先取樣 {n:,} 次，與精確聯合分布比較（最大絕對誤差）:")
    worst = 0.0
    for key in sorted(joint):
        empirical = float(np.mean(np.all(samples == np.array(key), axis=1)))
        worst = max(worst, abs(empirical - joint[key]))
    print(f"  最大絕對誤差 = {worst:.5f}")
    print("  （統計誤差量級約 1/sqrt(n) ≈ "
          f"{1 / np.sqrt(n):.5f}，符合即代表取樣正確）")


if __name__ == "__main__":
    main()
