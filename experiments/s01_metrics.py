"""
s01_metrics.py — §1 評估指標：為什麼準確率不夠
================================================
跑法（在 QBN 專案根目錄）：
    uv run python experiments/s01_metrics.py

本檔示範三件事：
  (1) 兩個模型準確率都是 95%，但一個危險、一個誠實 —— 準確率看不出來。
  (2) 類別不平衡時，準確率會說謊（95% 準確率的模型其實什麼都沒抓到）。
  (3) NLL 是嚴格評分規則，準確率不是；NLL 無界、Brier 有界。
"""

from __future__ import annotations

import numpy as np

from mlkit import (
    accuracy,
    brier,
    confidence,
    ece,
    nll,
    onehot,
    predict,
    prf_per_class,
    softmax,
)

C = 10
N = 1000
SEED = 20260601


# ---------------------------------------------------------------------------
# 1.1 兩個模型，同樣 95% 準確率
# ---------------------------------------------------------------------------
def build_model(conf_mean: float, conf_sd: float, correct_mask: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """
    把「每筆樣本多確定」還原成一張完整的機率矩陣 (N, C)。

    作法：先決定哪幾筆答對（前 950 筆），答錯的從其他類別裡挑一個當預測；
    信心全放在預測類別上，剩下的機率平均分給其餘 C-1 類。
    """
    n = y.size
    yhat = y.copy()
    wrong = np.where(~correct_mask)[0]
    for i in wrong:
        cand = [c for c in range(C) if c != y[i]]
        yhat[i] = cand[rng.integers(len(cand))]

    conf = np.clip(rng.normal(conf_mean, conf_sd, n), 0.05, 0.99999)
    P = np.full((n, C), 0.0)
    P[:] = ((1.0 - conf) / (C - 1))[:, None]
    P[np.arange(n), yhat] = conf
    return P / P.sum(axis=1, keepdims=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    y = np.repeat(np.arange(C), N // C)  # 10 類各 100 筆，完全平衡
    n_correct = 950
    correct_mask = np.zeros(N, dtype=bool)
    correct_mask[:n_correct] = True  # 前 950 筆答對

    # 模型 A（危險）：不管對錯，一律宣稱 99% 確定
    PA = build_model(0.990, 0.004, correct_mask, y, rng)
    # 模型 B（誠實）：信心 ≈ 準確率 95%。注意誠實模型答錯時**也是** 0.95，
    #                 因為「說 95% 就該有 5% 出錯」正是校準的定義。
    PB = build_model(0.952, 0.008, correct_mask, y, rng)

    print("=" * 74)
    print("§1.1 兩個模型，同樣的準確率，不同的下場")
    print("=" * 74)
    print(f"資料：{N} 筆、{C} 類（每類 100 筆，完全平衡）；答對 {n_correct} 筆\n")

    header = f"{'指標':<16}{'模型 A（宣稱 99%）':>22}{'模型 B（誠實）':>20}{'誰比較好':>14}"
    print(header)
    print("-" * 74)

    rows = []
    for name, P in (("A", PA), ("B", PB)):
        rows.append(
            {
                "name": name,
                "acc": accuracy(P, y),
                "nll": nll(P, y),
                "brier": brier(P, y),
                "ece": ece(P, y, M=10),
                "conf": float(confidence(P).mean()),
            }
        )

    a, b = rows
    print(f"{'準確率':<16}{a['acc']:>22.4f}{b['acc']:>20.4f}{'一樣':>14}")
    print(f"{'平均信心':<16}{a['conf']:>22.4f}{b['conf']:>20.4f}{'B 較誠實':>14}")
    print(f"{'NLL（越低越好）':<14}{a['nll']:>22.4f}{b['nll']:>20.4f}{'B 較好':>14}")
    print(f"{'Brier（越低越好）':<13}{a['brier']:>22.4f}{b['brier']:>20.4f}{'B 較好':>14}")
    print(f"{'ECE（越低越好）':<15}{a['ece']:>22.4f}{b['ece']:>20.4f}{'B 較好':>14}")
    print()
    print(f"→ 準確率完全相同（都是 {a['acc']:.4f}），但 ECE 差了 {a['ece'] / max(b['ece'], 1e-12):.0f} 倍。")
    print(f"→ NLL 差 {a['nll'] - b['nll']:+.4f}（A 較差），這是因為 A 答錯時仍宣稱 0.99，")
    print("  真類別被壓到 (1-0.99)/9 ≈ 0.0011，-log(0.0011) ≈ 6.8 —— NLL 對這種事懲罰極重。")
    print()

    # 為什麼 NLL 差的幅度看起來不大？因為只有 5% 的樣本答錯。
    # 拆開來看「答對的子集」與「答錯的子集」。
    ca, cb = predict(PA) == y, predict(PB) == y
    idx = np.arange(N)
    nllA_wrong = float(-np.log(PA[idx[~ca], y[~ca]]).mean())
    nllB_wrong = float(-np.log(PB[idx[~cb], y[~cb]]).mean())
    print("拆開看（同一批答錯的 50 筆）：")
    print(f"  模型 A 答錯時的平均 NLL = {nllA_wrong:.4f}")
    print(f"  模型 B 答錯時的平均 NLL = {nllB_wrong:.4f}")
    print(f"  → 差距 {nllA_wrong - nllB_wrong:+.4f}，擴大了 {(nllA_wrong - nllB_wrong) / max(a['nll'] - b['nll'], 1e-12):.1f} 倍。")
    print("  → 這就是為什麼「平均 NLL」會把危險模型洗白：95% 的答對樣本稀釋了 5% 的災難。")
    print()

    # -----------------------------------------------------------------------
    # 1.2 類別不平衡：準確率直接說謊
    # -----------------------------------------------------------------------
    print("=" * 74)
    print("§1.2 類別不平衡時，準確率會說謊")
    print("=" * 74)
    n_pos, n_neg = 50, 950
    yb = np.array([1] * n_pos + [0] * n_neg)          # 1 = 少數類（5%）

    # 「懶惰分類器」：永遠說 0（多數類），而且永遠 100% 確定
    P_lazy = np.zeros((yb.size, 2))
    P_lazy[:, 0] = 1.0

    # 「真的有在做事的分類器」：用一張指定的混淆矩陣反推機率。
    #   正類：TP=30, FN=20（召回 0.60）
    #   負類：FN=60, TP=890（也就是 60 筆負類被誤判成正類 → 精確率 30/90 = 0.333）
    #   整體準確率 = (30 + 890) / 1000 = 0.92 —— **比懶惰分類器還低！**
    P_good = np.zeros((yb.size, 2))
    P_good[0:30, 1], P_good[0:30, 0] = 0.80, 0.20          # 正類，抓到
    P_good[30:50, 1], P_good[30:50, 0] = 0.35, 0.65        # 正類，漏掉
    P_good[50:110, 0], P_good[50:110, 1] = 0.40, 0.60      # 負類，誤報（argmax 變成 1）
    P_good[110:, 0], P_good[110:, 1] = 0.90, 0.10          # 負類，正確

    print(f"測試集：{n_neg} 筆負類 + {n_pos} 筆正類（正類佔 {n_pos / yb.size:.1%}）\n")
    print(f"{'模型':<22}{'準確率':>10}{'精確率':>10}{'召回率':>10}{'F1':>10}")
    print("-" * 74)
    for name, P in (("永遠說「負類」", P_lazy), ("真的有在分類", P_good)):
        m = prf_per_class(P, yb, C=2)
        print(f"{name:<20}{accuracy(P, yb):>10.4f}{m['precision'][1]:>10.4f}{m['recall'][1]:>10.4f}{m['f1'][1]:>10.4f}")
    print()
    print("→ 懶惰分類器準確率 95%，召回率 0、F1 0：它一筆正類都沒抓到。")
    print("→ 真的在做事的模型準確率只有 92%（比較低！），卻抓到了 60% 的正類。")
    print("  如果你只看準確率，你會挑錯模型。這就是為什麼類別不平衡時")
    print("  一定要同時報 macro-F1 與 per-class 指標（ECE 也一樣，見 §2.5）。")
    print()

    # -----------------------------------------------------------------------
    # 1.3 NLL 是嚴格評分規則，準確率不是
    # -----------------------------------------------------------------------
    print("=" * 74)
    print("§1.3 嚴格評分規則（strictly proper scoring rule）")
    print("=" * 74)
    print("定義（Murphy, Advanced Topics, L43955–L43969）：")
    print("  S(p_theta, p*) <= S(p*, p*)    等號若且唯若 p_theta = p*")
    print("  白話：**只有誠實報告機率**才能拿到最好分數。")
    print("  準確率不是嚴格評分規則 —— 它只看 argmax，")
    print("  (0.99, 0.005, ...) 跟 (0.34, 0.33, 0.33) 在它眼裡一樣好。\n")

    # 三種「報告方式」，真實機率 p* 已知
    p_star = np.array([0.60, 0.25, 0.10, 0.05])
    reports = {
        "誠實  q = p*": p_star.copy(),
        "過度自信（往 0/1 推）": softmax(np.log(p_star) * 2.5),
        "過度保守（往均勻推）": 0.5 * p_star + 0.5 * np.full(4, 0.25),
    }

    # 先看「單一筆已實現結果」的分數 —— 這裡會出現反直覺的現象
    print("(a) 只看單一筆已實現結果（真實標籤剛好是類別 0）：")
    print(f"{'報告方式':<26}{'NLL':>10}{'Brier':>10}{'q_0':>10}{'top-1':>8}")
    print("-" * 74)
    for name, q in reports.items():
        print(f"{name:<24}{nll(q[None, :], np.array([0])):>10.4f}{brier(q[None, :], np.array([0])):>10.4f}{q[0]:>10.4f}{int(q.argmax()):>8}")
    print("  ⚠ 注意：過度自信的 NLL 反而最低！因為這一筆剛好答對了，")
    print("    把類別 0 推得更高自然得分更好。**這不是評分規則壞掉，")
    print("    而是「單一樣本的分數」本來就有運氣成分。**\n")

    # 正確的嚴格性檢查：對真實分布取期望
    def expected_nll(q: np.ndarray) -> float:
        return float(-(p_star * np.log(np.clip(q, 1e-12, None))).sum())

    def expected_brier(q: np.ndarray) -> float:
        return float(sum(p_star[c] * ((q - onehot(np.array([c]), 4)[0]) ** 2).sum() for c in range(4)))

    print("(b) 對真實分布 p* 取期望（這才是 proper scoring rule 的定義）：")
    print(f"{'報告方式':<26}{'E[NLL]':>12}{'E[Brier]':>12}")
    print("-" * 74)
    for name, q in reports.items():
        print(f"{name:<24}{expected_nll(q):>12.4f}{expected_brier(q):>12.4f}")
    print()
    print(f"  → 誠實報告的 E[NLL] = {expected_nll(p_star):.4f}，正好等於真實分布的熵 H(p*) = {-(p_star * np.log(p_star)).sum():.4f}。")
    print("  → 任何偏離 p* 的報告方式（更尖或更平）期望分數都更差。")
    print("    這就是嚴格評分規則（Gibbs 不等式）：**誠實是唯一的最佳策略**。")
    print("  → 準確率沒有這個性質：三種報告方式的 top-1 標籤完全一樣，")
    print("    它在 (a) 看不出差別，在 (b) 也完全無法區分。\n")

    # -----------------------------------------------------------------------
    # 1.4 NLL 無界、Brier 有界
    # -----------------------------------------------------------------------
    print("=" * 74)
    print("§1.4 為什麼 Brier 有界而 NLL 無界")
    print("=" * 74)
    print(f"{'真類別被給的機率 p':>20}{'NLL = -log p':>16}{'Brier（單筆, C=4）':>22}")
    print("-" * 74)
    for p in (0.9, 0.5, 0.1, 0.01, 1e-3, 1e-6, 1e-12):
        q = np.array([[p, (1 - p) / 3, (1 - p) / 3, (1 - p) / 3]])
        print(f"{p:>20.0e}{nll(q, np.array([0])):>16.4f}{brier(q, np.array([0])):>22.6f}")
    print()
    print("→ p → 0 時 NLL → ∞，Brier 卻飽和在 2（除以 C 後是 0.5）。")
    print("  推論：**如果你的模型偶爾會把正確類別壓到極低機率，NLL 會被那幾筆支配。**")
    print("  實務上要同時報兩個：Brier 看整體、NLL 看尾端災難。")
    print()

    # 逆向檢查：一個「作弊」的評分方式會獎勵說謊
    print("=" * 74)
    print("§1.5 反例：什麼樣的指標會被騙")
    print("=" * 74)
    honest = np.tile(np.array([0.95, 0.05]), (100, 1))
    liar = np.tile(np.array([0.999, 0.001]), (100, 1))
    ybin = np.array([0] * 95 + [1] * 5)
    both = np.vstack([honest, liar])
    print(f"{'模型':<26}{'準確率':>10}{'ECE':>10}{'NLL':>10}{'Brier':>10}")
    print("-" * 74)
    for name, P in (("誠實（說 0.95）", honest), ("吹牛（說 0.999）", liar)):
        print(f"{name:<24}{accuracy(P, ybin):>10.4f}{ece(P, ybin, M=10):>10.4f}{nll(P, ybin):>10.4f}{brier(P, ybin):>10.4f}")
    print()
    print("→ 準確率一模一樣；ECE / NLL / Brier 一致地把吹牛的那個抓出來。")
    print("  這三個指標就是本專案「不讓模型靠大聲取勝」的整套防線。")
    print("=" * 74)


if __name__ == "__main__":
    main()
