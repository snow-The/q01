"""
s05_seeds_stats.py — §5 統計嚴謹性：5 個種子該怎麼報
======================================================
跑法：
    uv run python experiments/s05_seeds_stats.py

示範：
  (1) 5 個種子 [7, 21, 42, 84, 168] 的 mean ± std 報告格式。
  (2) 配對 vs 非配對檢定該怎麼選。
  (3) **為什麼 n = 5 的 p 值不能當強證據** —— 用模擬把檢定力算出來。
  (4) 該改報什麼：效果量 + 自助法信賴區間。
  (5) 可以直接抄進 limitation 的誠實聲明 → figs/06_power_n5.png
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from demo_setup import fig_font_setup
from mlkit import (
    accuracy,
    bootstrap_ci,
    cohens_d_independent,
    ece,
    fit_temperature,
    hedges_g,
    logits_of,
    make_readout_dataset,
    nll,
    paired_t_test,
    softmax,
    student_t_sf,
    train_softmax,
    welch_t_test,
)

fig_font_setup()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SEEDS = [7, 21, 42, 84, 168]  # 本專案規格常數（採 Wang et al. 2026，F3【L331】）


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 兩個「模型」：把 separation 當成一個可調的物理條件。
#   條件 A（separation 3.0）= 教學上的「量子層（未去相位）」
#   條件 B（separation 2.75）= 「量子層（去相位後）」
# ⚠ 這裡的重點是**統計協定**，不是物理結論。實際實驗時把這兩行換成
#   你的 CUDA-Q 電路（去相位前 / 後）即可，下面的統計程式一行都不用改。
# ---------------------------------------------------------------------------
def run_condition(separation: float, seed: int) -> dict:
    X, y, _ = make_readout_dataset(N=3000, C=10, D=5, separation=separation, noise=1.0, seed=42)
    Xtr, ytr = X[:300], y[:300]
    Xva, yva = X[1800:2400], y[1800:2400]
    Xte, yte = X[2400:], y[2400:]
    m = train_softmax(Xtr, ytr, Xva, yva, C=10, seed=seed, epochs=2000, lr=0.05, batch=64, l2=0.0)
    Zv, Zt = logits_of(m, Xva), logits_of(m, Xte)
    T = fit_temperature(Zv, yva)
    return {
        "acc": accuracy(softmax(Zt), yte),
        "acc_ts": accuracy(softmax(Zt / T), yte),
        "ece": ece(softmax(Zt), yte, 10),
        "ece_ts": ece(softmax(Zt / T), yte, 10),
        "nll_ts": nll(softmax(Zt / T), yte),
        "T": T,
    }


def main() -> None:
    section("§5.1 五個種子：資料怎麼收、怎麼報")
    print(f"種子集合 = {SEEDS}（本專案規格常數，採 Wang et al. 2026，F3 抽取檔【L331】）")
    print("每個種子都要重跑**完整訓練流程**（含權重初始化與資料順序），")
    print("只有種子不同，其他超參數一個字都不能改。\n")
    print("跑兩組條件（A = 未去相位、B = 去相位後）…")

    resA = [run_condition(3.00, s) for s in SEEDS]
    resB = [run_condition(2.75, s) for s in SEEDS]

    def col(res, k):
        return np.array([r[k] for r in res])

    print()
    print(f"{'seed':>6}{'A acc':>10}{'A ECE':>10}{'A acc(TS)':>12}{'A ECE(TS)':>12}{'T*':>8}")
    print("-" * 78)
    for s, r in zip(SEEDS, resA):
        print(f"{s:>6}{r['acc']:>10.4f}{r['ece']:>10.4f}{r['acc_ts']:>12.4f}"
              f"{r['ece_ts']:>12.4f}{r['T']:>8.3f}")
    print(f"{'mean':>6}{col(resA, 'acc').mean():>10.4f}{col(resA, 'ece').mean():>10.4f}"
          f"{col(resA, 'acc_ts').mean():>12.4f}{col(resA, 'ece_ts').mean():>12.4f}")
    print(f"{'std':>6}{col(resA, 'acc').std(ddof=1):>10.4f}{col(resA, 'ece').std(ddof=1):>10.4f}"
          f"{col(resA, 'acc_ts').std(ddof=1):>12.4f}{col(resA, 'ece_ts').std(ddof=1):>12.4f}")
    print()
    print(f"{'seed':>6}{'B acc':>10}{'B ECE':>10}{'B acc(TS)':>12}{'B ECE(TS)':>12}{'T*':>8}")
    print("-" * 78)
    for s, r in zip(SEEDS, resB):
        print(f"{s:>6}{r['acc']:>10.4f}{r['ece']:>10.4f}{r['acc_ts']:>12.4f}"
              f"{r['ece_ts']:>12.4f}{r['T']:>8.3f}")
    print(f"{'mean':>6}{col(resB, 'acc').mean():>10.4f}{col(resB, 'ece').mean():>10.4f}"
          f"{col(resB, 'acc_ts').mean():>12.4f}{col(resB, 'ece_ts').mean():>12.4f}")
    print(f"{'std':>6}{col(resB, 'acc').std(ddof=1):>10.4f}{col(resB, 'ece').std(ddof=1):>10.4f}"
          f"{col(resB, 'acc_ts').std(ddof=1):>12.4f}{col(resB, 'ece_ts').std(ddof=1):>12.4f}")

    print()
    print("★ 報告格式（直接抄）")
    print(f"   A（未去相位） 準確率 {col(resA, 'acc').mean() * 100:.2f} ± "
          f"{col(resA, 'acc').std(ddof=1) * 100:.2f} %，"
          f"ECE {col(resA, 'ece').mean():.4f} ± {col(resA, 'ece').std(ddof=1):.4f}")
    print(f"   B（去相位後） 準確率 {col(resB, 'acc').mean() * 100:.2f} ± "
          f"{col(resB, 'acc').std(ddof=1) * 100:.2f} %，"
          f"ECE {col(resB, 'ece').mean():.4f} ± {col(resB, 'ece').std(ddof=1):.4f}")
    print(f"   （以上為 {len(SEEDS)} 個種子的平均 ± 標準差；")
    print("     標準差用 ddof=1 的樣本標準差，在論文中要寫明。）")

    # -----------------------------------------------------------------------
    section("§5.2 配對還是非配對？以及該報什麼")
    aA, aB = col(resA, "ece"), col(resB, "ece")
    print("配對（paired）：同一個種子下兩個條件各跑一次 → 種子帶來的差異被消掉，")
    print("                檢定力最高。**只要種子能一一對應，就一定要用配對。**")
    print("非配對（unpaired）：種子數不同、或某個種子跑失敗 → 只能用 Welch t 檢定。")
    print()
    pt = paired_t_test(aA, aB)   # A 減 B
    wt = welch_t_test(aA, aB)
    print(f"{'檢定':<28}{'統計量':>12}{'自由度':>10}{'p 值':>12}")
    print("-" * 78)
    print(f"{'配對 t 檢定':<26}{pt['t']:>12.4f}{pt['df']:>10}{pt['p']:>12.5f}")
    print(f"{'Welch t 檢定（非配對）':<22}{wt['t']:>12.4f}{wt['df']:>10.3f}{wt['p']:>12.5f}")
    print()
    print(f"平均差（A − B, ECE）= {pt['mean_diff']:+.4f}；配對 SD = {pt['sd_diff']:.4f}")
    print(f"Cohen's d_z（配對效果量） = {pt['dz']:+.3f}")
    print(f"Hedges' g（獨立效果量，含小樣本修正） = {hedges_g(aA, aB):+.3f}")

    d_ = aA - aB
    est, lo, hi = bootstrap_ci(d_, n_boot=20000, seed=0)
    print(f"自助法 95% 信賴區間（配對差, 20000 次重抽）= [{lo:+.4f}, {hi:+.4f}]")
    est_e, lo_e, hi_e = bootstrap_ci(aA, n_boot=20000, seed=0)
    print(f"自助法 95% CI（A 的 ECE 本身）           = [{lo_e:.4f}, {hi_e:.4f}]")
    print()
    print("★ 效果量的粗略判讀（Cohen 的慣例，僅供參考）：")
    print("   |d| < 0.2 小、0.2–0.5 中、0.5–0.8 大、> 0.8 很大")
    print(f"   本示範 |d_z| = {abs(pt['dz']):.3f}")
    print()
    print("★ 只有 5 個種子時，建議的報告順序（由強到弱）：")
    print("   1. 兩組的 mean ± std 與**配對差**")
    print("   2. 配對差的自助法（或 t 分布）95% 信賴區間 ← 最重要")
    print("   3. 效果量（配對用 Cohen's d_z、獨立用 Hedges' g）")
    print("   4. p 值：**只能當輔助**，且必須附上 n 與自由度")
    print("   5. 明確寫出樣本數不足的限制（見 §5.5）")

    # -----------------------------------------------------------------------
    section("§5.3 ★ 為什麼 n = 5 的 p 值不能當強證據（模擬實測）")
    print("模擬設定：假設真實效果 d_z（配對差的標準化平均值）已知，")
    print("每次實驗抽 n 個配對差 d_i ~ N(d_z, 1)，跑雙尾配對 t 檢定，")
    print("統計『拒絕 H0（p < 0.05）』的比例 = 檢定力（power）。\n")
    rng = np.random.default_rng(2026)
    REPS = 20000
    print(f"{'真實 d_z':>10}{'n = 5 檢定力':>16}{'n = 10':>12}{'n = 20':>12}{'n = 50':>12}")
    print("-" * 78)
    power_curve = {}
    for dz in (0.2, 0.5, 0.8, 1.2, 2.0):
        row = []
        for n in (5, 10, 20, 50):
            d = rng.normal(dz, 1.0, size=(REPS, n))
            dm, sd = d.mean(axis=1), d.std(axis=1, ddof=1)
            t = dm / (sd / math.sqrt(n))
            p = np.array([2 * student_t_sf(abs(tt), n - 1) for tt in t])
            row.append((p < 0.05).mean())
        power_curve[dz] = row
        print(f"{dz:>10.1f}" + "".join(f"{v:>12.3f}" if i else f"{v:>16.3f}"
                                       for i, v in enumerate(row)))
    print()
    print("★ 讀法：真實效果 d_z = 0.8（學界常說的『大效果』）時，")
    print(f"  n = 5 只有 {power_curve[0.8][0] * 100:.0f}% 的機率測得到；")
    print(f"  n = 20 有 {power_curve[0.8][2] * 100:.0f}%。")
    print("  → 若你用 n = 5 得到 p > 0.05，**不能**說『沒有差異』——")
    print("    你只是沒有足夠的樣本去偵測它。這叫『不顯著 ≠ 無效果』。")

    print()
    print("★ 反向的陷阱：就算 p < 0.05，那個 p 值本身也很不穩定。")
    print("  在 d_z = 0.8、n = 5 且『剛好顯著』的那些實驗裡，來看看效果量長什麼樣：")
    d = rng.normal(0.8, 1.0, size=(REPS, 5))
    dm, sd = d.mean(axis=1), d.std(axis=1, ddof=1)
    t = dm / (sd / math.sqrt(5))
    p = np.array([2 * student_t_sf(abs(tt), 4) for tt in t])
    sig = p < 0.05
    dz_hat = dm[sig] / sd[sig]
    print(f"  顯著的實驗數：{sig.sum()} / {REPS}（{sig.mean() * 100:.1f}%）")
    print(f"  這些實驗的 d_z 估計值：中位數 {np.median(dz_hat):.2f}，"
          f"範圍 [{dz_hat.min():.2f}, {dz_hat.max():.2f}]")
    print(f"  真實值只有 0.80，但估計值從 {dz_hat.min():.2f} 到 {dz_hat.max():.2f} ——")
    print("  這就是『贏者詛咒（winner's curse）』：只有效果被高估的實驗才會顯著。")
    print("  → 這也是為什麼**效果量 + 信賴區間**比 p 值誠實得多。")

    # -----------------------------------------------------------------------
    section("§5.4 那要幾個種子才夠？")
    print("依上面的檢定力模擬（雙尾 α = 0.05，目標檢定力 0.8）：")
    print(f"{'真實 d_z':>10}{'需要 n':>10}")
    print("-" * 78)
    for dz in (0.5, 0.8, 1.2, 2.0):
        need = None
        for n in range(3, 120):
            dd = rng.normal(dz, 1.0, size=(6000, n))
            dm2, sd2 = dd.mean(axis=1), dd.std(axis=1, ddof=1)
            t2 = dm2 / (sd2 / math.sqrt(n))
            p2 = np.array([2 * student_t_sf(abs(tt), n - 1) for tt in t2])
            if (p2 < 0.05).mean() >= 0.80:
                need = n
                break
        print(f"{dz:>10.1f}{str(need) if need else '> 119':>10}")
    print()
    print("→ 要達到慣用的 0.8 檢定力：中等效果（d_z = 0.5）需要 n ≈ 33，")
    print("  大效果（d_z = 0.8）也需要 n ≈ 15。")
    print("  5 個種子**只夠支撐描述性統計**（mean ± std）與誠實的限制聲明。")
    print("  本專案沿用的是 Wang et al. 2026 的協定（5 個種子），")
    print("  因此我們**沿用協定**但**不假裝它足以做推論統計**。")

    # -----------------------------------------------------------------------
    section("§5.5 可以直接抄進 limitation 的誠實聲明")
    print("-" * 78)
    print("""本研究依循 Wang et al. (2026) 的多種子協定，以 5 個隨機種子
（7、21、42、84、168）重複每一組實驗，並以平均 ± 樣本標準差（ddof = 1）報告。

我們必須明確指出這個設計的統計限制：在 5 個種子下，配對 t 檢定的自由度
僅為 4，對中等效果量（Cohen's d_z ≈ 0.5）的檢定力遠低於常用的 0.8 水準。
因此本研究**不將 p 值作為主要證據**，也不以「未達顯著」推論「沒有差異」。
我們改以（a）配對差的平均與自助法 95% 信賴區間、（b）效果量（Cohen's d_z
與 Hedges' g）作為量化不確定性的主要依據，並在主文中同時呈現個別種子的
數值，讓讀者能自行判斷離散程度。

在後續工作中，我們建議將種子數提高到 20 以上，或改用重複量測的階層式
模型（hierarchical model）同時估計種子內與種子間的變異，以取得足以支撐
推論統計的檢定力。""")
    print("-" * 78)
    print("（英文版見指南 §5.6，可直接放進論文。）")

    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax = axes[0]
    ns = [5, 10, 20, 50]
    for dz, row in power_curve.items():
        ax.plot(ns, row, "o-", label=f"d_z = {dz}")
    ax.axhline(0.8, color="#d62728", ls="--", lw=1.5, label="慣用檢定力門檻 0.80")
    ax.axvline(5, color="#7f7f7f", ls=":", lw=1.5, label="本專案的 n = 5")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(x) for x in ns])
    ax.set_xlabel("種子數 n（對數刻度）")
    ax.set_ylabel("檢定力 P(p < 0.05)")
    ax.set_title("配對 t 檢定的檢定力：n = 5 幾乎測不到中等效果")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(dz_hat, bins=40, color="#ff7f0e", edgecolor="white", alpha=0.9)
    ax.axvline(0.8, color="#d62728", ls="--", lw=2, label="真實 d_z = 0.8")
    ax.axvline(float(np.median(dz_hat)), color="#1f77b4", ls="-", lw=2,
               label=f"顯著實驗的中位數 = {np.median(dz_hat):.2f}")
    ax.set_xlabel("在 n = 5 下『剛好顯著』的實驗所估計出的 d_z")
    ax.set_ylabel("次數")
    ax.set_title("贏者詛咒：只有效果被高估的實驗才會顯著")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p6 = os.path.join(FIGS, "06_power_n5.png")
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    print()
    print(f"[圖] 已存 figs/{os.path.basename(p6)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
