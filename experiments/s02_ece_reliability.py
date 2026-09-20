"""
s02_ece_reliability.py — §2 ECE 與可靠度圖（本指南的核心）
============================================================
跑法：
    cd C:\\Users\\qq134\\source\\repos\\QBN
    uv run python experiments/s02_ece_reliability.py

示範：
  (1) 用一個真實訓練出來的 Linear(5→10) softmax 頭算 ECE。
  (2) 可靠度圖（含對角線、落差長條、每箱樣本數）→ figs/01_*.png
  (3) 分母用 N 還是 B？把 Murphy 式 14.19 的 OCR 疑點實際算給你看。
  (4) ECE 對分箱數 M 的敏感度（M = 5..50）→ figs/02_*.png
  (5) ECE 的兩種偏差：有限樣本造成的高估、分箱抹平造成的低估。
  (6) ECE 只看 top-1 的後果：MCE 與 class-wise ECE → figs/03_*.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from demo_setup import fig_font_setup, load_demo
from mlkit import (
    accuracy,
    brier,
    classwise_ece,
    confidence,
    ece,
    fit_temperature,
    mce,
    nll,
    predict,
    reliability_bins,
    softmax,
)

fig_font_setup()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def ece_oscillating(M: int, A: float = 0.15, N: int = 400000, freq: float = 10.0, seed: int = 1) -> float:
    """
    「箱內震盪」的校準模型：信心 s ~ U(0.5, 1)，在信心 s 下答對的機率

        w(s) = clip(s + A * sin(2π * freq * s), 0, 1)

    當 freq * 箱寬 ≈ 整數時，每個 bin 內的 sin 平均掉 → 分箱 ECE ≈ 0，
    但真實校準誤差 E|w(s) − s| 一直存在。這是 ECE「向下偏差」的乾淨示範。
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.5, 1.0, N)
    w = np.clip(s + A * np.sin(2 * np.pi * freq * s), 0.0, 1.0)
    correct = (rng.uniform(size=N) < w).astype(float)
    bins = np.clip(np.ceil(np.clip(s, 0.0, 1.0) * M).astype(int) - 1, 0, M - 1)
    val = 0.0
    for b in range(M):
        m = bins == b
        n = int(m.sum())
        if n == 0:
            continue
        val += (n / N) * abs(correct[m].mean() - s[m].mean())
    return float(val)


def main() -> None:
    d = load_demo()
    P, y = d["Pte"], d["yte"]
    C = P.shape[1]
    N = y.size
    acc = accuracy(P, y)
    conf_mean = float(confidence(P).mean())
    e10 = ece(P, y, M=10)

    section("§2.1 一個真實模型的校準現況（Linear(5→10)，60 個參數）")
    print(f"訓練 {d['n_train']} 筆 / 驗證 {d['Xva'].shape[0]} 筆 / 測試 {N} 筆；{C} 類完全平衡")
    print(f"（訓練 2000 epochs、關閉 L2 —— 刻意製造現實中常見的過度自信）\n")
    print(f"測試集準確率 = {acc:.4f}")
    print(f"平均信心     = {conf_mean:.4f}")
    print(f"信心 − 準確率 = {conf_mean - acc:+.4f}   ← 正值就是過度自信")
    print(f"NLL          = {nll(P, y):.4f}")
    print(f"Brier        = {brier(P, y):.4f}")
    print(f"ECE (M=10)   = {e10:.4f}")
    print()
    print(f"→ 準確率只有 {acc:.1%}，模型卻平均宣稱 {conf_mean:.1%} 的把握。")
    print("  ECE 就是把這個落差量化成一個數字：分箱後「平均信心」與「經驗準確率」")
    print("  的加權平均差。要注意它與準確率是**兩件不同的事**（見 §3）。")

    # -----------------------------------------------------------------------
    section("§2.2 可靠度圖（reliability diagram）：一張圖看完校準")
    rb = reliability_bins(P, y, M=10)
    print(f"{'bin':>4}{'信心區間':>16}{'樣本數':>8}{'平均信心':>10}{'經驗準確率':>12}{'落差':>10}")
    print("-" * 78)
    for r in rb["rows"]:
        rng_txt = f"({r['lo']:.1f}, {r['hi']:.1f}]"
        print(f"{r['bin']:>4}{rng_txt:>16}{r['count']:>8}{r['conf']:>10.4f}{r['acc']:>12.4f}{r['gap']:>10.4f}")
    print()
    n_below = sum(1 for r in rb["rows"] if r["acc"] < r["conf"])
    print(f"→ {len(rb['rows'])} 個非空 bin 之中，有 {n_below} 個的經驗準確率低於平均信心。")
    print("  理想校準時每一列兩者應該相等，可靠度圖上的點會全部落在對角線上。")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.4, label="完美校準線 y = x")
    ax.bar(
        rb["conf"], rb["acc"], width=0.085, alpha=0.75, color="#3b7dd8",
        edgecolor="white", label="各箱經驗準確率（柱高）",
    )
    for c_, a_, n_ in zip(rb["conf"], rb["acc"], rb["count"]):
        ax.plot([c_, c_], [a_, c_], color="#d62728", lw=2.2, solid_capstyle="butt")
        ax.annotate(f"n={n_}", (c_, max(a_, c_) + 0.03), ha="center", fontsize=7.5, color="#333333")
    ax.plot([], [], color="#d62728", lw=2.2, label="落差 |acc - conf|（Murphy L44022）")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("預測信心 conf(B_b)")
    ax.set_ylabel("經驗準確率 acc(B_b)")
    ax.set_title(f"可靠度圖：Linear(5->10) 測試集\nECE(M=10) = {e10:.4f}，準確率 = {acc:.4f}")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(confidence(P), bins=np.linspace(0, 1, 21), color="#8c8c8c", edgecolor="white")
    ax.set_xlabel("預測信心")
    ax.set_ylabel("樣本數")
    ax.set_title("信心分布：樣本集中在高信心區\n（ECE 因此主要由最右邊幾箱決定）")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p1 = os.path.join(FIGS, "01_reliability_diagram.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[圖] 已存 figs/{os.path.basename(p1)}")

    # -----------------------------------------------------------------------
    section("§2.3 分母到底是 N 還是 B？（Murphy 式 14.19 的 OCR 疑點）")
    e_N = ece(P, y, M=10, denominator="N")
    e_B = ece(P, y, M=10, denominator="B")
    print(f"ECE（分母 N = {N}，正確）      = {e_N:.6f}")
    print(f"ECE（分母 B = 10，OCR 版本）   = {e_B:.6f}")
    print(f"放大倍率 = N / B = {N} / 10 = {e_B / e_N:.1f} 倍")
    print()
    print("→ 抽取檔 F5（L405–L406）把式 14.19 的分母錄成 B（bin 數）。")
    print("  這在數學上不可能成立：|B_b| 是**樣本數**，加總起來是 N 不是 B。")
    print("  除以 B 的後果不是「縮小 M 倍」，而是**ECE 會隨測試集大小膨脹**：")
    print("  同一個模型換到 10 倍大的測試集，分母 B 不變，ECE 就變成 10 倍。")
    print("  這種指標無法跨論文比較，也無法用來判斷「校準好不好」。")
    print()
    print("  本專案一律採：ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|，並在論文寫死。")
    print("  TODO(核實)：建議回查 Murphy 原書第 14 章 PDF 確認式 14.19 的分母。")

    # -----------------------------------------------------------------------
    section("§2.4 ECE 對分箱數 M 的敏感度（M = 5..50）")
    Ms = list(range(5, 51))
    eces = np.array([ece(P, y, M=m) for m in Ms])
    gap_identity = conf_mean - acc
    print(f"先看一個重要事實：平均信心 − 準確率 = {conf_mean:.6f} − {acc:.6f} = {gap_identity:.6f}")
    print(f"                  ECE (M=10)                        = {e10:.6f}")
    print(f"                  兩者相差 {abs(gap_identity - e10):.2e}\n")
    print("★ 當模型在**每一個 bin 都過度自信**（acc_b < conf_b）時，ECE 會退化成")
    print("  『平均信心 − 準確率』這個恆等式：")
    print("      ECE = Σ_b (|B_b|/N)(conf_b − acc_b) = mean(conf) − mean(acc)")
    print("  因為絕對值裡的每一項同號，|Σ| = Σ|·|。**這與 M 完全無關。**")
    print("  本模型的可靠度表（§2.2）8 個 bin 全部 acc < conf，所以 M=5..11 的 ECE")
    print("  一模一樣；M 再大時才有少數樣本數極少的 bin 出現反向落差。\n")
    print(f"{'M':>4}{'ECE':>12}    {'M':>4}{'ECE':>12}    {'M':>4}{'ECE':>12}")
    print("-" * 78)
    for i in range(0, len(Ms), 3):
        row = ""
        for j in range(3):
            k = i + j
            row += f"{Ms[k]:>4}{eces[k]:>12.6f}    " if k < len(Ms) else " " * 20
        print(row)
    print()
    print(f"M=5  時 ECE = {ece(P, y, 5):.6f}")
    print(f"M=10 時 ECE = {e10:.6f}   ← 本專案採用值")
    print(f"M=50 時 ECE = {ece(P, y, 50):.6f}")
    print(f"最小 {eces.min():.6f}（M = {Ms[int(eces.argmin())]}）／"
          f"最大 {eces.max():.6f}（M = {Ms[int(eces.argmax())]}）")
    print(f"極差 {eces.max() - eces.min():.6f}（只佔 M=10 值的 "
          f"{(eces.max() - eces.min()) / e10 * 100:.1f}%）")
    print()
    print("→ **當落差方向一致時，ECE 對 M 幾乎不敏感**（本模型就是這種情況）。")
    print("  但方向一旦在不同 bin 之間翻轉，M 的影響就會非常大 ——")
    print("  用 §2.5 那個『箱內震盪』的模型實測，同樣掃 M = 5..50：")

    # 震盪模型：落差方向在 bin 之間反覆翻轉 → M 敏感度極大
    sos = np.array([ece_oscillating(M=m, seed=1) for m in Ms])
    print(f"{'M':>4}{'震盪模型 ECE':>16}    {'M':>4}{'震盪模型 ECE':>16}")
    print("-" * 78)
    for i in range(0, len(Ms), 2):
        row = ""
        for j in range(2):
            k = i + j
            row += f"{Ms[k]:>4}{sos[k]:>16.6f}    " if k < len(Ms) else ""
        print(row)
    print()
    print(f"震盪模型：最小 {sos.min():.6f}（M={Ms[int(sos.argmin())]}）／"
          f"最大 {sos.max():.6f}（M={Ms[int(sos.argmax())]}）")
    print(f"          極差 {sos.max() - sos.min():.6f}，是 M=10 值的 "
          f"{(sos.max() - sos.min()) / sos[Ms.index(10)]:.1f} 倍")
    print()
    print("★ 兩個模型放在一起，結論才完整：")
    print(f"   - 真實模型（落差同號）：M = 5..50 的 ECE 範圍 {eces.min():.4f} ~ {eces.max():.4f}")
    print(f"   - 震盪模型（落差反覆變號）：M = 5..50 的 ECE 範圍 {sos.min():.4f} ~ {sos.max():.4f}")
    print("  所以論文中報 ECE 一定要寫明 M；本專案固定 M = 10（採 Wang et al. 2026，")
    print("  F3 抽取檔【L490】），並在附錄附上這張敏感度曲線當作穩健性檢查。")
    print("  要比較兩個模型的 ECE 時，**必須用同一個 M、同一份測試集**。")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax = axes[0]
    ax.plot(Ms, eces, "o-", ms=3.5, color="#1f77b4", label="本指南的真實模型")
    ax.axhline(gap_identity, color="#1f77b4", ls=":", lw=1.2, alpha=0.8,
               label=f"平均信心 − 準確率 = {gap_identity:.4f}")
    ax.axvline(10, color="#d62728", ls="--", lw=1.5, label="本專案採用 M = 10")
    ax.set_xlabel("分箱數 M")
    ax.set_ylabel("ECE")
    ax.set_title("落差方向一致時：ECE 幾乎與 M 無關\n（退化成 mean(conf) − acc 的恆等式）")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(Ms, sos, "s-", ms=3.5, color="#ff7f0e", label="箱內震盪模型")
    ax.axvline(10, color="#d62728", ls="--", lw=1.5, label="本專案採用 M = 10")
    ax.set_xlabel("分箱數 M")
    ax.set_ylabel("ECE")
    ax.set_title("落差方向在各 bin 之間翻轉時：\nECE 對 M 極度敏感")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p2 = os.path.join(FIGS, "02_ece_bin_sensitivity.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[圖] 已存 figs/{os.path.basename(p2)}")

    # -----------------------------------------------------------------------
    section("§2.5 ECE 的兩種偏差：有限樣本『高估』、分箱抹平『低估』")
    print("★ 偏差一：有限樣本雜訊 → ECE 被高估（upward bias）")
    print("  模擬：真實機率 p* 從 Dirichlet 抽，標籤 y ~ Categorical(p*)。")
    print("  這樣造出來的模型**在母體上完美校準**，理論 ECE = 0；")
    print("  但有限樣本下每個 bin 的 conf 與 acc 都是雜訊估計 → ECE 期望值 > 0。\n")
    rng = np.random.default_rng(0)
    print(f"{'樣本數 N':>10}{'重複次數':>10}{'平均 ECE(M=10)':>18}{'標準差':>12}")
    print("-" * 78)
    for Nsim, reps in ((100, 400), (1000, 400), (10000, 200), (50000, 60)):
        vals = []
        for _ in range(reps):
            alpha = rng.uniform(0.3, 3.0, size=10)
            Ptrue = rng.dirichlet(alpha, size=Nsim)
            ys = np.array([rng.choice(10, p=Ptrue[i]) for i in range(Nsim)])
            vals.append(ece(Ptrue, ys, M=10))
        vals = np.array(vals)
        print(f"{Nsim:>10}{reps:>10}{vals.mean():>18.5f}{vals.std(ddof=1):>12.5f}")
    print()
    print("  → N=100 時光是雜訊就貢獻約 0.067 的 ECE —— 比很多論文報的『改進量』還大。")
    print("    ★ 硬約束：測試集太小，『A 模型 ECE 0.012、B 模型 0.009』可能純屬雜訊。")
    print("      必須 (a) 用足夠大的測試集，(b) 對 ECE 本身做自助法信賴區間（§5）。")

    print()
    print("★ 偏差二：分箱把箱內的校準曲線抹平 → ECE 被低估（downward bias）")
    print("  構造一個『箱內震盪』的模型：信心 s ~ U(0.5, 1)，")
    print("  在信心 s 之下答對的機率 w(s) = s + A·sin(2π·10·s)，A = 0.15。")
    print("  它的真實校準誤差 E|w(s) − s| 是固定的，但等寬分箱看不到箱內震盪。\n")
    rng2 = np.random.default_rng(1)
    Nbig = 400000
    s = rng2.uniform(0.5, 1.0, Nbig)
    A = 0.15
    w = np.clip(s + A * np.sin(2 * np.pi * 10 * s), 0.0, 1.0)
    true_ece = float(np.abs(w - s).mean())
    print(f"  真實校準誤差 E|w(s) − s| = {true_ece:.4f}")
    print(f"{'M':>6}{'箱寬':>10}{'分箱 ECE':>14}{'低估比例':>12}")
    print("-" * 78)
    for M in (5, 10, 20, 50, 100):
        val = ece_oscillating(M)
        print(f"{M:>6}{1.0 / M:>10.3f}{val:>14.6f}{(1 - val / true_ece) * 100:>11.1f}%")
    print()
    print("  → M=5 與 M=10 的分箱 ECE 只有真實值的幾個百分點，因為 sin 的週期（0.1）")
    print("    正好是箱寬的整數倍，箱內正負互相抵消。M=20 之後才看得見震盪。")
    print("  ★ 兩個偏差方向相反、都隨 M 變化 —— 這就是為什麼 ECE 不是絕對真理，")
    print("    只能當『同一個 M、同一份測試集下的相對比較』。")

    # -----------------------------------------------------------------------
    section("§2.6 ECE 只看 top-1 的後果：MCE 與 class-wise ECE")
    print("ECE 只看『被選為 top-1 的那個類別』的信心（Murphy L44028）。")
    print("其餘 C−1 個類別的機率**完全沒被檢查**。做一個實驗：")
    print("把非 top-1 的機率全部塞給『下一個類別』，同時保證 top-1 標籤不變。\n")

    top = predict(P)
    pmax = confidence(P)
    nxt = (top + 1) % C
    tail = 1.0 - pmax
    to_nxt = np.minimum(tail, 0.95 * pmax)          # 上限 0.95·pmax，保證 argmax 不變
    share = (tail - to_nxt) / (C - 2)               # 其餘 C-2 類平分剩下的機率
    P_bad = np.full_like(P, 0.0)
    P_bad[np.arange(N), top] = pmax
    P_bad[np.arange(N), nxt] = to_nxt
    others = np.ones((N, C), dtype=bool)
    others[np.arange(N), top] = False
    others[np.arange(N), nxt] = False
    P_bad[others] = np.repeat(share, C - 2)

    ok = np.array_equal(predict(P_bad), top)
    print(f"構造檢查：每列機率和 ∈ [{P_bad.sum(1).min():.6f}, {P_bad.sum(1).max():.6f}]；"
          f"max_c P 的最小值 = {pmax.min():.4f}")
    print(f"          top-1 標籤是否完全沒變：{ok}\n")
    assert ok, "構造失敗：top-1 標籤被改變了"
    print(f"{'模型':<30}{'準確率':>10}{'ECE':>10}{'MCE':>12}{'NLL':>10}")
    print("-" * 78)
    for name, Q in (("原模型（合理分配剩餘機率）", P), ("尾部亂塞（top-1 完全相同）", P_bad)):
        print(f"{name:<28}{accuracy(Q, y):>10.4f}{ece(Q, y, 10):>10.4f}"
              f"{mce(Q, y, 10):>12.6f}{nll(Q, y):>10.4f}")
    print()
    print("→ 兩者的 top-1 標籤、準確率、**ECE 完全相同**（因為 top-1 信心一個字都沒動），")
    print(f"  但 NLL 從 {nll(P, y):.4f} 暴增到 {nll(P_bad, y):.4f}，MCE 也放大 "
          f"{mce(P_bad, y, 10) / mce(P, y, 10):.1f} 倍。")
    print("  ★ 結論：本專案必須**同時報 ECE 與 MCE**（Murphy L44030–L44038；")
    print("    [Nix+19] 已展示 ECE 好但 MCE 差的案例）。")

    cw = classwise_ece(P, y, M=10)
    print()
    print("class-wise ECE（one-vs-rest：每一類用自己的機率 P[:,c] 對 1[y=c] 分箱）：")
    print("  類別   : " + "".join(f"{c:>8}" for c in range(C)))
    print("  ECE_c  : " + "".join(f"{v:>8.4f}" for v in cw))
    print(f"  max = {np.nanmax(cw):.4f}（類別 {int(np.nanargmax(cw))}）／"
          f"min = {np.nanmin(cw):.4f}（類別 {int(np.nanargmin(cw))}）／"
          f"平均 = {np.nanmean(cw):.4f}")
    print()
    print("  ⚠ 這裡的 ECE_c 與整體 ECE **不是同一個量、不能直接比大小**：")
    print("    整體 ECE 看的是 max_c P（top-1 信心），ECE_c 看的是 P[:,c] 這個座標。")
    print("    它的用途是『相對比較』：哪一類的機率座標最不校準。")
    print(f"    本模型 max/min 差 {np.nanmax(cw) / np.nanmin(cw):.1f} 倍，"
          f"類別 {int(np.nanargmax(cw))} 最差 → 去查那一類的樣本與特徵分布。")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.bar(np.arange(C), cw, color="#2ca02c", alpha=0.8, edgecolor="white")
    ax.axhline(float(np.nanmean(cw)), color="#d62728", ls="--", lw=1.5,
               label=f"各類平均 = {np.nanmean(cw):.4f}")
    ax.set_xticks(np.arange(C))
    ax.set_xlabel("類別")
    ax.set_ylabel("class-wise ECE")
    ax.set_title("各類別的 one-vs-rest ECE：抓『某一類的機率座標爛掉』")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    rb2 = reliability_bins(P_bad, y, M=10)
    ax.plot([0, 1], [0, 1], "k--", lw=1.4)
    ax.bar(rb2["conf"], rb2["acc"], width=0.085, alpha=0.75, color="#ff7f0e", edgecolor="white")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("預測信心")
    ax.set_ylabel("經驗準確率")
    ax.set_title(f"尾部亂塞模型的可靠度圖\nECE = {ece(P_bad, y, 10):.4f}（與原模型相同！）")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p3 = os.path.join(FIGS, "03_classwise_ece_and_blindspot.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"[圖] 已存 figs/{os.path.basename(p3)}")

    # -----------------------------------------------------------------------
    section("§2.7 對照組：溫度縮放後的 ECE（§3 的預告）")
    T = fit_temperature(d["Zva"], d["yva"])
    P_T = softmax(d["Zte"] / T)
    print(f"{'':<18}{'準確率':>10}{'ECE':>10}{'MCE':>12}{'NLL':>10}")
    print("-" * 78)
    print(f"{'未校準':<16}{acc:>10.4f}{e10:>10.4f}{mce(P, y, 10):>12.6f}{nll(P, y):>10.4f}")
    print(f"{'溫度縮放 T*':<14}{accuracy(P_T, y):>10.4f}{ece(P_T, y, 10):>10.4f}"
          f"{mce(P_T, y, 10):>12.6f}{nll(P_T, y):>10.4f}")
    print()
    print(f"→ T* = {T:.4f}；準確率完全不變（{acc:.4f}），ECE 從 {e10:.4f} 降到 {ece(P_T, y, 10):.4f}。")
    print("  這就是本專案的關鍵防禦點，§3 會完整拆解。")
    print("=" * 78)


if __name__ == "__main__":
    main()
