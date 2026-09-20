"""
s03_temperature.py — §3 溫度縮放：決定論文成敗的防禦點
========================================================
跑法：
    uv run python experiments/s03_temperature.py

示範：
  (1) q = softmax(z / T)，T 用驗證集最大似然（＝最小化 NLL）估計。
  (2) **掃 T 並逐一驗證 top-1 標籤完全不變**（Murphy L44113）。
  (3) ECE(T) 與 NLL(T) 兩條曲線：NLL 最佳 T 與 ECE 最佳 T 通常不同。
  (4) 「你只是沒做校準後處理」這句質疑的量化版本。
  (5) 為什麼必須贏過 temperature-scaled 基線 → figs/04_temperature_scaling.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from demo_setup import fig_font_setup, load_demo
from mlkit import accuracy, brier, confidence, ece, fit_temperature, nll, predict, softmax

fig_font_setup()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    d = load_demo()
    Zva, yva = d["Zva"], d["yva"]
    Zte, yte = d["Zte"], d["yte"]
    P = softmax(Zte)

    section("§3.1 未校準模型的基準數字")
    print(f"準確率 = {accuracy(P, yte):.4f}")
    print(f"平均信心 = {confidence(P).mean():.4f}   （信心 − 準確率 = "
          f"{confidence(P).mean() - accuracy(P, yte):+.4f}）")
    print(f"NLL      = {nll(P, yte):.4f}")
    print(f"ECE(M=10)= {ece(P, yte, 10):.4f}")
    print()
    print("溫度縮放要做的事：把 logits 全部除以一個純量 T，讓分布『不那麼尖』。")
    print("    q = softmax(z / T),   T > 0")
    print("T 只有一個參數，在**驗證集**上以最大似然估計 —— 也就是最小化驗證集 NLL。")

    # -----------------------------------------------------------------------
    section("§3.2 掃 T：驗證 top-1 標籤恆不變（Murphy L44113）")
    Ts = np.exp(np.linspace(np.log(0.25), np.log(20.0), 240))
    base_top = predict(P)
    accs, eces, nlls, flips = [], [], [], 0
    for T in Ts:
        Q = softmax(Zte / T)
        topT = predict(Q)
        if not np.array_equal(topT, base_top):
            flips += 1
        accs.append(accuracy(Q, yte))
        eces.append(ece(Q, yte, 10))
        nlls.append(nll(Q, yte))
    accs, eces, nlls = map(np.array, (accs, eces, nlls))

    print(f"掃描 T ∈ [{Ts[0]:.4f}, {Ts[-1]:.4f}]，共 {Ts.size} 個值")
    print(f"其中 top-1 標籤與 T=1 不同的次數：**{flips} 次**")
    uniq = sorted({round(float(v), 10) for v in accs})
    print(f"準確率的取值集合：{uniq}  ← 只有一個值")
    print()
    print("★ 為什麼保證不變？softmax 是**單調**的：對固定的 x，")
    print("    argmax_c softmax(z/T)_c = argmax_c z_c    對所有 T > 0 都成立。")
    print("  除以正數只會重新縮放 logits，不會改變它們的大小順序。")
    print("  → **溫度縮放是「零準確率損失改善校準」的免費午餐。**")
    print("  → 反過來說：任何「我們的模型校準比較好」的主張，")
    print("    只要對手也能做溫度縮放，就必須贏過**已經縮放過**的對手。")

    print()
    print(f"{'T':>10}{'準確率':>10}{'ECE(M=10)':>12}{'NLL':>10}")
    print("-" * 78)
    for T in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0):
        Q = softmax(Zte / T)
        print(f"{T:>10.2f}{accuracy(Q, yte):>10.4f}{ece(Q, yte, 10):>12.4f}{nll(Q, yte):>10.4f}")

    # -----------------------------------------------------------------------
    section("§3.3 學習 T：驗證集上最小化 NLL")
    T_star = fit_temperature(Zva, yva)
    T_va_ece = float(Ts[int(np.argmin([ece(softmax(Zva / t), yva, 10) for t in Ts]))])
    T_te_ece = float(Ts[int(np.argmin(eces))])
    P_T = softmax(Zte / T_star)

    print(f"T*（驗證集 NLL 最佳）        = {T_star:.4f}")
    print(f"T（驗證集 ECE 最佳）         = {T_va_ece:.4f}")
    print(f"T（測試集 ECE 最佳，僅供對照，實務不可用）= {T_te_ece:.4f}")
    print()
    print(f"{'':<22}{'準確率':>10}{'ECE':>10}{'NLL':>10}{'Brier':>10}")
    print("-" * 78)
    P_te_ece = softmax(Zte / T_te_ece)
    for name, Q in (
        ("未校準（T=1）", P),
        (f"溫度縮放 T*={T_star:.3f}", P_T),
        (f"T={T_te_ece:.3f}（ECE 最佳）", P_te_ece),
    ):
        print(f"{name:<20}{accuracy(Q, yte):>10.4f}{ece(Q, yte, 10):>10.4f}"
              f"{nll(Q, yte):>10.4f}{brier(Q, yte):>10.4f}")
    print()
    print(f"→ 準確率三列完全相同（{accuracy(P, yte):.4f}）。")
    print(f"→ NLL 最佳 T*={T_star:.3f} 讓 ECE 從 {ece(P, yte, 10):.4f} 降到 {ece(P_T, yte, 10):.4f}"
          f"（{(1 - ece(P_T, yte, 10) / ece(P, yte, 10)) * 100:.1f}% 的降幅）。")
    print(f"→ 但 ECE 自己最喜歡的 T 是 {T_te_ece:.3f}，能到 {ece(P_te_ece, yte, 10):.4f}。")
    print("  ★ 誠實提醒：**溫度縮放最小化的是 NLL，不是 ECE。**")
    print("    ECE 的改善是副產品。Murphy（L44056–L44113）引用的 [Guo+17] 觀察到")
    print("    它在多個 DNN 上『剛好』也拿到最低 ECE，但這不是定理。")
    print("    所以本專案必須同時報 NLL 與 ECE，不能只挑一個講。")
    print("  ★ 也**不可以**用測試集去挑 T —— 那是資料洩漏。上表第三列只是示範。")

    # -----------------------------------------------------------------------
    section("§3.4 這對本專案的意義：量子層的校準優勢必須贏過 TS 基線")
    e_base, e_ts = ece(P, yte, 10), ece(P_T, yte, 10)
    print("把 §3.3 的數字翻譯成論證檢查表：")
    print()
    print(f"  [1] 你的量子模型（未校準）ECE      = ______   （本示範：{e_base:.4f}）")
    print(f"  [2] 你的量子模型（溫度縮放後）ECE  = ______   （本示範：{e_ts:.4f}）")
    print(f"  [3] 古典基線（未校準）ECE          = ______")
    print(f"  [4] 古典基線（溫度縮放後）ECE      = ______   ← **這一格才是你的對手**")
    print()
    print("  若 [2] < [4]，你的結論是：『在相同校準後處理下，量子層帶來更好的校準。』")
    print("  若只比 [1] < [3]，你的結論只能寫：『我們的模型沒做校準後處理，對手也沒做。』")
    print("  → 後者會被一句話打掉：『你只是沒做溫度縮放而已。』")
    print()
    print("  ★ 本專案硬規則：**RQ2（去相位消融）的每一組對照，都要同時報")
    print("    未校準與溫度縮放後兩組 ECE。** 只報未校準的那一組不算數。")
    print("  ★ 額外要求 [4] 也要用**同一個驗證集**去擬合 T（不能各擬各的測試集），")
    print("    而且要去相位前後的兩個模型共用同一組資料切分。")

    # -----------------------------------------------------------------------
    section("§3.5 溫度縮放不是萬靈丹：它修不了什麼")
    print("溫度縮放只調整『整體的尖銳程度』（一個純量），因此：")
    print("  ✗ 修不了『同一個信心水準下，不同類別的準確率不同』（class-conditional 失準）")
    print("  ✗ 修不了分布偏移（Murphy L44125：[Ova+19] 指出良好校準的模型，")
    print("    遇到不同分布的輸入時常變得校準不良）")
    print("  ✗ 修不了『箱內震盪』型的失準（§2.5 偏差二）")
    print("  ✗ 不改變 top-1，所以對準確率一點幫助都沒有")
    print("  ✓ 只修得了『整條可靠度曲線一起被過度放大』這種最常見的失準")
    print()
    print("還有一個 T 治不了的例子：**類別條件失準（class-conditional miscalibration）**。")
    print("把 logits 做一個跟類別有關的變形：前 5 類乘 1.6（更尖）、後 5 類乘 0.6（更平）。")
    print("這樣『同一條可靠度曲線』的兩半需要相反的 T，一個純量 T 救不了。\n")
    Z_slice = Zte.copy()
    Z_slice[:, :5] *= 1.6
    Z_slice[:, 5:] *= 0.6
    Zv_slice = Zva.copy()
    Zv_slice[:, :5] *= 1.6
    Zv_slice[:, 5:] *= 0.6
    P_slice = softmax(Z_slice)
    T_slice = fit_temperature(Zv_slice, yva)
    P_slice_T = softmax(Z_slice / T_slice)

    print(f"{'':<34}{'準確率':>10}{'ECE':>10}{'NLL':>10}")
    print("-" * 78)
    for name, Q in (
        ("類別條件失準（T=1）", P_slice),
        (f"類別條件失準 + 全域 T*={T_slice:.3f}", P_slice_T),
        ("── 對照組 ──", None),
        ("原本的均勻失準（T=1）", P),
        (f"原本的均勻失準 + 全域 T*={T_star:.3f}", P_T),
    ):
        if Q is None:
            print(f"{name:<32}{'':>10}{'':>10}{'':>10}")
            continue
        print(f"{name:<32}{accuracy(Q, yte):>10.4f}{ece(Q, yte, 10):>10.4f}{nll(Q, yte):>10.4f}")
    print()
    drop_slice = (1 - ece(P_slice_T, yte, 10) / ece(P_slice, yte, 10)) * 100
    drop_uni = (1 - ece(P_T, yte, 10) / ece(P, yte, 10)) * 100
    print(f"→ 均勻失準：全域 T* 讓 ECE 從 {ece(P, yte, 10):.4f} 降到 {ece(P_T, yte, 10):.4f}"
          f"（降 {drop_uni:.1f}%）。")
    print(f"→ 類別條件失準：全域 T* 讓 ECE 從 {ece(P_slice, yte, 10):.4f} 降到 "
          f"{ece(P_slice_T, yte, 10):.4f}")
    print(f"  雖然也降了 {drop_slice:.1f}%，但**剩下的殘量 "
          f"{ece(P_slice_T, yte, 10):.4f} 仍是均勻失準修好後（{ece(P_T, yte, 10):.4f}）的 "
          f"{ece(P_slice_T, yte, 10) / ece(P_T, yte, 10):.1f} 倍**。")
    print("  因為一個純量 T 只能整體伸縮，無法同時滿足『前 5 類要更平、後 5 類要更尖』。")
    print("  要修它得用**每類一個溫度**（class-wise temperature），但那已經是多參數模型，")
    print("  而且需要更多驗證資料才估得穩。")
    print()
    print("  → 這正是「量子層若真能改善校準，就不只是等價於一個溫度參數」的論證骨架：")
    print("    你要能指出**溫度縮放結構上做不到**的那種改善。")
    print("    ⚠ 反過來也要誠實：如果你的量子層改善**剛好等於一個 T**，")
    print("      那就必須承認『這個改善可以被一個純量的校準後處理取代』。")

    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(Ts, eces, color="#1f77b4", lw=1.8)
    axes[0].axvline(T_star, color="#d62728", ls="--", lw=1.5, label=f"T* = {T_star:.3f}（NLL 最佳）")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("溫度 T（對數刻度）")
    axes[0].set_ylabel("ECE (M=10)")
    axes[0].set_title("ECE 隨 T 的變化")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.25)

    axes[1].plot(Ts, nlls, color="#ff7f0e", lw=1.8, label="NLL")
    axes[1].plot(Ts, eces, color="#1f77b4", lw=1.8, label="ECE")
    axes[1].axvline(T_star, color="#d62728", ls="--", lw=1.5, label=f"T* = {T_star:.3f}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("溫度 T（對數刻度）")
    axes[1].set_ylabel("指標值")
    axes[1].set_title("NLL 與 ECE 的最佳 T 不一定相同")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.25)

    axes[2].plot(Ts, accs, color="#2ca02c", lw=2.5)
    axes[2].set_xscale("log")
    axes[2].set_ylim(accuracy(P, yte) - 0.05, accuracy(P, yte) + 0.05)
    axes[2].set_xlabel("溫度 T（對數刻度）")
    axes[2].set_ylabel("準確率")
    axes[2].set_title(f"準確率對 T 完全不變（{flips} 次標籤改變）")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    p4 = os.path.join(FIGS, "04_temperature_scaling.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print()
    print(f"[圖] 已存 figs/{os.path.basename(p4)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
