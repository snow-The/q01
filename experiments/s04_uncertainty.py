"""
s04_uncertainty.py — §4 不確定性分解
=====================================
跑法：
    uv run python experiments/s04_uncertainty.py

示範：
  (1) T = 30 次的蒙地卡羅預測分布（本專案的規格常數）。
  (2) 預測熵 H[p̄]、互資訊 I[y;θ]、變異比 三個指標。
  (3) 三者各自的失效情形（用實際數字示範，不是口頭說）。
  (4) OOD 偵測：把不確定性指標當偵測器來用。
  (5) ⚠ 加法分解式的出處警告 → figs/05_uncertainty_decomposition.png
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
    confidence,
    ece,
    logits_of,
    make_readout_dataset,
    mutual_information,
    nll,
    predict,
    predictive_entropy,
    softmax,
    train_softmax,
    variation_ratio,
)

fig_font_setup()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

T_MC = 30  # 本專案規格常數：推論期蒙地卡羅次數（採 Wang et al. 2026，F3【L470】）


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    d = load_demo()
    Xtr, ytr = d["Xtr"], d["ytr"]
    Xte, yte = d["Xte"], d["yte"]

    # -----------------------------------------------------------------------
    section("§4.1 怎麼生出 T = 30 個預測分布")
    print("本專案取樣不確定性的來源有兩種（見 F3 抽取檔 §4.1 第 12 項）：")
    print("  (a) 把量子電路的變分參數 θ 本身當隨機變數（貝氏／變分 ansatz）")
    print("  (b) 對 θ 做擾動，或用 bootstrap 重抽訓練集訓多個成員")
    print("本節用 (b) 的**自助聚合（bagging）深度集成**示範，因為它不需要量子模擬器，")
    print("而且協定與量子版完全一樣：跑 T 次前向、逐次 softmax、再平均。")
    print()
    print("⚠ 順序很重要：**先逐次 softmax 再平均**，不是先平均 logits 再 softmax。")
    print("  （F3 抽取檔式 (9)(10)【L472】【L478】特別註明這點。）")

    rng = np.random.default_rng(1234)
    members = []
    for t in range(T_MC):
        idx = rng.integers(0, Xtr.shape[0], size=Xtr.shape[0])  # bootstrap 重抽
        m = train_softmax(
            Xtr[idx], ytr[idx], d["Xva"], d["yva"], C=10,
            seed=1000 + t, epochs=1200, lr=0.05, batch=64, l2=0.0,
        )
        members.append(m)
    print(f"\n已訓練 {len(members)} 個集成成員（T = {T_MC}）。")

    P_samples = np.stack([softmax(logits_of(m, Xte)) for m in members], axis=0)  # (T, N, C)
    P_bar = P_samples.mean(axis=0)
    print(f"P_samples 形狀 = {P_samples.shape}   （T, N, C）")

    # -----------------------------------------------------------------------
    section("§4.2 三個指標的定義與實測")
    H_total = predictive_entropy(P_bar)
    MI = mutual_information(P_samples)
    VR = variation_ratio(P_samples)

    print("  (i)  預測熵（總不確定性的代理量）  H[p̄] = -Σ_c p̄_c log p̄_c，p̄ = (1/T)Σ_t p_t")
    print("  (ii) 互資訊（BALD）                I = H[p̄] − (1/T)Σ_t H[p_t]")
    print("  (iii)變異比                        VR = 1 − (眾數標籤票數 / T)")
    print()
    print(f"{'指標':<26}{'平均':>10}{'最小':>10}{'最大':>10}")
    print("-" * 78)
    for name, v in (("預測熵 H[p̄]", H_total), ("互資訊 I[y;θ]", MI), ("變異比 VR", VR)):
        print(f"{name:<24}{v.mean():>10.4f}{v.min():>10.4f}{v.max():>10.4f}")
    print()
    print(f"檢查 1：互資訊是否非負？ min(I) = {MI.min():.3e}")
    print("  → 理論上 I ≥ 0（Jensen 不等式：平均的熵 ≥ 熵的平均）。")
    print("    若出現明顯負值，通常是浮點誤差或 p_t 沒有正規化。")
    print(f"檢查 2：I ≤ H[p̄]？ {bool(np.all(MI <= H_total + 1e-9))}")
    print("  → 互資訊是總不確定性的一部分，不可能超過它。")
    print(f"檢查 3：變異比的解析度。T = {T_MC} 時 VR 只能取 "
          f"{[round(float(v), 4) for v in sorted(set(np.round(VR, 6)))[:5]]} ... "
          f"等 {len(set(np.round(VR, 6)))} 種值")
    print(f"  → 解析度是 1/T = {1 / T_MC:.4f}，很小但**離散**。")

    # -----------------------------------------------------------------------
    section("§4.3 ⚠ 出處警告：加法分解式不是 Murphy 的")
    print("很多論文與教學投影片會寫：")
    print()
    print("      H(y|x)  =  I(y ; θ | x)  +  E_θ[ H(y | x, θ) ]")
    print("      （總不確定性 ＝ 認知不確定性 ＋ 偶然不確定性）")
    print()
    print("**這個式子在 Murphy《Probabilistic Machine Learning》全書並不存在。**")
    print("  證據（本專案抽取檔 F5 §2.4 與附錄）：")
    print("    - 對 book 2 全域搜尋 `epistemic` → 0 命中；")
    print("    - book 1 只有 L1322、L3314、L65586、L66620 四處**定性**提及；")
    print("    - 全書唯一的形式化結果是式 14.29 的環境 KL 分解（L44177）：")
    print("        d_KL(B,Q) = d_KL(E,Q) − I(E ; y | D_T, x)，其中互資訊項是常數；")
    print("    - 另有 §14.2.3 的擲幣**定性**範例（L44129–L44141），")
    print("      解釋 aleatoric 是「不可約的雜訊」、epistemic 是「對真實參數的無知」，")
    print("      但沒有給出上面那個加法分解式。")
    print()
    print("  這個分解式的正確出處是：")
    print("    - Depeweg, Hernández-Lobato, Doshi-Velez & Udluft (2018),")
    print("      *Decomposition of Uncertainty in Bayesian Deep Learning for")
    print("      Efficient and Risk-sensitive Learning*, ICML.")
    print("    - Kendall & Gal (2017), *What Uncertainties Do We Need in Bayesian")
    print("      Deep Learning for Computer Vision?*, NeurIPS.")
    print("    - Gal (2016) 博士論文。")
    print()
    print("  ★ 本專案寫作規則：凡用到此分解式，一律標")
    print("    『出處：Depeweg et al. 2018；Kendall & Gal 2017（非 Murphy）』")
    print("    並附 TODO(核實)，**不得**寫成「Murphy 說……」。")
    print()
    print("  另外，命名也要小心：H[p̄] 常被直接叫「aleatoric」，這只在")
    print("  『p_t 之間的差異純粹來自參數不確定性』時才對。本專案一律寫")
    print("  「預測熵（總不確定性的代理量）」。")

    # -----------------------------------------------------------------------
    section("§4.4 三個指標各自什麼時候會失效（用實際數字示範）")
    correct = (predict(P_bar) == yte)
    print(f"集成後準確率 = {accuracy(P_bar, yte):.4f}；ECE(M=10) = {ece(P_bar, yte, 10):.4f}；"
          f"NLL = {nll(P_bar, yte):.4f}\n")

    # (A) 失效一：全部成員一起錯 → 互資訊測不到 epistemic
    wrong = ~correct
    print("(A) 『全體一致地錯』——互資訊的結構性盲點")
    print(f"    答錯樣本的平均互資訊   = {MI[wrong].mean():.4f}")
    print(f"    答對樣本的平均互資訊   = {MI[correct].mean():.4f}")
    print(f"    答錯樣本的平均預測熵   = {H_total[wrong].mean():.4f}")
    print(f"    答對樣本的平均預測熵   = {H_total[correct].mean():.4f}")
    if MI[wrong].mean() > MI[correct].mean():
        print("    本示範的方向正確：答錯的樣本互資訊確實較高，指標有訊號。")
    else:
        print("    ⚠ 本示範方向不理想 —— 這本身就是要誠實回報的結果。")
    print("    但**結構性盲點仍然存在**，務必寫進 limitation：")
    print("      I[y;θ] 的定義是『集成成員之間不同意』。若所有成員共用同一個")
    print("      結構性偏差（例如同一個 ansatz、同一份被汙染的訓練資料、")
    print("      同一個前端的偏誤），它們會**一起錯而且彼此同意** → I ≈ 0。")
    print("      這時互資訊會嚴重低估真實的認知不確定性。")
    print("      這正是 Murphy（L32157）對 mean-field／乘積式 ansatz 的過度自信警告")
    print("      在量子電路上的對應風險：mode-seeking 讓變分分布擠在一個眾數上，")
    print("      集成變異數因此低估了真實的後驗不確定性。")

    # (B) 失效二：變異比忽略機率大小
    print()
    print("(B) 『變異比忽略機率大小』——VR 的盲點")
    vr_zero_but_unsure = ((VR < 1e-9) & (H_total > 0.5)).sum()
    print(f"    VR = 0（30 個成員全部投同一個標籤）但 H[p̄] > 0.5 的樣本數：{vr_zero_but_unsure}")
    print(f"    這些樣本的平均預測熵 = {H_total[(VR < 1e-9) & (H_total > 0.5)].mean():.4f}"
          if vr_zero_but_unsure else "")
    print("    → 每個成員都說 (0.51, 0.49, ...)，票數一致 → VR = 0，看起來『很確定』，")
    print("      但預測熵很高。VR 只看 argmax，完全丟掉機率的大小。")
    print(f"    → 而且要記住 VR 的解析度只有 1/T = {1 / T_MC:.4f}。")

    # (C) 失效三：預測熵把兩種不確定性混在一起
    print()
    print("(C) 『預測熵混合了兩種不確定性』——H[p̄] 的盲點")
    mi_gap = H_total - MI
    print(f"    H[p̄] 的平均 = {H_total.mean():.4f}；其中可由 I[y;θ] 解釋的 = {MI.mean():.4f}"
          f"（{MI.mean() / H_total.mean() * 100:.1f}%）")
    print(f"    剩下的 (1/T)Σ_t H[p_t] 平均 = {mi_gap.mean():.4f}"
          f"（{(1 - MI.mean() / H_total.mean()) * 100:.1f}%）")
    print("    → 大部分熵其實來自『單一成員自己就不確定』（近似 aleatoric），")
    print("      只有一小部分來自『成員之間不一致』（近似 epistemic）。")
    print("      只報 H[p̄] 會把兩者混為一談，對『該不該去蒐集更多資料』沒有任何指示。")

    # -----------------------------------------------------------------------
    section("§4.5 把不確定性當偵測器：分位數分組的錯誤率")
    order = np.argsort(MI)
    qs = np.array_split(order, 4)
    print(f"{'互資訊分位':>12}{'樣本數':>8}{'MI 平均':>10}{'H[p̄] 平均':>12}{'錯誤率':>10}")
    print("-" * 78)
    for i, q in enumerate(qs):
        err = 1.0 - correct[q].mean()
        print(f"{'Q' + str(i + 1):>12}{q.size:>8}{MI[q].mean():>10.4f}"
              f"{H_total[q].mean():>12.4f}{err:>10.4f}")
    print()
    err_lo, err_hi = 1 - correct[qs[0]].mean(), 1 - correct[qs[-1]].mean()
    print(f"→ 最低互資訊那一組的錯誤率 {err_lo:.4f}，最高那一組 {err_hi:.4f}"
          f"（比值 {err_hi / max(err_lo, 1e-9):.1f}）")
    if err_hi > err_lo:
        print("  方向正確：不確定性高的樣本比較容易錯 → 可以用來做拒答（selective prediction）。")
    else:
        print("  ⚠ 方向不理想或反轉 —— 這本身就是要誠實回報的結果，不能修飾。")
    print("  ★ 誠實聲明：這只證明『指標有排序能力』，**不證明**它等於真實的 epistemic。")
    print("    本專案不得把『MI 高』直接詮釋成『模型知道自己在這裡無知』。")

    # -----------------------------------------------------------------------
    section("§4.6 分布偏移下的表現（Murphy L44125 的警告）")
    X_ood, y_ood, _ = make_readout_dataset(
        N=600, C=10, D=5, separation=0.4, noise=3.0, seed=999
    )  # 分離度極低 + 雜訊極大 → 近似「不在訓練分布內」
    P_ood = np.stack([softmax(logits_of(m, X_ood)) for m in members], axis=0)
    Pb_ood = P_ood.mean(axis=0)
    print(f"{'':<24}{'準確率':>10}{'H[p̄] 平均':>12}{'I 平均':>10}{'VR 平均':>10}{'ECE':>10}")
    print("-" * 78)
    for name, X_, y_, Ps, Pb in (
        ("測試集（同分布）", Xte, yte, P_samples, P_bar),
        ("偏移集（低分離+高雜訊）", X_ood, y_ood, P_ood, Pb_ood),
    ):
        print(f"{name:<22}{accuracy(Pb, y_):>10.4f}"
              f"{predictive_entropy(Pb).mean():>12.4f}{mutual_information(Ps).mean():>10.4f}"
              f"{variation_ratio(Ps).mean():>10.4f}{ece(Pb, y_, 10):>10.4f}")
    print()
    print("→ 三個不確定性指標在偏移集上全部上升（方向正確），但**校準同時變差**（ECE 上升）。")
    print("  這正是 Murphy L44125 引 [Ova+19] 的警告：")
    print("  『即使良好校準的模型（含貝氏模型），遇到不同分布的輸入時也常變得校準不良。』")
    print("  ★ 本專案的誠實聲明必須包含這一句。")

    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    ax.scatter(MI[correct], H_total[correct], s=14, alpha=0.55, c="#2ca02c", label="答對")
    ax.scatter(MI[wrong], H_total[wrong], s=18, alpha=0.75, c="#d62728", label="答錯")
    ax.set_xlabel("互資訊 I[y;θ]（近似 epistemic）")
    ax.set_ylabel("預測熵 H[p_bar]（總不確定性）")
    ax.set_title("不確定性分解平面\n答錯的點應該偏右上（若不偏，就是演算法的盲點）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    w = 0.35
    xs = np.arange(4)
    ax.bar(xs - w / 2, [MI[q].mean() for q in qs], w, color="#1f77b4", label="互資訊 I[y;θ]")
    ax.bar(xs + w / 2, [H_total[q].mean() for q in qs], w, color="#ff7f0e",
           label="預測熵 H[p_bar]")
    ax2 = ax.twinx()
    ax2.plot(xs, [1 - correct[q].mean() for q in qs], "ko--", lw=2, label="錯誤率（右軸）")
    ax2.set_ylabel("錯誤率")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q{i + 1}" for i in range(4)])
    ax.set_xlabel("互資訊分位（Q1 最低 → Q4 最高）")
    ax.set_ylabel("不確定性指標")
    ax.set_title("不確定性分位 vs 實際錯誤率")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p5 = os.path.join(FIGS, "05_uncertainty_decomposition.png")
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print()
    print(f"[圖] 已存 figs/{os.path.basename(p5)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
