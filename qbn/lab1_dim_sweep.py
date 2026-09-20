"""Lab 1：維度瓶頸掃描（純古典探針）。

研究問題
--------
RQ-dim：把 potion-multilingual-128M 的 256 維句向量降到 5 維，語意損失有多大？
        這個損失比文獻指出的 $d=64$ 斷崖更嚴重嗎？

方法
----
1. 對雙語情感語料（80 句繁中 + 80 句英文）取 256 維嵌入。
2. 比較三種降維路徑（**全部只在訓練摺內 fit，避免資料洩漏**）：
   - `pca_first_k`：直接取 PCA 前 k 個主成分（**直覺做法，懷疑是錯的**）
   - `abtt_k`：丟棄前 2 個主成分（ABTT）後取 k 個，再逐維標準化（**規格常數建議的做法**）
   - `pca_std_k`：PCA 後逐維標準化但不丟棄（對照組）
3. 對每個 $k \in \{2,4,5,8,16,32,64,128,256\}$ 訓練邏輯斯迴歸，用**分層 5 摺交叉驗證**報告準確率。
4. 對照組：不降維（$k=256$）的邏輯斯迴歸，作為天花板。

決策規則（先寫下來，避免事後合理化）
------------------------------------
- 若 $k=5$ 的準確率 < 0.70 → **確認 5 維角度編碼不可行**，必須改用資料重上傳或振幅編碼。
- 若 $k=32$ 仍 >= 0.90 → 振幅編碼是可行的替代路線。
- 若 `abtt_k` 在所有 k 都顯著優於 `pca_first_k` → 證實 ABTT 的必要性（PC1/PC2 是語言身分）。
- 若 $k=32$ 與 $k=256$ 差距 < 0.03 → 瓶頸不在維度，而在別的地方（值得寫進論文）。

執行（WSL 內，需要 model2vec）：
    /root/qbn/.venv/bin/python dev/lab1_dim_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "results" / "lab1"
CACHE.mkdir(parents=True, exist_ok=True)

MODEL_ID = "minishlab/potion-multilingual-128M"
K_VALUES = [2, 4, 5, 8, 16, 32, 64, 128, 256]
N_FOLDS = 5
SEED = 42


# ---------------------------------------------------------------------------
# 1. 取得嵌入（有快取，重跑不用重新編碼）
# ---------------------------------------------------------------------------
def get_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emb_file = CACHE / "embeddings.npz"
    if emb_file.exists():
        d = np.load(emb_file, allow_pickle=True)
        print(f"  從快取載入 {emb_file.name}")
        return d["X"], d["y"], d["lang"]

    from data_bilingual_sentiment import load_dataset
    from model2vec import StaticModel

    texts, labels, langs = load_dataset()
    print(f"  語料：{len(texts)} 句（繁中 {langs.count('zh')}、英文 {langs.count('en')}）")
    print(f"  載入 {MODEL_ID} ...")
    t0 = time.perf_counter()
    model = StaticModel.from_pretrained(MODEL_ID)
    print(f"  載入耗時 {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    X = np.asarray(model.encode(texts), dtype=np.float64)
    print(f"  編碼 {len(texts)} 句耗時 {time.perf_counter() - t0:.2f} s")
    print(f"  嵌入維度 = {X.shape}")

    y = np.asarray(labels, dtype=np.int64)
    lang = np.asarray(langs)
    np.savez_compressed(emb_file, X=X, y=y, lang=lang)
    print(f"  已快取到 {emb_file}")
    return X, y, lang


# ---------------------------------------------------------------------------
# 2. 三種降維路徑（都只在訓練摺上 fit）
# ---------------------------------------------------------------------------
def fit_pca(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (mean, components)，components 的列是主成分（已排序）。"""
    mu = Xtr.mean(axis=0, keepdims=True)
    Xc = Xtr - mu
    # 用 SVD 做 PCA（比共變異數矩陣的 eigen 分解數值上更穩）
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu.ravel(), Vt


def project(X: np.ndarray, mu: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """投影到主成分空間（全部主成分）。"""
    return (X - mu) @ Vt.T


def reduce_features(
    Xtr: np.ndarray, Xte: np.ndarray, k: int, mode: str, abtt_drop: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """三種降維路徑。**只使用 Xtr 統計量**，避免資料洩漏。"""
    mu, Vt = fit_pca(Xtr)
    Ztr = project(Xtr, mu, Vt)
    Zte = project(Xte, mu, Vt)

    if mode == "pca_first_k":
        Atr, Ate = Ztr[:, :k], Zte[:, :k]
        # 不做標準化（這是直覺做法）
        return Atr, Ate

    if mode == "pca_std_k":
        Atr, Ate = Ztr[:, :k], Zte[:, :k]

    elif mode == "abtt_k":
        # ABTT：丟棄前 abtt_drop 個主成分，再取 k 個
        if k + abtt_drop > Ztr.shape[1]:
            k_eff = max(1, Ztr.shape[1] - abtt_drop)
        else:
            k_eff = k
        Atr = Ztr[:, abtt_drop:abtt_drop + k_eff]
        Ate = Zte[:, abtt_drop:abtt_drop + k_eff]
    else:
        raise ValueError(mode)

    # 逐維標準化（只用訓練摺的統計量）
    s = Atr.std(axis=0)
    s[s < 1e-12] = 1.0
    return (Atr - Atr.mean(axis=0)) / s, (Ate - Ate.mean(axis=0)) / s


# ---------------------------------------------------------------------------
# 3. 交叉驗證的邏輯斯迴歸（純 NumPy，避免額外依賴）
# ---------------------------------------------------------------------------
def fit_logreg(
    X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 400, lr: float = 0.5
) -> tuple[np.ndarray, float]:
    """梯度下降的 L2 正則邏輯斯迴歸（含截距）。"""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        gw = X.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return w, float(b)


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % n_folds].append(ix)
    return [np.asarray(sorted(f)) for f in folds]


def cross_val_accuracy(
    X: np.ndarray, y: np.ndarray, k: int, mode: str, folds: list[np.ndarray]
) -> tuple[float, float, np.ndarray]:
    """回傳 (平均準確率, 標準差, 每摺準確率)。"""
    accs = []
    for i in range(len(folds)):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        Atr, Ate = reduce_features(X[tr], X[te], k, mode)
        w, b = fit_logreg(Atr, y[tr].astype(float))
        pred = (predict_proba(Ate, w, b) >= 0.5).astype(int)
        accs.append(float((pred == y[te]).mean()))
    a = np.asarray(accs)
    return float(a.mean()), float(a.std()), a


def main() -> None:
    print("=" * 84)
    print("Lab 1：維度瓶頸掃描（純古典線性探針）")
    print("=" * 84)

    X, y, lang = get_embeddings()
    folds = stratified_folds(y, N_FOLDS, SEED)
    print(f"\n  {N_FOLDS} 摺分層交叉驗證，每摺測試集大小 = "
          f"{[len(f) for f in folds]}")
    print(f"  隨機基準線 = {max(y.mean(), 1 - y.mean()):.4f}"
          f"（類別平衡時為 0.5）")

    results: dict = {"k_values": K_VALUES, "modes": {}, "lang": lang.tolist()}

    for mode in ("pca_first_k", "abtt_k", "pca_std_k"):
        print(f"\n{'-' * 84}")
        label = {
            "pca_first_k": "直接取 PCA 前 k 維（不標準化）← 直覺做法",
            "abtt_k": f"ABTT（丟棄前 2 個主成分）+ 逐維標準化 ← 規格常數建議",
            "pca_std_k": "PCA 前 k 維 + 逐維標準化（對照）",
        }[mode]
        print(f"路徑：{label}")
        print(f"{'-' * 84}")
        print(f"  {'k':>5}  {'準確率':>10}  {'標準差':>8}  每摺")
        rows = []
        for k in K_VALUES:
            if mode == "abtt_k" and k + 2 > X.shape[1]:
                continue
            if k > X.shape[1]:
                continue
            m, s, a = cross_val_accuracy(X, y, k, mode, folds)
            rows.append({"k": k, "mean": m, "std": s, "folds": a.tolist()})
            bar = "█" * int(round(m * 40))
            print(f"  {k:>5}  {m:>10.4f}  {s:>8.4f}  "
                  f"{' '.join(f'{v:.2f}' for v in a)}  {bar}")
        results["modes"][mode] = rows

    # --- 決策規則判定 ---
    print("\n" + "=" * 84)
    print("依事前決策規則判定")
    print("=" * 84)

    abtt = {r["k"]: r["mean"] for r in results["modes"]["abtt_k"]}
    first = {r["k"]: r["mean"] for r in results["modes"]["pca_first_k"]}

    print(f"\n  規則 1：k=5 的準確率 < 0.70？")
    acc5 = abtt.get(5, float("nan"))
    print(f"     ABTT k=5 準確率 = {acc5:.4f} → "
          f"{'✅ 確認 5 維角度編碼不可行' if acc5 < 0.70 else '❌ 5 維竟然可行，需重新評估'}")

    print(f"\n  規則 2：k=32 的準確率 >= 0.90？")
    acc32 = abtt.get(32, float("nan"))
    print(f"     ABTT k=32 準確率 = {acc32:.4f} → "
          f"{'✅ 振幅編碼是可行替代路線' if acc32 >= 0.90 else '❌ 振幅編碼也不夠'}")

    print(f"\n  規則 3：ABTT 在所有 k 都優於「直接取前 k 維」？")
    common = sorted(set(abtt) & set(first))
    wins = [(k, abtt[k] - first[k]) for k in common]
    n_win = sum(1 for _, d in wins if d > 0)
    print(f"     ABTT 勝 {n_win}/{len(wins)} 個 k 值")
    for k, d in wins:
        mark = "✅" if d > 0 else "❌"
        print(f"       k={k:>3}: ABTT {abtt[k]:.4f} vs 直接 {first[k]:.4f}  "
              f"Δ={d:+.4f} {mark}")

    print(f"\n  規則 4：k=32 與 k=256 差距 < 0.03？")
    gap = abtt.get(256, float("nan")) - acc32
    print(f"     ABTT k=256 = {abtt.get(256, float('nan')):.4f}，"
          f"k=32 = {acc32:.4f}，差距 = {gap:.4f} → "
          f"{'✅ 瓶頸不在 32 維，在別的地方' if gap < 0.03 else '❌ 32 維確實仍是瓶頸'}")

    out = CACHE / "lab1_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  結果已存到 {out}")

    # --- 語言分解（額外分析）---
    print("\n" + "=" * 84)
    print("額外分析：語言分解（僅 ABTT k=5 與 k=32）")
    print("=" * 84)
    for k in (5, 32):
        for lang_name, mask in (("繁中", lang == "zh"), ("英文", lang == "en")):
            Xs, ys = X[mask], y[mask]
            f = stratified_folds(ys, min(N_FOLDS, len(np.unique(ys)) * 2), SEED)
            m, s, _ = cross_val_accuracy(Xs, ys, k, "abtt_k", f)
            print(f"  k={k:>3}  {lang_name}：準確率 {m:.4f} ± {s:.4f}  "
                  f"（n={int(mask.sum())}）")

    print("\nDONE")


if __name__ == "__main__":
    main()
