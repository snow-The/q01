"""
mlkit.py — QBN 專題「AI／ML 評估」共用工具庫
=================================================

設計原則
--------
1. **只用 NumPy**（+ matplotlib 只在畫圖的腳本裡）。不依賴 scikit-learn／scipy，
   讓物理系同學在任何環境都能跑。
2. 所有函式都在同一組介面上操作：
      P : 預測機率矩陣，shape (N, C)，每一列和為 1
      y : 真標籤，shape (N,)，整數 0..C-1
      信心 conf = P.max(axis=1)，預測標籤 yhat = P.argmax(axis=1)
3. 每個函式都附「這個函式在防什麼錯」的註解，因為本專案踩過的坑都在這裡。

對應文件
--------
Murphy, *Probabilistic Machine Learning: Advanced Topics*（F5 抽取檔行號）
  - proper scoring rule          L43955–L43969（式 14.14–14.15）
  - Brier score                  L43977（式 14.16）
  - 分箱準確率 acc(B_b)          L44012（式 14.17）
  - 分箱平均信心 conf(B_b)       L44018（式 14.18）
  - ECE                          L44024（式 14.19）
  - MCE                          L44030–L44038（式 14.20–14.21）
  - 溫度縮放                     L44058 / L44113
  - aleatoric vs epistemic       L44129–L44141
Murphy, *Probabilistic Machine Learning: An Introduction*（F4 抽取檔行號）
  - log loss / proper scoring    L13976
  - Brier score                  L13992
  - NLL（二元）                  L25667；多類 L27190
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12

# =============================================================================
# 0. 機率工具
# =============================================================================


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """數值穩定的 softmax。先減最大值，避免 exp 溢位。"""
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """log softmax，用 log-sum-exp 寫法。算 NLL 時比 log(softmax(z)) 穩定。"""
    z = np.asarray(z, dtype=float)
    m = z.max(axis=axis, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=axis, keepdims=True))


def onehot(y: np.ndarray, C: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    Y = np.zeros((y.size, C))
    Y[np.arange(y.size), y] = 1.0
    return Y


def confidence(P: np.ndarray) -> np.ndarray:
    """信心 conf_i = max_c P[i,c]（Murphy L44010–L44022）。"""
    return P.max(axis=1)


def predict(P: np.ndarray) -> np.ndarray:
    """MAP 標籤 yhat_i = argmax_c P[i,c]（Murphy L44010）。"""
    return P.argmax(axis=1)


# =============================================================================
# 1. 分類指標：準確率、精確率、召回率、F1
# =============================================================================


def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    return float((predict(P) == y).mean())


def confusion_matrix(P: np.ndarray, y: np.ndarray, C: int | None = None) -> np.ndarray:
    """回傳 C×C 混淆矩陣，cm[i, j] = 真實為 i、被預測為 j 的樣本數。"""
    y = np.asarray(y, dtype=int)
    C = C or int(max(y.max(), predict(P).max())) + 1
    yhat = predict(P)
    cm = np.zeros((C, C), dtype=int)
    np.add.at(cm, (y, yhat), 1)
    return cm


def prf_per_class(P: np.ndarray, y: np.ndarray, C: int | None = None) -> dict:
    """
    每個類別的 precision / recall / F1 / support。

    定義（多類別、one-vs-rest）：
      precision_c = TP_c / (TP_c + FP_c)    「說它是 c 的時候，有多少真的是 c」
      recall_c    = TP_c / (TP_c + FN_c)    「真的是 c 的時候，抓到多少」
      F1_c        = 2PR / (P + R)           調和平均
    """
    cm = confusion_matrix(P, y, C)
    C = cm.shape[0]
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, EPS), 0.0)
        rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, EPS), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, EPS), 0.0)

    w = support / max(support.sum(), EPS)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * w).sum()),
        "micro_f1": float(np.diag(cm).sum() / max(cm.sum(), 1)),
    }


# =============================================================================
# 2. 嚴格評分規則：NLL 與 Brier
# =============================================================================


def nll(P: np.ndarray, y: np.ndarray, from_logits: bool = False) -> float:
    """
    負對數似然（negative log-likelihood, NLL）＝ 多類別交叉熵：

        NLL = -(1/N) * sum_n log P[n, y_n]        （Murphy F4 L27190）

    它是**嚴格評分規則（strictly proper scoring rule）**：
    最小值唯一地在「P 等於真實機率」時達成（Murphy F4 L13976；F5 L43955）。
    因此它無法被「講話大聲」騙過去，而準確率可以。

    P 很小的時候 -log P 會爆掉 —— 這正是 NLL **無界**的來源，
    也是它對「自信地答錯」懲罰最重的原因。
    """
    y = np.asarray(y, dtype=int)
    if from_logits:
        lp = log_softmax(P)
    else:
        lp = np.log(np.clip(np.asarray(P, float), EPS, None))
    return float(-lp[np.arange(y.size), y].mean())


def brier(P: np.ndarray, y: np.ndarray, C: int | None = None, divide_by_C: bool = False) -> float:
    """
    Brier 分數（越小越好）：

        BS = (1/N) * sum_n sum_c (P[n,c] - onehot(y_n)[c])^2      （Murphy F4 L13992）

    Murphy 的《Advanced Topics》寫成

        S = -(1/C) * sum_c (p_theta(y=c|x) - I(y=c))^2            （式 14.16，L43977）

    兩者只差「除以 C」與「正負號」。**報 Brier 時一定要寫清楚用哪個慣例**，
    否則 0.28 與 0.028 會被誤讀成兩個數量級。
    本函式預設回傳 F4 的慣例（不除 C）；divide_by_C=True 時回傳式 14.16 的量級。

    與 NLL 的差別：Brier **有界**（單筆介於 0 與 2 之間，除以 C 後介於 0 與 2/C 之間），
    NLL **無界**。所以 Brier 對「罕見類別被壓到極低機率」較不敏感、
    對極端錯誤較不敏感，但也因此對尾端機率的錯誤比較不痛。
    """
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, float)
    C = C or P.shape[1]
    Y = onehot(y, C)
    per_sample = ((P - Y) ** 2).sum(axis=1)
    out = float(per_sample.mean())
    return out / C if divide_by_C else out


def brier_decomposition_terms(P: np.ndarray, y: np.ndarray, M: int = 10) -> dict:
    """
    可靠性圖的三個成分（僅供教學對照，數值不保證為標準 Murphy 分解）。
    這裡只示範「Brier 可拆成 可靠性 + 解析度 + 不確定性」的精神。
    """
    conf = confidence(P)
    correct = (predict(P) == y).astype(float)
    bins = _bin_index(conf, M)
    rel = res = 0.0
    ybar = correct.mean()
    for b in range(M):
        m = bins == b
        n = int(m.sum())
        if n == 0:
            continue
        rel += n * (conf[m].mean() - correct[m].mean()) ** 2
        res += n * (correct[m].mean() - ybar) ** 2
    N = y.size
    return {"reliability": rel / N, "resolution": res / N, "uncertainty": ybar * (1 - ybar)}


# =============================================================================
# 3. ECE、可靠性圖、MCE
# =============================================================================


def _bin_index(conf: np.ndarray, M: int) -> np.ndarray:
    """
    等寬分箱：信心落在 I_b = ((b-1)/M, b/M] 者歸入第 b 箱（b = 1..M），
    本函式回傳 0-based 索引 0..M-1。

    ⚠ 邊界：信心恰為 0 的樣本若用 ceil(0*M)=0 會落到 -1，
      故先夾到 [0,1]，並用 ceil 後減 1、再夾到 [0, M-1]。
      Murphy L44008 用的是「均勻 bin 寬」並以 ((b-1)/B, b/B] 定義，
      與 scikit-learn 的 np.digitize 慣例略有差異 —— 這是 ECE 實作
      最常見的「同一份程式碼兩個答案」來源，**必須在論文裡寫死你用哪一種**。
    """
    conf = np.clip(np.asarray(conf, float), 0.0, 1.0)
    idx = np.ceil(conf * M).astype(int) - 1
    return np.clip(idx, 0, M - 1)


def reliability_bins(P: np.ndarray, y: np.ndarray, M: int = 10) -> dict:
    """
    回傳每個非空 bin 的：樣本數、平均信心 conf(B_b)、經驗準確率 acc(B_b)、
    以及落差 |acc - conf|。

    acc(B_b)  = (1/|B_b|) sum_{n in B_b} 1[yhat_n = y_n]     （式 14.17，L44012）
    conf(B_b) = (1/|B_b|) sum_{n in B_b} phat_n              （式 14.18，L44018）
    """
    P = np.asarray(P, float)
    y = np.asarray(y, dtype=int)
    conf = confidence(P)
    correct = (predict(P) == y).astype(float)
    bins = _bin_index(conf, M)

    rows = []
    for b in range(M):
        m = bins == b
        n = int(m.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin": b + 1,
                "lo": b / M,
                "hi": (b + 1) / M,
                "count": n,
                "conf": float(conf[m].mean()),
                "acc": float(correct[m].mean()),
                "gap": float(abs(correct[m].mean() - conf[m].mean())),
            }
        )
    return {
        "rows": rows,
        "conf": np.array([r["conf"] for r in rows]),
        "acc": np.array([r["acc"] for r in rows]),
        "count": np.array([r["count"] for r in rows]),
        "gap": np.array([r["gap"] for r in rows]),
    }


def ece(P: np.ndarray, y: np.ndarray, M: int = 10, denominator: str = "N") -> float:
    """
    期望校準誤差（Expected Calibration Error, ECE）：

        ECE = sum_{b=1}^{M} (|B_b| / N) * |acc(B_b) - conf(B_b)|      （式 14.19，L44024）

    ★★ 分母的陷阱 ★★
    Murphy 抽取檔（F5 L405）把式 14.19 的分母錄成 **B（bin 數）**，
    但這在數學上不成立：|B_b| 是樣本數、加總起來是 N 而不是 B，
    除以 B 會讓 ECE 憑空縮小 M 倍，而且會隨你選幾個 bin 而改變尺度。
    本函式預設 denominator="N"（正確）。
    傳 denominator="B" 可以重現 OCR 那個版本，用來**親眼看到差多少**。

    ★ ECE 的三個已知問題（論文 limitation 一定要寫）
      1. 對分箱數 M 敏感（本指南 §2.6 會實測 M = 5..50 的範圍）。
      2. 有向下偏差（negative bias）：有限樣本下即使模型完美校準，
         ECE 的期望值仍大於 0（每個 bin 內 conf 與 acc 都是雜訊估計）。
      3. 只看 top-1（Murphy L44028）：其餘 C-1 個類別的機率完全沒被檢查，
         類別不平衡時問題會被掩蓋 → 必須同時報 MCE（L44030–L44038）。
    """
    y = np.asarray(y, dtype=int)
    rb = reliability_bins(P, y, M)
    counts = rb["count"].astype(float)
    gaps = rb["gap"]
    if counts.size == 0:
        return float("nan")
    if denominator == "N":
        denom = float(y.size)
    elif denominator == "B":
        denom = float(M)  # 重現 OCR 版本，僅供對照
    else:
        raise ValueError("denominator 只能是 'N' 或 'B'")
    return float((counts / denom * gaps).sum())


def mce(P: np.ndarray, y: np.ndarray, M: int = 10, weights: np.ndarray | None = None) -> float:
    """
    邊際校準誤差（Marginal Calibration Error, MCE；又名 static calibration error）：

        MCE = sum_c w_c * sum_b (|B_{b,c}| / N) * (acc(B_{b,c}) - conf(B_{b,c}))^2
                                                                  （式 14.20–14.21，L44030）

    這裡對**每一個類別 c** 各自做一次 one-vs-rest 的分箱校準，
    再把平方落差依類別權重 w_c 加總。與 ECE 的差別：
      - ECE 只檢查「被選為 top-1 的那個類別」的信心（L44028）；
      - MCE 檢查全部 C 個類別的機率值。
    [Nix+19] 展示了 ECE 很好但 MCE 很差的案例（L44038）。
    """
    P = np.asarray(P, float)
    y = np.asarray(y, dtype=int)
    N, C = P.shape
    if weights is None:
        weights = np.full(C, 1.0 / C)
    weights = np.asarray(weights, float)

    total = 0.0
    for c in range(C):
        pc = P[:, c]
        hit = (y == c).astype(float)
        bins = _bin_index(pc, M)
        acc_c = 0.0
        for b in range(M):
            m = bins == b
            n = int(m.sum())
            if n == 0:
                continue
            acc_c += (n / N) * (hit[m].mean() - pc[m].mean()) ** 2
        total += weights[c] * acc_c
    return float(total)


def classwise_ece(P: np.ndarray, y: np.ndarray, M: int = 10) -> np.ndarray:
    """
    類別化 ECE（class-wise ECE）：對每個類別 c 做一次 **one-vs-rest** 的校準檢查，
        ECE_c = sum_b (|B_{b,c}| / N) * |acc(B_{b,c}) - conf(B_{b,c})|
    其中第 c 類的「信心」是 P[:, c]、「正確」是 1[y == c]，**用全部 N 筆樣本**分箱。

    ⚠ 不要寫成「只取真實標籤為 c 的子集」：那樣 1[y==c] 恆為 1，
      acc(B) 全部等於 1，ECE_c 會退化成 1 - mean(P[:,c])，失去意義。
      這裡刻意用 one-vs-rest 全樣本版本，與 MCE 的定義保持一致。

    回傳長度 C 的陣列。用來抓「整體 ECE 很好，但某一類的機率爛掉」的情形。
    """
    P = np.asarray(P, float)
    y = np.asarray(y, dtype=int)
    N, C = P.shape
    out = np.full(C, np.nan)
    for c in range(C):
        pc = P[:, c]
        hit = (y == c).astype(float)
        bins = _bin_index(pc, M)
        val = 0.0
        for b in range(M):
            m = bins == b
            n = int(m.sum())
            if n == 0:
                continue
            val += (n / N) * abs(hit[m].mean() - pc[m].mean())
        out[c] = val
    return out


# =============================================================================
# 4. 溫度縮放（temperature scaling）
# =============================================================================


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """q = softmax(z / T)（Murphy 式見 L44058）。T > 1 讓分布變平（less peaky）。"""
    return softmax(np.asarray(logits, float) / float(T))


def fit_temperature(
    logits_val: np.ndarray,
    y_val: np.ndarray,
    grid: np.ndarray | None = None,
    refine: bool = True,
) -> float:
    """
    在**驗證集**上用最大似然（＝最小化 NLL）估計 T > 0。

    作法是先粗算對數網格，再在最佳點附近做黃金分割細修。
    ⚠ 三個必踩的坑：
      1. **只能用驗證集**，用測試集調 T 就是資料洩漏。
      2. T 是**單一純量**，不可能造成嚴重過擬合 —— 這是它比
         矩陣縮放（K×K 參數）安全的主因（Murphy L44046）。
      3. T 只改變機率，**不改變 argmax**（L44113）→ 準確率完全不動。
    """
    logits_val = np.asarray(logits_val, float)
    y_val = np.asarray(y_val, dtype=int)
    if grid is None:
        grid = np.exp(np.linspace(math.log(0.05), math.log(50.0), 400))

    nlls = np.array([nll(apply_temperature(logits_val, t), y_val) for t in grid])
    best = float(grid[int(nlls.argmin())])

    if refine:
        lo = float(grid[max(int(nlls.argmin()) - 1, 0)])
        hi = float(grid[min(int(nlls.argmin()) + 1, len(grid) - 1)])
        gr = (math.sqrt(5.0) - 1.0) / 2.0  # 黃金比例
        a, b = lo, hi
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc = nll(apply_temperature(logits_val, c), y_val)
        fd = nll(apply_temperature(logits_val, d), y_val)
        for _ in range(60):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = nll(apply_temperature(logits_val, c), y_val)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = nll(apply_temperature(logits_val, d), y_val)
            if abs(b - a) < 1e-8:
                break
        best = 0.5 * (a + b)
    return float(best)


# =============================================================================
# 5. 不確定性分解：預測熵、互資訊、變異比
# =============================================================================


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def predictive_entropy(P_bar: np.ndarray) -> np.ndarray:
    """
    預測熵（total uncertainty）H[ p̄ ]，p̄ = (1/T) sum_t p_t。

    ⚠ 命名紀律（本專案最重要的一條誠實規則）
    很多文獻把 H[p̄] 直接叫「aleatoric（偶然）不確定性」。**這只在
    「p_t 之間的差異純粹來自參數不確定性」時才對**。實務上 H[p̄] 是
    **總不確定性**；要拆出 aleatoric 必須再減掉互資訊。
    本專案一律寫「預測熵（總不確定性的代理量）」。
    """
    return _entropy(P_bar)


def mutual_information(P_samples: np.ndarray) -> np.ndarray:
    """
    互資訊（BALD）I[y ; theta | x]
        = H[ (1/T) sum_t p_t ]  -  (1/T) sum_t H[p_t]

    第一項是總不確定性，第二項是（在固定參數下）仍存在的
    「單次預測本身的分散度」，兩者相減得到「因為參數不確定而多出來的不確定性」。
    這是**變異數分解的精神**，但請注意下一個函式的警告。

    ⚠⚠ 出處警告（本專案強制標註）
    常見的**加法分解式**
        H(y|x)  =  I(y ; theta | x)  +  E_theta[ H(y | x, theta) ]
    （總不確定性 = 認知不確定性 + 偶然不確定性）
    **在 Murphy《Probabilistic Machine Learning》全書並不存在**：
    對 book 2 全域搜尋 `epistemic` 得 **0 命中**；書中唯一的
    aleatoric/epistemic 形式化結果是式 14.29 的環境 KL 分解
    d_KL(B,Q) = d_KL(E,Q) - I(E ; y | D_T, x)（L44177），
    以及 §14.2.3 的擲幣定性範例（L44129–L44141）。
    這個分解式的真正出處是：
      - Depeweg, Hernández-Lobato, Doshi-Velez & Udluft (2018),
        *Decomposition of Uncertainty in Bayesian Deep Learning for
        Efficient and Risk-sensitive Learning*, ICML.
      - Kendall & Gal (2017), *What Uncertainties Do We Need in Bayesian
        Deep Learning for Computer Vision?*, NeurIPS.
      - Gal (2016) 博士論文。
    → 本專案凡使用此分解式，一律標
      「出處：Depeweg et al. 2018；Kendall & Gal 2017（非 Murphy）」+
      `TODO(核實)`，並且**不得**寫成「Murphy 說……」。

    本函式以 MC 樣本 (T, N, C) 為輸入，回傳長度 N 的互資訊。
    """
    P_samples = np.asarray(P_samples, float)  # (T, N, C)
    T = P_samples.shape[0]
    P_bar = P_samples.mean(axis=0)
    H_total = _entropy(P_bar)
    H_expected = _entropy(P_samples).mean(axis=0)
    return H_total - H_expected


def variation_ratio(P_samples: np.ndarray) -> np.ndarray:
    """
    變異比（variation ratio）＝ 1 - (眾數標籤的出現次數 / T)。

    直觀意義：「T 次預測裡有多少比例不同意多數決」。
    ⚠ 失效情形：當 T 很小（本專案 T = 30），它的解析度只有 1/T ≈ 0.033，
    而且**完全忽略機率的大小** —— 一個每次都說 (0.51, 0.49) 的模型
    變異比是 0，看起來「很確定」，其實非常不確定。
    因此它只能當輔助指標，不能單獨作為 epistemic 的量測。
    """
    P_samples = np.asarray(P_samples, float)
    T = P_samples.shape[0]
    votes = P_samples.argmax(axis=2)  # (T, N)
    counts = np.zeros((T, votes.shape[1], P_samples.shape[2]))
    np.put_along_axis(counts, votes[:, :, None], 1.0, axis=2)
    return 1.0 - counts.sum(axis=0).max(axis=1) / T


# =============================================================================
# 6. 統計工具（純 NumPy，無 scipy）
# =============================================================================


def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-16) -> float:
    """連分數展開，用於正則化不完全 beta 函數（Numerical Recipes 6.4 風格）。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正則化不完全 beta 函數 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """P(T > t)，T ~ t(df)。雙尾 p 值 = 2 * student_t_sf(|t|, df)。"""
    x = df / (df + t * t)
    p = 0.5 * betainc(df / 2.0, 0.5, x)
    return p if t > 0 else 1.0 - p


def paired_t_test(a: np.ndarray, b: np.ndarray) -> dict:
    """
    配對 t 檢定（同一個種子下兩個模型的差異）。
    回傳 t 統計量、自由度、雙尾 p 值、以及配對效果量 Cohen's d_z。

    ★ 為什麼 5 個種子的 p 值不能當強證據 ★
      n = 5 時 df = 4，t 分布的尾端極厚：要達到 p < 0.05 需要 |t| > 2.776。
      檢定力（power）低到即使真實效果中等，也常常測不出來；
      反之若剛好 p < 0.05，那個 p 值本身的抽樣變異也很大（見 §5 的模擬）。
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = d.size
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n)
    t = float(d.mean() / se) if se > 0 else float("nan")
    df = n - 1
    p = float(2 * student_t_sf(abs(t), df)) if np.isfinite(t) else float("nan")
    dz = float(d.mean() / sd) if sd > 0 else float("nan")
    return {"n": n, "mean_diff": float(d.mean()), "sd_diff": sd, "se": se, "t": t, "df": df, "p": p, "dz": dz}


def welch_t_test(a: np.ndarray, b: np.ndarray) -> dict:
    """非配對（Welch）t 檢定。種子無法一一對應時才用這個。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = a.size, b.size
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    t = float((a.mean() - b.mean()) / se) if se > 0 else float("nan")
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = float(2 * student_t_sf(abs(t), df)) if np.isfinite(t) else float("nan")
    return {"t": t, "df": float(df), "p": p}


def cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    """獨立樣本 Cohen's d（用 pooled SD）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = a.size, b.size
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g：小樣本（n < 20）下對 Cohen's d 做偏誤修正。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = a.size + b.size
    d = cohens_d_independent(a, b)
    J = 1.0 - 3.0 / (4.0 * n - 9.0)
    return float(d * J)


def bootstrap_ci(
    sample: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    百分位自助法（percentile bootstrap）信賴區間。
    回傳 (點估計, 下界, 上界)。

    ★ 只有 5 個種子時的建議報告方式 ★
      不要報 p 值當主要證據，改報：
        1. 平均差異 + 自助法 95% 信賴區間
        2. 效果量（配對用 Cohen's d_z；獨立用 Hedges' g）
      自助法在 n = 5 時仍然很粗糙（重抽只有 5^5 = 3125 種可能），
      但它**誠實地把「n 很小」反映在很寬的區間上**，
      而 p 值會把這個不確定性壓縮成一個看起來很果斷的數字。
    """
    rng = np.random.default_rng(seed)
    sample = np.asarray(sample, float)
    n = sample.size
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = statistic(sample[idx], axis=1)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return float(statistic(sample)), lo, hi


# =============================================================================
# 7. 迷你線性 softmax 分類器（純 NumPy，用來產生真實的 logits）
# =============================================================================


def train_softmax(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    C: int | None = None,
    seed: int = 0,
    epochs: int = 400,
    lr: float = 0.05,
    batch: int = 64,
    l2: float = 1e-4,
) -> dict:
    """
    用 Adam 訓練一個線性 + softmax 頭：logits = X @ W + b。

    這正是本專案「5 維量子讀出 → Linear(5→K)」的分類頭，
    也是 §6 參數預算裡「5→10 softmax 頭 = 60 個參數」的那個模型。
    """
    X = np.asarray(X, float)
    y = np.asarray(y, dtype=int)
    N, D = X.shape
    C = C or int(y.max()) + 1
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0 / math.sqrt(D), size=(D, C))
    b = np.zeros(C)

    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    hist = []

    for ep in range(epochs):
        perm = rng.permutation(N)
        for s in range(0, N, batch):
            idx = perm[s : s + batch]
            xb, yb = X[idx], y[idx]
            Z = xb @ W + b
            P = softmax(Z)
            Y = onehot(yb, C)
            G = (P - Y) / xb.shape[0]
            gW = xb.T @ G + l2 * W
            gb = G.sum(axis=0)

            step += 1
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW**2
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * gb**2
            mW_hat = mW / (1 - beta1**step)
            vW_hat = vW / (1 - beta2**step)
            mb_hat = mb / (1 - beta1**step)
            vb_hat = vb / (1 - beta2**step)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        if X_val is not None:
            hist.append(nll(softmax(X_val @ W + b), y_val))
    return {"W": W, "b": b, "C": C, "val_nll_history": np.array(hist)}


def logits_of(model: dict, X: np.ndarray) -> np.ndarray:
    return np.asarray(X, float) @ model["W"] + model["b"]


def n_params_softmax_head(D: int, C: int, bias: bool = True) -> int:
    return D * C + (C if bias else 0)


# =============================================================================
# 8. 合成資料：5 個 <Z> 讀出值 → 10 類
# =============================================================================


def make_readout_dataset(
    N: int = 3000,
    C: int = 10,
    D: int = 5,
    separation: float = 2.6,
    noise: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    產生一份「量子讀出風格」的合成資料集。

    為什麼要合成資料而不是用 MNIST：
      - 本指南要示範的是**評估方法**，不是模型能力；
      - 合成資料的 ground truth 完全可控，可以精確檢查
        「準確率相同但校準不同」這類現象。
      - 仍然刻意對齊本專案的形狀：D = 5（5 個 <Z_i>，值域 [-1, 1]）、
        C = 10（10 類），所以 Linear(5→10) 就是 60 個參數。

    回傳 (X, y, centers)，X 已夾到 [-1, 1]（模擬 <Z> 的物理值域）。
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(C, D))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= separation

    y = np.repeat(np.arange(C), N // C)
    X = centers[y] + rng.normal(0.0, noise, size=(y.size, D))
    X = np.clip(X, -1.0, 1.0)  # <Z> 的物理值域

    perm = rng.permutation(y.size)
    return X[perm], y[perm], centers
