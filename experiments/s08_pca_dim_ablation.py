#!/usr/bin/env python3
"""s08_pca_dim_ablation.py — 用 PCA 統一多模型的維度消融（DIM1）

研究問題
--------
正式學術評估（`學術標準評估.md` §3.2）指出一個實驗設計缺陷：
`potion-multilingual-128M` 是**固定 256 維**，無法做維度消融；只有 `bekko`
支援 Matryoshka（可截 256/128/64）。若只在 bekko 內部做，**維度就不是共同的
受控變數**。

本腳本的核心想法：**維度不該依賴模型是否支援 Matryoshka**。
對所有模型統一套用 PCA 降維，讓 $k$ 成為共同的受控變數。

四個處理臂（全部只在訓練摺 fit，避免資料洩漏）
----------------------------------------------
| 臂 | 做法 |
|---|---|
| `pca_first_k` | 直接取 PCA 前 $k$ 維（直覺做法） |
| `abtt_k`      | 丟棄前 2 個主成分（ABTT）→ 逐維標準化 → 取 $k$ 維 |
| `abtt_nostd_k`| 丟棄前 2 個主成分但不標準化（隔離「標準化」的貢獻） |
| `pca_std_k`   | PCA 前 $k$ 維 + 逐維標準化（對照） |
| `trunc_k`     | 不做 PCA，直接截原生嵌入前 $k$ 維後 L2 正規化（＝ bekko 的 Matryoshka 路徑） |

ABTT 依據（文獻）
----------------
`docs/_extract/F10-static-word-embeddings-sentence.md`【L49】：
丟棄前 $r$ 個主成分，保留第 $(r+1)$…$(r+d')$ 個，$r \\le \\lfloor d/100 \\rfloor$。
$d = 256 \\Rightarrow r \\le 2$（potion / M2V），$d = 384 \\Rightarrow r \\le 3$（bekko）。
本腳本主臂固定 $r = 2$（讓 $r$ 也是共同常數），並對 bekko 另外做 $r$ 的敏感度檢查。
【L325】PC1/PC2 主要捕捉**語言身分**；移除後 en–zh F1 由 39.2 → 88.6（【L148】）。

資料集
------
`dev/data_bilingual_sentiment.py`：160 句（繁中 80 + 英文 80），二元情感標籤，
類別平衡（正 80 / 負 80），**不需下載**。

三個探針任務
------------
1. `sentiment_cv`（主）：混合語言情感分類，分層 5 摺交叉驗證 × 5 個種子。
2. `langid_cv`：語言辨識（zh vs en）——用來**直接檢驗「PC1/PC2 = 語言身分」**。
3. `transfer`：跨語言遷移（zh→en、en→zh）——F10 的 ABTT 效果（+49.4）是
   **跨語言**任務的數字，不是單語言任務，所以必須分開量。

★ 已知限制（誠實標示，不隱藏）
------------------------------
訓練摺樣本數 $n_{tr} = 128$，PCA 的秩上限是 $n_{tr} - 1 = 127$。
因此 **$k = 256$ 這一格在 PCA 路徑上不可能真正達成**，腳本會印出實際 $k_{eff}$
並另外提供「不降維（原始全維）」的探針作為天花板。
這是小語料的產物；C2 階段（MASSIVE zh-TW，11514 筆訓練）沒有這個限制。

執行（一律在 WSL2 內）
----------------------
    wsl -d QBN -u root -- bash -c "cd <repo> && \\
        /root/qbn/.venv/bin/python experiments/s08_pca_dim_ablation.py"

輸出
----
- `ml/out/s08_pca_dim_ablation.txt`（逐字真實輸出，由 Tee 同步寫入）
- `ml/out/s08_results.json`（結構化結果，供報告引用）
- `ml/figs/07_pca_dim_ablation.png` 與 `.pdf`
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 0. 路徑與常數
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../qbn-capacity-calibration/ml
PROJECT = HERE.parent                             # .../qbn-capacity-calibration
REPO = PROJECT.parent.parent                      # .../QBN
DEV = REPO / "dev"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DEV))

import mlkit  # noqa: E402  （專案共用指標：accuracy / prf_per_class）

OUT_DIR = HERE / "out"
FIG_DIR = HERE / "figs"
CACHE_DIR = OUT_DIR / "_cache"
for _d in (OUT_DIR, FIG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TXT_PATH = OUT_DIR / "s08_pca_dim_ablation.txt"
JSON_PATH = OUT_DIR / "s08_results.json"
FIG_PNG = FIG_DIR / "07_pca_dim_ablation.png"
FIG_PDF = FIG_DIR / "07_pca_dim_ablation.pdf"

SEEDS = [7, 21, 42, 84, 168]           # 協定 §5 指定的 5 個種子
N_FOLDS = 5                            # 分層 5 摺交叉驗證
K_GRID = [256, 128, 64, 32, 16, 8]     # 協定要求的 k 網格（由大到小）
ABTT_DROP = 2                          # 主臂固定丟棄前 2 個主成分（d=256 時 r ≤ 2）
C_REG = 1.0                            # L2 邏輯斯迴歸的 C（與 MTEB 預設同口徑）
MAX_NEWTON_ITER = 200
NEWTON_TOL = 1e-11

# 模型註冊表：依協定 §3，三者**都不需要前綴**（potion / bekko / M2V）
MODELS: dict[str, dict] = {
    "potion-multilingual-128M": {
        "hf_id": "minishlab/potion-multilingual-128M",
        "loader": "model2vec",
        "dim_declared": 256,
        "matryoshka": False,
    },
    "bekko-embedding-v1-a25m": {
        "hf_id": "hotchpotch/bekko-embedding-v1-a25m",
        "loader": "sentence_transformers",
        "dim_declared": 384,
        "matryoshka": True,
        "truncate_dims": [256, 128, 64],
    },
    "M2V_multilingual_output": {
        "hf_id": "minishlab/M2V_multilingual_output",
        "loader": "model2vec",
        "dim_declared": 256,
        "matryoshka": False,
    },
}
MODEL_ORDER = ["potion-multilingual-128M", "bekko-embedding-v1-a25m", "M2V_multilingual_output"]
BEKKO = "bekko-embedding-v1-a25m"        # 唯一支援 Matryoshka 的模型（第七節用）

ARMS = ["pca_first_k", "abtt_k", "abtt_nostd_k", "pca_std_k", "trunc_k"]
ARM_LABEL = {
    "pca_first_k": "PCA 前 k 維（不標準化）← 直覺做法",
    "abtt_k": f"ABTT（丟棄前 {ABTT_DROP} 個主成分）+ 逐維標準化",
    "abtt_nostd_k": f"ABTT（丟棄前 {ABTT_DROP} 個主成分）但不標準化",
    "pca_std_k": "PCA 前 k 維 + 逐維標準化（對照）",
    "trunc_k": "不做 PCA：原生嵌入前 k 維 + L2 正規化（bekko 即 Matryoshka）",
}
MODEL_SHORT = {
    "potion-multilingual-128M": "potion-128M",
    "bekko-embedding-v1-a25m": "bekko-a25m",
    "M2V_multilingual_output": "M2V-multiling",
}

T_START = time.perf_counter()
_PRINTED_FOLD_HEADER: set[tuple[str, str]] = set()   # 摺結構只印一次，避免輸出灌水


# ---------------------------------------------------------------------------
# 1. Tee：把 stdout 逐字同步寫進檔案（保證 .txt 是真實輸出，不是事後補寫）
# ---------------------------------------------------------------------------
class Tee:
    """同時寫到終端與檔案的 stdout 代理。"""

    def __init__(self, path: Path) -> None:
        self.file = path.open("w", encoding="utf-8", newline="\n")
        self.stdout = sys.stdout

    def write(self, data: str) -> int:
        self.stdout.write(data)
        self.file.write(data)
        return len(data)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.flush()
            self.file.close()
        finally:
            sys.stdout = self.stdout


def head(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


# ---------------------------------------------------------------------------
# 2. 環境紀錄（可重現性）
# ---------------------------------------------------------------------------
def package_versions() -> dict[str, str]:
    import importlib

    out: dict[str, str] = {}
    for name in ("numpy", "matplotlib", "model2vec", "sentence_transformers",
                 "transformers", "torch", "huggingface_hub", "scipy"):
        try:
            m = importlib.import_module(name)
            out[name] = str(getattr(m, "__version__", "?"))
        except Exception as e:  # noqa: BLE001
            out[name] = f"（未安裝：{type(e).__name__}）"
    return out


def hf_snapshot_revision(hf_id: str) -> str:
    """從 HF 快取目錄讀出實際使用的 snapshot 修訂碼（第一手出處證明）。"""
    slug = "models--" + hf_id.replace("/", "--")
    base = Path.home() / ".cache" / "huggingface" / "hub" / slug / "snapshots"
    if not base.is_dir():
        return "（快取中找不到）"
    revs = sorted(p.name for p in base.iterdir() if p.is_dir())
    return ", ".join(revs) if revs else "（無 snapshot）"


# ---------------------------------------------------------------------------
# 3. 嵌入抽取（含快取；載入失敗一律記錄，不靜默跳過）
# ---------------------------------------------------------------------------
def l2norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n


def truncate_and_norm(X_raw: np.ndarray, k: int) -> np.ndarray:
    """Matryoshka 式截斷：取前 k 維後 L2 正規化（順序與 sentence-transformers 一致）。"""
    return l2norm(X_raw[:, :k])


def encode_with_model(key: str, texts: list[str]) -> tuple[np.ndarray, dict]:
    """回傳 (原生嵌入矩陣, 中介資料)。載入失敗會拋出例外，由呼叫端記錄。"""
    cfg = MODELS[key]
    hf_id = cfg["hf_id"]
    meta: dict = {"hf_id": hf_id, "loader": cfg["loader"]}

    t0 = time.perf_counter()
    if cfg["loader"] == "model2vec":
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(hf_id)
        meta["load_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        X = np.asarray(model.encode(texts), dtype=np.float64)
        meta["encode_s"] = time.perf_counter() - t0
        try:
            meta["n_embedding_params"] = int(np.prod(model.embedding.shape))
            meta["normalize_on_encode"] = bool(getattr(model, "normalize", None))
            meta["model_dim_attr"] = int(getattr(model, "dim", -1))
            meta["base_model_name"] = str(getattr(model, "base_model_name", "?"))
        except Exception as e:  # noqa: BLE001
            meta["n_embedding_params"] = f"（取得失敗：{type(e).__name__}）"
        meta["dim_native"] = int(X.shape[1])
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(hf_id, device="cpu")
        meta["load_s"] = time.perf_counter() - t0
        meta["max_seq_length"] = int(getattr(model, "max_seq_length", -1) or -1)
        meta["st_version_truncate_attr"] = hasattr(model, "truncate_dim")
        t0 = time.perf_counter()
        # 原生輸出：**不正規化**，讓 Matryoshka 截斷能照 ST 的順序（先截再正規化）
        X = np.asarray(
            model.encode(texts, normalize_embeddings=False, convert_to_numpy=True),
            dtype=np.float64,
        )
        meta["encode_s"] = time.perf_counter() - t0
        meta["dim_native"] = int(X.shape[1])

        # --- 實測驗證：ST 內建 truncate_dim 是否等於「手動截前 k 維 + L2 正規化」---
        checks = []
        for k in cfg.get("truncate_dims", []):
            if k >= X.shape[1]:
                continue
            y_st = None
            try:
                model.truncate_dim = k
                y_st = np.asarray(
                    model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
                    dtype=np.float64,
                )
            except Exception as e:  # noqa: BLE001
                checks.append({"k": int(k), "error": f"{type(e).__name__}: {e}"})
                continue
            finally:
                try:                       # 還原（若屬性不可寫，這裡不能把整個程式帶走）
                    model.truncate_dim = None
                except Exception:  # noqa: BLE001
                    pass
            y_manual = truncate_and_norm(X, k)
            same = (y_st.shape == y_manual.shape and
                    bool(np.allclose(y_st, y_manual, atol=1e-9)))
            checks.append({
                "k": int(k),
                "st_shape": list(y_st.shape),
                "manual_shape": list(y_manual.shape),
                "max_abs_diff": float(np.abs(y_st - y_manual).max())
                if y_st.shape == y_manual.shape else float("nan"),
                "equal_1e-9": same,
            })
        meta["truncate_checks"] = checks
    return X, meta


def get_embeddings(key: str, texts: list[str]) -> tuple[np.ndarray, dict]:
    """有快取就用快取；快取不存在才真的跑模型。"""
    cache = CACHE_DIR / f"emb_{key.replace('/', '_')}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        meta["from_cache"] = True
        return d["X"], meta
    X, meta = encode_with_model(key, texts)
    meta["from_cache"] = False
    np.savez_compressed(cache, X=X, meta=json.dumps(meta, ensure_ascii=False))
    return X, meta


# ---------------------------------------------------------------------------
# 4. PCA 與四個處理臂（**只在訓練摺 fit**）
# ---------------------------------------------------------------------------
def fit_pca(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳 (mean, Vt, 奇異值)。Vt 的每一列是一個主成分方向（依變異數排序）。

    ⚠ full_matrices=False → Vt 只有 min(n_tr, d) 列，**秩上限 = n_tr − 1**。
    """
    mu = Xtr.mean(axis=0, keepdims=True)
    Xc = Xtr - mu
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu.ravel(), Vt, s


def pca_rank(s: np.ndarray, tol: float = 1e-10) -> int:
    """數值秩：奇異值大於 tol 的個數。用來誠實標示 k_eff。"""
    return int((s > tol).sum())


def standardize(Atr: np.ndarray, Ate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐維標準化，**只用訓練摺的均值與標準差**。"""
    mu = Atr.mean(axis=0)
    sd = Atr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Atr - mu) / sd, (Ate - mu) / sd


def build_arm(
    Ztr: np.ndarray, Zte: np.ndarray, arm: str, k: int, abtt_drop: int = ABTT_DROP
) -> tuple[np.ndarray, np.ndarray, int]:
    """PCA 空間的三個臂。回傳 (Atr, Ate, 取用的維度數)。"""
    d = Ztr.shape[1]
    if arm == "pca_first_k":
        kk = min(k, d)
        return Ztr[:, :kk], Zte[:, :kk], kk
    if arm == "pca_std_k":
        kk = min(k, d)
        Atr, Ate = standardize(Ztr[:, :kk], Zte[:, :kk])
        return Atr, Ate, kk
    if arm == "abtt_k" or arm == "abtt_nostd_k" or arm.startswith("abtt_r"):
        if arm == "abtt_k" or arm == "abtt_nostd_k":
            drop = ABTT_DROP
        else:
            drop = int(arm.replace("abtt_r", ""))
        avail = max(d - drop, 1)
        kk = min(k, avail)
        Atr, Ate = Ztr[:, drop:drop + kk], Zte[:, drop:drop + kk]
        if arm != "abtt_nostd_k":           # abtt_nostd_k 刻意**不**標準化（隔離標準化的效果）
            Atr, Ate = standardize(Atr, Ate)
        return Atr, Ate, kk
    raise ValueError(f"未知的臂：{arm}")


def arm_features(
    Ztr: np.ndarray, Zte: np.ndarray, Xraw_tr: np.ndarray, Xraw_te: np.ndarray,
    arm: str, k: int, rank: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """統一入口：依臂別組出特徵，並回報**實際維度** k_eff（受 PCA 秩上限限制）。"""
    if arm == "trunc_k":
        kk = min(k, Xraw_tr.shape[1])
        return truncate_and_norm(Xraw_tr, kk), truncate_and_norm(Xraw_te, kk), kk
    Atr, Ate, kk = build_arm(Ztr, Zte, arm, k)
    return Atr, Ate, min(kk, rank)


# ---------------------------------------------------------------------------
# 5. 線性探針：L2 正則邏輯斯迴歸，Newton–IRLS 精確求解（純 NumPy）
# ---------------------------------------------------------------------------
def _objective(A: np.ndarray, y: np.ndarray, theta: np.ndarray, C: float) -> float:
    """J(θ) = 0.5‖w‖² + C·Σ logloss（**不懲罰截距**，與 sklearn 預設一致）。

    `A` 已含截距欄（最後一欄），`theta` 長度 = d + 1。
    """
    z = A @ theta
    loss = np.logaddexp(0.0, z).sum() - float(y @ z)  # y ∈ {0,1}
    w = theta[:-1]
    return 0.5 * float(w @ w) + C * loss


def fit_logreg_l2(
    X: np.ndarray, y: np.ndarray, C: float = C_REG, max_iter: int = MAX_NEWTON_ITER,
    tol: float = NEWTON_TOL,
) -> dict:
    """精確求解 L2 邏輯斯迴歸（Newton–IRLS + Levenberg 阻尼）。

    為什麼不用梯度下降：PC 各維尺度差異極大（PC1 變異數可達數十，PC_k 趨近 0），
    固定學習率的梯度下降會**收斂不良**，讓「低維看起來比較好」變成假的。
    Newton 對線性重參數化不變（affine invariant），所以各臂可以公平比較。

    回傳 dict：w, b, n_iter, grad_norm（KKT 憑證）, converged, objective
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    theta = np.zeros(d + 1)
    obj = _objective(A, y, theta, C)
    damping = 1e-8
    grad_norm = float("inf")
    n_iter = 0
    converged = False

    for n_iter in range(1, max_iter + 1):
        z = A @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        wq = p * (1.0 - p)
        g = np.empty(d + 1)
        g[:-1] = theta[:-1] + C * (X.T @ (p - y))
        g[-1] = C * float((p - y).sum())
        grad_norm = float(np.linalg.norm(g))

        H = C * (A * wq[:, None]).T @ A            # C·Aᵀ diag(w) A
        H[np.arange(d), np.arange(d)] += 1.0       # L2 懲罰（截距不罰）
        H[np.arange(d + 1), np.arange(d + 1)] += 1e-10 * max(
            1.0, float(np.trace(H)) / (d + 1))     # 數值抖動（遠小於 H 的尺度）

        accepted = False
        for _ in range(40):
            try:
                step = np.linalg.solve(H + damping * np.eye(d + 1), g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H + damping * np.eye(d + 1), g, rcond=None)[0]
            cand = theta - step
            obj_cand = _objective(A, y, cand, C)
            if np.isfinite(obj_cand) and obj_cand <= obj:
                theta, new_obj, accepted = cand, obj_cand, True
                damping = max(damping * 0.3, 1e-12)
                break
            damping *= 10.0
        if not accepted:
            converged = True                       # 已到數值極限
            break
        if abs(obj - new_obj) < tol * max(1.0, abs(obj)):
            obj = new_obj
            converged = True
            break
        obj = new_obj

    return {
        "w": theta[:-1], "b": float(theta[-1]), "n_iter": n_iter,
        "grad_norm": grad_norm, "converged": bool(converged), "objective": float(obj),
    }


def probe_scores(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> dict:
    """訓練探針並回傳 accuracy / macro-F1 / 收斂診斷。"""
    m = fit_logreg_l2(Xtr, ytr)
    z = Xte @ m["w"] + m["b"]
    p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    P = np.column_stack([1.0 - p1, p1])
    return {
        "acc": mlkit.accuracy(P, yte),
        "macro_f1": mlkit.prf_per_class(P, yte, 2)["macro_f1"],
        "grad_norm": m["grad_norm"],
        "n_iter": m["n_iter"],
        "converged": m["converged"],
    }


# ---------------------------------------------------------------------------
# 6. 分摺（分層）與單摺評估
# ---------------------------------------------------------------------------
def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """分層 K 摺：每個類別內部先洗牌再輪流分配，保證每摺類別比例一致。"""
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            buckets[i % n_folds].append(int(ix))
    return [np.asarray(sorted(b), dtype=int) for b in buckets]


def run_one_split(
    Xn: np.ndarray, Xraw: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
    arms: list[str], ks: list[int],
) -> dict:
    """在單一 (訓練摺, 測試摺) 上跑完所有臂 × k。

    ★ 洩漏防護的核心：`fit_pca` 與 `standardize` 都只接到 `tr` 的資料；
      測試摺 `te` 只在預測時出現。
    """
    mu, Vt, s = fit_pca(Xn[tr])
    rank = pca_rank(s)
    Ztr = (Xn[tr] - mu) @ Vt.T
    Zte = (Xn[te] - mu) @ Vt.T
    out: dict = {}
    for arm in arms:
        for k in ks:
            Atr, Ate, k_eff = arm_features(Ztr, Zte, Xraw[tr], Xraw[te], arm, k, rank)
            sc = probe_scores(Atr, y[tr], Ate, y[te])
            out[(arm, k)] = {**sc, "k_eff": int(k_eff), "rank": int(rank)}
    return out


# ---------------------------------------------------------------------------
# 7. 核心：對一個任務跑完整掃描
# ---------------------------------------------------------------------------
def sweep_task(
    task: str, X_raw: np.ndarray, y: np.ndarray, arms: list[str], ks: list[int],
    split_mode: str = "cv", lang: np.ndarray | None = None, extra_arms: dict | None = None,
) -> dict:
    """對單一 (模型, 任務) 跑 k × 臂 × 種子 的完整掃描。

    split_mode:
      - "cv"       : 分層 5 摺交叉驗證，seed 控制分摺（主協定）
      - "transfer" : 固定切分（來源語言訓練 → 目標語言測試），無隨機性
    extra_arms: {臂名: 丟棄數 r}，例如 {"abtt_r3": 3}
    """
    Xn = l2norm(X_raw)
    arms = list(arms) + list((extra_arms or {}).keys())
    results: dict = {}
    diag: dict = {"max_grad_norm": 0.0, "n_fits": 0, "n_not_converged": 0,
                  "k_eff": {}, "rank": None, "raw_acc_by_seed": {}}

    def accumulate(arm: str, k: int, key: str, acc: float, f1: float, k_eff: int) -> None:
        slot = results.setdefault(arm, {}).setdefault(k, {})
        d = slot.setdefault(key, {"acc": [], "f1": []})
        d["acc"].append(float(acc))
        d["f1"].append(float(f1))
        diag["k_eff"][f"{arm}@{k}"] = int(k_eff)

    if split_mode == "transfer":
        assert lang is not None
        langs = sorted(set(lang.tolist()))
        pairs = [(l, np.where(lang == l)[0], np.where(lang != l)[0]) for l in langs]
        print(f"    固定切分（無隨機性）："
              + "、".join(f"{l}(n={len(tr)})→{str(lang[te][0])}" for l, tr, te in pairs))
        # ★ 刻意跑滿 5 個種子：切分不含隨機成分、求解器確定性 → 5 次結果應完全相同。
        #    這是「可重現性」的實測證據，而不是口頭聲明。
        for seed in SEEDS:
            per_dir: dict[tuple[str, int], list[tuple[float, float]]] = {}
            for src_l, tr, te in pairs:
                tgt = str(lang[te][0])
                r = run_one_split(Xn, X_raw, y, tr, te, arms, ks)
                for (arm, k), sc in r.items():
                    accumulate(arm, k, f"tgt_{tgt}", sc["acc"], sc["macro_f1"], sc["k_eff"])
                    per_dir.setdefault((arm, k), []).append((sc["acc"], sc["macro_f1"]))
                    diag["max_grad_norm"] = max(diag["max_grad_norm"], sc["grad_norm"])
                    diag["n_fits"] += 1
                    diag["n_not_converged"] += int(not sc["converged"])
                    diag["rank"] = sc["rank"]
                    diag["raw_acc_by_seed"].setdefault(f"{arm}@{k}@{tgt}", []).append(
                        (seed, sc["acc"]))
            for (arm, k), vals in per_dir.items():
                accumulate(arm, k, "mean", float(np.mean([v[0] for v in vals])),
                           float(np.mean([v[1] for v in vals])), diag["k_eff"][f"{arm}@{k}"])
    else:
        per_seed: dict[tuple[str, int], list[float]] = {}
        per_seed_f1: dict[tuple[str, int], list[float]] = {}
        first_time = (task, split_mode) not in _PRINTED_FOLD_HEADER
        for seed in SEEDS:
            folds = stratified_folds(y, N_FOLDS, seed)
            if first_time:
                print(f"    種子 {seed}: 摺大小 {[len(f) for f in folds]}")
            for fi in range(N_FOLDS):
                te = folds[fi]
                tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fi])
                r = run_one_split(Xn, X_raw, y, tr, te, arms, ks)
                for (arm, k), sc in r.items():
                    per_seed.setdefault((arm, k), []).append(sc["acc"])
                    per_seed_f1.setdefault((arm, k), []).append(sc["macro_f1"])
                    diag["max_grad_norm"] = max(diag["max_grad_norm"], sc["grad_norm"])
                    diag["n_fits"] += 1
                    diag["n_not_converged"] += int(not sc["converged"])
                    diag["k_eff"][f"{arm}@{k}"] = sc["k_eff"]
                    diag["rank"] = sc["rank"]
                if fi == 0 and first_time:
                    print(f"      摺 1/{N_FOLDS}: 訓練 {len(tr)}、測試 {len(te)}、"
                          f"PCA 數值秩 = {r[(arms[0], ks[0])]['rank']}"
                          f"（rank 上限 = n_tr − 1 = {len(tr) - 1}）")
                    _PRINTED_FOLD_HEADER.add((task, split_mode))
        for (arm, k), accs in per_seed.items():
            A = np.asarray(accs).reshape(len(SEEDS), N_FOLDS)
            F = np.asarray(per_seed_f1[(arm, k)]).reshape(len(SEEDS), N_FOLDS)
            slot = results.setdefault(arm, {}).setdefault(k, {})
            slot["seed"] = {
                "acc_mean": float(A.mean(axis=1).mean()),
                "acc_std": float(np.std(A.mean(axis=1), ddof=1)),
                "f1_mean": float(F.mean(axis=1).mean()),
                "f1_std": float(np.std(F.mean(axis=1), ddof=1)),
                "acc_per_seed_mean": [float(v) for v in A.mean(axis=1)],
                "fold_std_within_seed": float(A.std(axis=1, ddof=1).mean()),
                "n_seeds": len(SEEDS), "n_folds": N_FOLDS,
            }
            slot["fold"] = {
                "acc_mean": float(A.mean()),
                "acc_std": float(A.std(ddof=1)) if A.size > 1 else 0.0,
                "f1_mean": float(F.mean()),
                "f1_std": float(F.std(ddof=1)) if F.size > 1 else 0.0,
                "n": int(A.size),
            }
            slot["folds_raw"] = {"acc": [float(v) for v in A.ravel()],
                                 "f1": [float(v) for v in F.ravel()],
                                 "shape_seed_by_fold": [len(SEEDS), N_FOLDS]}

    # --- 統一彙總：每個 key 都補上 *_mean / *_std（transfer 的 tgt_*、mean 需要）---
    for arm in arms:
        for k in ks:
            for _key, d in results.get(arm, {}).get(k, {}).items():
                if "acc_mean" in d:
                    continue
                a = np.asarray(d["acc"], dtype=float)
                f = np.asarray(d["f1"], dtype=float)
                d["acc_mean"] = float(a.mean())
                d["acc_std"] = float(a.std(ddof=1)) if a.size > 1 else 0.0
                d["f1_mean"] = float(f.mean())
                d["f1_std"] = float(f.std(ddof=1)) if f.size > 1 else 0.0
                d["n"] = int(a.size)
    return {"task": task, "split_mode": split_mode, "arms": arms, "ks": ks,
            "results": results, "diag": diag}


# ---------------------------------------------------------------------------
# 8. 洩漏防護與求解器自我測試
# ---------------------------------------------------------------------------
def self_test(X_raw: np.ndarray, y: np.ndarray) -> dict:
    """三個真的會抓到問題的測試。

    A. 標籤置換：整體洗牌標籤後重跑，準確率必須回落到隨機水準（≈0.5）。
       若 PCA／標準化偷看了測試摺或標籤，分數會異常維持在高位。
    B. 訓練側不變性：同一訓練摺、換掉測試摺內容，訓練側特徵矩陣必須**逐位元相同**
       （證明轉換只依賴訓練摺）。
    C. 合成可分資料：兩團高斯分開的 1 維資料，探針必須達到 1.0
       （抓符號／標籤編碼錯誤）。
    """
    out: dict = {}
    Xn = l2norm(X_raw)
    rng = np.random.default_rng(0)

    # --- A. 標籤置換 ---
    y_perm = y.copy()
    rng.shuffle(y_perm)
    folds = stratified_folds(y_perm, N_FOLDS, SEEDS[0])
    for arm, k in (("abtt_k", 32), ("pca_first_k", 128)):
        accs = []
        for fi in range(N_FOLDS):
            te = folds[fi]
            tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fi])
            r = run_one_split(Xn, X_raw, y_perm, tr, te, [arm], [k])
            accs.append(r[(arm, k)]["acc"])
        a = np.asarray(accs)
        out[f"permuted_labels_{arm}_k{k}"] = {
            "acc_mean": round(float(a.mean()), 4),
            "acc_std": round(float(a.std(ddof=1)), 4),
            "chance": 0.5,
            "within_3sigma_or_0.05": bool(abs(a.mean() - 0.5) <= 3 * a.std(ddof=1) + 0.05),
        }

    # --- B. 訓練側不變性 ---
    tr = np.arange(0, 128)
    te_a = np.arange(128, 160)
    te_b = rng.permutation(np.arange(128, 160))
    mu_a, Vt_a, s_a = fit_pca(Xn[tr])
    mu_b, Vt_b, s_b = fit_pca(Xn[tr])
    rank = pca_rank(s_a)
    Ztr_a = (Xn[tr] - mu_a) @ Vt_a.T
    Ztr_b = (Xn[tr] - mu_b) @ Vt_b.T
    Atr_a, Ate_a, _ = arm_features(Ztr_a, (Xn[te_a] - mu_a) @ Vt_a.T, X_raw[tr], X_raw[te_a],
                                   "abtt_k", 32, rank)
    Atr_b, Ate_b, _ = arm_features(Ztr_b, (Xn[te_b] - mu_b) @ Vt_b.T, X_raw[tr], X_raw[te_b],
                                   "abtt_k", 32, rank)
    out["train_side_invariance"] = {
        "mu_identical": bool(np.array_equal(mu_a, mu_b)),
        "Vt_identical": bool(np.array_equal(Vt_a, Vt_b)),
        "Ztr_identical": bool(np.array_equal(Ztr_a, Ztr_b)),
        "standardised_train_features_identical": bool(np.array_equal(Atr_a, Atr_b)),
        "test_features_differ": bool(not np.array_equal(Ate_a, Ate_b)),
        "ok": bool(np.array_equal(Atr_a, Atr_b)),
    }

    # --- C. 合成可分資料 ---
    rng2 = np.random.default_rng(3)
    Xs = np.concatenate([rng2.normal(-3.0, 0.5, (40, 1)), rng2.normal(3.0, 0.5, (40, 1))])
    ys = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])
    sc = probe_scores(Xs[:60], ys[:60], Xs[60:], ys[60:])
    out["synthetic_separable_probe"] = {
        "acc": round(sc["acc"], 4), "grad_norm": float(f"{sc['grad_norm']:.3e}"),
        "n_iter": sc["n_iter"], "ok": bool(sc["acc"] == 1.0),
    }
    return out


# ---------------------------------------------------------------------------
# 9. 主流程
# ---------------------------------------------------------------------------
PC_COLUMN_SETS: list[tuple[str, list[int]]] = [
    ("PC1", [0]), ("PC2", [1]), ("PC1+PC2", [0, 1]), ("PC3+PC4", [2, 3]),
    ("PC3..PC10", list(range(2, 10))), ("PC3..PC66", list(range(2, 66))),
]


def pc_content_diagnostic(
    X_raw: np.ndarray, y_sent: np.ndarray, y_lang: np.ndarray
) -> dict:
    """前幾個主成分「各自帶什麼資訊」——ABTT 前提是否成立，取決於這一節。

    對每一組主成分欄位，分別用情感標籤與語言標籤各訓練一次線性探針，
    走**同一個** 5 摺 × 5 種子協定（PCA 只在訓練摺 fit）。
    """
    Xn = l2norm(X_raw)
    out: dict = {}
    for label, cols in PC_COLUMN_SETS:
        acc_s, acc_l = [], []
        for seed in SEEDS:
            folds = stratified_folds(y_sent, N_FOLDS, seed)
            for fi in range(N_FOLDS):
                te = folds[fi]
                tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fi])
                mu, Vt, _ = fit_pca(Xn[tr])
                Ztr = (Xn[tr] - mu) @ Vt.T
                Zte = (Xn[te] - mu) @ Vt.T
                acc_s.append(probe_scores(Ztr[:, cols], y_sent[tr],
                                          Zte[:, cols], y_sent[te])["acc"])
                acc_l.append(probe_scores(Ztr[:, cols], y_lang[tr],
                                          Zte[:, cols], y_lang[te])["acc"])
        out[label] = {
            "sentiment": {"mean": float(np.mean(acc_s)), "std": float(np.std(acc_s, ddof=1))},
            "language": {"mean": float(np.mean(acc_l)), "std": float(np.std(acc_l, ddof=1))},
            "n_dims": len(cols),
        }
    return out


def main() -> int:
    head("s08：用 PCA 統一多模型的維度消融（DIM1）")
    print(f"  執行時間（本機）   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  工作目錄           : {Path.cwd()}")
    print(f"  Python             : {platform.python_version()}（{platform.platform()}）")
    print(f"  腳本位置           : {Path(__file__).resolve()}")
    print()
    print("  套件版本：")
    vers = package_versions()
    for name, v in vers.items():
        print(f"    {name:24s} {v}")

    from data_bilingual_sentiment import load_dataset

    texts, labels, langs = load_dataset()
    y = np.asarray(labels, dtype=np.int64)
    lang = np.asarray(langs)
    y_lang = (lang == "en").astype(np.int64)   # 語言辨識的整數標籤
    print()
    print(f"  語料：{len(texts)} 句（繁中 {int((lang == 'zh').sum())}、"
          f"英文 {int((lang == 'en').sum())}）")
    print(f"  標籤：正 {int((y == 1).sum())}、負 {int((y == 0).sum())}"
          f"（平衡 → 隨機基準線 0.5000）")
    print("  ⚠ 語料性質：zh 與 en 兩半互為翻譯對，所以跨語言任務衡量的是"
          "「同內容換語言」——正是 ABTT 聲稱要解決的情境。")

    # ---------------- 嵌入 ----------------
    head("一、嵌入抽取（依協定 §3：三者都不需要前綴，編碼時不加任何前綴）")
    embs: dict[str, np.ndarray] = {}
    metas: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for key in MODEL_ORDER:
        cfg = MODELS[key]
        print()
        print(f"  [{key}]  {cfg['hf_id']}")
        try:
            t0 = time.perf_counter()
            X, meta = get_embeddings(key, texts)
            dt = time.perf_counter() - t0
            embs[key] = X
            metas[key] = {**meta, "hf_revision": hf_snapshot_revision(cfg["hf_id"])}
            print(f"    ✅ 載入成功：原生維度 {X.shape[1]}（宣告 {cfg['dim_declared']}）"
                  f"，耗時 {dt:.2f} s"
                  f"{'（來自快取）' if meta.get('from_cache') else '（本次實際編碼）'}")
            print(f"       HF snapshot 修訂：{metas[key]['hf_revision']}")
            nrm = np.linalg.norm(X, axis=1)
            print(f"       原生 L2 範數：min {nrm.min():.4f} / mean {nrm.mean():.4f} / "
                  f"max {nrm.max():.4f}")
            for fld in ("load_s", "encode_s", "n_embedding_params", "max_seq_length"):
                if fld in meta:
                    print(f"       {fld} = {meta[fld]}")
            if meta.get("truncate_checks"):
                print("       Matryoshka 等價性實測（ST 內建 truncate_dim vs 手動截前 k 維）：")
                for c in meta["truncate_checks"]:
                    if "error" in c:
                        print(f"         k={c['k']:>4d}：⚠ {c['error']}")
                    else:
                        print(f"         k={c['k']:>4d}：shape {c['st_shape']} vs "
                              f"{c['manual_shape']}，max|Δ| = {c['max_abs_diff']:.3e} → "
                              f"{'✅ 完全等價' if c['equal_1e-9'] else '❌ 不等價'}")
        except Exception as e:  # noqa: BLE001
            failures[key] = f"{type(e).__name__}: {e}"
            print(f"    ❌ 載入失敗：{type(e).__name__}: {e}")
            print("       traceback（末 8 行）：")
            for line in traceback.format_exc().splitlines()[-8:]:
                print(f"         {line}")

    print()
    print(f"  成功載入的模型：{list(embs.keys())}")
    print(f"  失敗的模型    ：{failures if failures else '（無）'}")
    if not embs:
        print("\n  沒有任何模型可載入 → 中止（不偽造結果）。")
        return 1
    loaded = [m for m in MODEL_ORDER if m in embs]

    # ---------------- 自我測試 ----------------
    head("二、洩漏防護與求解器自我測試（先驗證，再相信數字）")
    st = self_test(embs[loaded[0]], y)
    for kk, vv in st.items():
        print(f"  {kk}:")
        for k2, v2 in vv.items():
            print(f"    {k2:42s} {v2}")
    perm_ok = all(v["within_3sigma_or_0.05"] for k2, v in st.items()
                  if k2.startswith("permuted"))
    print()
    print(f"  A 標籤置換後準確率回落至 0.5 附近：{'✅ 通過' if perm_ok else '❌ 未通過'}")
    print(f"  B 轉換只依賴訓練摺（訓練側逐位元相同）："
          f"{'✅ 通過' if st['train_side_invariance']['ok'] else '❌ 未通過'}")
    print(f"  C 合成可分資料可達 1.0："
          f"{'✅ 通過' if st['synthetic_separable_probe']['ok'] else '❌ 未通過'}")

    # ---------------- PCA 描述統計 ----------------
    head("三、各模型的 PCA 結構與**秩上限**（描述統計；評估用的 PCA 一律只在訓練摺 fit）")
    pca_info: dict = {}
    folds0 = stratified_folds(y, N_FOLDS, SEEDS[0])
    te0 = folds0[0]
    tr0 = np.concatenate([folds0[j] for j in range(N_FOLDS) if j != 0])
    for key in loaded:
        Xn = l2norm(embs[key])
        mu, Vt, s = fit_pca(Xn)      # 全資料：**只用來描述，不進任何評估路徑**
        var = s ** 2
        evr = var / var.sum()
        _, _, s_tr = fit_pca(Xn[tr0])
        rank_tr = pca_rank(s_tr)
        pca_info[key] = {
            "dim_native": int(embs[key].shape[1]),
            "evr_pc1": float(evr[0]), "evr_pc2": float(evr[1]),
            "evr_pc1_2": float(evr[:2].sum()),
            "evr_first_8": float(evr[:8].sum()),
            "evr_first_64": float(evr[:64].sum()),
            "rank_train_fold": rank_tr,
        }
        print(f"  {key}")
        print(f"    原生維度 {embs[key].shape[1]}；PC1 解釋變異 {evr[0]:.4f}、"
              f"PC2 {evr[1]:.4f}、前 2 合計 {evr[:2].sum():.4f}")
        print(f"    前 8 維累積 {evr[:8].sum():.4f}；前 64 維累積 {evr[:64].sum():.4f}")
        print(f"    ★ 訓練摺（n={len(tr0)}）的 PCA 數值秩 = {rank_tr}"
              f" → PCA 路徑的實際維度上限是 {rank_tr}，k=256 不可能真正達成")

    # ---------------- 主掃描 ----------------
    head("四、主協定：混合語言情感分類，分層 5 摺 CV × 5 種子")
    print(f"  k 網格 = {K_GRID}；臂 = {ARMS}")
    print(f"  探針 = L2 邏輯斯迴歸（0.5‖w‖² + C·Σlogloss，C={C_REG}，不懲罰截距），")
    print(f"         Newton–IRLS 精確求解至 KKT 梯度 < {NEWTON_TOL:g}")
    print("  ★ 所有 PCA 與標準化**只在訓練摺 fit**（測試摺從不參與 fit）")

    tasks: dict[str, dict] = {}
    t0 = time.perf_counter()
    for key in loaded:
        print()
        print(f"  ── 模型：{key} ──")
        tasks.setdefault("sentiment_cv", {})[key] = sweep_task(
            "sentiment_cv", embs[key], y, ARMS, K_GRID, split_mode="cv")
        r = tasks["sentiment_cv"][key]
        print(f"     完成：{r['diag']['n_fits']} 次擬合，"
              f"最大 KKT 梯度範數 = {r['diag']['max_grad_norm']:.3e}，"
              f"未收斂 {r['diag']['n_not_converged']} 次")
    print(f"\n  主掃描總耗時 {time.perf_counter() - t0:.1f} s")

    print("\n  ── 實際維度 k_eff（受 PCA 秩上限限制；已=要求值則未受限制）──")
    for key in loaded:
        res = tasks["sentiment_cv"][key]
        print(f"  {key}（PCA 秩上限 {res['diag']['rank']}）")
        print("  " + f"{'要求 k':>7}" + "".join(f"{a:>16}" for a in res["arms"]))
        for k in K_GRID:
            line = f"  {k:>7}"
            for a in res["arms"]:
                ke = res["diag"]["k_eff"].get(f"{a}@{k}")
                flag = "" if ke == k or a == "trunc_k" else " *"
                line += f"{str(ke) + flag:>16}"
            print(line)
        print("    （* = 被秩上限截斷，實際維度小於要求值）")

    def table(res: dict, field: str, metric: str = "acc") -> None:
        ks = res["ks"]
        print("  " + f"{'k':>5}" + "".join(f"{a:>24}" for a in res["arms"]))
        for k in ks:
            line = f"  {k:>5}"
            for a in res["arms"]:
                d = res["results"].get(a, {}).get(k, {}).get(field)
                if d is None:
                    line += f"{'—':>24}"
                else:
                    line += (f"{d[metric + '_mean']:>15.4f} ± "
                             f"{d[metric + '_std']:<7.4f}")
            print(line)

    for key in loaded:
        print(f"\n  {key}：accuracy（mean ± std across 5 seeds，每種子先平均 5 摺）")
        table(tasks["sentiment_cv"][key], "seed", "acc")
    print("\n  ── macro-F1（同一組配置）──")
    for key in loaded:
        print(f"\n  {key}")
        table(tasks["sentiment_cv"][key], "seed", "f1")

    # ---------------- PC 內容診斷 ----------------
    head("四之二、前幾個主成分各自帶什麼資訊（ABTT 的前提檢驗）")
    print("  對每組主成分欄位，分別用「情感標籤」與「語言標籤」各訓練一次線性探針，")
    print("  協定與第四節完全相同（5 摺 × 5 種子，PCA 只在訓練摺 fit）。")
    print("  ★ ABTT 的前提是「PC1/PC2 = 語言身分」；若某模型的前 2 個主成分其實裝的是")
    print("     任務訊號（情感），則丟掉它們不是去語言偏誤，而是直接把答案丟掉。")
    pc_diag: dict = {}
    for key in loaded:
        print()
        print(f"  ── 模型：{key} ──")
        pc_diag[key] = pc_content_diagnostic(embs[key], y, y_lang)
        print(f"  {'欄位':>11} {'維度':>5} {'情感 acc':>18} {'語言 acc':>18}")
        for label, d in pc_diag[key].items():
            print(f"  {label:>11} {d['n_dims']:>5} "
                  f"{d['sentiment']['mean']:>11.4f} ± {d['sentiment']['std']:<6.4f} "
                  f"{d['language']['mean']:>11.4f} ± {d['language']['std']:<6.4f}")
        pc12 = pc_diag[key]["PC1+PC2"]
        verdict = ("PC1/PC2 主要是語言身分 → ABTT 前提成立"
                   if pc12["language"]["mean"] - pc12["sentiment"]["mean"] > 0.15 else
                   "PC1/PC2 主要是任務訊號 → **ABTT 前提不成立**")
        print(f"    → {verdict}"
              f"（語言 {pc12['language']['mean']:.4f} vs 情感 {pc12['sentiment']['mean']:.4f}）")

    # ---------------- 配對檢定 ----------------
    head("四之三、ABTT 與「直接取前 k 維」的配對檢定（同一 (種子, 摺) 配對）")
    print("  ⚠ 配對單位是 25 個 (種子, 摺) 組合，但同一批 160 句在不同種子下重複使用，")
    print("     觀察值**不獨立**，所以 p 值只能當描述性參考，不能當推論證據。")
    paired: dict = {}
    for key in loaded:
        res = tasks["sentiment_cv"][key]
        print(f"\n  {key}")
        print(f"  {'k':>5} {'Δ(abtt−前k)':>14} {'t':>9} {'p':>10} {'dz':>8}   "
              f"{'Δ(abtt−abtt_nostd)':>20}")
        for k in K_GRID:
            a = np.asarray(res["results"]["abtt_k"][k]["folds_raw"]["acc"])
            b = np.asarray(res["results"]["pca_first_k"][k]["folds_raw"]["acc"])
            c = np.asarray(res["results"]["abtt_nostd_k"][k]["folds_raw"]["acc"])
            tt = mlkit.paired_t_test(a, b)
            tt2 = mlkit.paired_t_test(a, c)
            paired.setdefault(key, {})[k] = {
                "abtt_minus_first": tt, "abtt_minus_abtt_nostd": tt2}
            print(f"  {k:>5} {tt['mean_diff']:>+14.4f} {tt['t']:>9.3f} {tt['p']:>10.4g} "
                  f"{tt['dz']:>8.3f}   {tt2['mean_diff']:>+20.4f}")

    # ---------------- 語言辨識 ----------------
    head("五、語言辨識探針（直接檢驗 F10【L325】：PC1/PC2 編碼語言身分）")
    print("  任務：zh vs en（80/80 平衡，隨機基準線 0.5000）")
    LANG_ARMS = ["pca_first_k", "abtt_k", "trunc_k"]
    for key in loaded:
        print()
        print(f"  ── 模型：{key} ──")
        tasks.setdefault("langid_cv", {})[key] = sweep_task(
            "langid_cv", embs[key], y_lang, LANG_ARMS, K_GRID, split_mode="cv")
        print(f"  {key}：語言辨識 accuracy（mean ± std across 5 seeds）")
        table(tasks["langid_cv"][key], "seed", "acc")

    print("\n  ── 只用 PC1+PC2 做語言辨識（2 維探針）──")
    print("  若準確率接近 1.0 → 前兩個主成分幾乎就是「語言身分」軸（與 F10【L325】一致）。")
    pc12: dict = {}
    for key in loaded:
        Xn = l2norm(embs[key])
        accs = []
        for seed in SEEDS:
            folds = stratified_folds(y_lang, N_FOLDS, seed)
            for fi in range(N_FOLDS):
                te = folds[fi]
                tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fi])
                mu, Vt, _ = fit_pca(Xn[tr])
                Ztr = (Xn[tr] - mu) @ Vt.T
                Zte = (Xn[te] - mu) @ Vt.T
                accs.append(probe_scores(Ztr[:, :2], y_lang[tr], Zte[:, :2], y_lang[te])["acc"])
        a = np.asarray(accs)
        pc12[key] = {"acc_mean": float(a.mean()), "acc_std": float(a.std(ddof=1))}
        print(f"    {key:28s} PC1+PC2 準確率 = {a.mean():.4f} ± {a.std(ddof=1):.4f}"
              f"（{len(a)} 次摺×種子）")

    # ---------------- 跨語言遷移 ----------------
    head("六、跨語言遷移（zh→en 與 en→zh）——F10 的 ABTT 效果所在的情境")
    print("  固定切分：來源語言全部訓練（80 句）→ 目標語言全部測試（80 句）。")
    print("  ★ 此任務**沒有隨機性**（切分固定、求解器確定性），所以不報「5 種子」的")
    print("     標準差；mean 欄的 std 是兩個方向之間的差異（方向不對稱），不是種子變異。")
    for key in loaded:
        print()
        print(f"  ── 模型：{key} ──")
        tasks.setdefault("transfer", {})[key] = sweep_task(
            "transfer", embs[key], y, ["pca_first_k", "abtt_k"], K_GRID,
            split_mode="transfer", lang=lang)
        res = tasks["transfer"][key]
        print(f"\n  {key}")
        print("  " + f"{'k':>5}{'pca_first_k →en':>18}{'→zh':>10}"
              f"{'abtt_k →en':>15}{'→zh':>10}{'Δ(abtt−前k)':>16}")
        for k in K_GRID:
            pf = res["results"]["pca_first_k"][k]
            ab = res["results"]["abtt_k"][k]
            fe, fz = pf["tgt_en"]["acc"][0], pf["tgt_zh"]["acc"][0]
            ae, az = ab["tgt_en"]["acc"][0], ab["tgt_zh"]["acc"][0]
            print(f"  {k:>5}{fe:>18.4f}{fz:>10.4f}{ae:>15.4f}{az:>10.4f}"
                  f"{((ae + az) - (fe + fz)) / 2:>+16.4f}")
        print(f"    macro-F1（平均兩方向）：")
        for k in K_GRID:
            pf = res["results"]["pca_first_k"][k]["mean"]
            ab = res["results"]["abtt_k"][k]["mean"]
            print(f"      k={k:<4d} pca_first_k {pf['f1_mean']:.4f}   "
                  f"abtt_k {ab['f1_mean']:.4f}   Δ = {ab['f1_mean'] - pf['f1_mean']:+.4f}")

    # ---------------- bekko：Matryoshka vs PCA ----------------
    head("七、bekko：Matryoshka 截斷 vs PCA 降維（唯一能直接比較的模型）")
    bekko = "bekko-embedding-v1-a25m"
    matryoshka_cmp: dict = {}
    if bekko in embs:
        res = tasks["sentiment_cv"][bekko]
        rank = res["diag"]["rank"]
        print("  bekko 原生 384 維且支援 Matryoshka；potion / M2V 的 256 維是固定的。")
        print("    trunc_k     ＝ Matryoshka 截斷（取原生前 k 維 + L2 正規化）")
        print("    pca_first_k ＝ PCA 前 k 維")
        print("    abtt_k      ＝ ABTT（丟棄前 2 個主成分）+ 逐維標準化")
        print(f"  ⚠ 維度不對等的陷阱：trunc_k 在 k=256 時真的是 256 維，而 PCA 臂受訓練摺")
        print(f"     秩上限限制（k_eff 最高 {rank}）。這對 k ≥ 128 的比較**有利於 Matryoshka**，")
        print("     必須明說。")
        print()
        print(f"  {'k':>5}  {'trunc(Matryoshka)':>19}  {'pca_first_k':>19}  "
              f"{'abtt_k':>19}  {'k_eff(PCA)':>10}  {'最佳':>14}")
        names = ["trunc_k", "pca_first_k", "abtt_k"]
        for k in K_GRID:
            row = [res["results"][a][k]["seed"]["acc_mean"] for a in names]
            best = names[int(np.argmax(row))]
            ke = res["diag"]["k_eff"].get(f"pca_first_k@{k}")
            matryoshka_cmp[k] = {
                "trunc_k": row[0], "pca_first_k": row[1], "abtt_k": row[2],
                "k_eff_pca": ke, "winner": best, "delta_trunc_minus_pca": row[0] - row[1]}
            print(f"  {k:>5}  {row[0]:>19.4f}  {row[1]:>19.4f}  {row[2]:>19.4f}  "
                  f"{ke:>10}  {best:>14}")
        print("\n  參考（不做任何降維，天花板）：")
        for other in loaded:
            dim = int(embs[other].shape[1])
            rr = sweep_task("ref_full", embs[other], y, ["trunc_k"], [dim], split_mode="cv")
            tasks.setdefault("ref_full", {})[other] = rr
            dd = rr["results"]["trunc_k"][dim]["seed"]
            print(f"    {other:28s} 原生 {dim} 維 → accuracy "
                  f"{dd['acc_mean']:.4f} ± {dd['acc_std']:.4f}，"
                  f"macro-F1 {dd['f1_mean']:.4f} ± {dd['f1_std']:.4f}")
    else:
        print(f"  ❌ {bekko} 未載入，本節無法執行（原因見第一節）。")

    # ---------------- ABTT 丟棄數 r 的敏感度 ----------------
    head("八、ABTT 丟棄數 r 的敏感度（文獻：r ≤ ⌊d/100⌋ → d=384 允許 r ≤ 3）")
    r_sens: dict = {}
    if bekko in embs:
        extra = {f"abtt_r{r}": r for r in (1, 3, 4)}
        print(f"  對象：{bekko}（d = 384）；k ∈ {{256, 64}}；主臂 r=2 之外再掃 r=1,3,4")
        rs = sweep_task("r_sens", embs[bekko], y, ["abtt_k"], [256, 64],
                        split_mode="cv", extra_arms=extra)
        print(f"\n  {'r':>3}  {'k=256':>20}  {'k=64':>20}")
        for arm, r in [("abtt_r1", 1), ("abtt_k", 2), ("abtt_r3", 3), ("abtt_r4", 4)]:
            d256 = rs["results"][arm][256]["seed"]
            d64 = rs["results"][arm][64]["seed"]
            r_sens[r] = {"k256": d256["acc_mean"], "k256_std": d256["acc_std"],
                         "k64": d64["acc_mean"], "k64_std": d64["acc_std"]}
            print(f"  {r:>3}  {d256['acc_mean']:>13.4f} ± {d256['acc_std']:<6.4f}  "
                  f"{d64['acc_mean']:>13.4f} ± {d64['acc_std']:<6.4f}")
    else:
        print("  （跳過：bekko 未載入）")

    # ---------------- 自動判定 ----------------
    head("九、依事前問題自動判定（數字全部來自上面幾節）")

    print("\n  Q1. ABTT 在低維時是否真的必要？（abtt_k vs pca_first_k，逐 k）")
    q1: dict = {}
    for key in loaded:
        res = tasks["sentiment_cv"][key]
        print(f"    {key}")
        wins = 0
        for k in K_GRID:
            a = res["results"]["abtt_k"][k]["seed"]
            b = res["results"]["pca_first_k"][k]["seed"]
            delta = a["acc_mean"] - b["acc_mean"]
            wins += int(delta > 0)
            q1.setdefault(key, {})[k] = {
                "abtt": a["acc_mean"], "abtt_std": a["acc_std"],
                "pca_first": b["acc_mean"], "pca_first_std": b["acc_std"], "delta": delta}
            print(f"      k={k:<4d} abtt {a['acc_mean']:.4f}±{a['acc_std']:.4f} vs "
                  f"前 k {b['acc_mean']:.4f}±{b['acc_std']:.4f}  "
                  f"Δ = {delta:+.4f}  {'✅ ABTT 勝' if delta > 0 else '❌ 直接取較好'}")
        print(f"      → ABTT 勝 {wins}/{len(K_GRID)} 個 k 值")
    print("\n    ★ 但「ABTT 是否必要」不能只看勝負，要看前提是否成立（第四之二節）：")
    for key in loaded:
        p12 = pc_diag[key]["PC1+PC2"]
        print(f"      {key:28s} PC1+PC2：語言 {p12['language']['mean']:.4f}、"
              f"情感 {p12['sentiment']['mean']:.4f} → "
              f"{'丟掉的是語言偏誤' if p12['language']['mean'] > p12['sentiment']['mean'] + 0.15 else '丟掉的是任務訊號'}")
    abtt_all = [np.mean([q1[key][k]["abtt"] for k in K_GRID]) for key in loaded]
    first_all = [np.mean([q1[key][k]["pca_first"] for k in K_GRID]) for key in loaded]
    print(f"      六個 k 的平均：abtt {np.mean(abtt_all):.4f} vs pca_first "
          f"{np.mean(first_all):.4f}（三個模型平均）")

    print("\n  Q2. 不同模型的最佳 k 是否相同？")
    best_k: dict = {}
    for key in loaded:
        res = tasks["sentiment_cv"][key]
        scores = {k: res["results"]["abtt_k"][k]["seed"]["acc_mean"] for k in K_GRID}
        bk = max(scores, key=lambda kk: scores[kk])
        plateau = sorted([k for k in K_GRID if scores[bk] - scores[k] <= 0.02], reverse=True)
        best_k[key] = {"k": bk, "acc": scores[bk], "all": scores, "plateau": plateau}
        print(f"    {key:28s} 最佳 k = {bk:<4d}（acc {scores[bk]:.4f}）；"
              f"距最佳 2 個百分點內：{plateau}")
    ks_seen = {v["k"] for v in best_k.values()}
    print(f"    → 最佳 k 的集合 = {sorted(ks_seen, reverse=True)}；"
          f"{'相同' if len(ks_seen) == 1 else '**不同**'}")

    print("\n  Q3. bekko：Matryoshka 截斷 vs PCA 降維")
    q3: dict = {}
    for k in K_GRID:
        d = matryoshka_cmp.get(k)
        if d:
            q3[k] = d
            print(f"    k={k:<4d} Matryoshka {d['trunc_k']:.4f} vs "
                  f"PCA 前 k {d['pca_first_k']:.4f}（實際 {d['k_eff_pca']} 維）"
                  f" → Δ = {d['delta_trunc_minus_pca']:+.4f}，"
                  f"{'Matryoshka 勝' if d['delta_trunc_minus_pca'] > 0 else 'PCA 勝'}")

    print("\n  Q4. 斷崖在哪？（每個臂各自看：由大 k 往小 k 走時的最大單步降幅，")
    print("      以及首度跌破「該臂最佳值 − 5 個百分點」的 k）")
    cliff: dict = {}
    for key in loaded:
        res = tasks["sentiment_cv"][key]
        print(f"    {key}")
        cliff[key] = {}
        for arm in res["arms"]:
            scores = {k: res["results"][arm][k]["seed"]["acc_mean"] for k in K_GRID}
            ordered = sorted(K_GRID)
            # 由大 k 往小 k 走：比較 (k_hi → k_lo) 的降幅
            drops = [(ordered[i + 1], ordered[i], scores[ordered[i + 1]] - scores[ordered[i]])
                     for i in range(len(ordered) - 1)]
            k_hi, k_lo, cd = max(drops, key=lambda t: t[2])
            best = max(scores.values())
            below = [k for k in ordered if scores[k] < best - 0.05]
            cliff[key][arm] = {
                "largest_drop_from": k_hi, "largest_drop_to": k_lo,
                "largest_drop": cd,
                "first_below_best_minus_5pct": below[0] if below else None,
                "scores": scores}
            print(f"      {arm:14s} 最大降幅 k={k_hi} → {k_lo}（Δ = −{cd:.4f}）；"
                  f"首度跌破最佳值 5 個百分點：k = "
                  f"{below[0] if below else '（掃描範圍內未跌破）'}")

    print("\n  Q5. 這個方案是否真的解決了「維度不是共同受控變數」？")
    ctrl: dict = {}
    for key in loaded:
        res = tasks["sentiment_cv"][key]
        dim = int(embs[key].shape[1])
        ceiling = tasks["ref_full"][key]["results"]["trunc_k"][dim]["seed"]["acc_mean"]
        curve = {k: res["results"]["pca_first_k"][k]["seed"]["acc_mean"] for k in K_GRID}
        # 「需要多少維」＝**最小**的 k，使 PCA 前 k 維已在天花板 −1 個百分點之內
        need = min([k for k in K_GRID if curve[k] >= ceiling - 0.01], default=None)
        ctrl[key] = {
            "dim_native": dim, "ceiling_full_dim": ceiling,
            "k_eff_at_256": res["diag"]["k_eff"].get("pca_first_k@256"),
            "abtt_256": res["results"]["abtt_k"][256]["seed"]["acc_mean"],
            "pca_first_8": curve[8], "abtt_8": res["results"]["abtt_k"][8]["seed"]["acc_mean"],
            "k_needed_within_1pct_of_ceiling": need,
            "spread_at_k8": None,
        }
        print(f"    {key:28s} 原生 {dim} 維天花板 {ceiling:.4f}（trunc 全維）；"
              f"PCA 前 k 只要 **k = {need}** 就到達「天花板 −1 個百分點」之內")
        print(f"      {'':28s} PCA 在 k=256 的實際維度 = {ctrl[key]['k_eff_at_256']}；"
              f"k=8：abtt {ctrl[key]['abtt_8']:.4f} vs 前 k {ctrl[key]['pca_first_8']:.4f}")
    for k in (256, 64, 8):
        vals = [tasks["sentiment_cv"][key]["results"]["pca_first_k"][k]["seed"]["acc_mean"]
                for key in loaded]
        print(f"    → 固定 k={k}（pca_first_k 臂，同一降維算子）時，三個模型的準確率"
              f"極差 = {max(vals) - min(vals):.4f}（{min(vals):.4f} ~ {max(vals):.4f}）")
    spread = max(v["abtt_256"] for v in ctrl.values()) - min(v["abtt_256"] for v in ctrl.values())
    print(f"    → 同一 k（k=256、abtt 臂）下三個模型的準確率極差 = {spread:.4f}")

    # ---------------- 存檔 ----------------
    head("十、輸出檔案")
    payload = {
        "meta": {
            "script": "s08_pca_dim_ablation.py", "agent": "DIM1",
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": platform.python_version(), "packages": vers,
            "seeds": SEEDS, "n_folds": N_FOLDS, "k_grid": K_GRID,
            "abtt_drop": ABTT_DROP, "C_reg": C_REG, "n_sentences": len(texts),
            "models_loaded": loaded, "models_failed": failures,
            "hf_revisions": {k: metas[k]["hf_revision"] for k in loaded},
            "embedding_dims": {k: int(embs[k].shape[1]) for k in loaded},
            "model_meta": {k: {kk: vv for kk, vv in metas[k].items()
                               if not isinstance(vv, np.ndarray)} for k in loaded},
        },
        "self_test": st, "pca_info": pca_info, "pc1_pc2_langid": pc12,
        "pc_content_diagnostic": pc_diag, "paired_tests": paired,
        "tasks": {t: {m: {"results": r["results"], "diag": r["diag"]}
                      for m, r in d.items()} for t, d in tasks.items()},
        "matryoshka_vs_pca": matryoshka_cmp, "abtt_r_sensitivity": r_sens,
        "verdicts": {"q1_abtt_vs_first": q1, "q2_best_k": best_k, "q3_matryoshka": q3,
                     "q4_cliff": cliff, "q5_control": ctrl,
                     "ceiling_spread_k256": spread},
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                         encoding="utf-8")
    print(f"  ✅ {JSON_PATH}")

    make_figure(tasks, loaded, matryoshka_cmp, pc_diag)
    print(f"  ✅ {FIG_PNG}")
    print(f"  ✅ {FIG_PDF}")
    print(f"\n  總耗時 {time.perf_counter() - T_START:.1f} s")
    print("\nDONE")
    return 0


# ---------------------------------------------------------------------------
# 10. 圖：維度 vs 準確率（多個臂 × 多個模型）
# ---------------------------------------------------------------------------
def make_figure(tasks: dict, loaded: list[str], matryoshka_cmp: dict,
                pc_diag: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "legend.frameon": False,
    })
    # 圖上文字一律用英文（WSL 內無 CJK 字型，中文會變成豆腐方塊）
    colors = {"potion-multilingual-128M": "#1f77b4",
              "bekko-embedding-v1-a25m": "#d62728",
              "M2V_multilingual_output": "#2ca02c"}
    xk = sorted(K_GRID)
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))

    def curve(res, arm, kk, field="seed", metric="acc"):
        d = res["results"].get(arm, {}).get(kk, {}).get(field)
        return (d[f"{metric}_mean"], d[f"{metric}_std"]) if d else (float("nan"), 0.0)

    def proxy_legend(ax, models, arm_styles, loc="best", ncol=2, fontsize=6.0):
        """代理圖例：顏色 = 模型、線型 = 臂。避免每個 (模型 × 臂) 都佔一個條目。"""
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], color=colors.get(m, "#444444"), lw=2.0,
                          label=MODEL_SHORT.get(m, m)) for m in models]
        handles += [Line2D([], [], color="black", lw=1.6, ls=ls, marker=mk, ms=3.5,
                           label=lab) for ls, mk, lab in arm_styles]
        ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=fontsize)

    MODEL_STYLES = [("-", "o", "ABTT (drop PC1-2)"),
                    ("--", "s", "PCA top-$k$")]

    # --- (a) 主協定 ---
    ax = axes[0][0]
    for m in loaded:
        res = tasks["sentiment_cv"][m]
        for arm, ls, mk in (("abtt_k", "-", "o"), ("pca_first_k", "--", "s")):
            y = [curve(res, arm, k)[0] for k in xk]
            e = [curve(res, arm, k)[1] for k in xk]
            ax.errorbar(xk, y, yerr=e, ls=ls, marker=mk, ms=3.5, lw=1.4,
                        color=colors.get(m, "#444444"),
                        alpha=0.95 if arm == "abtt_k" else 0.55)
    ax.axhline(0.5, color="gray", lw=1.0, ls=":")
    ax.text(xk[-1], 0.505, "chance 0.50", ha="right", va="bottom", fontsize=7, color="gray")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xk)
    ax.set_xticklabels([str(k) for k in xk])
    ax.set_xlabel("$k$ (requested dimension)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.44, 1.03)
    ax.set_title("(a) Main protocol — bilingual sentiment\n"
                 "stratified 5-fold CV $\\times$ 5 seeds (160 sentences)", fontsize=9)
    proxy_legend(ax, loaded, MODEL_STYLES, loc="lower left", ncol=2)

    # --- (b) Δ = abtt − top-k（含不標準化的 ABTT）---
    ax = axes[0][1]
    for m in loaded:
        res = tasks["sentiment_cv"][m]
        d1 = [curve(res, "abtt_k", k)[0] - curve(res, "pca_first_k", k)[0] for k in xk]
        d2 = [curve(res, "abtt_nostd_k", k)[0] - curve(res, "pca_first_k", k)[0]
              for k in xk]
        ax.plot(xk, d1, marker="o", ms=4, lw=1.5, color=colors.get(m, "#444444"))
        ax.plot(xk, d2, marker="^", ms=3.5, lw=1.1, ls="--",
                color=colors.get(m, "#444444"), alpha=0.65)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xk)
    ax.set_xticklabels([str(k) for k in xk])
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\Delta$ accuracy (ABTT $-$ PCA top-$k$)")
    ax.set_title("(b) Is ABTT necessary at low $k$?\n(above 0 = ABTT better)", fontsize=9)
    proxy_legend(ax, loaded,
                 [("-", "o", "ABTT + standardise"), ("--", "^", "ABTT, no standardise")],
                 loc="best", ncol=2)

    # --- (c) 對照臂 ---
    ax = axes[0][2]
    for m in loaded:
        res = tasks["sentiment_cv"][m]
        ax.plot(xk, [curve(res, "pca_std_k", k)[0] for k in xk], marker="^", ms=3.5,
                lw=1.4, color=colors.get(m, "#444444"))
        ax.plot(xk, [curve(res, "trunc_k", k)[0] for k in xk], marker="x", ms=4,
                lw=1.1, ls=":", color=colors.get(m, "#444444"))
    ax.axhline(0.5, color="gray", lw=1.0, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xk)
    ax.set_xticklabels([str(k) for k in xk])
    ax.set_xlabel("$k$")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.46, 1.03)
    ax.set_title("(c) Control arms — standardised PCA\nvs plain truncation (Matryoshka-style)", fontsize=9)
    proxy_legend(ax, loaded,
                 [("-", "^", "PCA top-$k$ + standardise"), (":", "x", "truncate, no PCA")],
                 loc="lower right", ncol=1)

    # --- (d) 跨語言遷移 ---
    ax = axes[1][0]
    if "transfer" in tasks:
        for m in loaded:
            if m not in tasks["transfer"]:
                continue
            res = tasks["transfer"][m]
            for arm, ls, mk in (("abtt_k", "-", "o"), ("pca_first_k", "--", "s")):
                y = [curve(res, arm, k, "mean")[0] for k in xk]
                ax.plot(xk, y, ls=ls, marker=mk, ms=3.5, lw=1.5, color=colors.get(m, "#444444"),
                        alpha=0.95 if arm == "abtt_k" else 0.55)
        # 確定性標註：對照組是「同一 (臂,k,方向) 在 5 個種子下的唯一值個數」
        _m0 = next((mm for mm in loaded if mm in tasks["transfer"]), None)
        if _m0 is not None:
            det = tasks["transfer"][_m0]["diag"]["raw_acc_by_seed"]
            n_keys = len(det)
            n_identical = sum(1 for v in det.values()
                              if len({round(x[1], 12) for x in v}) == 1)
            ax.text(0.02, 0.02,
                    f"fixed split, deterministic solver\n"
                    f"{n_identical}/{n_keys} configs identical across 5 seeds",
                    transform=ax.transAxes, fontsize=6.5, color="dimgray")
    ax.axhline(0.5, color="gray", lw=1.0, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xk)
    ax.set_xticklabels([str(k) for k in xk])
    ax.set_xlabel("$k$")
    ax.set_ylabel("accuracy (mean of zh$\\rightarrow$en, en$\\rightarrow$zh)")
    ax.set_ylim(0.44, 1.03)
    ax.set_title("(d) Cross-lingual transfer — the regime where\n"
                 "F10 reports ABTT to be decisive", fontsize=9)
    proxy_legend(ax, loaded, MODEL_STYLES, loc="lower left", ncol=2)

    # --- (e) 語言辨識 ---
    ax = axes[1][1]
    if "langid_cv" in tasks:
        for m in loaded:
            res = tasks["langid_cv"][m]
            for arm, ls, mk in (("pca_first_k", "--", "s"), ("abtt_k", "-", "o")):
                y = [curve(res, arm, k)[0] for k in xk]
                e = [curve(res, arm, k)[1] for k in xk]
                ax.errorbar(xk, y, yerr=e, ls=ls, marker=mk, ms=3.5, lw=1.4,
                            color=colors.get(m, "#444444"),
                            alpha=0.95 if arm == "abtt_k" else 0.55)
    ax.axhline(0.5, color="gray", lw=1.0, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xk)
    ax.set_xticklabels([str(k) for k in xk])
    ax.set_xlabel("$k$")
    ax.set_ylabel("language-ID accuracy (zh vs en)")
    ax.set_ylim(0.44, 1.10)
    ax.set_title("(e) What PC1+PC2 actually encode\n"
                 "(language ID vs sentiment, 2-D probe)", fontsize=9)
    if pc_diag:
        lines = []
        for m in loaded:
            c = pc_diag.get(m, {}).get("PC1+PC2")
            if c:
                lines.append(f"{MODEL_SHORT.get(m, m)}: lang {c['language']['mean']:.2f}"
                             f" / sent {c['sentiment']['mean']:.2f}")
        if lines:
            ax.text(0.02, 0.98, "PC1+PC2 only:\n" + "\n".join(lines),
                    transform=ax.transAxes, fontsize=6.2, va="top", color="black",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    proxy_legend(ax, loaded, MODEL_STYLES, loc="lower right", ncol=1)

    # --- (f) bekko：Matryoshka vs PCA ---
    ax = axes[1][2]
    if matryoshka_cmp and BEKKO in tasks.get("sentiment_cv", {}):
        m = BEKKO
        res = tasks["sentiment_cv"][m]
        ax.plot(xk, [curve(res, "trunc_k", k)[0] for k in xk], marker="D", ms=4, lw=1.6,
                color="#9467bd", label="Matryoshka truncation (384$\\rightarrow k$)")
        ax.plot(xk, [curve(res, "pca_first_k", k)[0] for k in xk], marker="s", ms=4,
                lw=1.6, color="#8c564b", label="PCA top-$k$")
        ax.plot(xk, [curve(res, "abtt_k", k)[0] for k in xk], marker="o", ms=4, lw=1.6,
                color="#d62728", label="ABTT")
        eff = res["diag"]["k_eff"].get("pca_first_k@256") or res["diag"]["rank"]
        ax.axvline(eff, color="black", lw=1.0, ls="-.", alpha=0.5)
        ax.text(eff * 0.93, 0.50, f"PCA rank cap {eff}", rotation=90,
                fontsize=6.2, color="black", va="bottom", ha="right")
        ax.axhline(0.5, color="gray", lw=1.0, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xk)
        ax.set_xticklabels([str(k) for k in xk])
        ax.set_xlabel("$k$")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.47, 1.03)
        ax.set_title("(f) bekko-a25m — Matryoshka vs PCA\n"
                     "(PCA path capped at $n_{tr}-1$ = 127 dims)", fontsize=9)
        ax.legend(fontsize=6.5, loc="center left")

    fig.suptitle("s08 — PCA as a common dimension-reduction operator for cross-model "
                 "dimension ablation (DIM1, WSL2)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG_PNG)
    fig.savefig(FIG_PDF)
    plt.close(fig)


if __name__ == "__main__":
    tee = Tee(TXT_PATH)
    sys.stdout = tee
    code = 1
    try:
        code = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    finally:
        print(f"\n[exit code: {code}]")
        tee.close()
    sys.exit(code)
