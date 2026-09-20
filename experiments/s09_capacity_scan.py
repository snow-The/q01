"""s09：容量掃描【快速驗證版】（RQ1 缺失的那條曲線）。

設計（每一條都是為了讓曲線有意義）：
  1. **固定任務**：8 類合成分類、256 維輸入，類別訊號只在前 12 個主成分。
  2. **嵌套輸入**：n 個 qubit 用前 n 個 PCA 分量 ⇒ 更多 qubit ＝ 同一任務的更多資訊。
  3. **固定讀出**：前 3 個 qubit 的基底模式 → 8 類（與 n 無關）。
  4. **批次化模擬**：狀態形狀 (B, 2^n)，閘沿批次軸作用 —— 逐樣本迴圈會慢 600 倍。
  5. **梯度**：參數平移只作用在量子層機率向量，古典連鎖律另外算
     （專案已知的坑：直接平移整個損失是錯的，見 dev/verify_ps_correct.py）。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "src"))

N_CLASS, N_PC, DIM = 8, 12, 256
N_TRAIN, N_TEST = 80, 300        # demo 用較小訓練集；正式掃描再放大
DEPTH, STEPS = 2, 40
SEEDS = (0, 1, 2)
NS = (6, 8)


class BatchedSim:
    """批次狀態向量模擬（big-endian：軸 k ↔ qubit k；閘沿批次軸以廣播作用）。"""

    def __init__(self, n: int, batch: int) -> None:
        self.n = n
        self.psi = np.zeros((batch, 1 << n), dtype=np.complex128)
        self.psi[:, 0] = 1.0

    def apply_gate_1q(self, kind: str, th, t: int) -> None:
        """th 可以是純量（可訓練參數）或 (B,) 向量（角度編碼）；一律廣播。"""
        th = np.asarray(th, dtype=np.float64).reshape(-1, 1, 1)
        if kind == "ry":
            c, s = np.cos(th / 2), np.sin(th / 2)
            u00, u01, u10, u11 = c, -s, s, c
        elif kind == "rz":
            u00, u01 = np.exp(-1j * th / 2), 0.0
            u10, u11 = 0.0, np.exp(1j * th / 2)
        else:
            raise KeyError(kind)
        b = self.psi.shape[0]
        v = self.psi.reshape(b, -1, 2, 1 << (self.n - 1 - t))
        a, bb = v[:, :, 0, :], v[:, :, 1, :]
        na = u00 * a + u01 * bb
        bb[...] = u10 * a + u11 * bb
        a[...] = na

    def apply_cx(self, c: int, t: int) -> None:
        b = self.psi.shape[0]
        v = self.psi.reshape((b,) + (2,) * self.n)
        mv = np.moveaxis(v, (1 + c, 1 + t), (1, 2))
        sub = mv[:, 1]
        sub[...] = sub[:, ::-1].copy()

    def class_probs(self) -> np.ndarray:
        """前 3 個 qubit（qubit 0,1,2）的基底模式 → 8 類，**與 n 無關**。

        索引約定是 big-endian：平坦索引 = Σ qubit_k · 2^(n-1-k)，所以 qubit 0 是最高位。
        要固定讀出 qubit 0,1,2，就得把它們放在**前面的軸**：
            (B, 2^n) → (B, 8, 2^(n-3)) → 對最後一軸求和
        （舊寫法 reshape(B, 2^(n-3), 8).sum(axis=1) 取到的是**最低位**的 3 個 qubit，
         即 qubit n-3,n-2,n-1；因為編碼是 PC i → qubit i，n>3 時讀出頭會漂移到低訊號的
         PC 上，讓容量曲線失去意義。2026-09-19 修正。）
        """
        p = np.abs(self.psi) ** 2
        return p.reshape(p.shape[0], N_CLASS, 1 << (self.n - 3)).sum(axis=2)


def ry_mat(th):
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_mat(th):
    return np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)]).astype(np.complex128)


def ring(n, step):
    return [(i, (i + step) % n) for i in range(n)]


def probs_for(n, Xa, w):
    sim = BatchedSim(n, Xa.shape[0])
    for i in range(n):
        sim.apply_gate_1q("ry", Xa[:, i], i)          # 角度編碼（逐樣本角度）
    for d in range(DEPTH):
        for i in range(n):
            sim.apply_gate_1q("ry", w[d, i, 0], i)    # 可訓練（純量）
            sim.apply_gate_1q("rz", w[d, i, 1], i)
        for c, tt in ring(n, 1 if d % 2 == 0 else 2):
            sim.apply_cx(c, tt)
    return sim.class_probs()


def loss_and_grad(n, Xa, y, w):
    P = np.clip(probs_for(n, Xa, w), 1e-12, None)
    loss = float(-np.log(P[np.arange(len(y)), y]).mean())
    dLdP = np.zeros_like(P)
    dLdP[np.arange(len(y)), y] = -1.0 / (P[np.arange(len(y)), y] * len(y))
    grad = np.zeros_like(w)
    for d in range(DEPTH):
        for i in range(n):
            for k in range(2):
                wp = w.copy(); wp[d, i, k] += np.pi / 2
                wm = w.copy(); wm[d, i, k] -= np.pi / 2
                grad[d, i, k] = float((dLdP * (probs_for(n, Xa, wp) - probs_for(n, Xa, wm)) / 2).sum())
    return loss, grad


def encode(x):
    return 2.0 * np.arcsin(np.sqrt(np.clip(x, 0.0, 1.0)))


def make_data(seed=12345):
    rng = np.random.default_rng(seed)
    proto = rng.normal(size=(N_CLASS, N_PC)) * 1.6
    y_tr = rng.integers(0, N_CLASS, N_TRAIN)
    y_te = rng.integers(0, N_CLASS, N_TEST)
    X_tr = rng.normal(size=(N_TRAIN, DIM)) * 0.9
    X_te = rng.normal(size=(N_TEST, DIM)) * 0.9
    X_tr[:, :N_PC] += proto[y_tr]
    X_te[:, :N_PC] += proto[y_te]
    mu = X_tr.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_tr - mu, full_matrices=False)
    Z_tr, Z_te = (X_tr - mu) @ Vt[:N_PC].T, (X_te - mu) @ Vt[:N_PC].T
    lo, hi = Z_tr.min(axis=0), Z_tr.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return np.clip((Z_tr - lo) / span, 0, 1), y_tr, np.clip((Z_te - lo) / span, 0, 1), y_te


def train_one(n, Z_tr, y_tr, Z_te, y_te, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.3, size=(DEPTH, n, 2))
    Xa, Xb = encode(Z_tr[:, :n]), encode(Z_te[:, :n])
    lr, hist = 0.3, []
    for step in range(STEPS):
        loss, grad = loss_and_grad(n, Xa, y_tr, w)
        w -= lr * grad
        if step % 20 == 0 or step == STEPS - 1:
            hist.append(round(loss, 4))
    def acc(X, y):
        return float((probs_for(n, X, w).argmax(axis=1) == y).mean())
    return {"n_qubit": n, "n_trainable": DEPTH * n * 2, "seed": seed,
            "train_loss": round(loss, 4), "train_acc": round(acc(Xa, y_tr), 4),
            "test_acc": round(acc(Xb, y_te), 4), "loss_history": hist}


def main() -> int:
    Z_tr, y_tr, Z_te, y_te = make_data()
    print("=" * 86)
    print("s09 容量掃描可行性 demo（8 類、256 維、訊號在前 12 PC、批次化模擬）")
    print(f"訓練 {N_TRAIN}／測試 {N_TEST}；depth={DEPTH}；{STEPS} 步；種子 {SEEDS}")
    print("=" * 86)
    print(f"  {'n':>3} {'可訓練角度':>10} {'輸入維':>7} {'訓練損失':>10} {'訓練準確':>9} {'測試準確':>14} {'耗時':>9}")
    rows, t_all = [], time.perf_counter()
    for n in NS:
        t0 = time.perf_counter()
        res = [train_one(n, Z_tr, y_tr, Z_te, y_te, s) for s in SEEDS]
        dt = time.perf_counter() - t0
        rows += res
        te = np.array([r["test_acc"] for r in res])
        print(f"  {n:3d} {res[0]['n_trainable']:10d} {n:7d} "
              f"{np.mean([r['train_loss'] for r in res]):10.4f} "
              f"{np.mean([r['train_acc'] for r in res]):9.4f} "
              f"{te.mean():.4f} ± {te.std():.4f}   {dt:8.1f}s")
        print(f"      損失軌跡（種子 0）: {res[0]['loss_history']}")
    print()
    print(f"  總耗時 {time.perf_counter() - t_all:.1f} s")
    print("  ⚠ 這是**可行性 demo**（小訓練集、60 步）。正式掃描必須逐 n 確認收斂，")
    print("    且 n≥12 建議改用 torch 後端的 autograd（一次 backward 取代 2p 次前向）。")
    out = ROOT / "projects" / "qbn-capacity-calibration" / "ml" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "s09_capacity_scan_demo.json").write_text(
        json.dumps({"config": {"depth": DEPTH, "steps": STEPS, "seeds": list(SEEDS),
                               "n_train": N_TRAIN, "n_test": N_TEST}, "rows": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  已寫出 {out / 's09_capacity_scan_demo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
