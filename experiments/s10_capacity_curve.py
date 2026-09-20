"""s10：容量掃描正式版（RQ1 缺失的那條曲線）。

設計（每一條都為了讓曲線有意義）：
  1. **固定任務**：8 類、256 維輸入，訊號只在前 12 個主成分（沿用 s09 的 make_data）。
  2. **嵌套輸入**：n 個 qubit 用前 n 個 PCA 分量 ⇒ 更多 qubit ＝ 同一任務的更多資訊。
  3. **固定讀出**：前 3 個 qubit（qubit 0,1,2）的基底模式 → 8 類，**與 n 無關**
     （2026-09-19 修正 s09.class_probs 的軸向錯誤；舊版讀出的是最低位 3 個 qubit，
       n>3 時會漂移到低訊號主成分，曲線因此失去意義）。
     ⇒ n 必須 ≥ 3：8 個類別機率最少需要 3 個 qubit 的投影讀出（這本身就是編碼的硬限制）。
  4. **梯度**：參數平移（s09 的 loss_and_grad），所以每步成本 ≈ (8n+1) 次前向
     —— 這一項會被同時量測，用來支撐「瓶頸是梯度不是算力」的結論。
  5. 每個 (n, seed) 都記錄 wall-clock，所以成本曲線與準確率曲線來自同一次執行。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s09_capacity_scan as s09  # noqa: E402

NS = tuple(range(3, 11))          # 8 類需要 3 個讀出 qubit ⇒ 從 3 起
SEEDS = (0, 1, 2, 3, 4)
STEPS = 100   # 舊值 60 會讓模型停在最佳化不足的狀態（train_acc 僅 ~0.5）
DEPTH = 2
LR = 0.3
OUT = HERE / "out" / "s10_capacity_curve.json"


def train_one(n, Z_tr, y_tr, Z_te, y_te, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.3, size=(DEPTH, n, 2))
    Xa, Xb = s09.encode(Z_tr[:, :n]), s09.encode(Z_te[:, :n])
    hist, t0 = [], time.perf_counter()
    for step in range(STEPS):
        loss, grad = s09.loss_and_grad(n, Xa, y_tr, w)
        w -= LR * grad
        if step % 20 == 0 or step == STEPS - 1:
            hist.append(round(loss, 4))
    dt = time.perf_counter() - t0

    def acc(X, y):
        return float((s09.probs_for(n, X, w).argmax(axis=1) == y).mean())

    return {"n_qubit": n, "n_trainable": DEPTH * n * 2, "seed": seed,
            "train_loss": round(loss, 4), "train_acc": round(acc(Xa, y_tr), 4),
            "test_acc": round(acc(Xb, y_te), 4), "seconds": round(dt, 2),
            "sec_per_step": round(dt / STEPS, 4), "forwards_per_step": 8 * n + 1,
            "loss_history": hist}


def main() -> int:
    s09.DEPTH = DEPTH
    Z_tr, y_tr, Z_te, y_te = s09.make_data()
    print("s10 容量掃描：8 類、256 維、訊號在前 12 PC；n=%s；%d seeds；%d steps；depth=%d"
          % (list(NS), len(SEEDS), STEPS, DEPTH), flush=True)
    print("%5s %8s %10s %10s %11s %12s" % ("n", "params", "train_acc", "test_acc", "sec/step", "seed"), flush=True)
    rows = []
    t_all = time.perf_counter()
    for n in NS:
        for seed in SEEDS:
            r = train_one(n, Z_tr, y_tr, Z_te, y_te, seed)
            rows.append(r)
            print("%5d %8d %10.4f %10.4f %11.4f %12d" % (
                n, r["n_trainable"], r["train_acc"], r["test_acc"], r["sec_per_step"], seed), flush=True)
        sub = [r for r in rows if r["n_qubit"] == n]
        print("   n=%2d 平均 test_acc = %.4f  (逐 seed: %s)" % (
            n, np.mean([s["test_acc"] for s in sub]),
            [s["test_acc"] for s in sub]), flush=True)

    summary = []
    for n in NS:
        sub = [r for r in rows if r["n_qubit"] == n]
        summary.append({"n_qubit": n, "n_trainable": sub[0]["n_trainable"],
                        "test_acc_mean": round(float(np.mean([s["test_acc"] for s in sub])), 4),
                        "test_acc_std": round(float(np.std([s["test_acc"] for s in sub])), 4),
                        "train_acc_mean": round(float(np.mean([s["train_acc"] for s in sub])), 4),
                        "sec_per_step": round(float(np.mean([s["sec_per_step"] for s in sub])), 4)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment": "s10_capacity_curve", "classes": s09.N_CLASS, "dim": s09.DIM,
        "n_pc": s09.N_PC, "n_train": s09.N_TRAIN, "n_test": s09.N_TEST,
        "steps": STEPS, "depth": DEPTH, "lr": LR, "seeds": list(SEEDS),
        "gradient": "parameter-shift on quantum layer + classical chain rule (s09.loss_and_grad)",
        "readout": "first 3 qubits -> 8 classes (fixed, n>=3 required)",
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "summary": summary, "rows": rows,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print("total %.1f s；寫入 %s" % (time.perf_counter() - t_all, OUT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
