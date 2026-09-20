"""s11：容量斷崖完整網格（RQ1 的圖 fig:cliff）。

論文（paper.typ §貢獻摘要、§結果/容量斷崖）承諾的是
    「qubit 數 × 電路深度」的掃描，圖說要「準確率與 ECE」。
本腳本產出這兩條曲線。

設計：
  * 任務沿用 s09.make_data（8 類、256 維、訊號在前 12 個主成分）。
  * 嵌套輸入：n 個 qubit 吃前 n 個主成分。
  * **固定讀出**：qubit 0,1,2 的基底模式 → 8 類（與 n、depth 都無關）。
    ⇒ n ≥ 3：8 個類別機率最少需要 3 個 qubit 的投影讀出。
  * 每個 (n, depth, seed) 都記錄：train/test 準確率、ECE、NLL、Brier、秒數。
  * 梯度＝參數平移（量子層）＋古典連鎖律，所以每步 ≈ (8·n·depth+1) 次前向；
    這一項同時被量測，用來支撐「瓶頸是梯度」的結論。
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

NS = tuple(range(3, 11))          # 3..10
DEPTHS = (1, 2, 3, 4)
SEEDS = (0, 1, 2)
STEPS = 100
LR = 0.3
OUT = HERE / "out" / "s11_capacity_grid.json"


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """期望校準誤差（confidence 分箱後 |準確率 − 平均信心| 的加權和）。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    hit = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        out += float(m.mean()) * abs(hit[m].mean() - conf[m].mean())
    return float(out)


def nll_brier(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(probs, 1e-12, 1.0)
    nll = float(-np.log(p[np.arange(len(y)), y]).mean())
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(((p - onehot) ** 2).sum(axis=1).mean())
    return nll, brier


def train_one(n, depth, Z_tr, y_tr, Z_te, y_te, seed):
    s09.DEPTH = depth
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.3, size=(depth, n, 2))
    Xa, Xb = s09.encode(Z_tr[:, :n]), s09.encode(Z_te[:, :n])
    t0 = time.perf_counter()
    for step in range(STEPS):
        loss, grad = s09.loss_and_grad(n, Xa, y_tr, w)
        w -= LR * grad
    dt = time.perf_counter() - t0
    Pa, Pb = s09.probs_for(n, Xa, w), s09.probs_for(n, Xb, w)
    nll, brier = nll_brier(Pb, y_te)
    return {"n_qubit": n, "depth": depth, "seed": seed, "n_trainable": depth * n * 2,
            "train_loss": round(float(loss), 4),
            "train_acc": round(float((Pa.argmax(axis=1) == y_tr).mean()), 4),
            "test_acc": round(float((Pb.argmax(axis=1) == y_te).mean()), 4),
            "test_ece": round(ece(Pb, y_te), 4), "test_nll": round(nll, 4),
            "test_brier": round(brier, 4),
            "mean_max_conf": round(float(Pb.max(axis=1).mean()), 4),
            "seconds": round(dt, 2), "sec_per_step": round(dt / STEPS, 4),
            "forwards_per_step": 8 * n * depth + 1}


def main() -> int:
    Z_tr, y_tr, Z_te, y_te = s09.make_data()
    chance = 1.0 / s09.N_CLASS
    print("s11 容量網格：n=%s × depth=%s × %d seeds × %d steps（8 類，隨機 = %.3f）"
          % (list(NS), list(DEPTHS), len(SEEDS), STEPS, chance), flush=True)
    print("%4s %5s %8s %9s %9s %9s %9s %10s" % ("n", "depth", "params", "train_acc", "test_acc", "ECE", "sec/step", "forwards"), flush=True)
    rows, t_all = [], time.perf_counter()
    for n in NS:
        for depth in DEPTHS:
            for seed in SEEDS:
                r = train_one(n, depth, Z_tr, y_tr, Z_te, y_te, seed)
                rows.append(r)
                print("%4d %5d %8d %9.4f %9.4f %9.4f %9.4f %10d" % (
                    n, depth, r["n_trainable"], r["train_acc"], r["test_acc"],
                    r["test_ece"], r["sec_per_step"], r["forwards_per_step"]), flush=True)
            sub = [r for r in rows if r["n_qubit"] == n and r["depth"] == depth]
            print("      -> n=%2d depth=%d: test_acc %.4f ± %.4f | ECE %.4f | train_acc %.4f" % (
                n, depth, np.mean([s["test_acc"] for s in sub]), np.std([s["test_acc"] for s in sub]),
                np.mean([s["test_ece"] for s in sub]), np.mean([s["train_acc"] for s in sub])), flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"experiment": "s11_capacity_grid", "partial": True,
                                   "classes": s09.N_CLASS, "n_train": s09.N_TRAIN, "n_test": s09.N_TEST,
                                   "steps": STEPS, "seeds": list(SEEDS), "rows": rows,
                                   "completed_n": n}, indent=1, ensure_ascii=False), encoding="utf-8")

    cells = []
    for n in NS:
        for depth in DEPTHS:
            sub = [r for r in rows if r["n_qubit"] == n and r["depth"] == depth]
            cells.append({"n_qubit": n, "depth": depth, "n_trainable": depth * n * 2,
                          "test_acc_mean": round(float(np.mean([s["test_acc"] for s in sub])), 4),
                          "test_acc_std": round(float(np.std([s["test_acc"] for s in sub])), 4),
                          "train_acc_mean": round(float(np.mean([s["train_acc"] for s in sub])), 4),
                          "test_ece_mean": round(float(np.mean([s["test_ece"] for s in sub])), 4),
                          "test_nll_mean": round(float(np.mean([s["test_nll"] for s in sub])), 4),
                          "sec_per_step": round(float(np.mean([s["sec_per_step"] for s in sub])), 4)})
    OUT.write_text(json.dumps({"experiment": "s11_capacity_grid", "partial": False,
                               "classes": s09.N_CLASS, "dim": s09.DIM, "n_pc": s09.N_PC,
                               "n_train": s09.N_TRAIN, "n_test": s09.N_TEST, "steps": STEPS,
                               "depth": "swept", "lr": LR, "seeds": list(SEEDS), "chance": chance,
                               "gradient": "parameter-shift (quantum layer) + classical chain rule",
                               "readout": "qubits 0,1,2 -> 8 classes (fixed; requires n>=3)",
                               "total_seconds": round(time.perf_counter() - t_all, 1),
                               "cells": cells, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("total %.1f s；寫入 %s" % (time.perf_counter() - t_all, OUT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
