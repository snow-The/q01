"""s16：容量斷崖正式版（論文 fig:cliff 的資料來源）。

走過的兩個錯誤（都由實測否證，記錄下來以免重蹈）：
  1. s10/s11 用參數平移、100 步、SGD ⇒ 欠擬合（train_acc 0.2–0.5），量到的是最佳化不足。
  2. s14 用 Adam、2000 步仍卡在 train_acc 0.55 ⇒ 一度以為是表達能力上限；
     s15 把深度加到 8 立刻突破（train 0.84 / test 0.61）⇒ **上限是深度不足，不是硬限制**。
因此正式版固定用 **Adam + 1200 步**，並讓 depth 成為真正的掃描軸。

讀出固定為 qubit 0,1,2 → 8 類（與 n、depth 無關）；n ≥ 3 是 8 類投影讀出的硬需求。
梯度＝torch autograd 穿過批次狀態向量模擬器（s12，已對帳：機率 2.2e-16、梯度 1.3e-15）。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s09_capacity_scan as s09  # noqa: E402
import s12_torch_capacity as s12  # noqa: E402

NS = (3, 4, 6, 8, 10)
DEPTHS = (1, 2, 4, 6, 8)
SEEDS = (0, 1)
STEPS = 1200
LR = 0.05
OUT = HERE / "out" / "s16_capacity_final.json"


def ece(P, y, bins=10):
    conf, pred = P.max(axis=1), P.argmax(axis=1)
    hit = (pred == y).astype(float)
    edges, out = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += float(m.mean()) * abs(hit[m].mean() - conf[m].mean())
    return float(out)


def main() -> int:
    torch.set_default_dtype(torch.float64)
    Z_tr, y_tr, Z_te, y_te = s09.make_data()
    chance = 1.0 / s09.N_CLASS
    print("s16 正式容量曲線：n=%s × depth=%s × %d seeds × %d 步（Adam lr=%.2f）；隨機 = %.3f"
          % (list(NS), list(DEPTHS), len(SEEDS), STEPS, LR, chance), flush=True)
    print("%4s %5s %7s %9s %9s %9s %9s %8s" % ("n", "depth", "params", "train_acc", "test_acc", "ECE", "loss", "秒"), flush=True)
    rows, t_all = [], time.perf_counter()
    for n in NS:
        for depth in DEPTHS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                w = torch.tensor(rng.normal(0, 0.3, size=(depth, n, 2)), requires_grad=True)
                Xa, Xb = s09.encode(Z_tr[:, :n]), s09.encode(Z_te[:, :n])
                opt = torch.optim.Adam([w], lr=LR)
                t0 = time.perf_counter()
                loss = None
                for _ in range(STEPS):
                    loss = s12.ce_loss(n, depth, Xa, y_tr, w)
                    opt.zero_grad(); loss.backward(); opt.step()
                dt = time.perf_counter() - t0
                with torch.no_grad():
                    Pa = s12.probs_torch(n, depth, Xa, w).numpy()
                    Pb = s12.probs_torch(n, depth, Xb, w).numpy()
                rows.append({"n_qubit": n, "depth": depth, "seed": seed, "n_trainable": depth * n * 2,
                             "train_acc": round(float((Pa.argmax(1) == y_tr).mean()), 4),
                             "test_acc": round(float((Pb.argmax(1) == y_te).mean()), 4),
                             "test_ece": round(ece(Pb, y_te), 4),
                             "train_loss": round(float(loss.detach()), 4), "seconds": round(dt, 1)})
                print("%4d %5d %7d %9.4f %9.4f %9.4f %9.4f %8.1f" % (
                    n, depth, depth * n * 2, rows[-1]["train_acc"], rows[-1]["test_acc"],
                    rows[-1]["test_ece"], rows[-1]["train_loss"], rows[-1]["seconds"]), flush=True)
            sub = [r for r in rows if r["n_qubit"] == n and r["depth"] == depth]
            print("      -> n=%2d d=%d: train %.4f | test %.4f | ECE %.4f" % (
                n, depth, np.mean([s["train_acc"] for s in sub]),
                np.mean([s["test_acc"] for s in sub]), np.mean([s["test_ece"] for s in sub])), flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"experiment": "s16_capacity_final", "partial": True, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    cells = []
    for n in NS:
        for depth in DEPTHS:
            sub = [r for r in rows if r["n_qubit"] == n and r["depth"] == depth]
            cells.append({"n_qubit": n, "depth": depth, "n_trainable": depth * n * 2,
                          "train_acc_mean": round(float(np.mean([s["train_acc"] for s in sub])), 4),
                          "test_acc_mean": round(float(np.mean([s["test_acc"] for s in sub])), 4),
                          "test_ece_mean": round(float(np.mean([s["test_ece"] for s in sub])), 4)})
    OUT.write_text(json.dumps({
        "experiment": "s16_capacity_final", "partial": False,
        "classes": s09.N_CLASS, "dim": s09.DIM, "n_pc": s09.N_PC,
        "n_train": s09.N_TRAIN, "n_test": s09.N_TEST, "steps": STEPS, "lr": LR,
        "optimizer": "Adam", "seeds": list(SEEDS), "chance": chance,
        "gradient": "torch autograd through batched statevector simulator (s12, verified 1.3e-15 vs parameter-shift)",
        "readout": "qubits 0,1,2 -> 8 classes (fixed; n>=3 required)",
        "baselines": {"nearest_centroid_6pc": 0.8233, "logistic_6pc": 0.8033,
                      "nearest_centroid_3pc": 0.7067, "nearest_centroid_12pc": 0.8467},
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "cells": cells, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("total %.1f s；寫入 %s" % (time.perf_counter() - t_all, OUT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
