"""s12：torch 可微分批次模擬器 + 容量網格。

動機（來自 s09/s10/s11 的實測）：參數平移每步要 8·n·depth+1 次前向，訓練因此在
固定步數下**嚴重欠擬合**（train_acc 只有 0.2–0.5，隨機 = 0.125）——曲線量到的是
「最佳化不足」而不是「容量到頂」。

做法：把 s09 的批次模擬器改寫成 torch，讓 autograd 一次 backward 取代 8·n·depth+1 次前向。
  * 閘作用方式與 s09 完全相同（big-endian：軸 k ↔ qubit k），差別只在可微分。
  * CX 用**預先算好的置換索引**（index_select 可微分），避免 in-place 破壞計算圖。
  * **自檢**：機率必須與 s09 的 NumPy 版一致（<1e-12），梯度必須與參數平移一致（<1e-9）。
    自檢沒過就不跑網格。
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

NS = tuple(range(3, 11))
DEPTHS = (1, 2, 3, 4)
SEEDS = (0, 1, 2)
STEPS = 400
LR = 0.15
OUT = HERE / "out" / "s12_torch_capacity_grid.json"
_CDT = torch.complex128


class TorchSim:
    """可微分批次狀態向量模擬器（big-endian：軸 k ↔ qubit k）。"""

    def __init__(self, n: int, batch: int, device="cpu"):
        self.n, self.device = n, device
        psi = torch.zeros((batch, 1 << n), dtype=_CDT, device=device)
        psi[:, 0] = 1.0
        self.psi = psi

    def _view(self, t: int):
        b = self.psi.shape[0]
        return self.psi.reshape(b, -1, 2, 1 << (self.n - 1 - t))

    def ry(self, th, t: int) -> None:
        th = th.reshape(-1, 1, 1)
        c, s = torch.cos(th / 2), torch.sin(th / 2)
        v = self._view(t)
        a, b = v[..., 0, :], v[..., 1, :]
        self.psi = torch.stack([c * a - s * b, s * a + c * b], dim=-2).reshape(self.psi.shape)

    def rz(self, th, t: int) -> None:
        th = th.reshape(-1, 1, 1)
        v = self._view(t)
        a, b = v[..., 0, :], v[..., 1, :]
        self.psi = torch.stack([torch.exp(-0.5j * th) * a, torch.exp(0.5j * th) * b],
                               dim=-2).reshape(self.psi.shape)

    def cx(self, c: int, t: int) -> None:
        self.psi = self.psi[:, _cx_perm(self.n, c, t, self.device)]

    def class_probs(self) -> torch.Tensor:
        p = self.psi.abs() ** 2
        return p.reshape(p.shape[0], s09.N_CLASS, 1 << (self.n - 3)).sum(dim=2)


_PERM: dict = {}


def _cx_perm(n: int, c: int, t: int, device):
    key = (n, c, t)
    if key not in _PERM:
        idx = np.arange(1 << n)
        cbit = (idx >> (n - 1 - c)) & 1
        perm = idx ^ ((1 << (n - 1 - t)) * cbit)
        _PERM[key] = torch.as_tensor(perm.astype(np.int64))
    return _PERM[key].to(device)


def ring(n, step):
    return [(i, (i + step) % n) for i in range(n)]


def probs_torch(n, depth, Xa, w, device="cpu"):
    """Xa: (B,n) numpy float；w: (depth,n,2) torch tensor（可訓練）。"""
    X = torch.as_tensor(np.asarray(Xa), dtype=torch.float64, device=device)
    sim = TorchSim(n, X.shape[0], device)
    for i in range(n):
        sim.ry(X[:, i], i)
    for d in range(depth):
        for i in range(n):
            sim.ry(w[d, i, 0], i)
            sim.rz(w[d, i, 1], i)
        for c, tt in ring(n, 1 if d % 2 == 0 else 2):
            sim.cx(c, tt)
    return sim.class_probs()


def ce_loss(n, depth, Xa, y, w, device="cpu"):
    P = torch.clamp(probs_torch(n, depth, Xa, w, device), min=1e-12)
    return -torch.log(P[torch.arange(len(y), device=device), torch.as_tensor(y, device=device)]).mean()


def selftest() -> bool:
    print("=== 自檢：torch 版 vs s09 NumPy 版 ===", flush=True)
    ok = True
    rng = np.random.default_rng(7)
    for n, depth in ((3, 1), (4, 2), (5, 2)):
        w_np = rng.normal(0.0, 0.5, size=(depth, n, 2))
        Xa = rng.random((6, n))
        s09.DEPTH = depth
        P_np = s09.probs_for(n, Xa, w_np)
        w_t = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
        P_t = probs_torch(n, depth, Xa, w_t)
        d_p = float(np.max(np.abs(P_t.detach().numpy() - P_np)))
        y = rng.integers(0, s09.N_CLASS, size=6)
        loss = ce_loss(n, depth, Xa, y, w_t)
        loss.backward()
        g_t = w_t.grad.numpy()
        _, g_np = s09.loss_and_grad(n, Xa, y, w_np)
        d_g = float(np.max(np.abs(g_t - g_np)))
        rel = d_g / max(float(np.max(np.abs(g_np))), 1e-30)
        print("  n=%d depth=%d: |dP|=%.3e  |dgrad|=%.3e (rel %.2e)" % (n, depth, d_p, d_g, rel), flush=True)
        ok = ok and d_p < 1e-12 and rel < 1e-9
    print("自檢", "通過" if ok else "**失敗**", flush=True)
    return ok


def main() -> int:
    if not selftest():
        return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device, flush=True)
    Z_tr, y_tr, Z_te, y_te = s09.make_data()
    rows, t_all = [], time.perf_counter()
    print("%4s %5s %8s %9s %9s %9s %9s" % ("n", "depth", "params", "train_acc", "test_acc", "ECE", "sec/step"), flush=True)
    for n in NS:
        for depth in DEPTHS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                w = torch.tensor(rng.normal(0.0, 0.3, size=(depth, n, 2)), dtype=torch.float64,
                                 device=device, requires_grad=True)
                Xa, Xb = s09.encode(Z_tr[:, :n]), s09.encode(Z_te[:, :n])
                t0 = time.perf_counter()
                for _ in range(STEPS):
                    loss = ce_loss(n, depth, Xa, y_tr, w, device)
                    loss.backward()
                    with torch.no_grad():
                        w -= LR * w.grad
                        w.grad = None
                dt = time.perf_counter() - t0
                with torch.no_grad():
                    Pa = probs_torch(n, depth, Xa, w, device).cpu().numpy()
                    Pb = probs_torch(n, depth, Xb, w, device).cpu().numpy()
                conf = Pb.max(axis=1)
                hit = (Pb.argmax(axis=1) == y_te).astype(float)
                edges, ecc = np.linspace(0, 1, 11), 0.0
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (conf > lo) & (conf <= hi)
                    if m.any():
                        ecc += float(m.mean()) * abs(hit[m].mean() - conf[m].mean())
                row = {"n_qubit": n, "depth": depth, "seed": seed, "n_trainable": depth * n * 2,
                       "train_loss": round(float(loss), 4),
                       "train_acc": round(float((Pa.argmax(axis=1) == y_tr).mean()), 4),
                       "test_acc": round(float((Pb.argmax(axis=1) == y_te).mean()), 4),
                       "test_ece": round(float(ecc), 4), "seconds": round(dt, 2),
                       "sec_per_step": round(dt / STEPS, 5),
                       "forwards_per_step_if_paramshift": 8 * n * depth + 1}
                rows.append(row)
                print("%4d %5d %8d %9.4f %9.4f %9.4f %9.5f" % (
                    n, depth, row["n_trainable"], row["train_acc"], row["test_acc"],
                    row["test_ece"], row["sec_per_step"]), flush=True)
            sub = [r for r in rows if r["n_qubit"] == n and r["depth"] == depth]
            print("      -> n=%2d depth=%d: test %.4f ± %.4f | train %.4f | ECE %.4f" % (
                n, depth, np.mean([s["test_acc"] for s in sub]), np.std([s["test_acc"] for s in sub]),
                np.mean([s["train_acc"] for s in sub]), np.mean([s["test_ece"] for s in sub])), flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"experiment": "s12_torch_capacity_grid", "partial": True,
                                   "device": device, "steps": STEPS, "rows": rows,
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
                          "sec_per_step": round(float(np.mean([s["sec_per_step"] for s in sub])), 6)})
    OUT.write_text(json.dumps({"experiment": "s12_torch_capacity_grid", "partial": False,
                               "device": device, "classes": s09.N_CLASS, "n_train": s09.N_TRAIN,
                               "n_test": s09.N_TEST, "steps": STEPS, "lr": LR, "seeds": list(SEEDS),
                               "chance": 1.0 / s09.N_CLASS,
                               "gradient": "torch autograd through the batched statevector simulator",
                               "readout": "qubits 0,1,2 -> 8 classes (fixed; requires n>=3)",
                               "total_seconds": round(time.perf_counter() - t_all, 1),
                               "cells": cells, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("total %.1f s；寫入 %s" % (time.perf_counter() - t_all, OUT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
