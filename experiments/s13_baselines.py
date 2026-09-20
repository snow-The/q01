"""s13：同一任務的古典基線 —— 判斷量子層的 0.2–0.4 到底是好是壞。

沒有基線就無法解讀容量曲線：若最近質心/logistic 在同一組主成分上就能拿到 0.9，
那量子層量到的是「最佳化失敗」而不是「容量到頂」。
"""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import s09_capacity_scan as s09

Z_tr, y_tr, Z_te, y_te = s09.make_data()
chance = 1.0 / s09.N_CLASS
print("任務：%d 類、訓練 %d、測試 %d、隨機 = %.4f" % (s09.N_CLASS, len(y_tr), len(y_te), chance))
print("%6s %14s %14s %14s" % ("n_pc", "最近質心", "logistic", "多項式(2次) logistic"))

def acc(p, y):
    return float((p == y).mean())

for n in (3, 4, 5, 6, 8, 10, 12):
    A, B = Z_tr[:, :n], Z_te[:, :n]
    # 最近質心
    cent = np.stack([A[y_tr == c].mean(axis=0) for c in range(s09.N_CLASS)])
    pred_nc = ((B[:, None, :] - cent[None]) ** 2).sum(-1).argmin(axis=1)
    # logistic（梯度下降，含標準化）
    mu, sd = A.mean(0), A.std(0) + 1e-9
    As, Bs = (A - mu) / sd, (B - mu) / sd
    W = np.zeros((n, s09.N_CLASS)); b = np.zeros(s09.N_CLASS)
    for _ in range(800):
        s = As @ W + b
        s -= s.max(1, keepdims=True)
        P = np.exp(s); P /= P.sum(1, keepdims=True)
        Y = np.zeros_like(P); Y[np.arange(len(y_tr)), y_tr] = 1
        G = (P - Y) / len(y_tr)
        W -= 2.0 * (As.T @ G); b -= 2.0 * G.sum(0)
    pred_lr = (Bs @ W + b).argmax(1)
    # 二次特徵 logistic
    A2 = np.hstack([As, As ** 2]); B2 = np.hstack([Bs, Bs ** 2])
    W2 = np.zeros((A2.shape[1], s09.N_CLASS)); b2 = np.zeros(s09.N_CLASS)
    for _ in range(800):
        s = A2 @ W2 + b2
        s -= s.max(1, keepdims=True)
        P = np.exp(s); P /= P.sum(1, keepdims=True)
        Y = np.zeros_like(P); Y[np.arange(len(y_tr)), y_tr] = 1
        G = (P - Y) / len(y_tr)
        W2 -= 2.0 * (A2.T @ G); b2 -= 2.0 * G.sum(0)
    pred_q = (B2 @ W2 + b2).argmax(1)
    print("%6d %14.4f %14.4f %14.4f" % (n, acc(pred_nc, y_te), acc(pred_lr, y_te), acc(pred_q, y_te)))
