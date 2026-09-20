"""s15：表達能力上限探針 —— train_acc 卡在 0.55 是深度造成的嗎？

s14 的實測：SGD 400/2000 步與 Adam 2000/4000 步，train_acc 全部落在 0.52–0.55，
換優化器與加步數都無效。這指向**表達能力上限**而非最佳化不足。
本腳本把深度一路加到 8，若上限不動，就是這個 ansatz/讀出架構的硬限制。
"""
import pathlib, sys, time
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import s09_capacity_scan as s09
import s12_torch_capacity as s12

Z_tr, y_tr, Z_te, y_te = s09.make_data()
torch.set_default_dtype(torch.float64)
N, DEPTHS, SEEDS, STEPS, LR = 6, (1, 2, 4, 6, 8), (0, 1), 1500, 0.05

print("n=%d、Adam lr=%.2f、%d 步、%d seeds；比較不同深度能否突破 train_acc 0.55" % (N, LR, STEPS, len(SEEDS)), flush=True)
print("%6s %8s %10s %10s %9s %8s" % ("depth", "params", "train_acc", "test_acc", "loss", "秒"), flush=True)
for depth in DEPTHS:
    tas, tes, ls, dts = [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        w = torch.tensor(rng.normal(0, 0.3, size=(depth, N, 2)), requires_grad=True)
        Xa, Xb = s09.encode(Z_tr[:, :N]), s09.encode(Z_te[:, :N])
        opt = torch.optim.Adam([w], lr=LR)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            loss = s12.ce_loss(N, depth, Xa, y_tr, w)
            opt.zero_grad(); loss.backward(); opt.step()
        dt = time.perf_counter() - t0
        with torch.no_grad():
            ta = float((s12.probs_torch(N, depth, Xa, w).argmax(1) == y_tr).float().mean())
            te = float((s12.probs_torch(N, depth, Xb, w).argmax(1) == y_te).float().mean())
        tas.append(ta); tes.append(te); ls.append(float(loss)); dts.append(dt)
    print("%6d %8d %10.4f %10.4f %9.4f %8.1f" % (
        depth, depth * N * 2, np.mean(tas), np.mean(tes), np.mean(ls), np.mean(dts)), flush=True)
print("對照：最近質心（古典，同 6 個主成分）test_acc = 0.8233；隨機 = 0.125", flush=True)
