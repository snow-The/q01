"""s14：最佳化探針 —— 量子層是「容量不夠」還是「沒訓練起來」？

SGD(lr=0.15, 400 步) 只到 train_acc 0.55，而最近質心在同一任務上有 0.71–0.85。
這裡比較 SGD vs Adam、加長步數，看訓練集能不能被擬合。
"""
import pathlib, sys, time
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import s09_capacity_scan as s09
import s12_torch_capacity as s12

Z_tr, y_tr, Z_te, y_te = s09.make_data()
torch.set_default_dtype(torch.float64)

def run(n, depth, seed, opt, steps, lr):
    rng = np.random.default_rng(seed)
    w = torch.tensor(rng.normal(0, 0.3, size=(depth, n, 2)), requires_grad=True)
    Xa, Xb = s09.encode(Z_tr[:, :n]), s09.encode(Z_te[:, :n])
    optim = torch.optim.Adam([w], lr=lr) if opt == "adam" else None
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = s12.ce_loss(n, depth, Xa, y_tr, w)
        if optim:
            optim.zero_grad(); loss.backward(); optim.step()
        else:
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad; w.grad = None
    dt = time.perf_counter() - t0
    with torch.no_grad():
        ta = float((s12.probs_torch(n, depth, Xa, w).argmax(1) == y_tr).float().mean())
        te = float((s12.probs_torch(n, depth, Xb, w).argmax(1) == y_te).float().mean())
    return ta, te, dt, float(loss)

print("%4s %5s %6s %6s %7s %9s %9s %9s %8s" % ("n", "depth", "opt", "lr", "steps", "train_acc", "test_acc", "loss", "秒"))
for n, depth in ((6, 3), (8, 4)):
    for opt, lr, steps in (("sgd", 0.15, 400), ("sgd", 0.15, 2000), ("adam", 0.05, 2000), ("adam", 0.02, 4000)):
        tas, tes, dts, ls = [], [], [], []
        for seed in (0, 1, 2):
            ta, te, dt, l = run(n, depth, seed, opt, steps, lr)
            tas.append(ta); tes.append(te); dts.append(dt); ls.append(l)
        print("%4d %5d %6s %6s %7d %9.4f %9.4f %9.4f %8.1f" % (
            n, depth, opt, lr, steps, np.mean(tas), np.mean(tes), np.mean(ls), np.mean(dts)), flush=True)
