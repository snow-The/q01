#!/usr/bin/env python3
"""SimSIMD／NumKong 在**我們的** workload 上到底有沒有用？"""
import time

import numpy as np

for name in ("simsimd", "numkong"):
    try:
        mod = __import__(name)
        print(f"== {name} {getattr(mod, '__version__', '?')} ==")
        api = [x for x in dir(mod) if not x.startswith("_")]
        print("  API:", ", ".join(api[:60]))
    except Exception as e:
        print(f"== {name}: 未安裝（{type(e).__name__}）==")
print()


def best(fn, reps=50, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


rng = np.random.default_rng(0)

print("=" * 84)
print("A. 我們的量子端：complex128 內積（⟨ψ|ψ⟩ 這種，長度 2^18）")
print("=" * 84)
n = 1 << 18
a = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
b = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
t_np = best(lambda: np.vdot(a, b))
print(f"  numpy.vdot            : {t_np*1e6:9.1f} us   ({n*16/t_np/1e9:6.2f} GB/s)")
try:
    import simsimd
    for nm in ("dot", "vdot"):
        fn = getattr(simsimd, nm, None)
        if fn is None:
            continue
        try:
            t = best(lambda: fn(a, b))
            print(f"  simsimd.{nm:14s}   : {t*1e6:9.1f} us   ({n*16/t/1e9:6.2f} GB/s)")
        except Exception as e:
            print(f"  simsimd.{nm}: 不支援此 dtype（{type(e).__name__}: {str(e)[:50]}）")
except ImportError:
    pass
try:
    import numkong
    for nm in ("dot", "vdot", "inner"):
        fn = getattr(numkong, nm, None)
        if fn is None:
            continue
        try:
            t = best(lambda: fn(a, b))
            print(f"  numkong.{nm:14s}   : {t*1e6:9.1f} us   ({n*16/t/1e9:6.2f} GB/s)")
        except Exception as e:
            print(f"  numkong.{nm}: 不支援（{type(e).__name__}: {str(e)[:50]}）")
except ImportError:
    pass

print()
print("=" * 84)
print("B. 我們的 ML 端：potion 嵌入的 1-to-many 餘弦相似度（1 x 11514 x 256）")
print("=" * 84)
db = rng.normal(size=(11514, 256)).astype(np.float32)
q = rng.normal(size=256).astype(np.float32)
db64 = db.astype(np.float64)
q64 = q.astype(np.float64)


def np_cos_f32():
    qn = q / np.linalg.norm(q)
    dn = db / np.linalg.norm(db, axis=1, keepdims=True)
    return dn @ qn


def np_cos_f64():
    qn = q64 / np.linalg.norm(q64)
    dn = db64 / np.linalg.norm(db64, axis=1, keepdims=True)
    return dn @ qn


t32 = best(np_cos_f32, reps=20)
t64 = best(np_cos_f64, reps=20)
print(f"  numpy f32（OpenBLAS） : {t32*1e3:8.3f} ms")
print(f"  numpy f64（OpenBLAS） : {t64*1e3:8.3f} ms")
try:
    import simsimd
    fn = getattr(simsimd, "cdist", None)
    if fn is not None:
        for dt, arr in (("f32", db), ("f64", db64)):
            try:
                t = best(lambda: fn(arr, q.reshape(1, -1) if dt == "f32" else q64.reshape(1, -1), metric="cosine"), reps=10)
                print(f"  simsimd.cdist {dt}      : {t*1e3:8.3f} ms   (x{t32/t:5.2f} vs numpy f32)")
            except Exception as e:
                print(f"  simsimd.cdist {dt}: 失敗 {type(e).__name__}: {str(e)[:60]}")
except ImportError:
    pass

print()
print("=" * 84)
print("C. 精度：f32 輸入、不同累加器（NumKong 宣稱 f32 用 f64 累加）")
print("=" * 84)
a32 = rng.normal(size=2048).astype(np.float32)
b32 = rng.normal(size=2048).astype(np.float32)
ref = float(np.dot(a32.astype(np.float64), b32.astype(np.float64)))
d_np = float(np.dot(a32, b32))
print(f"  精確（f64 累加）      = {ref!r}")
print(f"  numpy.dot(f32, f32)   = {d_np!r}   相對誤差 = {abs(d_np-ref)/abs(ref):.3e}")
for name, mod in (("simsimd", "simsimd"), ("numkong", "numkong")):
    try:
        m = __import__(mod)
        fn = getattr(m, "dot", None)
        if fn is None:
            continue
        v = float(fn(a32, b32))
        print(f"  {name}.dot(f32, f32)  = {v!r}   相對誤差 = {abs(v-ref)/abs(ref):.3e}")
    except Exception as e:
        print(f"  {name}: {type(e).__name__}: {str(e)[:60]}")