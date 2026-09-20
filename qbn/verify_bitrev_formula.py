"""驗算：format(idx, '0nb')[::-1] 與 bit_reverse_permutation 是否等價。

背景：CQ1 主張兩者數學上相同，而我先前在常數檔把 format 版列為「錯誤」。
這件事必須用實際計算判定，不能靠推論。
"""

from __future__ import annotations

import numpy as np


def bit_reverse_permutation(n: int) -> np.ndarray:
    """位元反轉置換：把 k 的二進位表示反轉後重新解讀。"""
    dim = 2 ** n
    perm = np.zeros(dim, dtype=int)
    for k in range(dim):
        r = 0
        for b in range(n):
            if (k >> b) & 1:
                r |= 1 << (n - 1 - b)
        perm[k] = r
    return perm


def fmt_reverse(n: int) -> np.ndarray:
    """字串反轉版：format(k, '0nb')[::-1] 再解讀為二進位。"""
    dim = 2 ** n
    return np.array([int(format(k, f"0{n}b")[::-1], 2) for k in range(dim)])


print("=" * 78)
print("驗算一：兩種換算是否逐項相同")
print("=" * 78)
print(f"{'n':>3}  {'相等?':>8}  {'首 8 項（bit_reverse）':<34} {'首 8 項（format[::-1]）'}")
for n in range(1, 9):
    brp = bit_reverse_permutation(n)
    fmt = fmt_reverse(n)
    eq = bool(np.array_equal(brp, fmt))
    show = min(8, len(brp))
    print(f"{n:>3}  {str(eq):>8}  {str(list(brp[:show])):<34} {list(fmt[:show])}")

print()
print("=" * 78)
print("驗算二：n=5 逐項對照（本專案的實際情形）")
print("=" * 78)
n = 5
brp = bit_reverse_permutation(n)
fmt = fmt_reverse(n)
print(f"{'k':>3}  {'k 的二進位':>10}  {'bit_reverse':>12}  {'其二進位':>10}  "
      f"{'format[::-1]':>13}  {'其二進位':>10}  {'相同?':>6}")
for k in range(32):
    same = "✓" if brp[k] == fmt[k] else "✗"
    print(f"{k:>3}  {format(k, '05b'):>10}  {brp[k]:>12}  "
          f"{format(brp[k], '05b'):>10}  {fmt[k]:>13}  {format(fmt[k], '05b'):>10}  {same:>6}")

print()
print("=" * 78)
print("驗算三：array[::-1] 是否也相同（我原本以為它與 format 版等價）")
print("=" * 78)
arr_rev = np.arange(32)[::-1]
print(f"array[::-1]              = {list(arr_rev[:8])} ...")
print(f"bit_reverse_permutation  = {list(brp[:8])} ...")
print(f"format[::-1]             = {list(fmt[:8])} ...")
print()
print(f"array[::-1] == bit_reverse ? {bool(np.array_equal(arr_rev, brp))}")
print(f"array[::-1] == format[::-1] ? {bool(np.array_equal(arr_rev, fmt))}")
print()
print("array[::-1] 的定義是 k ↦ (2^n − 1 − k)，也就是把整個索引順序倒過來。")
print("位元反轉是 k ↦ rev(k)。兩者只在 n ≤ 2 時相同。")

print()
print("=" * 78)
print("結論")
print("=" * 78)
all_eq = all(np.array_equal(bit_reverse_permutation(n), fmt_reverse(n))
             for n in range(1, 9))
print(f"  對 n = 1..8，format 版與 bit_reverse 逐項相同：{all_eq}")
print()
if all_eq:
    print("  → CQ1 的主張成立。兩者數學上等價（都是把位元反轉後重新解讀）。")
    print("  → 常數檔把 format 版列為「錯誤」是錯的，必須更正。")
    print("  → 真正該排除的只有 array[::-1]（k ↦ 2^n−1−k）。")
else:
    print("  → 兩者不全面等價，需逐 n 檢查。")
print()
print("  為什麼我先前會搞混：實際量到 2.383e-01 的那個錯誤版本是 array[::-1]，")
print("  不是 format 版。我把兩者混為一談，於是把錯的標籤貼到了對的公式上。")
