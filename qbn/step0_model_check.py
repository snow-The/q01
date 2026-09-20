#!/usr/bin/env python
"""步驟 0：確認 potion-multilingual-128M 能載入，並量測維度與繁簡一致性。

這是端到端驗證的第一個關卡。若這裡失敗，後面全部不用談。

執行（WSL 內）：
    /root/qbn/.venv/bin/python dev/step0_model_check.py
"""

from __future__ import annotations

import time

import numpy as np
from model2vec import StaticModel

MODEL_ID = "minishlab/potion-multilingual-128M"

print("=" * 78)
print(f"載入 {MODEL_ID}")
print("=" * 78)
t0 = time.perf_counter()
model = StaticModel.from_pretrained(MODEL_ID)
print(f"  載入耗時 {time.perf_counter() - t0:.1f} s")
print(f"  型別 = {type(model).__name__}")

# --- 維度與基本屬性 ---
print("\n" + "=" * 78)
print("模型屬性")
print("=" * 78)
for attr in ("dim", "dimension", "embedding_dim", "vocab_size", "max_length",
             "normalize", "tokenizer"):
    val = getattr(model, attr, None)
    if val is not None and not callable(val):
        s = str(val)
        print(f"  {attr} = {s[:90]}")

# --- 編碼測試 ---
sents = [
    "這部電影真的很好看。",
    "這部電影真的很難看。",
    "This movie is really good.",
    "This movie is really bad.",
]
print("\n" + "=" * 78)
print("編碼測試")
print("=" * 78)
embs = model.encode(sents)
print(f"  encode(list) → type={type(embs)}, shape={np.asarray(embs).shape}, "
      f"dtype={np.asarray(embs).dtype}")

E = np.asarray(embs, dtype=np.float64)
print(f"\n  各句的 L2 範數: {np.round(np.linalg.norm(E, axis=1), 6)}")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\n  餘弦相似度矩陣:")
print("            " + "".join(f"{i:>10}" for i in range(len(sents))))
labels_short = ["zh+", "zh-", "en+", "en-"]
for i in range(len(sents)):
    row = "".join(f"{cos(E[i], E[j]):>10.4f}" for j in range(len(sents)))
    print(f"    {labels_short[i]:>6} {row}")

print(f"\n  ★ 繁體中文正/負的相似度 = {cos(E[0], E[1]):.4f}（應明顯 < 1）")
print(f"  ★ 英文正/負的相似度     = {cos(E[2], E[3]):.4f}")
print(f"  ★ 跨語言同極性 zh+/en+ = {cos(E[0], E[2]):.4f}")

# --- 繁簡一致性測試（書中的未核實項）---
print("\n" + "=" * 78)
print("繁簡一致性測試（potion 只用 zh 標籤，無 zh-Hant 區分）")
print("=" * 78)
pairs = [
    ("這部電影真的很好看", "这部电影真的很好看"),
    ("服務態度非常好", "服务态度非常好"),
    ("價格合理品質也不錯", "价格合理品质也不错"),
    ("房間乾淨明亮", "房间干净明亮"),
    ("老師講解得很清楚", "老师讲解得很清楚"),
    ("產品品質超出預期", "产品品质超出预期"),
    ("這款手機的拍照功能很強", "这款手机的拍照功能很强"),
    ("客服回應迅速", "客服回应迅速"),
    ("環境很安靜適合專心工作", "环境很安静适合专心工作"),
    ("整體來說非常值得", "整体来说非常值得"),
]
sims = []
for trad, simp in pairs:
    e = np.asarray(model.encode([trad, simp]), dtype=np.float64)
    s = cos(e[0], e[1])
    sims.append(s)
    print(f"  {s:.4f}   {trad[:14]:<16} ↔ {simp[:14]}")

sims_arr = np.asarray(sims)
print(f"\n  平均相似度 = {sims_arr.mean():.4f}，最小 = {sims_arr.min():.4f}")
if sims_arr.min() > 0.95:
    print("  → 判定：繁簡高度一致，可放心混用。")
elif sims_arr.min() > 0.85:
    print("  → 判定：大致一致，但仍有差異，論文中應列為 limitation。")
else:
    print("  → 判定：⚠️ 繁簡差異明顯！必須在論文中明確列出，並考慮只用繁體訓練。")

print("\nDONE")
