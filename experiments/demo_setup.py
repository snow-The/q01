"""
demo_setup.py — 本指南所有章節共用的示範設定（單一真相來源）
==============================================================
把「資料怎麼生、模型怎麼訓、切分怎麼切」集中在這裡，
確保 §2 ~ §5 每一節講的是**同一個模型**，數字可以互相對照。

設計對齊本專案：
  - 輸入 5 維，值域 [-1, 1]  →  模擬 5 個 qubit 的 <Z_i> 讀出
  - 輸出 10 類（完全平衡）    →  Linear(5→10) 剛好 5*10 + 10 = 60 個參數
  - 訓練集刻意只給 300 筆、跑 2000 個 epoch、關掉 L2
    →  製造出**現實中深度網路最常見的過度自信**
      （訓練集上完美，測試集上信心遠高於準確率）
    這不是為了讓數字好看，而是因為「校準」這個問題只在過度自信時才看得見。
"""

from __future__ import annotations

import numpy as np

from mlkit import logits_of, make_readout_dataset, softmax, train_softmax

SEED = 42
N_TRAIN = 300
N_VAL = 600
EPOCHS = 2000


def load_demo(seed: int = SEED, n_train: int = N_TRAIN) -> dict:
    """回傳統一格式的資料切分與訓練好的模型。"""
    X, y, centers = make_readout_dataset(N=3000, C=10, D=5, separation=3.0, noise=1.0, seed=SEED)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[1800 : 1800 + N_VAL], y[1800 : 1800 + N_VAL]
    Xte, yte = X[2400:], y[2400:]

    model = train_softmax(
        Xtr, ytr, Xva, yva, C=10, seed=seed, epochs=EPOCHS, lr=0.05, batch=64, l2=0.0
    )
    return {
        "model": model,
        "Xtr": Xtr, "ytr": ytr,
        "Xva": Xva, "yva": yva,
        "Xte": Xte, "yte": yte,
        "Zva": logits_of(model, Xva),
        "Zte": logits_of(model, Xte),
        "Pva": softmax(logits_of(model, Xva)),
        "Pte": softmax(logits_of(model, Xte)),
        "centers": centers,
        "n_train": n_train,
        "seed": seed,
    }


def fig_font_setup() -> None:
    """matplotlib 中文字型設定。放在這裡讓每張圖都一致。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
