"""
s06_param_budget.py — §6 古典基線與參數量計算器（公平比較協定）
================================================================
跑法：
    uv run python experiments/s06_param_budget.py

示範：
  (1) 一支可重複使用的參數量計算器（電路配置 + 基線模型 → 參數量）。
  (2) 驗證本專案的已知數字：1 層電路 = 10 個可訓練角度、5→10 softmax 頭 = 60。
  (3) 為什麼「1023 vs 31」是不公平的比較（類別錯誤）。
  (4) 同一份表格直接用於論文的「公平比較協定」。
"""

from __future__ import annotations

import unicodedata

import numpy as np


def dw(s) -> int:
    """字串的**顯示寬度**：全形字（中日韓）算 2 欄，其餘算 1 欄。

    為什麼需要它：Python 的 f-string 對齊格式 `{s:<20}` 算的是**字元數**，
    中文標籤會被排得歪七扭八。所有含中文的表格都用這個函式自己補空白。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))


def pad(s, w: int, align: str = "<") -> str:
    s = str(s)
    n = max(w - dw(s), 0)
    return (" " * n + s) if align == ">" else (s + " " * n)


def row(*cells) -> str:
    """cells = [(內容, 寬度, 對齊), ...]"""
    return "".join(pad(c, w, a) for c, w, a in cells)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# 6.1 參數量計算器
# ===========================================================================
def circuit_trainable_angles(n_qubit: int, n_layer: int, rot_per_qubit: int = 2) -> int:
    """
    量子電路的**可訓練角度**數量。

    本專案的電路結構（見 docs/_summary/00-專案規格常數.md §二）：
        RY 角度編碼（不可訓練） → 環形 CX 糾纏（不可訓練）
        → 可訓練 RY/RZ 層 → （可選第二層糾纏）
    每個可訓練層：每個 qubit 一個 RY + 一個 RZ → 2 個角度/qubit。
    糾纏閘（CX）沒有可訓練參數，所以不計入。

    ⚠ 這是「可訓練」參數量，**不是**希爾伯特空間維度、也不是密度矩陣的自由度。
    """
    return rot_per_qubit * n_qubit * n_layer


def readout_params(n_qubit: int, n_class: int, mode: str = "z_linear", bias: bool = True) -> int:
    """
    古典讀出層的參數量。

    mode = "z_linear"   ：5 個 <Z_i> 期望值 → Linear(n_qubit → K)
                          （採 Wang et al. 2026 的做法，F3【L256】【L261】）
    mode = "prob_linear"：2^n 維測量機率向量 → Linear(2^n → K)
                          （本專案的「第二版／消融」讀出，參數多 2^n·K 個）
    mode = "none"       ：直接取 argmax，沒有可訓練參數
    """
    if mode == "none":
        return 0
    d = n_qubit if mode == "z_linear" else 2**n_qubit
    return d * n_class + (n_class if bias else 0)


def encoder_params(input_dim: int, n_qubit: int, bias: bool = True) -> int:
    """角度編碼前的降維層 Linear(input_dim → n_qubit)。若前端已是 n_qubit 維則為 0。"""
    if input_dim <= 0:
        return 0
    return input_dim * n_qubit + (n_qubit if bias else 0)


def classical_params(kind: str, input_dim: int, n_class: int, hidden: int = 0, bias: bool = True) -> int:
    """古典基線的參數量。kind ∈ {softmax_head, logreg, mlp}。"""
    b = 1 if bias else 0
    if kind in ("softmax_head", "logreg"):
        return input_dim * n_class + b * n_class
    if kind == "mlp":
        if hidden <= 0:
            raise ValueError("mlp 需要 hidden > 0")
        return input_dim * hidden + b * hidden + hidden * n_class + b * n_class
    raise ValueError(f"未知的 kind：{kind}")


def quantum_total(
    n_qubit: int = 5,
    n_layer: int = 1,
    n_class: int = 10,
    encoder_in: int = 0,
    readout: str = "z_linear",
    bias: bool = True,
) -> dict:
    """把一個完整的混合模型拆成三段，回傳明細與總和。"""
    enc = encoder_params(encoder_in, n_qubit, bias) if encoder_in else 0
    ang = circuit_trainable_angles(n_qubit, n_layer)
    ro = readout_params(n_qubit, n_class, readout, bias)
    return {"encoder": enc, "circuit": ang, "readout": ro, "total": enc + ang + ro}


def state_space_facts(n_qubit: int) -> dict:
    """
    三個很容易被混為一談的數字：
      hilbert_dim   ：態向量的複數維度 = 2^n
      dm_real_dof   ：密度矩陣的**實自由參數**數 = (2^n)^2 − 1 = 4^n − 1
                      （厄米、跡為 1 → 對角 n 個實數 + 上三角 2·C(n,2) 個複數）
      softmax_free  ：K 類 softmax 的**自由**參數量 = K − 1（和為 1 的約束）
    """
    dim = 2**n_qubit
    return {"hilbert_dim": dim, "dm_real_dof": dim * dim - 1}


def main() -> None:
    section("§6.1 先驗證本專案的已知數字")
    a1 = circuit_trainable_angles(5, 1)
    print(f"5 qubit、1 層可訓練 RY/RZ：2 × 5 × 1 = {a1} 個可訓練角度")
    print(f"  規格書寫的『1 層電路 = 10 個可訓練角度』→ "
          f"{'✅ 一致' if a1 == 10 else '❌ 不一致'}")
    h = readout_params(5, 10, "z_linear")
    print(f"5 → 10 softmax 頭（5 個 <Z_i> 讀出）：5×10 + 10 = {h} 個參數")
    print(f"  規格書寫的『5→10 softmax 頭 = 60 個參數』→ "
          f"{'✅ 一致' if h == 60 else '❌ 不一致'}")
    p32 = readout_params(5, 10, "prob_linear")
    print(f"32 維機率向量 → 10 類：32×10 + 10 = {p32} 個參數（第二版讀出）")
    print(f"  規格書寫的『32×K 讀出矩陣』→ 矩陣本身 32×10 = {32 * 10}，"
          f"加上偏差 {10} 得 {p32}")

    # -----------------------------------------------------------------------
    section("§6.2 三個數字不要混為一談（『1023 vs 31』的類別錯誤）")
    f = state_space_facts(5)
    print(row(("數字", 44, "<"), ("值", 8, ">"), ("  這是什麼", 26, "<")))
    print("-" * 78)
    rows = [
        ("態向量的複數維度 2^5", f["hilbert_dim"], "狀態空間的維度"),
        ("密度矩陣的實自由參數 32²−1", f["dm_real_dof"], "『一個任意態』需要幾個實數"),
        ("1 層電路的可訓練角度 2·5·1", circuit_trainable_angles(5, 1), "電路真的在學幾個數"),
        ("2 層電路的可訓練角度 2·5·2", circuit_trainable_angles(5, 2), "電路真的在學幾個數"),
        ("5→10 讀出層 5·10+10", readout_params(5, 10, "z_linear"), "古典分類頭學幾個數"),
        ("10 類 softmax 的自由參數 10−1", 10 - 1, "扣掉『和為 1』後的自由度"),
        ("32 類 softmax 的自由參數 32−1", 32 - 1, "同上，類別數 32 時"),
    ]
    for name, v, what in rows:
        print(row((name, 44, "<"), (v, 8, ">"), ("  " + what, 26, "<")))
    print()
    print("★ 為什麼『1023 vs 31』是錯的比較：")
    print("  1023 是**一個任意 5-qubit 混態需要多少個實數才能寫下來**（狀態空間的自由度），")
    print("  31 是**一個 32 類 softmax 需要學多少個自由參數**（參數化族的自由度）。")
    print("  一個是『物件的維度』，一個是『模型的參數量』—— 這是類別錯誤（category error），")
    print("  就像拿『一張照片有 1920×1080 個像素』去比『我這台相機有 3 個旋鈕』。")
    print()
    print("  正確的問法有兩個，而且答案不一樣：")
    print("  (a) 「這個電路**學**了幾個數？」 → 10（1 層）或 20（2 層）。")
    print("      古典對手就該給差不多數量的參數 → 5→10 softmax 頭剛好 60 個，同量級。")
    print("  (b) 「這個電路**能表達**多大的空間？」 → 狀態在 1023 維的實流形上跑，")
    print("      但只能沿著 10 個角度所張開的**低維子流形**移動。")
    print("      **可達子流形的維度是 ≤ 10，不是 1023** —— 這才是 RQ1『容量斷崖』的關鍵。")
    print()
    print("  ★ 所以論文裡絕不能寫『量子層有 1023 個參數』。")
    print("    要寫：『5-qubit 暫存器的狀態空間為 32 維（混態 1023 個實自由度），")
    print("    而 1 層可訓練電路僅提供 10 個可訓練角度，可達子流形維度 ≤ 10。』")

    # -----------------------------------------------------------------------
    section("§6.3 容量掃描：狀態空間指數成長，可訓練參數線性成長")
    print(row(("qubits", 8, ">"), ("希爾伯特維度", 14, ">"), ("混態實自由度", 14, ">"),
              ("可訓練角度(1層)", 18, ">"), ("可訓練角度(2層)", 18, ">"), ("讀出(10類)", 12, ">")))
    print("-" * 78)
    for n in range(1, 9):
        ff = state_space_facts(n)
        print(row((n, 8, ">"), (ff["hilbert_dim"], 14, ">"), (ff["dm_real_dof"], 14, ">"),
                  (circuit_trainable_angles(n, 1), 18, ">"),
                  (circuit_trainable_angles(n, 2), 18, ">"),
                  (readout_params(n, 10, "z_linear"), 12, ">")))
    print()
    print("★ 這張表就是 RQ1（容量斷崖）的參數預算骨架：")
    print("  - 狀態空間 2^n 指數成長（n=8 時 256 維、混態 65535 個實自由度）")
    print("  - 但可訓練角度只有 2n 個（n=8、1 層時只有 16 個）")
    print("  - 讀出層 10n+10 個（n=8 時 90 個）")
    print("  → 『容量』增加時，**模型能學的參數量成長得遠比狀態空間慢**，")
    print("    因此出現非單調行為（先升後崩）在參數預算上是有空間的（假設 H1）。")
    print("  ⚠ 但這只是『有可能』，不是證明。真正的證據只能從實驗曲線來。")

    # -----------------------------------------------------------------------
    section("§6.4 公平比較協定：模型的完整參數預算表")
    print("情境：前端 potion-multilingual-128M 凍結，輸出 256 維（見規格書 §一）。")
    print("      以下只計**可訓練**參數（前端凍結，故不列入）。\n")
    input_dim = 256
    K = 10
    QBN_ENC = 1285  # Linear(256→5) + bias = 256*5 + 5
    scen = [
        ("QBN 5q / 1 層 / 5 個 <Z> 讀出",
         quantum_total(5, 1, K, encoder_in=input_dim, readout="z_linear")),
        ("QBN 5q / 2 層 / 5 個 <Z> 讀出",
         quantum_total(5, 2, K, encoder_in=input_dim, readout="z_linear")),
        ("QBN 5q / 1 層 / 32 維機率讀出",
         quantum_total(5, 1, K, encoder_in=input_dim, readout="prob_linear")),
        ("QBN 5q / 1 層，前端已降到 5 維（不計編碼層）",
         quantum_total(5, 1, K, encoder_in=0, readout="z_linear")),
    ]
    print(row(("模型", 44, "<"), ("編碼層", 8, ">"), ("電路", 8, ">"),
              ("讀出", 8, ">"), ("總計", 10, ">")))
    print("-" * 78)
    for name, t in scen:
        print(row((name, 44, "<"), (t["encoder"], 8, ">"), (t["circuit"], 8, ">"),
                  (t["readout"], 8, ">"), (t["total"], 10, ">")))
    print()
    print(row(("古典基線", 44, "<"), ("", 24, ">"), ("總計", 10, ">")))
    print("-" * 78)
    base = [
        ("softmax 頭（5 維 → 10 類，同讀出預算）", classical_params("softmax_head", 5, K)),
        ("邏輯斯迴歸（256 維 → 10 類）", classical_params("logreg", input_dim, K)),
        ("邏輯斯迴歸（5 維 → 10 類）", classical_params("logreg", 5, K)),
        ("MLP 256 → 32 → 10（ReLU）", classical_params("mlp", input_dim, K, hidden=32)),
        ("MLP 256 → 64 → 10（ReLU）", classical_params("mlp", input_dim, K, hidden=64)),
        ("MLP 256 → 128 → 10（ReLU）", classical_params("mlp", input_dim, K, hidden=128)),
    ]
    for name, v in base:
        print(row((name, 44, "<"), ("", 24, ">"), (v, 10, ">")))
    print()
    q1 = quantum_total(5, 1, K, encoder_in=input_dim, readout="z_linear")["total"]
    q_noenc = quantum_total(5, 1, K, encoder_in=0, readout="z_linear")["total"]
    print("★ 三個公平比較的紀律：")
    print(f"  1. **同一個前端、同一個讀出頭**。若量子模型的讀出頭是 5→10（60 個），")
    print(f"     那 softmax 基線也必須是 5→10（60 個）—— 這就是『同參數預算』。")
    print(f"  2. **要嘛都算編碼層、要嘛都不算**。含 Linear(256→5) 時量子模型是 {q1} 個，")
    print(f"     不含時是 {q_noenc} 個。混著比就是拿 {input_dim * 5 + 5} 個參數的編碼層")
    print(f"     去跟一個沒有編碼層的基線比 —— 這是最常見的不公平來源。")
    print(f"  3. **不要拿狀態空間維度當參數量**（見 §6.2）。")
    print()
    print(f"  ★ 若論文要報『量子層的貢獻』，最乾淨的對照是：")
    print(f"     (i) 完整 QBN：Linear(256→5) + 1 層電路 + Linear(5→10) = {q1} 個可訓練參數")
    print(f"     (ii) 古典同構：Linear(256→5) + Linear(5→10)         = "
          f"{encoder_params(input_dim, 5) + classical_params('softmax_head', 5, K)} 個")
    print(f"     兩者**只差在那一層量子電路（10 個角度）**，其餘完全相同 → 單一變因。")

    # -----------------------------------------------------------------------
    section("§6.5 公平比較協定檢查表（可直接貼進論文的方法章）")
    checks = [
        "相同的資料切分（同一個 seed 產生的 train/val/test），且測試集只用一次",
        "相同的前端與凍結方式（potion-multilingual-128M 不得只對其中一組解凍）",
        "相同的降維方式（PCA/ABTT 參數只在訓練集上擬合，見規格書 §一）",
        "相同的分類頭維度（5→10）與相同的參數量預算（60 個）",
        "相同的優化器、學習率、batch size、epoch 數",
        "相同的正則化（不得只對基線加 dropout / weight decay）",
        "相同的 shots 數與推論期蒙地卡羅次數 T = 30",
        "相同的校準後處理（溫度縮放要用**同一個驗證集**；未校準與已校準兩組都要報）",
        "相同的種子集合 [7, 21, 42, 84, 168]，且每個種子都重跑完整流程",
        "相同的評估指標組合（準確率、macro-F1、NLL、Brier、ECE(M=10)、MCE）",
        "**參數量的計算規則要寫出來**（含不含偏差、含不含編碼層）",
    ]
    for i, c in enumerate(checks, 1):
        print(f"  [ ] {i:>2}. {c}")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
