# q01 — 量子貝氏網路分類器的 QML 工具箱

在**原生 Windows、純 CPU**上做量子機器學習研究：量子電路、QBN 分類器、校準與不確定性評估、
以及真機（Quafu）送單。**不需要 WSL、不需要 CUDA、不需要安裝 CUDA-Q。**

```python
try:
    import cudaq              # 正式軌：裝了真 CUDA-Q 就用真的
except ImportError:
    import q01 as cudaq       # 練習軌：同一份程式碼，原生 Windows 可跑
```

## 為什麼存在

CUDA-Q 官方**沒有 Windows 原生安裝**（PyPI 上的 `cudaq` 只有原始碼包，二進位 wheel 只覆蓋
Linux 與 macOS ARM）。於是量子機器學習在教學與小型研究現場的實際門檻，不是算力，而是**作業系統**。

本專案把這道門檻拿掉，並且刻意把「**可以被獨立對帳**」當成第一原則：每一項主張都附上量測方式與數字，
包含我們自己的否定性結果。

## 安裝

```bash
pip install qml01c        # PyPI 上的發佈名
uv add qml01c
```

!!! note "發佈名與匯入名不同"
    PyPI 的專案名是 `qml01c`（`q01` 被 PyPI 判定與既有專案過於相似而保留），
    但**匯入名仍是 `q01`**：

    ```python
    import q01          # 不變
    ```

    這與 `pillow` → `PIL`、`scikit-learn` → `sklearn` 是同一種模式。

只有一個必需依賴：`numpy>=2.0`。選用：`torch`（自動微分）、`qiskit` / `cirq` / `pennylane`（跨框架對帳）、`mpmath`（高精度參考）。

## 三十秒上手

```python
import q01 as cudaq

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)      # Bell 態需要兩個 qubit
    cudaq.h(q[0])
    cudaq.cx(q[0], q[1])

print(cudaq.get_state(bell))
```

> `q01` 與 CUDA-Q 相同：```kernel` 會讀取函式的**原始碼**做 AST 追蹤，所以 kernel 必須
> 定義在真實的 `.py` 檔裡，不能寫在 `python -c` 的內嵌字串中（會得到
> `kernel 取得原始碼失敗` 的錯誤）。

## 驗證到什麼程度

| 檢查 | 結果 | 怎麼驗 |
|---|---|---|
| 黃金向量一致性（5 案例） | 最差 **2.78e-16** | 與真 CUDA-Q 0.16.0 產生的向量比對 |
| 算子級判準（U†V 全么正比對） | **2.47e-17** | 不只看作用在 `|0⟩` 上的結果 |
| 跨框架一致（Qiskit／Cirq／PennyLane） | 五套實作最差 **2.78e-16** | 同一份電路定義，各後端獨立模擬 |
| PyTorch autograd vs 參數平移 | **1.95e-19** | 兩種梯度算法互相對帳 |
| 與 mpmath 60 位元參考的精度 | **0.5 × eps** | 已在雙精度捨入底線 |
| CPU 單量子位閘就地切片 | **1.5–3.7×** | 同機前後對比，數值不變 |
| 真機鏈路（Quafu 官方模擬器） | max\|ΔP\| = **1.76e-03** @20k shots | 與散粒噪聲的 1/√N 行為一致 |

## 倉庫結構

```
src/q01/        量子電路核心：AST tracer、閘集、模擬器、位元序、gradients、torch 後端
tests/          核心測試、跨框架一致性、黃金向量（tests/golden/）
qbn/            QBN 分類器模型：電路、編碼、訓練迴圈、張量網路對應
mlkit/          評估工具：指標、ECE、可靠度、溫度縮放、不確定性量化
experiments/    研究腳本 s01–s16（容量掃描、消融、基線、優化器與深度探針）
tools/oracle/   對帳與量測（精度底線、CPU/GPU 對比、慣例探測）
tools/hardware/ IR → OpenQASM 2.0 匯出、真機送單與結果檢視（qres.py）
SPEC.md         規格書：判準、支援的 API 子集、精度與效能實測、版本史
notes/          各框架的**實測慣例**（位元序、參數平移、精度底線）
```

## 檢視真機結果

```bash
python tools/hardware/qres.py <taskid>                      # 狀態與計數
python tools/hardware/qres.py <taskid> --golden qbn5_book   # 順便對帳黃金向量
```

狀態碼語意（pyquafu 的 `status_map`）：`0 In Queue／1 Running／2 Completed／3 Canceled／4 Failed／5 Pending`。

## 它不是什麼

- **不是 CUDA-Q**，是它的**子集**：只實作教學與本研究用到的閘集與 API 形狀。
- **不是模擬器競賽**：在 n ≤ 20 的小電路上求正確與可讀，不追求大規模效能。
- **不宣稱量子優勢**：本專案的實測反而支持相反的結論（見下）。

## 研究背景

本庫源自一個量子貝氏網路（Quantum Bayesian Network, QBN）分類器研究。目前得到的**否定性結論**，
我們選擇照實報告而不是藏起來：

- 在同一任務上，量子層的測試準確率（最佳 **0.618**）**低於**最樸素的古典基線（最近質心，3 個主成分即有 **0.707**）。
- 容量掃描（n = 3–10 qubit × 電路深度 1–8）顯示：**深度是唯一有效的軸**（0.13 → 0.62），
  **增加 qubit 數沒有幫助**（峰值在 n = 4，n = 10 反而降到 0.493）。
- 梯度成本是真正的瓶頸：參數平移每步需要 `8·n·depth+1` 次前向；改用 autograd
  可讓單步成本從 0.36 s 降到 0.012 s（**約 30×**），這是把掃描做得起來的關鍵。

## 授權

MIT，見 `LICENSE`。
