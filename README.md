# q01

**CUDA-Q 的相容子集**：用純 NumPy 在**原生 Windows、純 CPU**上執行 CUDA-Q 語法的量子電路。
不需要 WSL、不需要 CUDA、不需要安裝 CUDA-Q。

```python
try:
    import cudaq              # 正式軌：裝了真 CUDA-Q 就用真的
except ImportError:
    import q01 as cudaq       # 練習軌：同一份程式碼，原生 Windows 可跑
```

## 為什麼需要它

CUDA-Q 官方**沒有 Windows 的原生安裝**——PyPI 上的 `cudaq` 只有原始碼包，二進位 wheel
只覆蓋 Linux 與 macOS ARM。於是「量子」在教學現場的實際門檻不是算力，而是**作業系統**。

`q01` 把這道門檻拿掉：同一份電路定義，在沒有 WSL 的 Windows 筆電上也能跑、能驗算，
能與真 CUDA-Q 的輸出逐位元對帳。

## 安裝

```bash
uv add qml01c                     # 從 PyPI 安裝（發佈名 qml01c；import 名仍是 q01）
pip install qml01c

# 或本機開發路徑
uv add --editable path/to/q01
pip install -e path/to/q01
```

## 三十秒上手

```python
import q01 as cudaq

@cudaq.kernel
def bell():
    q = cudaq.qubit()
    cudaq.h(q)
    cudaq.cx(q, q)

print(cudaq.get_state(bell))     # 只有 numpy 也能跑
```

## 驗證到什麼程度

這個專案對自己的要求是「**可以被獨立對帳**」，所以每一項主張都有對應的量測：

| 檢查 | 結果 | 怎麼驗 |
|---|---|---|
| 黃金向量一致性（5 個案例） | 最差 **2.78e-16** | 與真 CUDA-Q 0.16.0 產生的向量比對 |
| 算子級判準（U†V，全么正比對） | **2.47e-17** | 不只看作用在 `|0⟩` 上的結果 |
| 跨框架一致（Qiskit／Cirq／PennyLane） | 五套實作最差 **2.78e-16** | 同一份電路定義，各後端獨立模擬 |
| PyTorch 後端 autograd vs 參數平移 | **1.95e-19** | 兩種梯度算法互相對帳 |
| 與 mpmath 60 位元參考的精度 | **0.5 × eps** | 已在雙精度的捨入底線 |
| CPU 最佳化（單量子位閘就地切片） | **1.5–3.7×** | 同一台機器前後對比，數值不變 |

## 它不是什麼

- **不是 CUDA-Q**。它是**子集**：只實作教學與本研究用到的閘集與 API 形狀。
- **不是模擬器競賽**。它在 n ≤ 20 的小電路上求正確與可讀，不追求大規模效能。
- **不宣稱量子優勢**。用途是教學、驗算、交叉核對；作為子集，它不取代真 CUDA-Q。

## 專案結構

```
src/q01/          核心：tracer（AST）、閘集、模擬器、服務層（sampling／gradients）、位元序、真機前端
src/q01/adapters/ 各框架 emitter（Qiskit／Cirq／PennyLane）與 PyTorch 後端
tests/            核心測試、跨框架一致性、黃金向量（tests/golden/）
tools/oracle/     對帳與量測工具（精度底線、CPU/GPU 對比、慣例探測）
tools/hardware/   IR → OpenQASM 2.0 匯出，以及真機（Quantum Inspire／Tuna-17）驗證與結果檢視
SPEC.md           規格書（含每一項判準與版本史）
notes/            研究筆記（各框架的實際慣例與踩過的坑）
```

## 文件

- `SPEC.md`：規格、判準、支援的 API 子集、精度與效能實測。
- `notes/`：Qiskit / Cirq / PennyLane / CUDA-Q 的**實測慣例**（位元序、參數平移、
  精度底線），每一條都附量測方式。

## 授權

MIT，見 `LICENSE`。
