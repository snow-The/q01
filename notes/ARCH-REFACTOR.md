# q01 架構重構提案（v0.1，待作者審）

> 起因：作者提議「q01 分成語法抽象、核心與適配器重新設計」，並指出專案「到處要飯——
> 手頭有什麼免費額度就用什麼」。
> 本檔先給設計，**不動程式碼**。審過才進 P1.5。

---

## 1. 我同意什麼、不同意什麼

| 提議 | 我的判斷 | 理由 |
|---|---|---|
| 分層（核心 vs 適配器） | ✅ **同意** | 目前的分層是「事實上的」而不是「契約上的」：`kernel.py` 同時做語法追蹤與 IR、`simulator.py` 同時是核心與 numpy 後端。再往後接 torch／qiskit／cirq 會讓縫更模糊 |
| 語法抽象成中立 DSL | ❌ **不同意** | q01 的賣點就是「**學生的程式碼與論文軌一模一樣**」。中立化之後，書裡的程式碼既不是 CUDA-Q 也不是 Qiskit，變成第三種要學的東西，論文「跨實作一致」也失去參照 |
| IR 當 hub，adapter 當 emitter | ✅ **同意（這是正解）** | 學生語法不變，額外得到 N 個獨立實作；「同一顆電路用不同框架跑出同一個答案」本身就能寫進論文 |
| 接 Cirq / Braket | 部分同意 | Cirq ✅（純 Python wheel、Apache-2.0、13 依賴，是便宜的第四個 oracle）；Braket ❌（25+20 依賴，唯一好處是「將來能上真機」，而我們明確不主張量子優勢） |

**「到處要飯」的真正單點不是後端數量**，而是三件事（本重構只解得了其中兩件）：

| # | 單點 | 對策 | 狀態 |
|---|---|---|---|
| ① | 論文數字只有一台機器生得出來 | 黃金向量入庫 ＋ 可重現 recipe | 🟡 黃金向量已入庫，recipe 待做 |
| ② | 書裡可執行的程式碼綁在沒有 Windows wheel 的廠商 | q01 | ✅ 已達成（兩軌各 28 passed） |
| ③ | 工具鏈吃免費額度 | **硬規則：q01 執行期零網路、零 API key、零雲端** | 🟡 待寫進 SPEC §2.2 |

---

## 2. 目標與非目標

**目標**
- G-A 縫要**契約化**：每個模組的職責、公開符號、允許的 import 方向都寫死，並用測試守住。
- G-B 新增一個 adapter 的成本＝「一個 emitter ＋ 跑同一份 conformance」，**不得**新增核心分支。
- G-C q01 的 runtime 依賴保持只有 numpy；其他全走 extra。

**非目標**
- ❌ 不做中立 DSL；CUDA-Q 子集就是唯一語法層。
- ❌ 不做真實後端（雲端／真機）；adapter 的角色是**驗證**與**梯度**，不是硬體存取。
- ❌ 不重寫數值核心（P1 已與 CUDA-Q 對到 3.33e-16，重寫是純風險）。

---

## 3. 分層（依賴只能由上往下，禁止反向）

| 層 | 模組 | 職責 | 公開符號 | 允許 import |
|---|---|---|---|---|
| **syntax** | `syntax/kernel.py`、`syntax/gates.py` | `@kernel` 追蹤、閘名注入、`.ctrl`、`mz/mx/my`、`adjoint/control`、`check_kernel` | `kernel`、`qvector`、`qubit`、16 個閘、`mz/mx/my`、`adjoint`、`control`、`check_kernel` | ir |
| **ir** | `ir.py` | `Op`（單閘＋控制位）、`Trace`、`global_phase`、qubit 控制代碼 | `Op`、`Trace`、`QubitRef`、`QVector` | 無（最底層） |
| **core** | `core/statevector.py`、`core/measure.py` | 狀態向量演化、量測塌縮、抽樣、期望值；**dtype 由後端決定** | `StateVector` 協定、`sample_counts`、`expectation_pauli` | ir |
| **services** | `services.py` | `sample/run/get_state/observe/ParameterShift` 的**語意契約**（含線路中量測要報錯、`get_state` 塌縮、observe 決定性） | 對外那 5 個函式 | ir、core |
| **adapters** | `adapters/numpy.py`（預設）、`torch.py`、`qiskit_emit.py`、`cirq_emit.py` | 執行（numpy/torch）或**產生別的框架的電路物件**（qiskit/cirq） | 統一 `Adapter` 介面 | ir、core（**不得** import syntax） |
| **靶場** | `tests/conformance/` | 同一份 suite，對每個 adapter 跑 | runner + golden | 全部（測試例外） |

**禁止**：core → adapters、ir → 任何上層、adapters → syntax。
用一個測試守住：掃 import 圖（`test_architecture.py`）。

---

## 4. Adapter 契約

每個 adapter 必須提供：

| 方法 | 語意 |
|---|---|
| `name` / `available()` | 名稱與「這個環境裝得起來嗎」 |
| `run_ir(trace, shots=None) -> AdapterResult` | 執行 IR（numpy/torch），或 |
| `emit(trace) -> object` | 產生該框架的電路物件（qiskit/cirq，交由它自己的模擬器跑） |
| `unitary(trace) -> np.ndarray` | 全么正矩陣（L1 用） |

**conformance 三層（每個 adapter 都要過）**
1. **L0 公約**：位元順序、角度符號、全域相位 → 布林判準。
2. **L1 算子**：`max|U†V − cI| ≤ 1e-10`（tket 判準；**不可**只比單一輸入態）。
3. **L2 電路**：對 `tests/golden/cudaq_golden.json` 的 5 個案例 `max|ΔP| ≤ 1e-10`。

**已知陷阱清單**（寫進 runner 的註解，每個新 adapter 都要重新確認）：
Qiskit 的 `RZ ≠ P`、`SX` 差 `e^{iπ/4}`、`StatevectorSampler` 不支援線路中量測；
Cirq 用 `LineQubit`、索引順序與我們的 big-endian 內部表示不同；
PennyLane 的 big-endian 是它自己的約定，**不能**當位元順序的證據（oracle 只有 CUDA-Q）。

---

## 5. 檔案樹（before → after）

```
before（現況，P1）                       after（P1.5）
src/q01/                                src/q01/
├── __init__.py      對外門面            ├── __init__.py        對外門面（符號不變）
├── circuit.py       IR                 ├── ir.py              （原 circuit.py）
├── gates.py         語法＋閘矩陣         ├── syntax/
├── kernel.py        追蹤＋量測           │   ├── kernel.py      （追蹤／量測／adjoint）
├── simulator.py     核心＋numpy 後端     │   └── gates.py       （閘名、矩陣、.ctrl）
├── sampler.py       服務                ├── core/
├── spin.py          服務                │   ├── statevector.py （原 simulator.py 的數值部分）
├── gradients.py     服務                │   └── measure.py     （量測／抽樣／期望值）
├── targets.py       target/seed         ├── services.py        （sample/run/get_state/observe/gradients）
└── bitorder.py      ★ 唯一置換處         ├── adapters/
                                          │   ├── numpy.py       （預設）
                                          │   ├── torch.py       （P4）
                                          │   ├── qiskit_emit.py （P3）
                                          │   └── cirq_emit.py   （P3）
                                          ├── targets.py
                                          └── bitorder.py        ★ 唯一置換處（位置不變）
```

`q01.__all__` **完全不變**——學生看不到這次重構（這是驗收條件）。

---

## 6. 遷移切片（每一片都能單獨驗證）

| 切片 | 動作 | 驗收 |
|---|---|---|
| S1 | `circuit.py → ir.py`、`kernel.py+gates.py → syntax/` | 28 tests 全綠、`q01.__all__` 不變 |
| S2 | `simulator.py` 拆成 `core/statevector.py` ＋ `adapters/numpy.py` | 同上 ＋ 黃金比對仍 ≤1e-10 |
| S3 | `sampler/spin/gradients → services.py`、加 `adapters/` 協定與 registry | 同上 ＋ 新增 `test_architecture.py`（import 方向） |
| S4 | 抽出 `tests/conformance/` runner（把 test_golden.py 改成跑 runner） | 每個 adapter 都跑同一份 suite |

**風險**：S2 動到數值核心的檔案邊界——但**不動數學**；回歸靠既有 28 測試 ＋ 黃金向量。

---

## 7. 現在的基準（重構的回歸線，2026-09-18）

- 本地 Windows（uv 0.8.13 / Python 3.13.13 / numpy 2.5.3）：**28 passed，0.16 s**
- 參考機 Windows（uv 0.11.8 / Python 3.13.14）：**28 passed，0.69 s**
- 跨軌對帳（CUDA-Q 0.16.0 黃金向量）：最差 **3.33e-16**（判準 1e-10）
- 算子層：`max|U†V − cI| ≤ 2.47e-17`，`|c| = 1`

**重構不得讓任何一個數字變差。**

---

## 8. 待決事項

1. **要不要現在做 seam 重排（P1.5）？** 我的建議：要，而且趁現在（~900 行；P4 之後再做貴 3–5 倍）。
2. **Braket 要不要保留在路線圖？** 我的建議：不要（列為「明確不做」，理由寫進 SPEC 以免日後反覆）。
3. **Cirq 的定位**：第四個 oracle（建議）／還是學生的可選執行後端？前者只需 emitter＋conformance，後者要完整後端支援。
