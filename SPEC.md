# q01 規格書（SPEC v0.1，草案）

> **一句話**：`q01` 是 **CUDA-Q 的相容子集**，用 **NumPy / PyTorch** 在**純 CPU、原生 Windows** 上執行，
> 並與 **CUDA-Q 0.16.0** 逐項對帳。
> 它不是「另一個量子框架」，而是本書教學軌的執行引擎，兼論文的可重現性產物。

- 狀態：**草案，待作者審**
- 建立：2026-09-18
- 相關文件：`AGENTS.md`（專案公約）、`HANDOFF.md`（交接）、
  `docs/03-實作/無CUDA-Q的參考實作.md`（現行純 NumPy 路線）、
  `docs/03-實作/CUDA-Q核心API.md`（API 事實來源）
- 命名：**`q01`**（PyPI 查詢為 **FREE**，`import q01` 合法；備選 `qbnlite` 亦 FREE）

---

## 0. 這份文件怎麼用

| 讀者 | 該讀哪幾節 |
|---|---|
| 教學書作者 | §1 §2 §3 §9 §13 |
| 實作者 | §3 §4 §5 §5.5 §6 §10 §12 |
| 老師／委員 | §1 §2 §5 §11（精度與風險） |
| 未來的自己 | 全部，尤其 §4 的語意契約與 §11 的坑 |

---

## 1. 問題陳述（全部有實測依據）

### 1.1 學生端的真實障礙不是「沒有 CUDA」

| 事實 | 證據 |
|---|---|
| CUDA-Q **不需要 GPU** | 參考機（i7-11800H，無獨立 GPU）以 `qpp-cpu` 執行，`dev/verify_all.py` **20/20 通過**；全書 36 處 `set_target("qpp-cpu")` |
| CUDA-Q **沒有 Windows wheel** | PyPI `cudaq 0.16.0` 只有 `cudaq-0.16.0.tar.gz`；二進位走 `cuda-quantum-cu12/cu13`，僅 manylinux／macosx；uv 的錯誤訊息已記錄在本書〈無CUDA-Q的參考實作〉§1.1 |
| 但學生**必須**在 WSL2 才能跑 | 這是「純 WSL2」路線的真正成本：跨檔案系統、要開虛擬化、Windows 端編輯器與執行環境分離 |

**結論**：障礙是**安裝與作業系統**，不是算力。這件事值得在書裡明說，因為讀者是物理背景，
他們最容易被「量子 = 要 GPU」這種直覺誤導。

### 1.2 現有資產與缺口

| 已有 | 缺口 |
|---|---|
| `dev/qbn_sim.py`（826 行、63 項自我測試）純 NumPy 狀態向量核心 | 它是 **class API**（`sim.ry(θ, k)`），與書中的 CUDA-Q 寫法**是兩套語法** |
| `dev/qbn5_encoding.py`（825 行）5 qubit 編碼層 + 去相位消融 | 同上；學生看到的是兩個世界 |
| `docs/03-實作/無CUDA-Q的參考實作.md`（748 行） | 它的定位是「保險線／交叉驗證」，**不是**可以照著做練習的主線 |
| 全書 **54 個** CUDA-Q 程式碼區塊（7 檔，26 個在〈CUDA-Q 核心 API〉） | **這些區塊在 Windows 上一行都跑不動** |

### 1.3 為什麼不是「換成 Qiskit 教」

- 論文與既有 artifact 的主線是 CUDA-Q；換教材等於承認「我們換了一個套件」。
- Qiskit 有 Windows wheel（實測 `qiskit 2.5.2` win_amd64 9.5 MB；`qiskit-aer 0.17.2` 有 cp313 win_amd64 9.6 MB），
  所以它是**很好用的第二對照物**，但**不該取代** CUDA-Q 的敘事。

---

## 2. 目標與非目標

### 2.1 目標（可驗收）

| # | 目標 | 驗收方式 |
|---|---|---|
| G1 | **同一份程式碼兩軌可跑**：學生在 Windows 用 `q01`，我們在 WSL 用真 CUDA-Q | 全書 54 個區塊在兩種模式下 EXIT=0 |
| G2 | **精度與 CUDA-Q 同級** | 算子級 `|dψ|\max = 0`；電路級 `max|ΔP| ≤ 1e-15`（見 §5） |
| G3 | **三行上手**：`uv init` → `uv add q01 numpy matplotlib` → `uv run lesson.py` | 在乾淨 Windows 帳號實測 |
| G4 | **教學規模夠用**：5 qubit 主線，≤ 12 qubit 練習，實用上限 ~20 qubit | §6 的規模掃描 |
| G5 | **可微分**：PyTorch autograd + 參數平移兩條路 | §5.4 對帳 |

### 2.2 非目標（明確不做，避免範圍蔓延）

- ❌ 不主張量子優勢（沿用專案立場）。
- ❌ v1 不支援雜訊模型／density matrix target（書中 0 次使用）。
- ❌ 不做 transpiler／電路最佳化（tket 的研究是為了「對照」不是「實作」）。
- ❌ **不取代 CUDA-Q 作為論文正式後端**：論文報的數字仍由真 CUDA-Q 產生。
- ❌ 不改 `dev/verify_all.py` 的 20 項契約（只能讓它繼續綠）。
- ❌ **執行期不得需要網路、API key 或雲端**（可重現性是硬需求；oracle／emitter 一律列為選配 extra）。
  這條是對「到處要飯」的正面回答：**artifact 的價值在於不需要任何人的免費額度就能重現**。
- ❌ 不接 Amazon Braket（2026-09-18 決定）：唯一好處是「將來能上真機」，而本專案明確不主張量子優勢；
  成本是 25 + 20 個依賴。**明確不做**，以免日後反覆。

---

## 3. 對外介面（API 契約）

**原則**：只實作書中用到的子集；**超出子集一律 `NotImplementedError` 並附上原因與替代寫法**，
絕不靜默降級或算出「看起來差不多」的錯答案。

### 3.1 使用方式（兩軌共用）

```python
try:
    import cudaq              # 正式軌：裝了真 CUDA-Q 就用真的
except ImportError:
    import q01 as cudaq       # 練習軌：NumPy / PyTorch 後端
```

### 3.2 介面清單

| 名稱 | 簽章 | 狀態 | 備註 |
|---|---|---|---|
| `kernel` | `@cudaq.kernel` 裝飾器 | 必做 | 追蹤式（trace）實作，見 §4.4 |
| `qvector` | `qvector(n)` → 可索引、`front(k)`、`back()` | 必做 | |
| `qubit` | `qubit()` | 必做 | 單一 qubit |
| 閘 | `h x y z s sdg t tdg sx rx ry rz` | 必做 | 角度一律在前 |
| 糾纏閘 | `cx cz swap ccx` | 必做 | |
| `.ctrl()` 修飾 | `ry.ctrl(theta, [c0,c1,c2], t)`、`x.ctrl([...], t)`、`h.ctrl(c,t)` | 必做 | 與 CUDA-Q 實測一致（〈CUDA-Q 核心 API〉§3.2） |
| 鏈式 `.ctrl().ctrl()` | — | **不支援** | CUDA-Q 也不支援，錯誤訊息要照抄它的語意 |
| `adjoint` | `cudaq.adjoint(sub, q, θ)` | 必做 | **只支援自訂 kernel**，與 CUDA-Q 相同 |
| `control` | `cudaq.control(sub, [c], q, θ)` | 必做 | 同上 |
| `mz / mx / my` | kernel 內使用 | 必做 | 含線路中量測，見 §4.5 |
| `sample` | `sample(k, *args, shots_count=)` → `.counts`、`.probability()`、`__getitem__` | 必做 | 鍵＝位元字串，q0 在最左；**kernel 內含線路中量測時必須照 CUDA-Q 報錯並導向 `run`**（§4.5） |
| `run` / `run_async` | `run(k, *args, shots_count=)` | **必做（v0.5 新增）** | 線路中量測的**正式路徑**；`sample` 已不接受這種 kernel（`python/cudaq/runtime/sample.py:86-100`） |
| `get_state` | `get_state(k, *args)` → `np.ndarray` | 必做 | **little-endian**（原始碼已證：`QppCircuitSimulator.cpp:135-148`），且複製「kernel 內有 mz → 回傳塌縮態」行為 |
| `observe` | `observe(k, spin_op, *args).expectation()` | 必做 | **預設為決定性**（逐 term basis-rotate ＋精確 parity 加總，不是矩陣收縮：`QppCircuitSimulator.cpp:311-321`、`CircuitSimulator.h:1211-1236`）；只有 `shots_count>0` 才有 shot noise |
| `spin` | `spin.x/y/z(i)`、乘積 `spin.z(0)*spin.z(1)` | 必做 | |
| `gradients.ParameterShift` | `.compute(vec, loss_fn, loss_at_x)` | 必做 | |
| `set_target` | `"qpp-cpu"` 收下；額外接受 `"numpy" / "torch" / "qiskit" / "cudaq"` | 必做 | 學生不必改程式碼 |
| `set_random_seed` | 同 CUDA-Q | 必做 | 可重現性是硬需求 |
| `make_kernel` | 動態建構 | 選做（P3） | 書中 1 個區塊 |
| 雜訊／`evolve`／`dynamics`／`get_unitary`／`getSVGstring` | — | **不支援** | `NotImplementedError` + 指向 CUDA-Q |

### 3.3 錯誤訊息的規格

超出子集時，錯誤必須包含三件事：**（1）哪個 API 不支援；（2）CUDA-Q 的原文錯誤長什麼樣；
（3）使用者該改寫成什麼**。例：

```
NotImplementedError: q01 不支援 ry.ctrl(c0).ctrl(c1)(theta, t)（鏈式 .ctrl()）。
  真 CUDA-Q 0.16.0 的錯誤：unknown function call
  請改寫成： ry.ctrl(theta, [q[0], q[1]], q[2])
```

這是「教學工具」與「玩具」的分界線：**錯誤訊息本身就是教材**。

---

## 4. 語意契約（§ 公約，實作前先釘死）

### 4.1 位元順序（本書最危險的坑，實測確認）

| 來源 | 順序 | 實測 |
|---|---|---|
| `cudaq.get_state` | **little-endian**（q[0] 是最低位） | `x(q[0]) ` → 索引 **1**；`x(q[4])` → 索引 **16** |
| `cudaq.sample(...).counts` 的鍵 | **q[0] 在最左** | `x(q[0])` → `"10000"` |
| 本專案 NumPy 核心 | **big-endian**（q[0] 是最高位） | `x(0)` → 索引 **16** |

**契約**：`q01` 對外必須和 CUDA-Q 完全一致（`get_state` little-endian、字串 q0 在最左）；
bit-reversal **只在一個模組（`bitorder.py`）裡出現一次**，並由 §10 的測試守住。
（實測教訓：本 SPEC 的第一次基準腳本正是因為漏了這個 perm，得到假的 0.31 誤差。）

**原始碼層面的證據（v0.5 補：不再只靠實測）**

- `QppCircuitSimulator.cpp:135-148`：`convertQubitIndex(q) = log2(dim) - q - 1` ⇒ `get_state` **確定**是 little-endian。
- `CircuitSimulator.h:883-887`：`sample` 對 sampleQubits **升冪排序** ⇒ counts 的鍵 q0 在最左。
- 這兩條可直接寫進書中〈CUDA-Q 核心 API〉，取代當前的 `TODO(核實)`。

**Qiskit 的好消息（v0.5）**：Qiskit 的 NumPy 索引順序**與 CUDA-Q 相同**（q0 = LSB；
`qiskit/quantum_info/operators/operator.py:530-532`：「qubit 0 corresponds to the right-most position
in the tensor product」），只有 **counts 的字串**是反的（`statevector_sampler.py:238-239`）。
⇒ `bitorder.py` 的責任縮到最小：**矩陣／向量層零置換**，只在「樣本字串 ↔ 索引」這一處反轉。

### 4.2 角度與參數順序

- 閘一律 **角度在前**：`ry(theta, target)`、`ry.ctrl(theta, ctrls, target)`。
- 閘的矩陣定義與 CUDA-Q 逐元素相同（§5.2 實測 `|dψ|max = 0`）。

### 4.3 全域相位與等價判準

機率與期望值比較**不比相位**。狀態向量與么正矩陣的比較必須使用**除相位後的等價判準**。

**★ 算子層一律用 `U†V ≈ cI`**（來源：tket `tket/test/src/Simulation/ComparisonFunctions.cpp:144-179`，tol 1e-10）：

- **只比「某一個輸入態的輸出」是不夠的**：閘分解若在某個基底態上寫錯，單態重疊會漏掉它。
- 正確做法：建出整個么正矩陣，比較 `M = U†V`，檢查 `M ≈ cI`（`c = tr(M)/dim`，且必須 `|c| = 1`）。
- 成本：n qubit 的閘需要 2^n 次模擬建出矩陣；**n ≤ 3 是秒級，全部測**。
- ⚠️ 本規格 v0.1–v0.3 所謂「算子級 `|dψ|max = 0`」只用了 |0⟩ 一個輸入態，**強度不足**，v0.4 起改掉。

狀態向量層（電路層比較）用 `1 - |<ψ|ψ'>| ≤ 1e-14`，且必須搭配多組輸入態。

### 4.4 `@kernel` 的執行模型

- 採 **trace（追蹤）**：呼叫 kernel 時記錄閘序列成一條 `Circuit`，再由後端執行。
- **支援的 Python 構造以 CUDA-Q 的 AST 白名單為準**：該檔把 `generic_visit` 覆寫成**一律報錯**
  （`python/cudaq/kernel/ast_bridge.py:1913-1920`），所以「支援 = 存在對應的 `visit_*`」，共 **27 個**
  （含 `for`／`while`／`if`／`range`／索引／二元運算／函式呼叫；
  **不含** `assert`、`with`、`try`、`class`、`dict`、f-string、walrus `:=`）。
  → q01 的檢查表**直接照抄這 27 項**。
- **動態 qubit 數是允許的**（`ast_bridge.py:3832-3835`：`quake.AllocaOp` 的 size 是 MLIR Value，可來自執行期參數）
  → **q01 必須支援**；v0.1–v0.4 把它列為「不允許」是**寫反了**，那會拒絕真 CUDA-Q 接受的程式、直接破壞 G1。
- **q01 不引入自創旋鈕**：`@cudaq.kernel` **沒有** `strict` 參數（`kernel_decorator.py:708`），
  q01 也不加——否則同一份程式碼在正式軌會 `TypeError`。
  嚴格檢查改由**獨立函式**提供（`q01.check_kernel(fn)`），書中示範用、正式軌可省略。
- 已知差異（要寫進書中附錄）：trace 對 Python 的寬容度仍高於 CUDA-Q 的 AST 編譯器。
  **對策**：CI 在 WSL 內用真 CUDA-Q 跑同一批區塊（§7）。

### 4.5 量測語意

> v0.5 依 CUDA-Q 原始碼修正：`sample` **不接受**含線路中量測的 kernel（這是 0.14 起的既成事實，不是未來變更）。

| 情境 | 契約 |
|---|---|
| 電路尾端 `mz(q)` → `sample` | 用精確機率做多項式抽樣（**O(shots)**，不是逐 shot 重跑） |
| **線路中量測**（`mz` 之後還有閘／條件分支）→ `sample` | **照 CUDA-Q 報錯並導向 `run`**（`python/cudaq/runtime/sample.py:86-100`；`docs/.../sample_vs_run.rst:9-12`）。**不做「默默幫你跑軌跡」** |
| 線路中量測 → `run` / `run_async` | 每 shot 重跑一次軌跡（trajectory）；量測結果是具體 bool，故 `if` 天然可用。CUDA-Q 在 qpp-cpu 上也是逐 shot 重跑（`CircuitSimulator.h:668-678`：`supportsBufferedSample` 恆為 false） |
| `get_state` 且 kernel 內有 `mz` | **複製 CUDA-Q 的行為**：回傳塌縮態（機制：`QppCircuitSimulator.cpp:276-295` 原地覆寫 state ＋ `CircuitSimulator.h:1297-1308`） |
| `get_state` 且無 `mz` | 精確振幅 |

### 4.6 隨機性

`set_random_seed(n)` 之後，`sample` 必須**完全可重現**（同一台機器、同一後端）。
跨後端不保證同一條亂數序列，但**分布必須在統計上一致**（§7 L4）。

---

## 5. 精度規格與實測

### 5.1 判準

| 層級 | 判準 | 依據 |
|---|---|---|
| 算子（單閘、複合閘） | **`max|U†V - cI| ≤ 1e-10`**（全么正矩陣，除全域相位）且 `|c| = 1` | tket 判準（§4.3）；雙精度 eps = 2.22e-16 |
| 電路（機率向量） | `max|ΔP| ≤ 1e-15` | 一次電路 ~20 個閘，誤差累積 ~1e-15 |
| 狀態向量 | `1 - |<ψ|ψ'>| ≤ 1e-14` | 除全域相位後比較 |

### 5.2 實測（參考機 i7-11800H / WSL2 / numpy 2.5.3 / CUDA-Q 0.16.0 qpp-cpu）

| 項目 | 結果 |
|---|---|
**（A）算子層 —— 全么正矩陣，判準 `U†V ≈ cI`（v0.4 起；v7 實測）**

| 閘 | dim | `max|U†V - cI|` | 判定 |
|---|---|---|---|
| `h` | 2 | 2.24e-17 | PASS |
| `x` | 2 | **0.000e+00** | PASS |
| `rx` / `ry` / `rz` | 2 | 2.47e-17 | PASS |
| `cx` | 4 | **0.000e+00** | PASS |
| `cz` | 4 | **0.000e+00** | PASS |

`|c| = 1.000000000000000` 全部成立；最差 **2.47e-17**（約 0.1 × 雙精度 eps）。
公約自檢：`cx` 把 big-endian 索引 2（\|10⟩）送到 3（\|11⟩）✅

**（B）電路層 —— 32 維機率與狀態向量（v3 實測）**

| 項目 | 結果 |
|---|---|
| 僅 RY 編碼層（5 qubit） | `max|ΔP| = 0`、`1-|<ψ|ψ'>| ≤ 1.11e-16` |
| 編碼 + 環形 CX | `max|ΔP| = 0`、`1-|<ψ|ψ'>| ≤ 2.22e-16` |
| **完整 QBN（depth 2, ring，3 組隨機參數）** | **`max|ΔP| = 1.11e-16`、`1-|<ψ|ψ'>| ≤ 4.44e-16`** |

> **回答「至少是同一個算子精度？」**：在上述閘與電路上，我們與 CUDA-Q 是**同一個浮點結果**
> 或相差 0.1 × eps，遠優於「同一級」的要求。原因是雙方的閘都直接以 `cos/sin` 生成矩陣
> 並以相同順序縮併，沒有中間近似。

**（C）★ 覆蓋缺口（v7 實測；P1 必須補）**

現行 `dev/qbn_sim.py` 的 `StateVectorSim` **不支援**：
`y`、`z`、`s`、`sdg`、`t`、`tdg`、`sx`、`swap`、`ccx`。
而書中**全部用到**（出現次數：`z` 28、`s` 13、`swap` 4、`ccx` 4、`t` 3、`sdg` 2、`tdg` 2、`sx` 2、`y` 1）。
→ P1 必須補齊這 9 個閘，並**全部納入同一套 `U†V ≈ cI` 判準**（§4.3）。

**（D）★ 跨軌端到端對帳（P1 已完成；2026-09-18）**

**同一個檔案** `tools/oracle/cases.py` 在兩軌執行——一軌是真 CUDA-Q 0.16.0（參考機 WSL），
一軌是 q01 0.0.1（原生 Windows，不需要 WSL／CUDA-Q）——各自輸出同格式 JSON 後逐元素比對：

| 案例 | 維度 | `max|ΔP|` |
|---|---|---|
| `bell` | 4 | **0.000e+00** |
| `entangle_then_rotate` | 4 | **0.000e+00** |
| `mcry3`（3 控制位 RY） | 16 | **0.000e+00** |
| `qbn5_book`（書中 5 qubit 輸出層） | 32 | **2.08e-17** |
| `qbn5_depth2`（depth 2 環形） | 32 | **1.67e-16** |
| ⟨Z₀⟩ | — | 5.55e-17 |
| ⟨Z₀Z₁⟩ | — | 3.33e-16 |

**最差 3.33e-16**（判準 1e-10 → 好 **6 個數量級**）。

- 黃金向量入庫：`tests/golden/cudaq_golden.json`（記 backend／版本／numpy／python／platform）
- 測試：`tests/test_golden.py`（**無黃金檔時自動跳過**——學生端不需要 CUDA-Q）
- 兩軌測試皆 **28 passed**：本機 Windows（uv 0.8.13／Python 3.13.13／numpy 2.5.3）0.16 s；
  參考機 Windows（uv 0.11.8／Python 3.13.14）0.69 s
**（E）★ 跨框架對帳：5 套獨立實作（P3 核心，2026-09-18）**

把 IR 發射成 Qiskit／Cirq／PennyLane 的電路，用**它們自己的本地模擬器**（免費、無帳號、無雲端）
各算一次，全部對 CUDA-Q 的黃金向量比對（判準 `max|ΔP| ≤ 1e-10`）：

| 案例 | q01 核心 | Qiskit 2.5.2 | Cirq 1.7.0 | PennyLane 0.45.1 |
|---|---|---|---|---|
| `bell` | 0.000e+00 | 0.000e+00 | 2.22e-16 | 0.000e+00 |
| `mcry3`（3 控制位 RY） | 0.000e+00 | 2.22e-16 | 0.000e+00 | 0.000e+00 |
| `qbn5_book` | 2.08e-17 | 3.47e-17 | 2.22e-16 | 2.08e-17 |
| `qbn5_depth2` | 1.67e-16 | 1.11e-16 | 2.78e-16 | 1.67e-16 |
| `entangle_then_rotate` | 0.000e+00 | 0.000e+00 | 1.74e-17 | 0.000e+00 |

**最差 2.78e-16**（Cirq）；全部在 Windows 原生執行、離線可跑。

**L0 公約實測（實測得來，不是查文件）**：

| 框架 | `x(q0)` 的非零索引（n=3） | 對外契約 |
|---|---|---|
| CUDA-Q `get_state` | 1 | little-endian（原始碼 `QppCircuitSimulator.cpp:135-148`） |
| Qiskit `Statevector.data` | 1 | little-endian，**零置換** |
| Cirq `final_state_vector` | 4 | big-endian → 走 `bitorder` 轉換 |
| PennyLane `qml.state()` | 4 | big-endian → 走 `bitorder` 轉換 |

**兩個踩過的坑（已寫進 adapter 檔頭，每個新 adapter 都要重新確認）**：

1. **Cirq 預設是 `complex64`**（貝爾態機率 0.49999997，差 ~3e-8）→ **達不到 1e-10 判準**，
   必須 `cirq.Simulator(dtype=np.complex128)`。
2. **PennyLane 的 `qml.ctrl(<實例>, control=...)` 不會排進電路**（CNOT 靜默失效，
   貝爾態看起來像「少了糾纏」）→ 必須傳可呼叫物：`qml.ctrl(qml.PauliX, control=[0])(wires=1)`。

> 這也回答了「到處要飯」：**驗證用的 oracle 全部免費、本地、可離線**——
> 不需要任何人的雲端額度，就能把「5 套獨立實作一致到 2.78e-16」寫進論文。

**（F）相容性矩陣：默認零雲端，但可驗證相容（2026-09-18）**

**原則**（作者指示）：學生端**默認不需要任何雲端帳號**；「相容」不靠宣稱，靠三層可執行的檢查。

| 檢查 | 方法 | 結果 |
|---|---|---|
| ① 五套獨立實作 | 同一顆電路在 CUDA-Q／q01／Qiskit／Cirq／PennyLane 各跑一次 | 最差 **2.78e-16**（§5.2 E） |
| ② 硬體基底轉譯 | 轉譯成真機閘集後**用該框架自己的模擬器重跑** | Qiskit（`rz,sx,x,cx`）最差 **8.74e-16**；Cirq（CZ gateset）最差 **5.11e-15** |
| ③ 預設安裝不綁雲 | 檢查 provider 套件有沒有被自動裝上 | `qiskit_ibm_runtime`／`cirq_google` **皆未安裝**（`tests/test_compat.py` 守住） |

**CUDA-Q 0.16.0 共 33 個 target**（參考機 `cudaq.get_targets()` 實測）：

- **本地模擬（無需帳號）**：`qpp-cpu`、`density-matrix-cpu`、
  `nvidia`／`nvidia-fp64`／`nvidia-mgpu`／`nvidia-mqpu`（含 `-fp64`、`-mps`）、
  `tensornet`／`tensornet-mps`、`orca-photonics`、`dynamics`。
- **遠端供應商（預設用不到）**：`ionq`、`quantinuum`、`oqc`、`iqm`、`infleqtion`、
  `anyon`、`tii`、`scaleway`、`fermiq`、`braket`。
  多數支援 `emulate=True` → **沒有帳號也能在本地驗證提交形狀**（`docs/sphinx/using/backends/hardware/*.rst`）。
- **測試／benchmark**：`value-semantics-test`、`compiler-bench-*`。

> 兩點推論：① `braket` 本身就是 CUDA-Q 的一個 target ⇒ 我們**不需要自己裝 Braket SDK**
> 就已經具備 Braket 相容性（客戶端由 CUDA-Q 提供）。② 這也讓「不接 Braket」的決定更站得住：
> 要用的時候走 CUDA-Q 就好，不必在 q01 多背 45 個依賴。

### 5.3 我們**不**保證的事

- ❌ 不保證所有電路逐位元相同（只保證 §5.1 判準）。
- ❌ 不保證與 CUDA-Q 的**取樣序列**相同（只保證分布一致）。
- ❌ 不支援的路徑**不會**悄悄用 NumPy 近似（一律報錯）。

### 5.4 可微分路徑（P4 驗收）

| 路徑 | 對帳對象 | 判準 |
|---|---|---|
| `torch` 後端 autograd | 參數平移（解析） | 梯度 `max|Δg| ≤ 1e-9` |

**2026-09-19 實測（P4 完成）**：autograd vs 參數平移的梯度差 = **1.946e-19**（CPU 與 CUDA 相同），
損失值與 numpy 路徑一致到 1e-12；torch 後端對黃金向量的 L2 = **2.22e-16（CPU）／2.78e-16（CUDA）**。
判準是 1e-9，實測好 10 個數量級。
| 參數平移（自製） | CUDA-Q `gradients.ParameterShift` | `max|Δg| ≤ 1e-12` |

---

### 5.5 PyTorch／GPU 後端的設計約束（P4 前置實測，2026-09-18）

**問題**：能不能用 PyTorch 吃「通用 GPU」？—— 可以，而且是我們手上**唯一**的通用路線，
但有三個實測出來的硬約束。

**（1）裝置不是問題，wheel 才是**

參考機 venv 是 `torch 2.14.0+cpu`：`cuda.is_available()=False`、
`cuda.device_count()=0`、`torch.version.cuda=None` ⇒ **就算插了卡也沒有 GPU 路徑**。
要走 GPU 必須裝對應的 wheel（CUDA／ROCm 共用 `torch.cuda` API；Intel 走 `torch.xpu`；
Apple 走 `torch.mps`）。程式碼不變，換的是 wheel 與 device 字串。

**（2）★ 精度：complex64 過不了判準**

同一顆 5 qubit QBN 電路（對 float64 numpy 參考）：

| dtype | `max|ΔP|` | 對 §5.1 的 1e-10 判準 |
|---|---|---|
| `complex64` | **6.26e-08** | ❌ 差兩個數量級 |
| `complex128` | **1.39e-16** | ✅ |

⇒ **要維持判準，GPU 上就必須走 FP64（complex128）。**

> TODO(核實)：消費級顯卡的 FP64 吞吐被大幅削減、Apple MPS 的 float64／complex128 支援範圍、
> Intel XPU 的 complex 支援範圍——這三者直接決定「哪張卡能跑我們的 1e-10 判準」。
> **必須在真裝置上各測一次**，不可只在 CPU 上宣稱。查證前不得寫進書中當事實。

**（3）autograd 可用且精確**

實測：loss `⟨Z⟩=cos θ` 與梯度 `-sin θ` 的誤差 **0.0e+00**
（PyTorch 對 complex 用 Wirtinger 導數；實數參數 + 實數損失這條路成立）。

**（4）CPU 上 torch 沒有速度優勢**

numpy 0.479 ms vs torch 1.112 ms（complex128、同一顆電路、暖機後 best-of-100）⇒ torch 的價值是
**(a) 真 autograd (b) 換裝置就能吃 GPU**，不是 CPU 加速。

**（5）P4 的實作規則（由以上實測導出）**

- `set_target("torch", device=...)`：`cpu` / `cuda` / `mps` / `xpu`。
- 預設 `complex128`；`complex64` 必須**明示選用**，且接受**較鬆的判準**（不得宣稱 1e-10）。
- 提供 `q01.torch_info()`：印 device、可用 dtype、FP64 是否可用——學生「跑不動」時的第一個診斷點。
- 書中**不得**出現「用 GPU 就快」而未在該裝置實測的敘述。

### 5.6 精度底線稽核：我們還能更準嗎？（2026-09-19 實測）

**問題**（作者提出）：能不能用高精度算術（EFT／double-double／Ozaki）把 CPU 模擬的準確度再推上去？

**（1）實測：我們離「精確解」只剩 0.5 × eps**

用 **mpmath 60 位有效數字獨立重寫**同一顆電路（qbn5_book：5 qubit、20 個閘）當參考：

| 項目 | 值 |
|---|---|
| `max|P_fp64 − P_exact|` | **1.110e-16** |
| 雙精度 eps | 2.220e-16 |
| 我們的誤差 | **0.5 × eps** |
| 理論量級（√K 隨機遊走，K≈20 個閘） | 2–20 × eps |
| `sum(P_fp64)` | 1.00000000000000000 |

**⇒ 我們已經低於理論誤差模型一個量級。statevector 這條路沒有可提升的空間。**

**（2）在 numpy 裡做 DD 的成本（決定性）**

- `np.fma` **不存在**（只有純量 `math.fma`，無法向量化）。
- 沒有向量化 FMA，TwoProduct 必須走 Dekker 拆分：**有 FMA 是 2 次運算，沒有是 17 次**
  （見 Ogita–Rump–Oishi 的 EFT 文獻）。
- 我們的 1q 路徑本來就是**開銷／頻寬受限**（§6.7／§6.9），DD 會變成約 8–17× FLOPs ＋ 2× 記憶體，
  而換到的精度（1e-32）**對任何結論都沒有影響**：本專案要報的是準確率／NLL／ECE，
  由有限樣本決定（5 個種子、檢定力 0.275 → 統計不確定性 ~1e-2 級，比捨入誤差大 14 個數量級）。

**（3）★ 真正吃到精度的三個地方（改用精準打擊）**

| # | 場景 | 為什麼敏感 | 對策 |
|---|---|---|---|
| 1 | **去相位消融** | 要區分 **1.4e-17（≈0）** 與 **0.0502** —— 這是相消敏感判定 | DD／mpmath 參考解 + 明確誤差界 |
| 2 | **ECE 恆等式** | 專案實測誤差 2.78e-17；要主張「恆等」需要**誤差上界**，不是「數字很小」 | 同上（這也是論文能加分的點） |
| 3 | 歸約（`sum(P)`、期望值、ECE/NLL 統計） | 誤差隨項數累積 | **Kahan–Babuška–Neumaier 補償求和**（幾行、零成本） |

**（4）結論**

- ❌ **不把整個模擬器換成 DD／QD**：我們已在舍入底線，換了只會慢 8–17× 而買不到任何東西。
- ✅ **採用補償求和**在歸約處（幾行成本，讓 `= 1`／`= 0` 的主張精確到 eps²）。
- ✅ **建立 DD／mpmath 高精度 oracle**（`q01[oracle]`），把論文的精度主張從
  「兩套實作一致」升級為「**我們的誤差 0.5 eps，已在理論底線上，且精確參考解以 60 位算出**」。
- ❌ **Ozaki／混合精度 GEMM 不適用**：那是「在 FP16/INT8 張量單元上模擬 FP64」的**加速**技術，
  我們沒有大矩陣乘法，而且已經在捨入底線。

**（5）參考文獻（書中「精度」一節的閱讀清單）**

- **EFT 基礎**：Ogita, Rump, Oishi (2005) 原始論文；J-STAGE (2020) 驗證數值計算綜述
- **Dekker / Veltkamp split**、**TwoSum / FastTwoSum**、**TwoProduct（FMA）**
- **雙倍-雙倍（DD）在 AVX2**：幸谷智紀等，ARITH 2021 的 LU 分解論文（DD 相對純量 >3×）
- **快速 FMA 的誤差界**：Jeannerod, Joldes, Louvet, Muller（ARITH 2026）——有條件保證
- **Kahan 向量化**：CERN `ROOT::Math::KahanSum`（AVX2-double 建議 N=4 個累加器）
- **Ozaki 方案**：RIKEN 2025 研討會；**混合精度綜述**：Higham & Mary, Acta Numerica 2022

## 6. 效能規格與實測

測試機：**i7-11800H（16 核、無獨顯）、WSL2、numpy 2.5.3、CUDA-Q 0.16.0 `qpp-cpu`**。
所有數字為同一顆電路、同一批參數、暖機後量測（best-of-N 取 min，另記 median）。

### 6.1 延遲（5 qubit QBN 電路，depth 2 ring）

| 項目 | NumPy 核心 | CUDA-Q `qpp-cpu` |
|---|---|---|
| 首次呼叫 | 1.29 ms | **88.8 ms**（連空 kernel 都要 88.8 ms，純 JIT） |
| 穩態 best-of-300 | 0.593 ms | 0.289 ms |
| 穩態 median | 0.689 ms | 0.477 ms |

### 6.2 ★ 延遲穩定性（200 次連續呼叫 —— 本節最重要的一張表）

| 引擎 | n | min | **p50** | p90 | p99 | max | **> 5 ms 的比例** |
|---|---|---|---|---|---|---|---|
| NumPy 核心 | 5 | 0.186 | **0.231** | 0.278 | 0.361 | 8.77 | **1/200** |
| NumPy 核心 | 10 | 0.577 | **0.648** | 0.716 | 1.015 | 1.257 | **0/200** |
| CUDA-Q | 5 | 0.091 | 0.116 | **44.56** | 61.40 | 68.55 | **50/200** |
| CUDA-Q | 10 | 0.463 | 6.459 | **92.27** | 144.92 | 196.73 | **111/200** |

- **不是量測腳本的 GC 偽影**：`gc.disable()` 為 109/200、每次 `gc.collect()` 為 87/100，
  雙峰分布不變。
- `cudaq.sample`（100 shots × 100 次）同樣：p50 0.306 ms 但 p90 38.79 ms、26/100 超過 5 ms。
- 也就是說：**CUDA-Q 的最佳情況更快，但有 25–55% 的呼叫會停頓 40–200 ms**；
  我們的實作慢一點，卻幾乎完全可預測。
- **對教學的意義**：學生跑 1000 次訓練迭代時，CUDA-Q 的總時間與體感抖動由這些停頓主導；
  我們的 `0.65 ms × 1000 ≈ 0.65 s` 是學生能預期的。

> TODO(核實)：停頓原因尚未定位（推測與 runtime 每次呼叫的執行引擎／記憶體配置有關）。
> 已派 CUDA-Q 原始碼調查（見 `notes/refs-cudaq.md`）；同時待測：**原生 Linux（非 WSL2）是否同樣雙峰**。

### 6.3 規模對照（同一電路 H^n + 環形 CX）

| n | NumPy min | CUDA-Q min | min 比值 | CUDA-Q p50 | 狀態向量 |
|---|---|---|---|---|---|
| 5 | 0.263 | 0.138 | 0.53× | 25.49 | 0.00 MB |
| 8 | 0.441 | 0.229 | 0.52× | 0.54 | 0.00 MB |
| 10 | 0.730 | 0.571 | 0.78× | 26.37 | 0.02 MB |
| 12 | **1.683** | 3.803 | **2.26×** | 80.66 | 0.07 MB |
| 14 | **5.147** | 53.70 | **10.4×** | 136.7 | 0.26 MB |
| 16 | **30.42** | 271.1 | **8.9×** | 313.9 | 1.05 MB |
| 18 | **204.1** | 578.4 | **2.8×** | 608.1 | 4.19 MB |
| 20 | **1090** | 2055 | **1.9×** | 2233 | 16.78 MB |

（min 比值 < 1 表示 CUDA-Q 最佳情況較快；p50 欄顯示 CUDA-Q 的典型值被停頓拖高。）

**結論**：n ≤ 10 兩者同級（CUDA-Q 最佳情況略快）；**n ≥ 12 我們的 NumPy 更快（最多 10×）**；
n ≥ 18 兩邊都進入秒級 —— 這是教學上限的自然邊界。

### 6.4 取樣（教學最常用的一步）

| shots | NumPy 核心 | CUDA-Q `qpp-cpu` | 比值 |
|---|---|---|---|
| 10k | 0.70 ms | 4.56 ms | 6.5× |
| 100k | **4.2 ms** | 56.3–135.8 ms（兩次獨立量測） | **12–32×** |

原因：我們從精確機率向量做多項式抽樣（O(shots)），`qpp-cpu` 是逐 shot 重跑模擬。

### 6.5 記憶體上限（純 NumPy，單一狀態向量）

| n | 單次求機率 | 狀態向量 |
|---|---|---|
| 14 | 5.1 ms | 0.26 MB |
| 16 | 30 ms | 1.05 MB |
| 18 | 204 ms | 4.19 MB |
| 20 | 1.09 s | 16.8 MB |
| 24（外推） | — | 268 MB |
| 26（外推） | — | 1.07 GB |

### 6.6 效能目標（P1 驗收）

- 5 qubit 電路單次 ≤ 5 ms（現況 0.6–0.7 ms ✅）
- 1000 次訓練迭代（前向）≤ 5 s（現況 ~0.7 s ✅）
- 100k shots 取樣 ≤ 50 ms（現況 4.2 ms ✅）
- **延遲可預測性**：單次呼叫 > 5 ms 的比例 **< 5%**（現況 0/200 ✅）——這是相對 CUDA-Q 的明確優勢
- 最佳情況不得比 CUDA-Q 慢超過 3×（現況 n ≤ 12 為 0.52–0.78×）

---

### 6.7 FP32／FP64、SIMD 與真正的瓶頸（2026-09-18 實測）

**背景**：作者指出 NumPy 有些運算會因精度考量**掉回純量**，因此在現代 SIMD 機器上
FP32 不一定比 FP64 快。以下實測證實了這個直覺，但同時也證明**它對本專案幾乎不重要**。

測試機：i7-11800H（**AVX-512**），numpy 2.5.3 dispatch = `['X86_V3','X86_V4','AVX512_ICL','AVX512_SPR']`，
BLAS = scipy-openblas 0.3.34（Haswell, MAX_THREADS=64）。

**（1）FP32 vs FP64：純運算 vs 超越函數（1M 元素）**

| 運算 | 吞吐 | 備註 |
|---|---|---|
| 乘法 float32 | 17.59 GB/s | |
| 乘法 float64 | **22.41 GB/s** | **FP64 反而更快**：此規模是記憶體頻寬受限，不是算力受限 |
| `sin` float32 | 8.36 GB/s | 走向量化路徑 |
| `sin` float64 | **1.18 GB/s** | **慢 7 倍** → FP64 的超越函數**掉回純量 libm**（約 7 ns/元素） |

⇒ 作者所述「因精度而丟純量」確實存在，但方向是：**FP64 的 sin/cos/exp 掉純量，FP32 才有 SIMD**；
而純乘加的 FP32 lane 優勢，被「本來就不是算力瓶頸」抵銷。

**（2）但在我們的電路上，dtype 幾乎不影響速度**

同一顆 5 qubit QBN 電路：

| n | complex64 | complex128 | 比值 | 狀態向量（f64） |
|---|---|---|---|---|
| 5 | 0.344 ms | 0.345 ms | **1.00×** | 0.0005 MB |
| 10 | 1.056 ms | 1.126 ms | 1.07× | 0.02 MB |
| 12 | 1.750 ms | 1.996 ms | 1.14× | 0.07 MB |
| 14 | 4.503 ms | 5.708 ms | 1.27× | 0.26 MB |
| 16 | 18.675 ms | 32.007 ms | **1.71×** | 1.05 MB |
| 18 | 267.5 ms | 341.5 ms | 1.28× | 4.19 MB |

**（3）5 qubit 的 0.743 ms 花在哪**

| 項目 | 時間 | 佔比 |
|---|---|---|
| 完整電路（30 個閘） | 0.743 ms | 100% |
| 同樣次數的空 `tensordot` | 0.572 ms | **77.0%**（純呼叫開銷） |
| 同樣次數的純量 `np.cos` | 0.014 ms | 1.8% |

與 §8.5 那份加速調查**獨立**推出的成本模型（`0.205 ms 固定 + 8.5 ns/(振幅·閘)`，n=5 有 98% 是開銷）吻合。

**（4）★ 真正的瓶頸是受控閘的實作（不是 dtype）**

規模掃描是**超線性**的：n=14→16→18 為 4.5 → 18.7 → **267 ms**。振幅數只漲 4 倍，
時間漲 14 倍 ⇒ 有東西**按 2^n 分配中間陣列**：目前的 `apply_controlled_1q` 每個 CX 都重建
`np.arange(2^n)` 遮罩並用布林 fancy indexing；n=18 時每個閘配置 4 MB 索引陣列。

**（5）結論與 P4 的優先序**

1. **受控閘改用 reshape／strided 視圖**（不要每閘重建遮罩）← 最大一筆，且與 dtype 無關。
2. **消滅呼叫開銷**：扁平 opcode 陣列執行器（n ≤ 8 用 numba，預估 7–20×，見 §8.5）。
3. **不要**為了速度換 FP32：最多賺 1.7×（還要在 n ≥ 16 才看得到），代價是 1e-10 判準
   （complex64 實測 6.26e-08，見 §5.5）。**維持 complex128 預設。**

> 這一節同時是書中可用的教材：**「為什麼量子模擬不該用 float32 換速度」**——
> 因為在教學規模（n ≤ 12）差距是 1.0–1.14×，而精度代價是 6 個數量級。


**（6）修正後實測（受控閘改走視圖）**

把單一控制位（熱路徑）改成「moveaxis 視圖 + 切片 + matmul 寫回」，不再重建遮罩：

| n | 修正前 | 修正後 | 加速 |
|---|---|---|---|
| 5 | 0.345 ms | 0.201 ms | 1.7× |
| 10 | 1.126 ms | 0.581 ms | 1.9× |
| 12 | 1.996 ms | 1.154 ms | 1.7× |
| 14 | 5.71 ms | 4.70 ms | 1.2× |
| 16 | 32.0 ms | 35.8 ms | （雜訊內） |
| 18 | 341.5 ms | **165.1 ms** | **2.1×** |
| 20 | — | 844 ms | — |

**數值完全不變**：39 項測試（含黃金向量 5 案 + 算子 `U†V ≈ cI`）全綠。

> 成長仍超線性（每 +2 qubit 約 ×5，而非振幅的 ×4）：`matmul` 每個閘仍會配置一個
> 半狀態大小的臨時陣列。要再推只能上「扁平 opcode 陣列執行器」＋ n ≤ 8 的 numba 路徑（§8.5）。
### 6.8 ★ CPU vs GPU：為什麼差這麼多（2026-09-19 實測 + 原始碼拆解）

測試機：i7-11800H（16 核）+ **RTX 3060 Laptop 6 GB**（CC 8.6，驅動 616.92，WSL2 直通）。

**（1）實測：同一顆 H^n + 環形 CX 電路（穩態 best-of-N）**

| n | qpp-cpu (fp64) | CUDA-Q GPU fp32 | CUDA-Q GPU fp64 | GPU 加速比 |
|---|---|---|---|---|
| 5 | **0.103 ms** | 3.019 ms | 2.751 ms | **0.03×（GPU 慢 30 倍）** |
| 10 | **0.841 ms** | 3.487 ms | 3.015 ms | 0.24× |
| 14 | 57.4 ms | 3.821 ms | 3.686 ms | 15× |
| 18 | 493.6 ms | 4.392 ms | 6.108 ms | 112× |
| 20 | 2048 ms | 7.065 ms | 14.29 ms | 290× |
| 22 | 8482 ms | 20.52 ms | 45.49 ms | **413×** |
| 24 | — | 64.44 ms | 144.6 ms | — |

**（2）為什麼差這麼多？量有效頻寬就清楚了**

| 引擎 | n=22 狀態 | 閘數 | 移動量 | 時間 | **有效頻寬** |
|---|---|---|---|---|---|
| qpp-cpu | 67 MB | ~44 | ~5.9 GB | 8.48 s | **0.7 GB/s** |
| CUDA-Q GPU (fp32) | 67 MB | ~44 | ~5.9 GB | 20.5 ms | **288 GB/s**（≈ 這張卡的上限 ~336 GB/s） |

- GPU 是**貼著記憶體頻寬上限**在跑。
- qpp-cpu 卻只跑 0.7 GB/s，**離單執行緒 CPU 的頻寬能力（約 20–30 GB/s）還差 30–40 倍**。
- **根本原因（原始碼確認）**：qpp-cpu 的閘作用**是單執行緒的**——OpenMP 只出現在
  `calculateExpectationValue（QppCircuitSimulator.cpp:162-175）`，閘作用路徑沒有平行化。
  一核 vs RTX 3060 的數千核 + 每閘的 C++ 配置開銷。

**（3）「批次設計」的真相：CUDA-Q 的 batch 不是閘融合**

拆 0.16.0 的 `runtime/nvqir/custatevec/CuStateVecBatch.cpp` 得到：

- `applyMatrices()` 呼叫 `custatevecApplyMatrixBatched(handle, /*batchedSv=*/states, ...)`（:265,276）
  —— 批次化的對象是**狀態向量**（多條軌跡），不是閘。
- 註解自證用途：「Batched **trajectory** gates must have identical operands and layout」（:314）、
  `applyNoise`（:454）、`sample`（:664）；內積走 `cublasZgemmStridedBatched`。
- ⇒ **我們的精確純態 `get_state` 走的是「一次一個閘」的普通路徑，與我們相同。**

所以 GPU 的優勢不是批次，而是**單閘 kernel 的品質 + 頻寬**；CPU 的劣勢則是**單執行緒 + 每閘開銷**。

**（4）★ PyTorch 能複刻嗎？能（1.35×–4.45×）**

純 Python 原型（`tools/oracle/torch_gpu_proto.py`，~60 行，無自訂 kernel）：
**little-endian 佈局** ⇒ 單閘 = `view(-1, 2, 2^t) + matmul`（零拷貝）；
受控閘 = `permute 視圖 + v[1].copy_(v[1].flip(0))`（不建索引張量）。

| n | torch GPU (complex128) | CUDA-Q fp64 | torch/CQ | torch vs q01 numpy |
|---|---|---|---|---|
| 14 | 4.96 ms | 3.69 ms | 1.35× | 0.9× |
| 16 | 5.98 ms | 3.54 ms | 1.69× | 5.0× |
| 18 | 8.78 ms | 6.11 ms | 1.44× | 19× |
| 20 | 33.9 ms | 14.29 ms | 2.37× | 25× |
| 22 | 141.9 ms | 45.49 ms | 3.12× | ~24× |
| 24 | 643.5 ms | 144.6 ms | 4.45× | — |

（差距隨 n 變大是因為原型的 CX 用 `flip+copy_`，每個閘多配置半個狀態的臨時陣列；
cuStateVec 有專門的 kernel。補上一個特化 CX 就能再縮。）

**（5）其他兩個副產品**

- **「3 ms 地板」不是 D2H 傳輸**：實測 D2H+sync 在 n=5 是 0.060 ms、n=14 是 0.092 ms
  （n=22 才 8.8 ms）⇒ CUDA-Q 的 ~3 ms 是它自己的每次呼叫開銷。
- **第三次獨立確認「預設 FP64」**：torch GPU 的 complex64 只快 1.2–1.8×，
  但誤差 **5.99e-04（n=14）／1.58e-05（n=20）**——比 CUDA-Q 的 fp32 還差，完全不能用。

**（6）對本專案的結論**

- **書（5 qubit）**：繼續用 CPU；GPU 在這個規模**慢 30 倍**（每次呼叫有固定開銷）。
  交叉點約在 n ≈ 13–14。教材價值極高：「何時該上 GPU」有了完整的實測表。
- **論文 C1/C2**：若 ansatz 放大到 ≥16 qubit，**本地 GPU**（免費、不碰雲）是正確選擇。
- **q01（P4）**：torch 後端加 `device="cuda"` 後，n≥18 相對 numpy 有 19–25× 的實質收益；
  且難度低（原型已證明 ~60 行可達 CUDA-Q 的 1.35–4.45×）。

### 6.9 CPU 還有多少可優化？（2026-09-19 原型實測）

現況：q01 的 numpy 核心有效頻寬只有 **2–6 GB/s**，而單執行緒 CPU 的能力是 20–30 GB/s、
16 執行緒約 40–80 GB/s ⇒ 帳面上還有 10–30× 的頭。寫了三個原型實測（同電路、交錯量測、min-of-5）：

- **A**：現行 big-endian 核心（每閘 moveaxis + tensordot／einsum）
- **B**：little-endian 佈局 + view 就地寫回（單執行緒）
- **C**：B + 8 執行緒（1q 閘切塊；numpy 放 GIL）

正確性先驗：B/C 經 bit-reversal 後與 A 一致到 **2.2e-16** ✅（否則時間沒有意義）。

| n | A 現行 | B little-endian | C +8 執行緒 | B/A | C/A |
|---|---|---|---|---|---|
| 5 | 0.486 ms | **0.192 ms** | 5.71 ms | **2.53×** | 0.09× |
| 10 | 0.990 ms | **0.495 ms** | 13.1 ms | **2.00×** | 0.08× |
| 14 | 4.75 ms | 7.03 ms | 97.2 ms | **0.68×** | 0.05× |
| 16 | 22.05 ms | 33.3 ms | 125 ms | **0.66×** | 0.18× |
| 18 | 202.9 ms | 196.0 ms | **156.9 ms** | 1.03× | **1.29×** |
| 20 | 922 ms | 995 ms | **606 ms** | 0.93× | **1.52×** |

**★ 被實測否決的假設：「全面改 little-endian 會更快」。**
小 n 確實快 2.0–2.5×（省掉每閘 moveaxis 與配置），但 n=14–16 **反而慢 1.4–1.5×**：
因為此版的 little-endian CX 用 `moveaxis + sub[::-1].copy()`（跨步拷貝半個狀態），
而現行核心的 CX 已經是「視圖 + `np.matmul`」（走 BLAS，快取友善）。
⇒ **閘路徑要逐類型選擇，不能一刀切換佈局。**

**可行清單（依「實測增益 × 信心 ÷ 風險」排序）**

| # | 優化 | 實測／預估 | 適用 n | 風險 |
|---|---|---|---|---|
| 1 | **1q 閘走就地切片**（避免 tensordot/einsum 分發與配置） | **實測 2.0–2.5×** | n ≤ 12 | 低（數值已驗） |
| 2 | **多執行緒，但改用「跨閘批次」切法**（現在是每閘開 8 個 task，n≤12 反而慢 12×） | 實測 1.3–1.5×（改切法後預期更高） | n ≥ 18 | 中 |
| 3 | **numba 執行器**（消滅 Python 呼叫開銷） | 調查預估 7–20× | n ≤ 8 | 中（依賴 + numpy 版本封印） |
| 4 | **閘融合**（同 qubit 連續 1q 合併；不相干閘合併成一趟） | 未實測，預估 1.3–1.6× | 全範圍 | 低 |
| 5 | 全盤改 little-endian | **實測否決**（n=14–16 慢 0.66×） | — | — |

**結論**：教學範圍（n=5–12）今天就能拿到 **~2×**（第 1 項），且與 GPU 無關；
大 n（≥18）靠執行緒再拿 **~1.5×**；要更大就得動 numba（第 3 項）。
注意：**這些 CPU 優化對 5 qubit 教學的實際意義有限**（0.5 ms → 0.2 ms），
真正的價值在 n ≥ 14 之後——而那時 GPU 已經快 15–400×（§6.8）。


**（7）已實作：1q 閘就地切片（2026-09-19）**

把原型 B 的核心想法（就地、零配置）實作進 `simulator.py` 的 `apply_1q`：
`psi.reshape(-1, 2, 1 << (n-1-k))` 的中間軸就是 qubit k（big-endian 權重），全程視圖 + 就地寫回，
舊的 einsum／tensordot 路徑移除（實測在 n=18–20 兩者持平，n ≤ 12 則慢一倍以上）。

| n | 改前 | 改後 | 加速 |
|---|---|---|---|
| 5 | 0.201 ms | **0.134 ms** | 1.5× |
| 12 | 1.154 ms | **0.665 ms** | 1.7× |
| 14 | 4.704 ms | **1.903 ms** | 2.5× |
| 18 | 165.1 ms | **44.7 ms** | **3.7×** |
| 20 | 844 ms | **253 ms** | **3.3×** |

**數值完全不變**：39 項測試（黃金向量 5 案 + 算子 `U†V ≈ cI` + 跨框架）全綠；
真 CUDA-Q 重跑黃金向量差異 = **0.0**。

> 比原型的 2.0–2.5× 更好，因為原型只改了 1q 路徑的佈局，而這裡同時移除了 einsum 的整狀態配置。

### 6.10 外部 SIMD 函式庫的適用邊界（SimSIMD／NumKong 實測，2026-09-19）

作者指向 `github.com/ashvardanian/NumKong`（PyPI `numkong`）與 `simsimd`。
兩者皆 Apache-2.0、**零依賴**、有 win_amd64 與 cp313 輪子；NumKong 宣稱覆蓋到 128-bit 複數，
且**把累加器提升到更寬型別**（f16→f32、**f32→f64**、i8→i32）。以下在我們的 workload 上實測。

| workload | numpy | simsimd | numkong |
|---|---|---|---|
| **complex128 內積**（2^18 = 262144 元素；量子的 ⟨ψ\|ψ'⟩） | **13.8 µs（304 GB/s）** | 149 µs（28 GB/s） | 368 µs（11 GB/s） |
| **1×11514×256 餘弦相似度**（potion 嵌入檢索） | 4.45 ms（f32，OpenBLAS） | **0.604 ms（7.4×）** | — |
| **f32 點積的精度**（2048 維，對 f64 累加參考） | 相對誤差 **1.20e-07** | 3.48e-07 | **6.37e-16** |

**判讀**

- ❌ **量子端（q01 核心）用不到**：numpy 的 complex128 `vdot` 比它們快 **10–27×**；
  這兩家的強項是實數／量化型別（f16/bf16/i8/e5m2）的檢索與 GEMM，不是複數狀態向量。
  我們的瓶頸本來也不在 FLOPs（§6.7／§6.9：開銷與頻寬）。
- ✅ **ML 端（前端嵌入）有實質價值**：1-to-many 餘弦相似度快 **7.4×**（f32）／2.4× 對 numpy-f32（f64 版）。
  ⚠️ 這個比較的 numpy 基線**含每次重新正規化整個 (11514,256) 矩陣**；實際管線若只正規化一次，差距會縮小。
  真正穩固的用法是「**單一查詢對大庫**」的檢索路徑（也正是 SimSIMD 的設計目標）。
- ✅ **★ 精度這條是關鍵發現**：NumKong 的 `dot(f32, f32)` 用 **f64 累加**，
  相對誤差 **6.37e-16**（等於 f64 累加結果），而 numpy 的 f32 點積是 **1.20e-07**。
  ⇒ 若 ML 管線（PCA、邏輯迴歸、類別原型、kNN）跑在 f32，**NumKong 同時給速度與精度**。

**對本專案的行動建議**

| 對象 | 建議 |
|---|---|
| q01 核心（量子模擬） | **不引入**（已量化：numpy complex128 快 10–27×） |
| ML 前端的檢索／相似度（可選 extra `q01[fast-retrieval]`） | 評估 SimSIMD；若管線用 f32 且在意精度，優先 NumKong |
| 精度論述（§5.6） | 把「f32 累加會損失 1e-7」寫進書中，作為「為何預設 FP64」的第三個獨立證據 |

## 7. 對帳協定（oracle）

### 7.1 四層

| 層 | 比什麼 | 對象 | 判準 |
|---|---|---|---|
| L0 | 公約：位元順序、角度符號、全域相位 | CUDA-Q + Qiskit `Operator` | 布林 |
| L1 | 算子：閘與受控變體 | `cudaq.get_state` / Qiskit `Operator` | §5.1 |
| L2 | 電路：書中 54 個區塊 | 真 CUDA-Q | §5.1 |
| L3 | 演算法：QBN 訓練曲線、去相位 0.0502 / 1.4e-17、C1 4-qubit MNIST smoke | CUDA-Q 黃金向量 | 相對誤差 ≤ 1e-10 |
| L4 | 統計：取樣 z 檢定、種子可重現 | 解析值 | `max z ≤ 4` |

### 7.2 黃金向量（golden vectors）

- 由**參考機**（有真 CUDA-Q）產生：`tools/oracle/emit_golden.py` → `tests/golden/*.json`
- 格式：`{case, n_qubits, circuit_digest, params, probabilities[2^n], backend, shots, seed, tool_version}`
- **入庫**：學生不需要任何量子 SDK，就能用 `q01` 比對黃金向量自我檢查
- 產生環境必須記錄：CUDA-Q 版本、numpy 版本、CPU、日期

### 7.3 可重現性聲明（給論文）

> 「CUDA-Q 0.16.0 與 q01（純 NumPy）兩套獨立實作，在 5 qubit QBN 電路上逐元素一致至
> `max|ΔP| = 1.11e-16`；Qiskit 作為第三套實作交叉核對（P3）。」

這比現行書中「保險線」的說法更強：**是三個實作，且誤差有數字**。

---

## 8. 加速策略研究（待研究，repo 已 clone 到 `.refs/`）

| 候選 | 假設（要驗證或推翻） | 預期適用處 |
|---|---|---|
| **Numba**（`numba`） | n ≤ 12 時我們的成本幾乎全是 Python／NumPy 呼叫開銷（0.6 ms 對 32 個振幅）；JIT 一個電路執行器可能快 10–50× | 小 n 前向、訓練迴圈 |
| **Numexpr**（`numexpr`） | 只在記憶體受限的大陣列運算式上有用；閘作用的張量縮併已被 numpy 用 BLAS 吃滿 | n ≥ 18 的閘融合 |
| **Dask**（`dask`） | 對「參數掃描／多 shots／多電路」這種易平行工作有用，但 `concurrent.futures` 可能就夠 | 批次訓練、參數掃描 |
| **多執行緒** | numpy 在 BLAS 內放 GIL；真正的平行點是 shots 與參數掃描 | 取樣、梯度批次 |
| **mpmath**（任意精度） | 不是為了跑電路，而是當**第三個 oracle**：解析恆等式可驗到 30–50 位，不需要任何量子 SDK | 書中「解析釘死」測試、ECE 恆等式（2.78e-17）那類 |
| **PyTorch** | autograd 走計算圖；要確認 `complex128` 支援與梯度精度 | QML 訓練（G5） |
| **PennyLane / lightning.qubit** | 效能與架構對照：C++/Kokkos + OpenMP 的實作方式、gate fusion、adjoint diff | 選配後端、效能標竿 |
| **tket** | 電路最佳化／transpile 的對照物；**不打算實作** | 僅文獻對照、可能用來驗證閘分解等價 |
| **CUDA-Q 本身** | 讀 `qpp-cpu` 的實作與 AST bridge，確認我們的語意契約沒抄錯 | §4 契約 |

### 8.1 已回收的研究結論（2026-09-18）

**PennyLane 主 repo（`notes/refs-pennylane.md`）** —— 五個直接改動本規格的結論：

| 研究結論（附原始碼位置） | 對 q01 的決策 |
|---|---|
| **平行只在 tape（電路）層**；單一電路永遠單執行緒（`pennylane/devices/default_qubit.py:793-824`；shots 也沒有 shot 層平行） | §8 的「多執行緒」列**降為低優先**：5 qubit 單電路沒有 thread 可切；要平行只能多電路／多參數批次 |
| **einsum vs tensordot 有硬門檻常數**：`EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = 13`（`pennylane/devices/qubit/apply_operation.py:29-30`，判斷式 `:341-351`）→ **n ≤ 12 走 einsum** | `simulator.py` 的閘作用路徑按 n 分流（n ≤ 12 einsum／n > 12 tensordot）；門檻值列為【待實測】在本機複驗 |
| **效率來自特化 kernel**：X 用 `math.roll`（`:521-524`）、Z/T/S 半片乘常數（`:527-544`、`:574-609`）、CNOT 用 roll 半片而非 4×4（`:762-778`） | P1 的閘實作照此特化，**不要全部走通用張量縮併**（我們的 0.6 ms 幾乎全是通用路徑的呼叫開銷） |
| **gate fusion 是電路層 transform，訓練時失效並污染梯度**（`pennylane/transforms/optimization/single_qubit_fusion.py:27-30`、`optimization_utils.py:112-119`：`requires_grad` 時跳過快速路徑） | **q01 v1 不做 gate fusion**；若要做只掛推論／取樣路徑。同時修正「lightning 有 gate fusion」這個說法 |
| **backprop 不需要自訂 autograd Function**：state 更新走框架無關的 math 分派（`pennylane/math/__init__.py:14-35`），`torch.autograd.Function` 層只服務「不可微後端要梯度」（`workflow/interfaces/torch.py:118-197`） | §5.4 的 torch 路線 = **純 torch 複數張量免費 autograd**；v1 梯度只做參數平移，不自建 Function |

**順帶對帳（好消息）**：PennyLane 的抽樣實作與本規格 §4.5 **完全一致**
（`np.random.default_rng` + `rng.choice(..., p=probs)` 一次抽完：`pennylane/devices/qubit/sampling.py:513,527`；
線路中量測每 shot 重跑：`simulate.py:376-381`）。

> ⚠️ 但 `sampling.py:529-531` 的 `1 << np.arange(num_wires)[::-1]` 是 **PennyLane 自己的 big-endian 約定**，
> **不能**當作 §4.1 的證據——**位元順序的 oracle 仍然只有 CUDA-Q**。

**tket**：`notes/refs-tket.md`（結論：q01 v1 不需要電路最佳化；僅作文獻對照）。

**研究產出**：每個 repo 一頁，放 `notes/refs-<repo>.md`。

### 8.2 tket 的結論（`notes/refs-tket.md`）

| 研究結論（附原始碼位置） | 對 q01 的決策 |
|---|---|
| **等價判準是 `U†V ≈ cI`**（`tket/test/src/Simulation/ComparisonFunctions.cpp:144-179`，tol 1e-10），不是單一態重疊 | **§4.3／§5.1 已改**：L1 算子層改用全么正判準（v0.4） |
| **全域相位是 IR 的一級欄位**（`tket/include/tket/Circuit/Circuit.hpp:1644` 的 `Expr phase`；分解的殘餘相位走 `add_phase`：`Rebase.cpp:48`） | q01 的 trace IR **必須有 `global_phase` 欄位**，否則 `ry.ctrl` 的分解測試永遠差一個相位（見 §9） |
| tket 的模擬器**也不做 gate fusion**（`GateNodesBuffer.cpp:47-52` 只有一行 `apply_full_unitary`，融合是註解裡的 TODO），預設上限 11 qubit（`CircuitSimulator.hpp:39-40,53-55`） | 佐證 §8.1 的決定：v1 不做融合；真正的槓桿是 JIT／特化 kernel |
| **pytket 有 Windows wheel**（`release.yml:146-183` 建置、`:371-391` 發布；但 release matrix 只列到 CPython 3.12，QBN 用 3.13） | 拒絕 tket 的理由是**範圍**（我們不需要 transpiler），不是「Windows 裝不了」；cp313 需查 PyPI（【待實測】） |
| 受控閘是獨立 OpType、依控制數分派不同分解（`ControlledGates.cpp:752-802`；gray-code 需矩陣 `2^(n-1)` 次方根 `:703`） | **不要複製這條路**：q01 直接構造受控 2×2 矩陣（`np_ref.py:93-112`）更精確也更短 |

**結論：q01 v1 不需要 tket**（原本的傾向獲得驗證）。將來「最佳化對照」章節裡，
tket 最有價值的角色是**等價判準的 oracle（不需要安裝 tket）**，其次是反面教材
（`GreedyPauliSimp` 官方註明不保全域相位 `passes.cpp:992-993`；
`ZXGraphlikeOptimisation` 自承可能讓電路變貴 `PassLibrary.hpp:153-154`）。

### 8.3 CUDA-Q 原始碼的結論（`notes/refs-cudaq.md`）

考察基準：`.refs/cuda-quantum` HEAD `249034efc`（2026-09-18）——**這是 main 分支，不是 0.16.0 tag**。

| 研究結論 | 對 q01 的決策 |
|---|---|
| **QPP 數值核心不在這個 repo**（`tpls/qpp` 是空目錄；CUDA-Q 只做 `qpp::apply/applyCTRL/sample/measure` 轉址：`QppCircuitSimulator.cpp:260,263,372`） | 「逐位元相同」**只能靠實驗支撐，不能引用原始碼**；§5.2 的量測因此是本規格唯一的精度證據 |
| **`observe` 預設不是 ⟨ψ\|H\|ψ⟩**：`canHandleObserve()` 預設 false（`:311-321`、`CircuitSimulator.h:1072-1074`），改走逐 term basis-rotate ＋精確 parity 加總 | §3.2 已補「預設決定性」契約；§5.4 的梯度對帳必須在 `shots_count=-1` 下比 |
| **`sample` 自 0.14 起就拒絕含量測分支的 kernel**（`python/cudaq/runtime/sample.py:86-100`，導向 `run`） | **§4.5 已重寫**；`run`/`run_async` 納入 API 子集 |
| **動態 qubit 數是允許的**（`ast_bridge.py:3832-3835`：`quake.AllocaOp` 的 size 是 MLIR Value） | **§4.4 原本寫反了，已改**：q01 必須支援，否則會拒絕真 CUDA-Q 接受的程式（破壞 G1） |
| **AST 白名單是反向機制**：`generic_visit` 被覆寫成一律報錯（`ast_bridge.py:1913-1920`）⇒ 支援集合 = **27 個 `visit_*`**（含 `while`；不含 `assert/with/try/class/dict/f-string/walrus`） | q01 的檢查表直接照抄這 27 項 |
| **`@cudaq.kernel` 沒有 `strict` 參數**（`kernel_decorator.py:708`） | **q01 也不加**（自創旋鈕會讓正式軌 `TypeError`）；嚴格檢查改由獨立函式 `q01.check_kernel(fn)` 提供 |
| 模擬前有 MLIR 管線（`JITTargetPipeline.cpp:38`、`Pipelines.cpp:109-129`：canonicalize／UnitarySynthesis／ApplySpecialization） | 可解釋「為什麼能逐位元相同」（`Passes.td:2136-2148`：UnitarySynthesis 只管 custom_op），但**建議【待實測】dump Quake IR 逐閘確認** |
| 官方遷移頁 `docs/sphinx/using/migration/upcoming_changes.rst` **全文 6 行、內容是 "coming soon"** | **不猜未來介面**；現階段只做「縮限 `sample` 範圍 ＋ 納入 `run`」 |

### 8.4 Qiskit／Aer 的結論（`notes/refs-qiskit.md`）

考察基準：`.refs/qiskit`（2.6.0.dev0）、`.refs/qiskit-aer`（0.17.2）。

| 研究結論 | 對 q01 的決策 |
|---|---|
| **位元順序與 CUDA-Q 同序**（`operator.py:530-532`），只有 counts 字串相反 | §4.1 已更新：**矩陣／向量層零置換**，`bitorder.py` 只剩字串反轉 |
| **閘矩陣權威在 Rust**：`crates/circuit/src/gate_matrix.rs:1-516`（dispatch `standard_gate/mod.rs:319`） | L1 的逐元素清單照這張表抄 |
| ⚠️ **`RZ(θ)=diag(e^{-iθ/2}, e^{+iθ/2}) ≠ P(θ)=diag(1, e^{iθ})`**；**`SX` 與 √X 差全域相位 `e^{iπ/4}`**（`sx.py:51-62`） | 這正是 v0.4 改用 `U†V ≈ cI` 判準的另一個理由（不會被「處處差同一相位」騙） |
| **Qiskit 2.6.0.dev0 沒有 `qiskit.gradients`**（全 repo `ParameterShift` 0 命中）；實作在未 clone 的 `qiskit-algorithms`。公式：`plus=x+π/2`、`minus=x−π/2`、`g=(evs[:n//2]−evs[n//2:])/2` | **不對帳引入 qiskit-algorithms**；公式寫進 q01 `gradients.py`，oracle 改用**解析梯度**。【待實測】CUDA-Q `ParameterShift` 是否同為 π/2 與 1/2 |
| **Aer 在 n ≤ 13 是單執行緒**（`statevector_parallel_threshold=14`，`:331-337`）、**gate fusion 也要 n ≥ 14**（`fusion_threshold=14`，`:475-494`） | 教學範圍 5–12 qubit **不必做閘融合、也不必模仿 threading**（與 §8.1／§8.2 同結論）。該規模唯一生效的最佳化是 `statevector_sample_measure_opt=10`（一次建機率向量、每 shot O(1)）——與 §4.5 的 `sampler.py` 契約同向 |
| **P3 首選「零後端」**：`Operator`／`Statevector`／`StatevectorSampler` 全是純 Python（只依賴 `quantum_info`：`operator.py:116-124`、`statevector.py:99-100`、`statevector_sampler.py:26,51,187`） | q01 只要**能吐出 `QuantumCircuit`** 就免費獲得官方 oracle；真後端（`BackendV2`+`JobV1`，範本 `basic_simulator.py`）列為 P3 選配。注意 `StatevectorSampler` **不支援線路中量測**（`statevector_sampler.py:213-216`） |

### 8.5 加速策略的結論（`notes/refs-accel.md`）

| 候選 | 結論 | 依據 |
|---|---|---|
| **Numba** | **只在 n ≤ 8 值得**（7–20×）；做成 `q01[fast]` optional extra，**永不進 runtime 依賴** | 由 §6.2 反推的成本模型：**0.205 ms 固定 + 8.5 ns/(振幅·閘)** ⇒ n=5 有 **98%** 是呼叫開銷、n=12 只剩 20%、n≥16 <1%。原假設「10–50×」對 n=12 不成立（約 1.2×） |
| Numba 實作限制 | JIT 核心**不能照抄 tensordot**：numba 沒有 `np.tensordot`／`np.einsum`（全 repo `einsum` 0 命中）⇒ 必須改寫為「扁平 opcode 陣列 ＋ 索引算術半邊更新」(`lo=i&~bit; hi=lo|bit`) | complex128 的 exp/sqrt/sin/cos 有 nopython 實作 |
| Numba 版本封印 | 支援 `numpy>=1.22,<1.27` 或 `>=2.0,<2.6`；**0.59 起移除 object-mode fallback** ⇒ 編譯失敗是硬失敗 | q01 必須保留 NumPy fallback 路徑 ＋ 測試；`numpy` 上界只出現在 `[fast]` extra |
| **Numexpr** | **永不納入** | 純逐元素 VM，對 tensordot/matmul/einsum **0 命中**、無 gather ⇒ 與閘作用無交集（USE_VML 在原始碼中被註解掉） |
| **Dask** | **v1／v2 都不納入** | 官方自報每 task 開銷 **200 µs–1 ms** ≥ 我們 n≤12 每 task 的 0.2–1.0 ms；且要 7 個依賴。要平行就用 `concurrent.futures` |
| **mpmath** | **納入（測試期）**：`q01[oracle]` | 零依賴；PSLQ 可把「去相位 0.0502」「ECE 2.78e-17」反查成閉式，`mp.iv` 提供誤差上界。限制：numpy 只吃純量、n≥16 不實用 ⇒ **不進 runtime** |

**→ P1 因此不引入任何加速依賴（純 numpy）**；`q01[fast]` 留到 P4 之後再評估。

---

## 9. 套件結構與 uv 整合

```
QBN/
├── pyproject.toml                  # 改成 uv workspace：members = ["packages/*"]
├── packages/q01/
│   ├── pyproject.toml              # dependencies = ["numpy>=2"]
│   │                               # extras: [torch] / [qiskit] / [oracle]=mpmath / [fast]=numba(numpy<2.6)
│   ├── SPEC.md                     # 本檔
│   ├── src/q01/
│   │   ├── __init__.py             # 對外 API（與 cudaq 同名同簽章）
│   │   ├── bitorder.py             # ★ 全專案唯一的位元順序轉換處
│   │   ├── circuit.py              # trace 記錄 + 電路 IR（★ 必須有 global_phase 欄位，見 §8.2）
│   │   ├── gates.py                # 閘矩陣（與 CUDA-Q 逐元素相同）
│   │   ├── simulator.py            # 狀態向量核心（自 dev/qbn_sim.py 抽出）
│   │   ├── sampler.py              # 多項式抽樣 + 軌跡模式（線路中量測）
│   │   ├── spin.py                 # Pauli 代數與期望值
│   │   ├── gradients.py            # ParameterShift
│   │   └── backends/{numpy,torch,qiskit,cudaq}.py
│   └── tests/                      # 單元 + conformance（54 區塊）+ golden 比對
└── tools/oracle/                   # 只在參考機跑：產生黃金向量
```

**學生端三行**（Windows PowerShell，完全不碰 WSL2）：

```powershell
uv init my-qbn-lab; cd my-qbn-lab
uv add q01 numpy matplotlib
uv run lesson01.py
```

**發布**：先出本地 wheel（`uv build` → `uv pip install dist/q01-*.whl`）；
PyPI 發布需要作者帳號，列為後續決定。

---

## 10. 測試計畫

| 類別 | 內容 | 需要 CUDA-Q？ |
|---|---|---|
| 單元 | 閘矩陣、bit order、取樣統計、spin 代數、參數平移 | ❌ |
| 一致性 | 對 golden vectors 比對（§7.2） | ❌ |
| Conformance | 書中 54 個區塊 EXIT=0、輸出對得上書中貼的實測值 | ❌（比 golden） |
| 對帳 | 在參考機對真 CUDA-Q 跑 L1–L4 | ✅ |
| 回歸 | `dev/verify_all.py` 維持 20/20；`dev/qbn_sim.py` 維持 63/63 | ❌ |
| 邊界 | 超出子集的 API 必須丟 `NotImplementedError`（且有測試） | ❌ |

---

## 11. 風險與對策

| # | 風險 | 影響 | 對策 |
|---|---|---|---|
| R1 | **語意漂移**：trace 比 CUDA-Q 的 AST 寬鬆，學生本機過、WSL 掛 | 高 | 54 區塊 conformance + 參考機對帳；`strict` 模式主動拒絕 |
| R2 | **位元順序**寫錯 | 極高（答案看起來「差不多但全錯」） | 單一 `bitorder.py` + §4.1 測試 + 書中公約檢查電路 |
| R3 | **教學價值被 shim 取代**：學生以為學的是 CUDA-Q | 中 | 明示「這是練習軌」；正式數字仍由 CUDA-Q 產生；書中附錄列差異 |
| R4 | CUDA-Q 版本漂移（0.16 → 0.17，API 已在預告變更） | 中 | 黃金向量記錄版本；對帳腳本在參考機重跑 |
| R5 | 維護兩套（核心 + shim） | 中 | **核心只有一份**：shim 直接吃 `dev/qbn_sim.py` 抽出的模組 |
| R6 | 學生環境差異（Python 版本、BLAS） | 低 | `requires-python >= 3.11`；CI 測 3.11/3.13 |

---

## 12. 驗收條件（Definition of Done）

**P1（核心）**
- [ ] `pip install -e packages/q01` 後，在**沒有 CUDA-Q、沒有 WSL** 的 Windows 上跑完書中 5 qubit 範例
- [ ] §5.2 的精度表全部重現（算子 0、電路 ≤ 1e-15）
- [ ] `bitorder.py` 之外的檔案**不得**出現 bit-reversal
- [ ] 超出子集 → `NotImplementedError`，訊息含 CUDA-Q 原文與改寫建議
- [ ] `dev/verify_all.py` 仍 20/20；`dev/qbn_sim.py` 仍 63/63

**P2（對帳）**
- [ ] 54 個區塊在參考機對真 CUDA-Q 通過（L2）
- [ ] `tests/golden/*.json` 入庫，且在無 CUDA-Q 環境可驗
- [ ] L3 的 QBN 訓練曲線與去相位數字對上（0.0502 / 1.4e-17）

**P3（Qiskit）**：算子級交叉核對完成；`qiskit` 後端可選
**P4（PyTorch）**：autograd 與參數平移對帳完成（§5.4）
**P5（書與公約）**：環境章改 uv、`AGENTS.md` 規則 1 改寫、新章上 nav

---

## 13. 里程碑

| 期 | 內容 | 產出 |
|---|---|---|
| P1 | 核心 + API 子集 + 單元測試（Windows 可跑） | `packages/q01`、測試綠 |
| P2 | 參考機對帳 + 黃金向量 | `tests/golden/*.json`、對帳報告 |
| P3 | 讀 Qiskit 原始碼 → Qiskit 後端 + 算子級核對 | 第三方一致性報告 |
| P4 | PyTorch 後端 + 梯度對帳 | 可微分路徑 |
| P5 | 書與 `AGENTS.md` | 新版環境章、新章 |

---

## 附錄 A：書中 54 個區塊的 API 覆蓋（來源：全書掃描）

`@cudaq.kernel` 34 次、`sample` 15、`get_state` 14、`.ctrl()/adjoint/control` 12、
`observe/spin` 9、`mz` 9、`ParameterShift` 3；**`NoiseModel`／density matrix 0 次、`cudaq.torch` 0 次**。
→ 這是「子集夠小、可以做到位」的量化依據。

## 附錄 B：待研究清單

`.refs/`：`cuda-quantum`、`qiskit`、`qiskit-aer`、`pennylane`、`tket`、`numba`、`dask`、`numexpr`、`mpmath`
（淺克隆）。產出：`notes/refs-survey.md`（§8 表格）。

## 附錄 C：變更紀錄

| 日期 | 版本 | 變更 |
|---|---|---|
| 2026-09-18 | v0.1 | 初稿；含 L0–L4 對帳協定、實測精度與效能、API 契約、驗收條件 |
| 2026-09-18 | v0.2 | §5.2／§6 換成 v3–v5 實測（算子逐位元相同、電路 1.11e-16、延遲穩定性雙峰）；確認名稱 q01 在 PyPI 為 FREE |
| 2026-09-18 | v0.3 | 新增 §8.1：PennyLane 調查回收（einsum/tensordot 門檻、特化 kernel 策略、gate fusion 排除、torch autograd 路線、多執行緒降級）；tket 結論 |
| 2026-09-18 | v0.4 | **修正精度判準的方法論漏洞**：算子層改用 `U†V ≈ cI`（tket 判準）；新增 §8.2 tket 結論；trace IR 加 `global_phase`；§5.2 換成 v7 全么正實測（最差 2.47e-17）＋覆蓋缺口（9 個閘未支援） |
| 2026-09-19 | v1.5 | 新增 §6.10：SimSIMD／NumKong 實測 —— 量子端 numpy complex128 快 10–27×（不引入）；ML 端 1-to-many 餘弦快 7.4×；**NumKong 的 f32 點積用 f64 累加，誤差 6.37e-16 vs numpy 1.20e-07** |
| 2026-09-19 | v1.4 | 新增 §5.6 精度底線稽核：以 mpmath 60 位獨立重寫為參考，實測誤差 **0.5 × eps**（已在理論底線下）；說明為何不採用 DD（np.fma 不存在 → TwoProduct 需 17 次運算）；改採「補償求和 + 高精度 oracle」精準打擊三個相消敏感場景，並附參考文獻清單 |
| 2026-09-19 | v1.3 | §6.9 實作 1q 就地切片（實測 1.5–3.7×，數值不變）＋ **P4 完成**：torch 後端（autograd 對參數平移 1.95e-19、L2 對黃金向量 2.2e-16 CPU／2.8e-16 CUDA、支援 device=cpu/cuda、多控制位、明確的 n 上限） |
| 2026-09-19 | v1.2 | 新增 §6.9 CPU 優化實測：1q 走就地切片可得 2.0-2.5x（n<=12）、執行緒在 n>=18 得 1.3-1.5x；**實測否決「全面改 little-endian」**（n=14-16 慢 0.66x） |
| 2026-09-19 | v1.1 | 新增 §6.8 CPU vs GPU 對照（GPU 在 n>=14 快 15-413x，5 qubit 反而慢 30x）+ 原始碼拆解（qpp-cpu 閘作用單執行緒；CuStateVecBatch 批次的是軌跡不是閘）+ PyTorch GPU 原型複刻到 CUDA-Q 的 1.35-4.45x |
| 2026-09-18 | v1.0 | 新增 §6.7：FP32/FP64 與 SIMD 實測（FP64 乘法反快、FP64 sin 掉純量慢 7×；但本專案 complex64 只快 1.0–1.7×，n=5 有 77% 是呼叫開銷）；確認真正瓶頸是受控閘的遮罩實作（超線性），維持 complex128 預設 |
| 2026-09-18 | v0.9 | 新增 §5.5：PyTorch／GPU 後端的設計約束（實測：complex64 6.26e-08 過不了 1e-10；complex128 1.39e-16；autograd 誤差 0；CPU 上 torch 慢 2.3 倍；wheel≠card） |
| 2026-09-18 | v0.8 | 新增 §5.2(F) 相容性矩陣：默認零雲端但可驗證相容（五套實作 2.78e-16；硬體基底轉譯後 ≤5.11e-15；預設安裝零雲端 provider）；CUDA-Q 33 個 target 清單 |
| 2026-09-18 | v0.7 | 新增 §5.2(E)：跨框架對帳 —— Qiskit／Cirq／PennyLane 三個獨立第三方實作（本地、免費、無帳號）與 CUDA-Q 一致到 2.78e-16；記錄 L0 公約實測與兩個坑（Cirq complex64、PennyLane qml.ctrl） |
| 2026-09-18 | v0.6 | 新增 §5.2(D) 跨軌端到端對帳（最差 3.33e-16）＋ P1 完成；§2.2 加入「執行期零網路／零 API key」硬規則與「不接 Braket」決定；新增 `notes/ARCH-REFACTOR.md`（分層提案，待審） |
| 2026-09-18 | v0.5 | **依 CUDA-Q／Qiskit 原始碼修正四處實作契約**：①`sample` 遇線路中量測要報錯並導向新增的 `run`；②動態 qubit 數其實允許（原寫反）；③移除自創的 `strict` 旋鈕；④`observe` 預設決定性。新增 §8.3–§8.5（CUDA-Q／Qiskit／加速策略結論）；加速依賴全部排除，P1 維持純 numpy |
