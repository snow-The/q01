# CUDA-Q 原始碼考察：q01 要「抄對」什麼

- **考察對象**：`.refs/cuda-quantum`，git HEAD `249034efc43dd3a4d30a00c4c12b14a6c5218116`（2026-09-18，"fix(python): stop parsing CLI arguments on import (#5396)"）。
- **這不是 0.16.0 release tag**，是 main 分支 HEAD；下列行號只對此 checkout 有效。凡「0.16 是否已如此」屬【待實測】。
- **路徑記法**：`CQ/` = `.refs/cuda-quantum/`。例：`CQ/runtime/nvqir/qpp/QppCircuitSimulator.cpp:130` = `.refs/cuda-quantum/runtime/nvqir/qpp/QppCircuitSimulator.cpp` 第 130 行。
- **標記**：【原始碼確認】= 我在該行讀到的事實；【推論】= 由事實推得但原始碼未直述；【待實測】= 需在參考機驗證。

---
## 0. 先講最重要的一件事：QPP 本體不在這個 repo

| 事實 | 證據 |
|---|---|
| QPP 是 git submodule，指向外部 | `.gitmodules` → `[submodule "tpls/qpp"] url = https://github.com/softwareQinc/qpp.git` |
| 本機 `tpls/qpp/` **是空目錄**（子模組未初始化），Eigen 同 | 實測 `Get-ChildItem CQ/tpls/qpp` → 無輸出 |
| CUDA-Q 只做轉址呼叫 | `CQ/runtime/nvqir/qpp/QppCircuitSimulator.cpp:260` `state = qpp::apply(state, matrix, targets);`、`:263` `qpp::applyCTRL(...)`、`:372` `qpp::sample(shots, state, measuredBits, 2)` |

【原始碼確認】`qpp::apply / applyCTRL / measure / sample / kron / reset` 的**內部演算法**（tensordot？閘融合？OpenMP 粒度？多項式抽樣？）**無法從本 repo 證實**。
→ **決策**：任何「與 CUDA-Q 逐位元相同」的主張**只能靠實驗**（本專案已有 max|ΔP|=1.11e-16 的實測），**不能引用原始碼**。若論文要寫「同一演算法」，須另克隆 softwareQinc/qpp 並在書中附註版本。

---
## 1. qpp-cpu 後端：CPU 狀態向量模擬

| 問題 | 結論 | 證據（原始碼行） |
|---|---|---|
| 實作位置 | CUDA-Q 側在 `nvqir/qpp`；數值核心在外部 Q++ | `CQ/runtime/nvqir/qpp/QppCircuitSimulator.cpp`（430 行）、`QppDMCircuitSimulator.cpp:11` `#include "QppCircuitSimulator.cpp"` |
| 預設模擬器 | `target qpp-cpu` → `nvqir-simulation-backend: qpp` → `QppCircuitSimulator<qpp::ket>`（**狀態向量**，非 density matrix；DM 是另一個 backend） | `CQ/runtime/nvqir/qpp/qpp-cpu.yml`、`QppCircuitSimulator.cpp:429` `NVQIR_REGISTER_SIMULATOR(..., qpp)`、`:130` |
| 精度 | **fp64（double）** | `:117` `return cudaq::SimulationState::precision::fp64;`；`:130` `CircuitSimulatorBase<double>`；`qpp-cpu.yml` 帶 `-D CUDAQ_SIMULATION_SCALAR_FP64` |
| 狀態向量型別/佈局 | `qpp::ket` = Eigen **行向量** `complex<double>`，長度 2^n，稠密連續 | `:27` `qpp::ket state;`、`:209` `qpp::ket::Zero(stateDimension)` |
| 閘演算法 | **逐閘乘**：佇列中每個 gate task 各呼叫一次 `qpp::apply`/`applyCTRL`。**CUDA-Q 層沒有 gate fusion、沒有 tensordot、沒有批次合併** | `:247-263`；`CQ/runtime/nvqir/CircuitSimulator.h:1033-1060`（`while(!gateQueue.empty()) applyGate(next);`） |
| 2×2 還是 k×k | 取決於 target 數：`toQppMatrix(task.matrix, task.targets.size())` → 1 個 target 是 2×2，多個是 2^k×2^k 單次套用 | `:248`、`:181-191` |
| 多執行緒 | CUDA-Q 側**只有一處** OpenMP：`calculateExpectationValue` 的 parity 迴圈（`#pragma omp parallel for`）。**閘作用路徑完全沒有 OpenMP / std::thread** | `:160-178`；grep `#pragma omp|std::thread` 於 `runtime/nvqir` → 只命中 `:163`、`:171` |
| 官方文件卻說 OpenMP | doc：「CPU-only, **OpenMP threaded** Q++ library」→ 執行緒在 **Q++ 內部**，不在 CUDA-Q | `CQ/docs/sphinx/using/backends/sims/svsims.rst:11` 【推論】 |
| 位元順序 | CUDA-Q 的 q0 ↔ Q++ 的最右（LSB）：`convertQubitIndex(q) = log2(stateDimension) - q - 1`。**⇒ `get_state` 是 little-endian，q0 = bit 0** | `:135-148`（含 ASCII 對照圖註解） |
| 自訂么正矩陣的 ordering 旗標 | QPP **覆寫為 `msb`**（base 預設 `lsb`）→ 多 target 的自訂矩陣會被轉置 | `:297` `getQubitOrdering() → QubitOrdering::msb`；`CircuitSimulator.h:37`（enum）、`:1081`（預設 lsb）、`:1359`（消費處） |

【推論，但要提防】`convertQubitIndex` 是**每個閘、每次量測**都做一次 `std::log2`（`:147`）；q01 應在電路建立期一次算好，別在熱迴圈模仿。這是 q01 小 n 能比 CUDA-Q 快的原因之一（SPEC §6.1 的 0.49×–0.69×）。

---
## 2. `@cudaq.kernel` 的 AST bridge：什麼能寫、什麼不能

**判斷位置**：`CQ/python/cudaq/kernel/ast_bridge.py`（~7500 行，`class PyASTBridge` 在 :362）。

### 2.1 白名單機制是「反向」的：沒有 visitor = 直接編譯錯誤
【原始碼確認】`ast_bridge.py:1913-1920` `generic_visit` 被覆寫成**一律報錯**：
> `self.emitFatalError("CUDA-Q does not currently support " + f"{type(node).__name__} expressions", node)`

⇒ 支援集合 = 有 `visit_X` 的節點集合。全部 visitor（檔內共 33 個 class/def，其中 27 個是 `visit_*`）：`Module, FunctionDef, Pass, Expr, Lambda, Assign, Attribute, Call, ListComp, List, Constant, Subscript, For, While, BoolOp, Compare, IfExp, Assert, If, Return, Tuple, UnaryOp, Break, Continue, BinOp, AugAssign, Name`。

**⇒ 不在名單＝不可用**（`ast_bridge.py:1913` 會擋）：`with`、`try/except`、`class`、`dict` 字面值、`set`、`import`、`global/nonlocal`、`yield`、`async`、`del`、`*星號展開`、f-string（`JoinedStr`）、海象運算子 `NamedExpr`、裝飾器鏈、`match`。
**決策**：q01 的 `strict=True` 檢查表**直接照抄這 27 個名字**，不要自己發明。

### 2.2 關鍵構造逐一回答
| 構造 | 支援？ | 原始碼行為 | 行號 |
|---|---|---|---|
| `for i in range(a,b,step)` | ✅ | 直接建成 MLIR 迴圈；**step 必須是常數**、非 0 | `ast_bridge.py:5233-5253`、`:1598-1623`（`:1618` `'range step value must be a constant'`） |
| `for i,j in enumerate(...)` | ✅ | 特例化成以 range 邊界驅動 | `:5258-5287` |
| `while` | ✅（意外） | 轉成 `cc.LoopOp` | `:5366-5388` |
| `if` | ✅ | 轉成 `cc.IfOp`（**不是** Python 的常數摺疊；分支是編譯出來的） | `:5634-5661` |
| `assert` | ❌ | `visit_Assert` 一律報錯 | `:5622-5632` |
| list 索引 | ✅ 但**索引要能化約**；tuple 的非定值索引直接報錯 | `:5131-5140` `"tuple value cannot be modified via non-constant subscript"`、`:5185`、`:5231` |
| **閉包捕獲** | ✅ **以「提升成 kernel 參數」實作** | `:6140-6184`：找不到區域符號 → `recover_value_of_or_none(node.id, self.defFrame)` → `cudaq_runtime.appendKernelArgument(...)` + `signature.add_variable_capture(...)` | 
| 捕獲**量子物件** | ❌ | 量子值不在可提升型別內；`:6189-6202` 找不到就 `"Invalid variable name requested - ..."` |
| **動態 qubit 數** | ✅ **可**（與 SPEC §4.4 相反！） | `:3832-3835` `if IntegerType.isinstance(value.type): qubits = quake.AllocaOp(ty, size=value)` —— `size` 是 MLIR **Value**，可來自 runtime 參數；動態長度另有 `quake.VeqSizeOp`（`:5306`） |
| 空 list | ❌ | `:4492`、`:4823` `"creating empty lists is not supported in CUDA-Q"` |
| 巢狀 `@cudaq.kernel` | ❌ | `:1972` `"nested @cudaq.kernel definitions are not allowed"` |
| 內層函式 `return` | ❌ | `:5665-5668` |

### 2.3 `@cudaq.kernel` 裝飾器本身
【原始碼確認】`CQ/python/cudaq/kernel/kernel_decorator.py:708` `def kernel(function=None, external=False, backend_symbol=None, **kwargs)`，`PyKernelDecorator.__init__`（`:123-135`）的參數是 `verbose, defer_compilation, disable_quantum_optimization, module, kernelName, signature, location, overrideGlobalScopedVars, decorator, atomic_quantum_region`——**沒有 `strict`**。
→ **決策**：SPEC §4.4 的 `strict=True` 是 **q01 自創旋鈕，CUDA-Q 沒有**。它不能出現在共用程式碼的呼叫端，否則正式軌（真 CUDA-Q）會 `TypeError`。建議改成模組層開關（如 `q01.strict_mode()`）或環境變數。

---
## 3. `get_state` vs `sample`：兩條完全不同的路徑

### 3.1 `get_state`
【原始碼確認】Python `get_state`（`CQ/python/cudaq/runtime/state.py:16-51`）→ `cudaq_runtime.get_state_impl`（`CQ/python/runtime/cudaq/algorithms/py_state.cpp:207-214`）→ `detail::extractState`（`CQ/runtime/cudaq/algorithms/get_state.h:38-57`）：

1. 建 `ExecutionContext("extract-state")`（`get_state.h:47`）
2. 跑整個 kernel（`get_state.h:49`）
3. 模擬器在 `finalizeExecutionContext` 看到 `context.name == "extract-state"` → `context.simulationState = getSimulationState()`（`CQ/runtime/nvqir/CircuitSimulator.h:1297-1308`）
4. `getSimulationState()` 只做 `flushGateQueue()` 再把**當前** state 搬走（`QppCircuitSimulator.cpp:407-410`）

**⇒ 「kernel 內有 mz 時 get_state 回傳塌縮態」在原始碼哪裡體現**：`QppCircuitSimulator.cpp:276-295` `measureQubit()` 呼叫 `qpp::measure(state, I, {qubitIdx}, 2, /*destructive=*/false)` 後**直接覆寫 `state`** 為坍縮後的分支（`:285-292`）。之後 `:407-410` 取出的是這個已塌縮的 state。**沒有「重跑一次取乾淨態」的機制**。
→ **決策**：q01 只需在 `measure` 時原地覆寫狀態向量即可自然複製此行為，不需額外分支。

### 3.2 `sample`：從機率抽樣，還是逐 shot 重跑？
| 情境 | 行為 | 證據 |
|---|---|---|
| 一般（尾端隱式量測） | **一次呼叫抽完所有 shots**：`qpp::sample(shots, state, measuredBits, 2)`；CUDA-Q 不重跑模擬 | `QppCircuitSimulator.cpp:358-405`（`:372`） |
| `shots < 1` | 不算抽樣，回**精確** parity 期望值 | `:361-364` → `:151-178` |
| `explicit_measurements=True` | **每 shot 重跑一次 kernel**（Python 層 `while` 迴圈補滿 shots） | `CircuitSimulator.h:668-678` `getNumShotsToExec()`：`explicitMeasurements && !supportsBufferedSample` → 1；`supportsBufferedSample` 預設 false（`:109`）且 **QPP 從未設 true**（只有 Stim 設 true：`stim/StimCircuitSimulator.cpp:730`）；迴圈在 `CQ/python/cudaq/runtime/sample.py:174` |
| 有量測分支的 kernel | **0.14 起 `sample` 直接 raise**，導向 `run` | `sample.py:86-100`；`docs/sphinx/using/examples/sample_vs_run.rst:9-12` |

【推論】「一次呼叫抽完」⇒ Q++ 內部是 O(shots) 的機率累積抽樣（多項式/alias 級別），**不是** O(shots × 2^n)。與 SPEC §6.1「取樣 100k shots 只要 4.22 ms」一致。
→ **決策**：q01 的 `sample()` 必須 (a) 無分支時用單次多項式抽樣；(b) `explicit_measurements=True` 時逐 shot 重跑；(c) **不要**為「有量測分支」實作 sample 路徑（CUDA-Q 已移除），改成 `run`。

### 3.3 位元字串順序（可從原始碼證實，不必只靠實測）
【原始碼確認】`sample` 先把 `sampleQubits` **升冪排序去重**（`CircuitSimulator.h:883-887`），再由 `:367-391` 依序 `convertQubitIndex` 後組字串。`convertQubitIndex` 對小 index 給大 Q++ index，而 Q++ 位元依 `measuredBits` 順序輸出 ⇒ **bitstring[0] = CUDA-Q q[最小 index] = q0**。
→ **決策**：SPEC §4.1「counts 的鍵 q0 在最左」**已由原始碼證實**；`get_state` little-endian 同樣由 `:135-148` 證實。兩者可寫進書中當「為什麼」而不只是「實測如此」。

---
## 4. `observe` / `spin`：**預設不是 ⟨ψ|H|ψ⟩**

【原始碼確認】最容易抄錯的一條。

`QppCircuitSimulator.cpp:311-321`：
```cpp
bool canHandleObserve() override {
  auto ctx = cudaq::getExecutionContext();
  if (ctx && ctx->shots != static_cast<std::size_t>(-1)) return false;  // 有指定 shots → 不用矩陣路
  return !shouldObserveFromSampling();                                   // 預設 true → 回 false
}
```
- `shouldObserveFromSampling(defaultConfig = true)`（`CircuitSimulator.h:1072-1074`）→ 讀 `CUDAQ_OBSERVE_FROM_SAMPLING`，**未設時回 true**（`Environment.cpp:25-30`）。註解自述：「Default is to enable observe from sampling」（`CircuitSimulator.h:1065-1068`）。
- Python `observe(..., shots_count=-1)`（`CQ/python/cudaq/runtime/observe.py:54`）在 `shots_count<=0` 時建 `ExecutionContext('observe', 0, qpu_id)`（`:165-168`）⇒ `shots == 0 ≠ (size_t)-1` ⇒ `canHandleObserve()` 回 **false**。

⇒ **預設走的是逐 term 路徑**（`CircuitSimulator.h:1211-1236`）：對 H 的每個 term 做 basis change（X→`h`、Y→`rx(±π/2)`：`:1599-1605`）→ `measureSpinOp`（`:1566-1635`）→ shots=0 ⇒ `sample(q, 0)` ⇒ `calculateExpectationValue` 的**精確 parity 加總**（`QppCircuitSimulator.cpp:151-178`），最後乘係數相加。
- **這條路是精確的（無抽樣雜訊）**，但不是矩陣收縮。
- 真正的 `⟨ψ|H|ψ⟩` 在 `QppCircuitSimulator.cpp:323-346`（`state.dot(qpp::apply(state, asEigen, targets, 2)).real()`），**只有在 `CUDAQ_OBSERVE_FROM_SAMPLING=0` 且沒給 shots 時才會被呼叫**。
- 若 `shots_count > 0`：走同一條 term 路徑但用 `qpp::sample` 抽樣 ⇒ **有 shot noise**。

→ **決策**：q01 的 `observe` 契約要分三段：(a) 預設 → 精確、逐 term、等價於 ⟨ψ|P|ψ⟩；(b) `shots_count>0` → 抽樣估計；(c) 提供環境變數或參數對齊 `CUDAQ_OBSERVE_FROM_SAMPLING`。SPEC §3.2 只寫 `.expectation()` 不夠——**「預設精確、指定 shots 才抽樣」必須寫進契約**，否則 q01 用抽樣實作會在訓練迴圈裡產生假梯度雜訊。

---
## 5. `gradients.ParameterShift`

| 項目 | 結論 | 證據 |
|---|---|---|
| 實作位置 | `CQ/runtime/cudaq/algorithms/gradients/parameter_shift.h`（Python 綁定在 `python/runtime/cudaq/algorithms/py_optimizer.cpp:105-121`，類名 `ParameterShift`） |
| 位移量 | **±π/2**（`shiftScalar = 0.5`，乘 `M_PI`） | `parameter_shift.h:17` `double shiftScalar = 0.5;`、`:30` `tmpX[i] += shiftScalar * M_PI;`、`:33` `tmpX[i] -= 2 * shiftScalar * M_PI;` |
| 公式 | `dx[i] = (f(x+π/2) − f(x−π/2)) / 2` | `:37`、`:58` |
| 多參數批次策略 | **純序列 for 迴圈，無批次、無平行**；用 `tmpX` 就地加減還原 | `:28-38`（`for i in x.size()`）、`:49-59` |
| 期望值怎麼來 | `getExpectedValue()` → `cudaq::observe(ansatz_functor, h, x)` | `CQ/runtime/cudaq/algorithms/gradient.h:47-49` |
| ⇒ 連鎖效應 | 因 §4 的結論，**ParameterShift 的每次評估也走「逐 term parity」路徑**，預設精確 | 【推論】由 `:48` + §4 合成 |

→ **決策**：(1) 位移常數寫死 π/2，並**保留 `shiftScalar` 可改**的欄位（CUDA-Q 有）。(2) 批次策略照抄「序列、就地」即可；(3) SPEC §5.4 要對帳 `max|Δg| ≤ 1e-12` —— 只要 q01 的 `observe` 預設也精確，這條可達成；若 q01 誤用抽樣，會直接失敗。

---
## 6. 0.16 → 未來：已經公告的破壞性變更

| 變更 | 狀態 | 證據 |
|---|---|---|
| **`sample` 不再支援「依量測結果分支」的 kernel**（`if mz(q):` / `r=mz(q); if r:`）→ 須改用 `run` / `run_async` | **0.14.0 起即生效**（不是未來式） | `docs/sphinx/using/examples/sample_vs_run.rst:9-12`；原始碼實際 raise：`python/cudaq/runtime/sample.py:86-100` |
| 在 `sample` 的 kernel 內把量測結果**賦值給具名變數** → 已標記 deprecated | deprecated（未移除） | `sample_vs_run.rst:330-332`；`CircuitSimulator.h:896-904` 的 runtime warning |
| **`sample` 與 `observe` 這兩個 algorithmic primitive 將在未來版本改變** | 已發 FutureWarning，但**內容尚未公布** | `python/cudaq/__init__.py:627-633`：`warnings.warn("The CUDA-Q sample and observe algorithmic primitives will change in a future release. ...", FutureWarning, stacklevel=2)`（測試鎖定此訊息：`python/tests/utils/test_lazy_imports.py:14,36-37`） |
| 公告細節頁 | **是空殼**：只有標題 + 「Details about the planned changes and migration guidance are coming soon.」 | `docs/sphinx/using/migration/upcoming_changes.rst`（全文 6 行） |
| `sample_result.expectation_z()` | 已 deprecated，改用 `expectation()` | `python/runtime/common/py_SampleResult.cpp:145-157` |
| `observe_result.counts/expectation` 傳 `SpinOperator`（而非 `SpinOperatorTerm`） | 已 deprecated | `python/runtime/common/py_ObserveResult.cpp:102-113`、`:141-153` |
| 舊式 `cudaq::optimizer` / `cudaq::gradient` | deprecated | `docs/sphinx/specification/cudaq/algorithmic_primitives.rst:618,689` |

→ **決策（直接影響 §3.2 的 API 契約）**：
1. **不要**現在就猜新介面——官方遷移頁是空殼（`upcoming_changes.rst` 6 行），猜錯的成本高於晚對齊。
2. **必須**現在就做的是：在契約裡把 `sample` 的適用範圍限定為「**無量測分支**」，並把 `run` 納入 API 子集（書中 54 區塊有 9 處 `mz`，其中若有分支就會踩到）。
3. **必須**現在就把 SPEC §4.5 的「線路中量測 → 每 shot 重跑軌跡」這條**改寫**：CUDA-Q 的 `sample` 已不接受該情境（見上表第 1 列）。
4. 在 `q01` 的 warning 策略上鏡射 CUDA-Q：發出同等 `FutureWarning` 但**不改變行為**，讓黃金向量與書中輸出維持可比。

---
## 7. 對 q01 SPEC 的具體修正建議（依重要性）

| # | SPEC 現況 | 原始碼事實 | 建議動作 |
|---|---|---|---|
| 1 | §4.4「不允許：動態決定 qubit 數」 | `ast_bridge.py:3832-3835` 明確允許（`AllocaOp(size=<Value>)`） | **改掉這條**；否則 q01 會拒絕 CUDA-Q 接受的程式，破壞 G1「同一份程式碼兩軌可跑」 |
| 2 | §4.5「`sample` + 量測分支 → 每 shot 重跑軌跡」 | 0.14 起 `sample` 對此 raise（`sample.py:94-100`） | 改成「`sample` 僅限無分支；分支走 `run`」；並把 `run` 加入 §3.2 必做清單 |
| 3 | §3.2 `observe` 只寫 `.expectation()` | 預設**精確逐 term parity**、`shots_count>0` 才抽樣（§4） | 契約補上「預設決定性」；並把 `CUDAQ_OBSERVE_FROM_SAMPLING` 的語意寫明 |
| 4 | §4.4 `strict=True` | CUDA-Q 的 `@kernel` 無此參數（`kernel_decorator.py:708,123-135`） | 改成模組層/環境變數開關，避免兩軌簽章不一致 |
| 5 | §4.4 白名單「for/range、if、索引」 | 完整清單是 ast_bridge 的 27 個 `visit_*`（§2.1） | 直接採用該清單（含 `while`、排除 `assert/dict/with/try/class/f-string/walrus`） |
| 6 | §5.2「逐位元相同」 | CUDA-Q 在模擬前有 MLIR 管線：`canonicalize` → `emul-jit-prep-pipeline`（含 `UnitarySynthesis`、`Canonicalizer`、`ApplySpecialization`、`AggressiveInlining`） | 見 §8；主張限縮為「對已在正規形式的原始閘（h/x/rx/ry/rz/cx）逐位元相同」 |
| 7 | §3.2 `sample` → `.counts`、`__getitem__` | 存在，但 `expectation_z()` 已 deprecated，且具名 register 支援已警示將移除 | 契約保留 `.counts/.probability()/__getitem__`；**不要**實作 `expectation_z()`（或實作但發同款 warning） |

---
## 8. 一個容易漏掉的風險：模擬前的 MLIR 管線

【原始碼確認】任何 target 的 pass pipeline 都以 `canonicalize` 開頭（`runtime/internal/compiler/JITTargetPipeline.cpp:38`），接著（模擬 target 走 emulate 分支，`:45-51`）掛上 `emul-jit-prep-pipeline`，其內容為（`cudaq/lib/Optimizer/Transforms/Pipelines.cpp:109-129, 152-160`）：
`InjectImplicitOutput` → `AddDeallocs` → `QuakeAddMetadata` → `PropagateMetadata` → `UnwindLowering` → **`Canonicalizer`** → `ClassicalMemToReg` → `ClassicalOptimizationPipeline` → `GlobalizeArrayValues` → **`Canonicalizer`** → **`UnitarySynthesis`** → **`Canonicalizer`** → `LoopInductionFusion` → `ApplySpecialization` → `AggressiveInlining`。

【原始碼確認】其中 `UnitarySynthesis` 的職責是「把 **custom operation**（使用者提供的具體么正矩陣）拆成原生閘序列」（`cudaq/include/cudaq/Optimizer/Transforms/Passes.td:2136-2148`）——**不是**對原生 `h/x/rx/ry/rz/cx` 做融合。
【推論】因此對「全部由原生閘組成」的電路，這條管線大體是 no-op ⇒ 這**解釋了**為何實測能達到 `max|ΔP|=1.11e-16`（僅浮點捨入差）。但 `canonicalize`/`ApplySpecialization` 仍可能刪除或重寫特定樣式。
【待實測】用 `--target qpp-cpu` 搭配 `CUDAQ_PRINT_...`/`cudaq.translate` dump 出 QBN 電路的 Quake IR，確認閘序列與 Python 原始碼**逐閘同序**。若不同，書中「逐位元相同」的敘述必須加註條件。

---
## 9. 無法從本 repo 證實、必須另找來源的清單

| 項目 | 為什麼不能證實 | 替代做法 |
|---|---|---|
| `qpp::apply` 內部是否用 tensordot / 是否有多執行緒閘融合 | `tpls/qpp` 空目錄（§0） | 克隆 softwareQinc/qpp；或純以實驗（已做） |
| `qpp::sample` 是否為 O(shots) 多項式抽樣 | 同上 | 效能曲線（SPEC §6.1 已有） |
| 0.16.0 release 是否已含本 checkout 的所有變更（尤其 §6 的 `sample` raise） | 此 checkout 是 main HEAD（2026-09-18），非 tag | **在參考機以 0.16.0 實測** `sample` 對 `if mz(q):` kernel 的行為 |
| OpenMP 執行緒數與綁定策略 | 執行緒在 Q++ 內 | 量測 scaling；或讀 Q++ 原始碼 |

---
## 10. 三行決策摘要

1. **能抄的是「介面與語意」，不能抄的是「數值核心」**——數值核心在 Q++（不在 repo），一致性只能靠實測守住。
2. **三個必須立刻改的契約**：動態 qubit 數（可）、`sample` 對量測分支（已禁止）、`observe` 預設精確（非抽樣）。
3. **不要追未公布的未來介面**（遷移頁是空殼），但要**現在就把 `run` 納入子集**，因為 `sample` 的縮限已經生效。
