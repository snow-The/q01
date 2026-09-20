# tket 原始碼研讀：我們要不要它、能借它什麼

- 對象：`C:\Users\snow\source\repos\QBN\.refs\tket`（Quantinuum tket + pytket，`TKET_VERSION = 2.1.99`，59 檔 C++ 標頭）。
- **所有 file:line 均相對於 `.refs/tket/`**（例：`tket/src/Gate/Rotation.cpp:385`）。
- 方法：**只讀原始碼**，未安裝、未建置、未執行任何量子 SDK。標記規則：
  **【原始碼確認】**＝我讀到那一行；**【推論】**＝我的推論；**【待實測】**＝需實跑才能定案。
- 前置修正：`packages/q01/SPEC.md` **在本機不存在**；本機的實體是 `.remote/SPEC-q01.md`（422 行）。
  以下「SPEC §x」全部指該檔。本筆記的決策依據是它的 §2.2、§3.2、§4.1、§5.1、§8。

---

## 1. 問題一：tket 的電路 IR

### 1.1 結構（【原始碼確認】）

| 元件 | 事實 | 出處 |
|---|---|---|
| 圖 | `boost::adjacency_list`，頂點＝運算、邊＝線，`bidirectionalS`＋vertex_index（為 topological sort） | `tket/include/tket/Circuit/DAGDefs.hpp:53-68` |
| 頂點屬性 | `VertexProperties{ Op_ptr op; optional<string> opgroup; }` | `DAGDefs.hpp:34-41` |
| 指令 | `Command{ Op_ptr op_ptr; unit_vector_t args; /*indexed by port numbering*/ optional<string> opgroup; Vertex vert; }` | `tket/include/tket/Circuit/Command.hpp:34-90`（成員在 :86-89） |
| 量子/古典參數 | 由 op 的 signature 逐埠過濾（`EdgeType::Quantum`→qubits，`Classical`→bits） | `Command.hpp:51-70` |
| Op 抽象基底 | `type_` + `desc_`；介面含 `dagger/transpose/symbol_substitution/get_params/n_qubits/commuting_basis/get_signature/is_clifford/is_identity/get_unitary` | `tket/include/tket/Ops/Op.hpp:54`、`:100`、`:113-133`、`:141`、`:149-185`、`:206-208` |
| 閘 | `class Gate : public Op`，多一個 `params_` 與 `n_qubits_` | `tket/include/tket/Gate/Gate.hpp:28`、`:86-87` |
| **typed ops** | `enum class OpType` 佔 `OpType.hpp:30-821`，**125 個成員**（對該行距做 regex 計數）；含 `TK1:294`、`PhasedX:512`、`CnRy:610`、`Conditional:744`、`Measure:480`、`Reset:490`、`Collapse:485`、`Barrier:79` | `tket/include/tket/OpType/OpType.hpp` |
| **相位追蹤** | Circuit 有獨立成員 `Expr phase`；`get_phase()` 回「π 的倍數，範圍 [0,2)」；`add_phase(Expr)` | `tket/include/tket/Circuit/Circuit.hpp:1644`、`:1580-1586`、`:1593` |
| **implicit qubit permutation** | 一級公民：`implicit_qubit_permutation()`、`has_implicit_wireswaps()`、`permute_boundary_output()`；**電路相等性把 ImplicitPermutation 列為比較項** | `Circuit.hpp:1163`、`:1168`、`:1174`、`:1186` |
| 置換的補償點 | 只在模擬最後一步做：`matr = apply_qubit_permutation(matr, circ.implicit_qubit_permutation())` | `tket/src/Circuit/Simulation/CircuitSimulator.cpp:56` |
| 條件運算 | `class Conditional : public Op` **包住**另一個 op（width/value 為 little-endian 條件值） | `tket/include/tket/Circuit/Conditional.hpp:28`、`:35-39` |
| 退化的受控閘 | `CnRy` 只有 1 個 arg 時，直接降級成 `Ry`（CnRx→Rx、CnRz→Rz、CnX→X…） | `Circuit.hpp:1770-1787` |

### 1.2 我們這邊的 IR 現況（【原始碼確認】）

- 我們的參考核心**目前沒有電路 IR**：`np_ref.py:51 StateVectorSim` 是立即執行（`apply_1q` 於 `:81`，受控版 `apply_controlled_1q` 於 `:93`，`cry` 於 `:148`）。
- `.remote/qbn_circuit.py` 直接就是 `@cudaq.kernel`（`:114`、`:135`、`:154`、`:175`、`:202`），沒有自己的 IR。
- `packages/q01/src/q01/circuit.py`（trace 記錄 + 電路 IR）**仍只是 SPEC §9 的規劃**（`.remote/SPEC-q01.md:357`），尚未存在。

### 1.3 對 q01 的決策（【推論】）

1. **相位要獨立成一個欄位**，不要混進閘序列。tket 把單閘分解的殘餘相位直接 `add_phase(tk1_angles[3])`（`tket/src/Transformations/Rebase.cpp:48`）——這是「分解等價」測試能過的關鍵。若 q01 的 trace IR 沒有 `global_phase: float`，`ry.ctrl` 的分解測試會差一個相位而永遠對不上。
2. **typed ops 不要抄 125 個**。q01 只需要 SPEC §3.2 的 ~15 個閘；但要有「閘＝(型別, 參數陣列)」的型別化表示，而不是純字串。
3. **implicit permutation 我們不需要**（沒有 SWAP 最佳化、沒有硬體耦合圖）。但需要它的弱化版：**輸出 qubit 順序**。tket 的做法提醒我們，順序必須是 IR 的一級屬性、且**只在一個地方補償**——這正是 SPEC §4.1 「bit-reversal 只在 `bitorder.py` 出現一次」的同一個設計。

---

## 2. 問題二：閘集與分解

### 2.1 典範形式：任何單 qubit 么正 → TK1 + 全域相位（【原始碼確認】）

- `Gate::get_tk1_angles()` 回傳 `Rz(a)Rx(b)Rz(c)` 的三個角**加上一個全域相位** | `tket/include/tket/Gate/Gate.hpp:51`（單位是矩陣乘法序，即電路序的反序）
- 反解演算法：從 2×2 矩陣取 `s,x,y,z` 四個複數，**取絕對值最大者抽相位**以降低數值不穩定，再對 `u≈0`／`v≈0` 兩種退化單獨特判 | `tket/src/Gate/Rotation.cpp:385-482`（抽相位 :413-420；退化特判 :447-466；一般解 :467-479）
- 正解（由角度生成矩陣）在 `Rotation.cpp:504-519`；`OpType::TK1` 的定義在 `OpType.hpp:290-294`。

**分解的架構**（這是最值得抄的一段）：分解＝一個替換函式 `tk1_replacement(α,β,γ) → Circuit`，殘餘相位另外加。
`rebase_op()` 的流程是：`get_tk1_angles()` → `tk1_replacement(a,b,c)` → `add_phase(tk1_angles[3])`
| 出處 `tket/src/Transformations/Rebase.cpp:34-51`

tket 內建十多種替換（同一組角、不同目標閘集）：
`tk1_to_tk1` / `tk1_to_rzrx` / `tk1_to_rxry` / `tk1_to_u3`（並 `add_phase(-(α+γ)/2)`）/ `tk1_to_PhasedXRz` / `tk1_to_PhasedX` / `tk1_to_rzh` / `tk1_to_rzsx` / `tk1_to_rzxsx`
| 出處 `tket/src/Circuit/CircPool.cpp:1384`、`:1452-1555`

### 2.2 受控閘：不是修飾子，是獨立型別（【原始碼確認】）

- `CnRy`/`CnRx`/`CnRz`/`CnX` 各自是 OpType，且**依控制數 arity 分派不同分解策略**：arity 1 → 退化成無控制；2 → `CRy_using_CX`；3–8 → gray-code `CnU_gray_code_decomp`；>8 → `CnSU2_linear_decomp` | `tket/src/Circuit/ControlledGates.cpp:752-802`（`CRy_using_CX` 定義於 `CircPool.cpp:811`）
- Gray-code 分解要**矩陣的 2^(n−1) 次方根**：`nth_root(u, 1ULL << (n-1))`，因此有分支選擇問題 | `ControlledGates.cpp:689-708`（:703）
- `CnRz` 以 `CnRy` 加 H/S 基底變換實作；`CnRx` 加 S/Sdg | `ControlledGates.cpp:804-842`

**【推論】** 這正是「我們該不該做 `ry.ctrl` 分解」的答案：**不該**。tket 之所以分解，是因為它的輸出要送硬體（只有 CX＋單閘）。q01 的輸出是 2^n 狀態向量，直接構造受控 2×2 矩陣（`np_ref.py:93-112` 的做法）既精確又便宜。tket 的受控分解對我們只有**測試對照物**的價值，沒有實作價值。

### 2.3 等價判準：本筆記最有價值的發現（【原始碼確認】）

tket 的分解等價測試用 `tket_sim` 自帶的判準，不用手寫：

- `enum class MatrixEquivalence { EQUAL, EQUAL_UP_TO_GLOBAL_PHASE }`；`compare_statevectors_or_unitaries(m1, m2, eq, tolerance = 1e-10)` | `tket/test/src/Simulation/ComparisonFunctions.hpp:25-45`
- **實作**：EQUAL → `m1.isApprox(m2, tol)`；UP_TO_GLOBAL_PHASE → 令 `P = U†V`，先驗 `| |P(0,0)| − 1 | < tol`，再驗 `P ≈ (P(0,0)/|P(0,0)|)·I` | `tket/test/src/Simulation/ComparisonFunctions.cpp:144-179`
- 附帶守門：兩矩陣大小須相同且為 2^n（`:110-122`）；必須真的是么正矩陣／歸一態向量，否則丟例外（`:128-142`）
- 檔案頭有完整的數值誤差論證（為何是 `U†V ≈ cI` 而不是逐元素比對） | `ComparisonFunctions.cpp:29-103`

**與 SPEC §5.1/§4.3 的落差（【推論】）**：SPEC 用的判準是 `1 − |⟨ψ|ψ'⟩| ≤ 1e-14`——**只驗一個輸入態**。若某個閘分解只在某個基底態上錯，而 `|0…0⟩` 上恰好對，這個判準會漏掉。tket 的 `U†V ≈ cI` 驗的是**整塊矩陣**。
**建議**：q01 的 L1（算子層）與閘分解單元測試改用 `U†V` 判準；電路層（L2／54 區塊）維持態向量判準。

**【待實測】** `U` 是 4^n 元素：n=5 → 1024（可忽略）、n=8 → 65536、n=10 → 1e6（邊緣）、n=12 → 1.7e7（不可行）。故 `U†V` 判準**只用在 n ≤ 8~10 的算子級測試**。

---

## 3. 問題三：最佳化 pass 與 5–20 qubit 純模擬的實際價值

### 3.1 與變分電路最相關的 pass（【原始碼確認】，出處為原始碼行）

| Pass | 做什麼 | 出處 |
|---|---|---|
| `FullPeepholeOptimise` | **9 段管線**：remove_redundancies >> synthesise_tket >> two_qubit_squash(false) >> clifford_simp >> synthesise_tket >> two_qubit_squash >> three_qubit_squash >> clifford_simp >> synthesise_tket | `tket/src/Transformations/OptimisationPass.cpp:44-52` |
| `SynthesiseTket` / `SynthesiseTK` | 全部化為 CX+TK1+Phase ／ TK2+TK1+Phase | `pytket/binders/passes.cpp:583-587` |
| `RemoveRedundancies` | 移除互逆閘對、合併旋轉、移除 identity 旋轉、移除量測前的冗餘 | `passes.cpp:576-581` |
| `SquashRzPhasedX` / `SquashTK1` / `SquashCustom` | 把單 qubit 序列壓成 PhasedX+Rz／TK1／自訂目標閘集（`tk1_replacement` 由呼叫者給） | `passes.cpp:588-609`、`tket/include/tket/Predicates/PassLibrary.hpp:58-65` |
| `GreedyPauliSimp` | Pauli gadget＋相位摺疊＋貪婪重合成 | `passes.cpp:984-1018` |
| `NormaliseTK2` / `EulerAngleReduction` / `RoundAngles` | TK2 角正規化到 Weyl chamber ／ Euler 角化簡 ／ 角度取整 | `PassLibrary.hpp:102-118`、`PassGenerators.cpp:296` |
| `DelayMeasures` / `RemoveImplicitQubitPermutation` / `RemoveBarriers` | 量測延後、置換轉回 SWAP、移除屏障 | `PassLibrary.hpp:79`、`:144`、`:86` |

### 3.2 兩個必須知道的反例（【原始碼確認】）

1. **`GreedyPauliSimp` 不保持全域相位**：官方 docstring 明寫
   `"WARNING: this pass will not preserve the global phase of the circuit."` | `pytket/binders/passes.cpp:992-993`
   → 對 q01 的 §4.3 相位契約，這是紅旗：**不能拿它當等價性判準的生產者**，只能當「電路變短」的示範。
2. **`ZXGraphlikeOptimisation` 可能讓電路變貴**：「As a resynthesis pass, this will ignore almost all optimisations achieved beforehand and **may increase the cost of the circuit**.」 | `tket/include/tket/Predicates/PassLibrary.hpp:153-154`
   → 「最佳化 pass 一定變快」是錯的直覺。

### 3.3 對 5–20 qubit 純模擬：會變快嗎？（【原始碼確認】＋【推論】）

**【原始碼確認】決定性事實**：tket 的模擬器**不做 gate fusion**。
`GateNodesBuffer::Impl::push()` 整支只有一行 `node.apply_full_unitary(matrix, number_of_qubits);`，前面是註解
`// Later, we might add fancy optimisation here: storing the gate for later use, looking for other compatible gates acting on the same qubits to merge with this, etc. etc.`
| `tket/src/Circuit/Simulation/GateNodesBuffer.cpp:47-52`
（全 Simulation 目錄中 `merge/fuse` 只出現在這類註解：`GateNodesBuffer.hpp:44`、`GateNodesBuffer.cpp:50`；唯一實際路徑是 `GateNode::apply_full_unitary`，`GateNode.hpp:37`、`GateNode.cpp:263`。）
另：模擬器**預設上限 11 qubit**，超過丟 `TOO_MANY_QUBITS` | `tket/include/tket/Circuit/Simulation/CircuitSimulator.hpp:39-40`、`:53-55`、`CircuitSimulator.cpp:39-43`

**【推論】** 因此：pass 的成本模型是「模擬時間 ∝ 閘數 × 每個閘作用的 2^n 成本」。pass 只減少閘數 → **線性**收益，不改變量級。
- 對 5 qubit QBN（~20 閘、SPEC §6.1 穩態 0.593 ms）：最佳化的收益落在量測誤差內，**幾乎沒有價值**。
- 對 n ≥ 18（SPEC §6.3：n=18 203 ms、n=20 857 ms）：閘數減半才有意義，但真正的槓桿是 **gate fusion**（tket 未實作）與 **JIT**（SPEC §8 numba 那條）。
- **真正需要 pass 的場景是參數掃描**：如果可以「編譯一次、參數換 N 次」，最佳化才划算。而這需要**符號參數**——tket 用 SymEngine `Expr` 支援（`Op.hpp:76-77 symbol_substitution`、`Gate.cpp:696 get_tk1_angles` 對符號參數仍成立），我們的 float 參數做不到。**【推論】這才是「最佳化」在 QML 裡的本業，而不是「把電路變漂亮」。**

---

## 4. 問題四：q01 v1 需要 tket 嗎？

### 4.1 結論：**不需要**（傾向獲得驗證）

| 理由 | 證據 |
|---|---|
| 我們沒有 transpile 需求 | q01 輸出是狀態向量，沒有耦合圖、沒有閘集限制（SPEC §2.2 已列為非目標：「不做 transpiler／電路最佳化」） |
| pass 對 5–20 qubit 純模擬不改變量級 | 見 §3.3；且 tket 自己沒有 gate fusion（`GateNodesBuffer.cpp:47-52`） |
| 建置成本高 | `tket/CMakeLists.txt:20-28`：Boost、Eigen3、nlohmann_json、SymEngine＋5 個 in-tree 庫，C++20（`:30`）；pytket 建置還需 conan+cmake＋自家 conan remote（`.github/workflows/release.yml:161-173`） |
| 我們只用到它 <5% 的表面 | `OpType` 125 個成員、pass 有 60+ 個；SPEC §3.2 的需求是 15 個閘、6 個 API |

### 4.2 一個必須說清楚的反例（【原始碼確認】）

**「tket 在 Windows 上不能用」是錯的。** release 工作流**有** Windows wheel：
- `build_Windows_wheels`（runs-on `windows-2025-vs2026`，matrix python `3.10/3.11/3.12`） | `.github/workflows/release.yml:146-183`（matrix 在 `:151`）
- `test_Windows_wheels` | `release.yml:334-356`
- `publish_to_pypi` 把全部平台 wheel 發到 `https://pypi.org/p/pytket` | `release.yml:371-391`

→ 拒絕 tket 的理由是**範圍**（我們不需要 transpiler），**不是**「Windows 裝不了」。
**【待實測】** release matrix 只列到 CPython 3.12；QBN 專案用 Python 3.13（AGENTS.md「環境」節）。PyPI 上是否有 cp313 win_amd64 wheel，需查證後才能寫進書裡。

### 4.3 將來「電路最佳化對照」章節中，tket 的角色

| 角色 | 具體做法 | 需要裝 tket？ |
|---|---|---|
| **① 等價判準的 oracle（最推薦）** | 把 `ComparisonFunctions.cpp:144-179` 的 `U†V ≈ cI` 判準重寫成 ~15 行 NumPy，當 q01 的 **L1 閘分解測試**標準。這是免費的、不引入依賴 | ❌ |
| ② 術語與管線的對照 | 用 `FullPeepholeOptimise` 的 9 段（`OptimisationPass.cpp:44-52`）當「最佳化」一章的骨架，示範 remove_redundancies/squash/rebase/peephole 四種動作 | 只在寫作時 |
| ③ 實測對照（可選） | 若 §4.2 的 wheel 可用：同一條 5 qubit QBN 電路 → tket 最佳化 → 比閘數與執行時間。若不可用，數字在參考機（WSL/Linux）產生，與論文同一策略 | ✅（選配） |
| ④ 反面教材 | §3.2 兩個反例（GreedyPauliSimp 不保相位、ZX 可能變貴）寫成「最佳化不是免費的」 | ❌ |

---

## 5. 借／不借清單（決策用）

| 項目 | 決定 | 理由 |
|---|---|---|
| DAG + Command IR | **不借** | 我們的 trace IR 只需線性閘序列＋參數；DAG 是為可變換性付的成本 |
| `global_phase` 為 IR 一級欄位 | **借** | `Circuit.hpp:1644`、`Rebase.cpp:48`；否則分解測試永遠差相位 |
| 「典範單閘形式 + 替換函式」的分解架構 | **借** | `Rebase.cpp:34-51`；q01 若要提供 `ry.ctrl` 的分解示範，這是正確骨架 |
| `U†V ≈ cI` 等價判準（tol 1e-10） | **借（強烈）** | `ComparisonFunctions.cpp:144-179`；比 SPEC §4.3 的單態判準嚴格 |
| `tk1_angles_from_unitary` 的數值防護 | **借（概念）** | `Rotation.cpp:413-420` 的「取最大振幅抽相位」與退化特判 |
| 受控閘 gray-code / nth_root 分解 | **不借** | `ControlledGates.cpp:689-802`；我們直接構造受控矩陣更精確（`np_ref.py:93-112`） |
| OpType 125 個型別 | **不借** | 超出 SPEC §3.2 的子集 |
| implicit qubit permutation | **不借（借用其單點補償原則）** | `CircuitSimulator.cpp:56` ↔ SPEC §4.1 `bitorder.py` |
| 最佳化 pass | **不借** | 見 §3.3 |

## 6. 待實測清單

| # | 項目 | 影響 |
|---|---|---|
| T1 | PyPI `pytket` 是否有 cp313 win_amd64 wheel | 決定 §4.3 角色③是否可行、書中能否寫「Windows 直接 pip install」 |
| T2 | `U†V` 判準在 n=8／n=10 的實測耗時 | 決定 L1 測試的 qubit 上限（推估 n≤10） |
| T3 | `ry.ctrl` 的受控矩陣 vs tket `CRy_using_CX` 分解：用 §2.3 的判準比對 | 若通過，可作為書中「同一算子的兩種寫法」示範 |
| T4 | tket 對 5 qubit QBN 電路的閘數變化 | 只有要寫「最佳化對照」章節才需要 |

## 7. 可回溯指針

- tket 電路 IR：`tket/include/tket/Circuit/Circuit.hpp`、`Command.hpp`、`DAGDefs.hpp`、`tket/include/tket/Ops/Op.hpp`
- 分解：`tket/src/Transformations/Rebase.cpp`、`tket/src/Gate/Rotation.cpp`、`tket/src/Circuit/CircPool.cpp`、`tket/src/Circuit/ControlledGates.cpp`
- 等價判準：`tket/test/src/Simulation/ComparisonFunctions.{hpp,cpp}`
- pass：`tket/src/Predicates/PassGenerators.cpp`、`tket/include/tket/Predicates/PassLibrary.hpp`、`pytket/binders/passes.cpp`
- 模擬器：`tket/src/Circuit/Simulation/{CircuitSimulator,GateNodesBuffer,GateNode}.{hpp,cpp}`
- 封裝／平台：`.github/workflows/release.yml`、`tket/CMakeLists.txt`
