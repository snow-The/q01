# Qiskit / Qiskit-Aer 原始碼閱讀筆記（q01 的第三對照物）

- 讀的是**原始碼**，不是官方文件。路徑相對於 `.refs/qiskit/` 與 `.refs/qiskit-aer/`。版本：qiskit **2.6.0.dev0**（`qiskit/VERSION.txt:1`）、qiskit-aer **0.17.2**（`qiskit_aer/VERSION.txt:1`）。
- ⚠️ SPEC §1.3 實測的是 **qiskit 2.5.2 wheel**；本 checkout 是 main 分支開發版。閘矩陣與位元公約相同、細節 API 可能有差。
- 標記：【原始碼確認】＝有 file:line；【推論】＝由原始碼推得但未逐行驗證；【待實測】＝必須跑才敢寫。

## 0. 五個結論

| # | 結論 | 對 q01 的決策 |
|---|---|---|
| 1 | Qiskit 的**向量／矩陣索引與 CUDA-Q 完全相同**（q0＝最低位）；只有 **counts 的字串是反的** | L1 算子對帳**零置換**；`bitorder.py` 只要一個字串反轉函式 |
| 2 | 標準閘矩陣在 Qiskit ≥1.0 已搬到 **Rust** | 閘矩陣的單一權威是 `crates/circuit/src/gate_matrix.rs`，不是 Python |
| 3 | `Operator(qc)` / `Statevector(qc)` 兩行取么正矩陣／態向量 | L1 用這兩個就夠；`equiv()` 已內建除全域相位比較 |
| 4 | **Qiskit 2.6 沒有 `qiskit.gradients`**，ParameterShift 在另一包 `qiskit-algorithms` | 不要為對帳引入這包；公式兩行，自己實作 |
| 5 | Aer 的 statevector **在 n≤13 是單執行緒**（門檻 14）、gate fusion 也要 n≥14 | 教學規模（5–12 qubit）不必模仿 Aer 的 fusion／threading |

## 1. 位元順序（L0，最高風險項）

### 1.1 Qiskit 的 numpy 索引：q0＝最低位（LSB0）【原始碼確認】

- `quantum_info/operators/operator.py:530-532`（`Operator.compose` 把閘放上張量的地方）：`# Note that we must reverse the subsystem dimension order as / # qubit 0 corresponds to the right-most position in the tensor / # product, which is the last tensor wire index.`；`:535` `indices = [num_indices - 1 - qubit for qubit in qargs]`。
- `quantum_info/operators/operator.py:297`：`for qubit, char in enumerate(reversed(label))` → `Operator.from_label('IX')` 的 `X` 作用在 **q0**。
- `quantum_info/states/utils.py:159-161`：明文「In Qiskit, qubits are ordered using **little-endian** notation, with the least significant qubits having smaller indices… a four-qubit system is represented as $|q_3q_2q_1q_0\rangle$」。
- `quantum_info/states/statevector.py:775`：`pos = int(z_label, 2)` → `Statevector.from_label('01')` 的振幅在索引 **1**；`:166` 註解 `# Apply LSb0 qubit ordering (least-significant-bit is 0)`。
- `providers/basic_provider/basic_simulator.py:262`：`axis.remove(self._number_of_qubits - 1 - qubit)`（第三處獨立證據）。

### 1.2 但「字串」是反的【原始碼確認】

- `quantum_info/states/quantum_state.py:349-362` `_index_to_ket_array`：先 `kets[k] = (ind // 2**k) % 2`（k＝qubit k），再以 `np.char.add(row, str_kets)` 逐列**前插** → 最終字串 ＝ `bit_{n-1} … bit_1 bit_0`，**左邊是 q_{n-1}、右邊是 q_0**。
- `primitives/statevector_sampler.py:238-239`：註解「samples of `Statevector.sample_memory` will be in the order of **qubit_last, …, qubit_1, qubit_0**」。
- `result/utils.py:261-263`：`# Since bitstrings have qubit-0 as least significant bit`，`:268` `key[-idx - 1]`。
- 測試釘死：`test/python/quantum_info/states/test_statevector.py:766-771`：`Statevector.from_label("+0").probabilities_dict() == {"00": 0.5, "10": 0.5}`（`+0` ＝ q1 在 |+⟩）→ 變動的是**左邊**字元＝q1。

### 1.3 對齊表

| 對象 | 索引／字串約定 | 對 q01 的動作 |
|---|---|---|
| `Statevector.data`、`Operator.data`、`probabilities()` | q0 = LSB，長度 2^n | **與 `cudaq.get_state` 完全同序 → 直接逐元素比、零置換** |
| `probabilities_dict()` 鍵、`sample_counts`、`Counts`、`get_counts()` | 左＝q_{n-1}，右＝q_0 | 與 CUDA-Q `sample().counts`（q0 在最左）**恰好相反 → 字串要反轉** |

→ **決策**：SPEC §4.1「bit-reversal 只出現一次」的範圍可縮到最小——`bitorder.py` 只需要 `qiskit_key = "".join(reversed(q01_key))`，**矩陣與向量層完全不需要置換**。

## 2. 閘矩陣定義

### 2.1 權威位置【原始碼確認】

- `circuit/gate.py:49-61`：`Gate.to_matrix()` 回傳 `self.__array__(dtype=complex)`；沒有 `__array__` 就丟 `CircuitError` → 有 Python `__array__` 的參數化閘（RX/RY/RZ/…）走 Python；**Singleton 閘（X,Y,Z,H,S,T,SX,CX,CZ,SWAP,CCX）走 Rust**。
- Rust 總表 `crates/circuit/src/gate_matrix.rs`；dispatch `crates/circuit/src/standard_gate/mod.rs:319` `StandardGate::matrix()`。
- 控制閘巨集 `gate_matrix.rs:31-51` `make_n_controlled_gate!`：**前 n 個 qubit 是控制、最後一個是目標**，矩陣塞在索引 `DIM/2-1` 與 `DIM-1`（控制位＝低位）。

### 2.2 矩陣與行號

| 閘 | 定義 | 位置 |
|---|---|---|
| H / X / Y / Z | $\frac{1}{\sqrt2}[[1,1],[1,-1]]$ / $[[0,1],[1,0]]$ / $[[0,-i],[i,0]]$ / $diag(1,-1)$ | `gate_matrix.rs:53-56` / `:58` / `:62` / `:60` |
| **S / Sdg** | $diag(1,i)$ / $diag(1,-i)$ | `gate_matrix.rs:64` / `:66`；docstring `standard_gates/s.py:44-47`、`:162-165` |
| **T / Tdg** | $diag(1,e^{+i\pi/4})$ / $diag(1,e^{-i\pi/4})$ | `gate_matrix.rs:78` / `:80`；docstring `t.py:41-44`、`:117-120` |
| **SX / SXdg** | $\frac12[[1+i,1-i],[1-i,1+i]]$ / 共軛版 | `gate_matrix.rs:68-71` / `:73-76`；docstring `sx.py:38-41` |
| **RX(θ)** | $[[\cos\frac\theta2,-i\sin\frac\theta2],[-i\sin\frac\theta2,\cos\frac\theta2]]$ | `standard_gates/rx.py:140-142`＝ `gate_matrix.rs:296-301` |
| **RY(θ)** | $[[\cos\frac\theta2,-\sin\frac\theta2],[\sin\frac\theta2,\cos\frac\theta2]]$ | `ry.py:140-142`＝ `gate_matrix.rs:304-309` |
| **RZ(θ)** | $diag(e^{-i\theta/2},e^{+i\theta/2})$ | `rz.py:154-155`（`ilam2 = 0.5j*θ`）＝ `gate_matrix.rs:312-315` |
| P(λ) | $diag(1,e^{i\lambda})$ | `gate_matrix.rs:278-280` |
| **CX / CZ / CH / CCX / CCZ** | `make_n_controlled_gate!(X/Z/H, 1)`、`!(X/Z, 2)` | `gate_matrix.rs:87` / `:91` / `:85` / `:147` / `:149` |
| **SWAP**、RXX/RYY/RZZ/RZX、U/U1/U3 | $[[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]$、二量子位旋轉、通用閘 | `gate_matrix.rs:127-132`、`:419-472`、`:318-330` |

### 2.3 逐元素比對時的相位陷阱

- 【原始碼確認】`RX/RY/RZ` 矩陣**不含額外全域相位**（就是 $e^{-i\theta P/2}$）→ **可直接逐元素比**。
- 【原始碼確認】**`RZ \ne P`**：`RZ(θ)=diag(e^{-iθ/2},e^{+iθ/2})` 對 `P(θ)=diag(1,e^{iθ})` 差 $e^{iθ/2}$。混用會看到「處處差同一相位」的假失敗。
- 【原始碼確認】**`SX` 與 $\sqrt{X}$ 只差全域相位**：`sx.py:51-62` 明載「A global phase difference exists between the definitions of RX(π/2) and √X」；手算 $\sqrt X = e^{i\pi/4}RX(\pi/2)$ ✓（`SXdg` 同理，`sx.py:164-175`）。
- 【推論】Qiskit 只在走 `obj.definition` 路徑時補 `definition.global_phase`（`operator.py:835-841`）；標準閘都有 `to_matrix`，走不到那條 → S/T/SX 不含多餘相位。

## 3. `Operator` / `Statevector`：電路 → 么正矩陣／態向量

### 3.1 最小程式碼（L1 算子對帳）

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)
U   = Operator(qc).data        # (4,4) complex128，q0 = LSB
psi = Statevector(qc).data     # (4,)  complex128，|00> 起算
```

### 3.2 原始碼對應【原始碼確認】

| 需求 | 位置 | 備註 |
|---|---|---|
| 電路 → Operator | `operator.py:116-124`（`QuantumCircuit/Operation` → `_init_instruction`）、`:780-786`（`Operator(eye(2^n))` 後逐指令 compose） | 含 `measure`/`reset`/條件閘會丟 `QiskitError`（`:119-123` 註解明講） |
| 電路 → Statevector | `statevector.py:99-100` → `from_instruction`；`:839-843`（由 `|0…0>` 出發，`init[0]=1`） | — |
| **除全域相位比較** | `Operator.equiv` `operator.py:688-710`、`Statevector.equiv` `statevector.py:480-510` → 都走 `matrix_equal(..., ignore_phase=True)` | 只回 bool；q01 要誤差數字，自己算 $1-|\langle\psi|\psi'\rangle|$ |
| 單閘張量積／子系統重排 | `Operator.from_label('IXYZ')` `operator.py:245-300`；`reverse_qargs()` `:712-732` | L1 逐閘檢查最省事；重排本專案不需要（同序） |

### 3.3 決策

- **用 `Operator(qc)`，不要用 `Operator.from_circuit(qc)`**：後者會讀 `circuit.layout` 做置換（`operator.py:381-460`，`ignore_set_layout` 預設 False）。q01 的電路沒有 layout，用前者才不會被靜默轉座標。
- 【待實測】qiskit 2.5.2 與 2.6.0.dev0 在 `Operator(circuit)` 上的行為差異。

## 4. 參數平移（ParameterShift）

### 4.1 【原始碼確認】Qiskit 2.6.0.dev0 **沒有** `qiskit.gradients`

- 對整個 `qiskit/**/*.py` grep `ParameterShift` → **0 命中**。殘存的 `gradient` 只在 `synthesis/unitary/aqc/fast_gradient/`（AQC 專用，非參數平移）。
- 真正實作在**另一個套件** `qiskit-community/qiskit-algorithms`，**本機 `.refs/` 未 clone**。以下行號來自該 repo main 分支原始碼（非本機檔案）：

### 4.2 公式【原始碼確認（外部）】

- 位移量：`qiskit_algorithms/gradients/utils.py` `_make_param_shift_parameter_values`：`plus_offsets = parameter_values + offset * np.pi / 2`、`minus_offsets = parameter_values - offset * np.pi / 2`（`offset`＝對參數索引的 one-hot）。
- 差分：`qiskit_algorithms/gradients/param_shift/param_shift_estimator_gradient.py`：`gradient_ = (evs[: n // 2] - evs[n // 2 :]) / 2`。
- → 合起來 $\;\partial_i f = \dfrac{f(x+\frac\pi2 e_i) - f(x-\frac\pi2 e_i)}{2}$（同一檔 `SUPPORTED_GATES`：`x,y,z,h,rx,ry,rz,p,cx,cy,cz,ryy,rxx,rzz,rzx`）。

### 4.3 決策

- **不要**為對帳引入 `qiskit-algorithms`：多一包依賴，且要接 `Estimator` primitive（等於還要 Qiskit 的電路物件）。公式只有兩行 → 直接寫進 q01 `gradients.py`；L1 的 oracle 用**解析梯度**比第三方套件更強更省。
- 【待實測】CUDA-Q `gradients.ParameterShift` 是否同為 π/2 與 1/2 因數。SPEC §5.4 要求 `max|Δg| ≤ 1e-12`；位移量若不同，會是「剛好差 2 倍」的假失敗。

## 5. Aer 的模擬器策略與可調參數

### 5.1 方法 → 適用規模【原始碼確認】`qiskit_aer/backends/aer_simulator.py`

| method | 適用 | 成本／門檻 | 行號 |
|---|---|---|---|
| `statevector` | 理想電路、量測全在尾端；有雜訊時每 shot 抽一條噪聲電路 | $2^n\times16$ B | `:92-95` |
| `matrix_product_state` / `extended_stabilizer` | 低糾纏大 n（可截斷 bond dim）／Clifford+T（**近似法**） | bond dim 可調／`approximation_error` 預設 0.05 | `:109-113`、`:105-107`、`:384-387` |
| `density_matrix` / `stabilizer` | 混態雜訊／純 Clifford | $4^n\times16$ B／上限 2048 qubit | `:97-99`、`:101-103` |
| `unitary` / `superop` | **算子級**（`unitary` 正是 L1 要的；不支援 measure/reset/雜訊） | $4^n$／$16^n$ | `:115-124` |
| `tensor_network` | GPU only（cuTensorNet） | — | `:126-128` |

### 5.2 statevector 可調參數表（q01 效能調校的直接對照）

| 參數 | 預設 | 作用 | 行號 |
|---|---|---|---|
| `method` / `precision` | `"automatic"` / `"double"` | 選模擬器／精度 | `:181`、`:188-190` |
| **`max_parallel_threads`** | `0`（＝核心數） | OpenMP 執行緒上限 | `:220-222` |
| `max_parallel_experiments` / `max_parallel_shots` | `1`（**＝關閉**）／`0`（＝auto） | 電路層／shot 層並行；**兩者不能同時開** | `:224-236` |
| **`statevector_parallel_threshold`** | **`14`** | n > 此值才開 OpenMP | `:331-337` |
| **`statevector_sample_measure_opt`** | **`10`** | n > 此值才用「大 qubit 最佳化取樣」 | `:339-342` |
| **`fusion_enable` / `fusion_max_qubit` / `fusion_threshold`** | `True` / `None`→**5** / `None`→**14** | gate fusion 及其門檻 | `:468`、`:472-494` |
| `batched_shots_gpu` / `_max_qubits` | `False` / `16` | 多 shot 批次（**僅 GPU**） | `:272-287` |
| `enable_truncation` / `max_memory_mb` / `zero_threshold` | `True` / `0` / `1e-10` | 移除無效 qubit／記憶體上限（2^n×16 B）／小值截斷 | `:210-215`、`:238-243` |
| `shot_branching_enable` | `False` | 動態電路多 shot 分支 | `:294-304` |

### 5.3 對 q01 有決策價值的發現

1. **【原始碼確認】n ≤ 13 時 Aer 的 statevector 是單執行緒**：Python 端門檻 14（`:331-337`）對應 C++ 端 `src/simulators/statevector/qubitvector.hpp:481-482` `omp_threads_ = 1; omp_threshold_ = 14;` 與 `:486-489` `omp_threads_managed() = (num_qubits_ > omp_threshold_ && omp_threads_ > 1) ? omp_threads_ : 1`。
   → **教學範圍（5–12 qubit）我們與 Aer 是同一種競爭（單執行緒、記憶體頻寬受限）**，Aer 不會因執行緒而贏。與 SPEC §6.2「n=12 已 1.90× 快於 CUDA-Q」一致。
2. **【原始碼確認】gate fusion 在教學規模根本不啟動**（`fusion_threshold` 預設 14）→ q01 **不需要**實作閘融合；這也側面支持 SPEC §8「Numexpr 只在 n ≥ 18 有用」。
3. **【原始碼確認】Aer 取樣的最佳化＝「一次建機率向量 + 每 shot O(1)」**：`statevector_sample_measure_opt=10` 對應 `qubitvector.hpp:483` `sample_measure_index_size_ = 10` —— 正是 SPEC §4.5 對 q01 `sampler.py` 的契約（多項式抽樣 O(shots)，非逐 shot 重跑）→ **方向與 Aer 相同**。
4. 【原始碼確認】Aer 的並行＝**單一 Python 行程 + 底層 OpenMP/CUDA**：`docs/howtos/parallel.rst:6-10`「runs simulation jobs on a single-worker Python multiprocessing ThreadPool executor so that all parallelization is handled by low-level OpenMP and CUDA code」；要自訂電路層平行才用 `executor` + `max_job_size`（`:12-24`）→ q01 的平行點（SPEC §8）應是 shots 與參數掃描。

## 6. Qiskit 2.x 模組結構與最小後端映射（P3）

- 【原始碼確認】頂層套件：`circuit, compiler, converters, dagcircuit, passmanager, primitives, providers, qasm2, qasm3, qpy, quantum_info, result, synthesis, transpiler, utils, visualization, capi`。`qiskit/__init__.py:152-167` 根命名空間只 re-export `QuantumCircuit / QuantumRegister / ClassicalRegister / AncillaRegister / QiskitError / transpile / generate_preset_pass_manager / __version__`（`__all__` `:193-200`）。
- 【原始碼確認】標準閘矩陣與部分 transpiler pass 在 **Rust**（`crates/`）；`qiskit/_accelerate` 是 PyO3 單一共享庫，子模組在 `qiskit/__init__.py:48-145` 手動註冊。
- 【原始碼確認】**最小後端＝兩個類別**：`BackendV2` 必須實作 `target`（abstract property，`providers/backend.py:148-149`）、`max_circuits`（`:173-174`）、`_default_options`（classmethod，`:182-183`）、`run`（`:308-309`）；`JobV1` 只需 `submit/result/status/backend`，純同步可照抄 `providers/basic_provider/basic_provider_job.py:22-66`（`_async=False`、`status()` 回 `JobStatus.DONE`）。
- 【原始碼確認】**現成完整範本**：`providers/basic_provider/basic_simulator.py`（純 Python＋numpy 狀態向量，上限 24 qubit — `:70-82`）：閘作用用 `np.einsum`（`:232-248`）、`Target(num_qubits=None)` 告訴 transpiler「不要 resize」（`:143-150`）、basis gates 清單 `:151-205` 幾乎就是 q01 的閘集。
- 【原始碼確認】**更省事的一條路**：`primitives/statevector_sampler.py:51` `StatevectorSampler` 與 `primitives/statevector_estimator.py:31` `StatevectorEstimator` 都是**純 Python、只依賴 `quantum_info`**（`statevector_sampler.py:26`、`:187`）→ **不需要 Aer、不需要後端**。
- 【原始碼確認】**限制**：`StatevectorSampler` 不支援 control-flow 與**線路中量測**（`statevector_sampler.py:213-216` 丟 `QiskitError`）→ 書中「線路中量測」的教學塊不能用 Qiskit primitives 當對照物，只能用 `Operator`/`Statevector` 的**無量測前綴**。

### 決策（按成本排序）

1. **P3 首選（零後端）**：q01 只要能吐出一個 `QuantumCircuit`，就直接用官方 `Operator` / `Statevector` / `StatevectorSampler` / `StatevectorEstimator` 當 oracle。不寫後端程式碼，同時拿到官方實作當第三方證據（正是 SPEC §7.3 要的「三套實作」）。
2. **P3 進階（可選）**：寫 `BackendV2` + `JobV1`（照抄 BasicSimulator，約 150 行）→ 解鎖 `backend.run(qc).result().get_counts()`。
3. **不做**：`ProviderV1` + 完整 `Target` + transpile 支援 —— SPEC §2.2 已把 transpiler 列為非目標。

## 7. 未解／待實測（不可寫進論文）

- 【待實測】qiskit **2.5.2**（SPEC §1.3 實測版本）與本 checkout 2.6.0.dev0 的差異（本機無 2.5.2 原始碼）。
- 【待實測】Aer `get_counts()` 鍵是否真的照 LSB0（本機無 wheel、不得執行；僅由 `result/utils.py:261-268` 推得）。
- 【待實測】CUDA-Q `gradients.ParameterShift` 的位移量（需在參考機 WSL 讀 cuda-quantum 原始碼）。
- 【缺口】`qiskit-algorithms` 本機 `.refs/` **未 clone**；§4.2 行號來自 GitHub 官方 repo main 分支，非本機檔案。

## Recoverable

- 原始碼根：`.refs/qiskit/`（2.6.0.dev0）、`.refs/qiskit-aer/`（0.17.2）
- 閘矩陣權威：`.refs/qiskit/crates/circuit/src/gate_matrix.rs`（全表 1-516）；dispatch `crates/circuit/src/standard_gate/mod.rs:319`
- 位元順序三證據：`qiskit/quantum_info/operators/operator.py:530-535`、`qiskit/quantum_info/states/utils.py:159-165`、`qiskit/primitives/statevector_sampler.py:238-239`
- Aer 選項全表：`qiskit_aer/backends/aer_simulator.py:177-496`；C++ 執行緒門檻 `src/simulators/statevector/qubitvector.hpp:481-489`
- 最小後端範本：`qiskit/providers/basic_provider/basic_simulator.py`、`basic_provider_job.py`
- 未 clone 的外部依賴：`github.com/qiskit-community/qiskit-algorithms`（`qiskit_algorithms/gradients/utils.py`、`.../param_shift/param_shift_estimator_gradient.py`）
