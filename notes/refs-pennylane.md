# PennyLane 原始碼考察：高效 CPU 模擬該抄什麼

- 考察對象：`C:\Users\snow\source\repos\QBN\.refs\pennylane`
- **下文所有檔案路徑皆相對於 `.refs/pennylane/`**（例：`pennylane/devices/qubit/apply_operation.py:29`）
- 標記：【原始碼確認】= 我讀到那一行；【推論】= 由原始碼導出、未實測；【待實測】= 需跑數字
- 版本：`pennylane/_version.py:19` → `0.46.0-dev103`

## 0. 先講硬事實：這個 checkout 沒有 lightning 的 C++ 核心

| 事實 | 證據 |
|---|---|
| 這是 **PennyLane 核心 repo**，根目錄只有 `pennylane/`、`tests/`、`doc/`，**沒有 `pennylane_lightning/`** | 目錄列舉（`Test-Path .refs/pennylane/pennylane_lightning` → False） |
| 全 repo grep `Kokkos` 或 `OpenMP` 只有 17 命中，**全是文件字串**，無實作 | `pennylane/devices/default_qubit.py:479`、`pennylane/devices/preprocess.py:207`、`pennylane/concurrency/executors/external/dask.py:59` |
| lightning 的 C++ 在**另一個 repo** | `doc/releases/changelog-0.18.0.md:8` 指向 `PennyLaneAI/pennylane-lightning` |
| 「lightning 較快」目前只有官方文字，本 repo 無可查證的實作 | `doc/releases/changelog-0.18.0.md:48-52` |

> **結論**：Q1 問的「Kokkos？OpenMP？gate fusion 觸發條件」**無法用本 repo 回答**。
> 要答必須再 clone `PennyLaneAI/pennylane-lightning`。以下 Q1 只寫「Python 側能確認的部分」。

## 1. lightning.qubit 架構（可確認的部分）

| 問題 | 答案 | 證據 |
|---|---|---|
| 多執行緒在**哪一層**切分？ | **電路（tape）層**，不是單一 statevector 內部。單一 5 qubit 電路永遠單執行緒 | `pennylane/devices/default_qubit.py:793-808`（max_workers=None 時逐電路序列）／`:810-824`（才用 `executor.map(_simulate_wrapper, vanilla_circuits, …)` 平行多個 QuantumScript） |
| 梯度計算呢？ | 同樣是**多電路平行**，不是單電路內平行 | `pennylane/devices/default_qubit.py:845-846`、`:1062-1069` |
| shots 平行？ | 沒有 shot 層平行；shots 是在**同一個 numpy 呼叫**內一次抽完（見 §5） | `pennylane/devices/qubit/sampling.py:527` |
| 有沒有防過度訂閱？ | 有，明確警告 OMP×process 不可超過核心數 | `pennylane/devices/preprocess.py:205-228`、`pennylane/devices/default_tensor.py:268-272` |
| lightning 支援哪些微分？ | `adjoint` 與 `parameter-shift` | `doc/news/program_capture_sharp_bits.rst:82-85` |

**Gate fusion：不在模擬器內，是電路層 transform**【原始碼確認】

| 項目 | 內容 | 證據 |
|---|---|---|
| 位置 | `qp.transforms.single_qubit_fusion`，**非** default.qubit / lightning 的內部步驟 | `pennylane/transforms/optimization/single_qubit_fusion.py:27-30`；全 repo grep `fusion` 24 命中，無一在 `devices/qubit/` |
| 觸發條件 | 單 qubit 閘，且「下一個碰到同一條線的閘」也是單 qubit | `:253-263`、`:293-298` |
| 怎麼合 | 用 `qp.single_qubit_zyz_angles` 轉 ZYZ 角，再 `fuse_rot_angles` 逐顆併 | `:254-255`、`:294-299` |
| 何時停 | `find_next_gate` 遇 abstract 或非單 qubit 即停 | `pennylane/transforms/optimization/optimization_utils.py:36-43` |
| 消去 | 融合後 `abs(RZ 角和)` 與 `abs(RY 角)` 都 < `atol`（預設 1e-8）就整顆丟掉 | `:308-319`，預設值 `:29` |
| ⚠️ 訓練時失效 | `requires_grad` 或 abstract 時**跳過**快速路徑 `_try_no_fuse` | `optimization_utils.py:112-119`，快速路徑 `:46-68` |
| ⚠️ 梯度風險 | 官方自標 "not differentiable everywhere"，角度奇異點梯度 **NaN**；建議只用 64-bit | `:77-80`、`optimization_utils.py:96-99` |

## 2. `default.qubit`：閘作用怎麼寫（三層策略，量化門檻）

| 層 | 機制 | 證據 |
|---|---|---|
| L1 dispatch | `apply_operation` 是 `functools.singledispatch`，per-operator-type kernel | `pennylane/devices/qubit/apply_operation.py:258-259` |
| L2 通用 | einsum ↔ tensordot 二選一，門檻為**常數**：`EINSUM_OP_WIRECOUNT_PERF_THRESHOLD = 3`、`EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = 13` | `:29-30`、判斷式 `:341-351` |
| L3 特化 | 完全繞過矩陣乘法，只做記憶體搬移 / 逐元素乘法 | 見下表 |

**門檻的實際意義**（state 的 `ndim == qubit 數`，因 state 形狀是 `[2]*n`）：條件是 `len(op.wires) < 3 and ndim(state) < 13`
→ **n ≤ 12 用 einsum；n ≥ 13 用 tensordot**。`:341-351`

| 閘 | 實作 | 證據 |
|---|---|---|
| X | `math.roll(state, 1, axis)`（純搬移） | `:521-524` |
| Z | 半片乘 −1 後 `stack` | `:527-544` |
| S / T | 半片乘常數相位（`1j`／`exp(0.25j·π)`） | `:574-590`、`:593-609` |
| H（numpy 介面） | dtype-keyed 快取矩陣 + `_apply_single_qubit_np`（`np.tensordot` + `moveaxis`） | `:624-638`、`:34-41`、`:73-88` |
| RX/RY/RZ | 共用 `_apply_rotation_1q`；**numpy 且 n<13 時不建矩陣**，直接 slice+4 乘法+`np.stack`；n≥13 才建 2×2 走 tensordot | `:645-723`，關鍵分支 `:699-711`；係數 `:726-741` |
| CNOT | 對 control=1 的半片 `math.roll(state[sl_1], 1, target_axis)` 再 `stack`（**不做 4×4 tensordot**） | `:762-778` |
| MultiControlledX | `len(op.wires) < 9` 走通用路徑；否則 matrix-free（先 roll control=0 的軸 → transpose 成 `(-1, 2, 2^{k-1})` → 只 roll 最後一格 → 還原） | `:781-832` |
| 只有 sparse 矩陣的算子 | `full_state @ sparse.T`（CSR） | `:327-338`，觸發 `:344-345` |

**einsum 路徑細節**：以 `string.ascii_letters` 造下標，受影響下標用 `op.wires` 直接當索引，輸出下標做 `replace`；矩陣 reshape 成 `[2]*(2k)`，batched 時最前面加 batch 維。`:176-199`

**成本結構**【推論】
1. `simulate` / `get_final_state` / `measure_final_state` 都掛 `@debug_logger`，但未開 logging 時只多一次 `lgr.isEnabledFor`——**裝飾器不是瓶頸**。`pennylane/logging/decorators.py:50-61`、`:81`
2. 真正的固定成本是「**每閘一次 Python 呼叫 + singledispatch 解析 + 一次 numpy 呼叫**」；n≤12 時 numpy 呼叫開銷 > 算術開銷。
3. 沒有 gate fusion 把連續同線單閘併成一次搬移。
4. 【推論】default.qubit 與 lightning 的差距主要來自 (1)(2)(3) 的 Python 化與 (lightning 的) 編譯後 C++ 迴圈，不是數學不同。

## 3. adjoint differentiation

**演算法**：一次前向取得 |ψ⟩（ket），之後反向掃描電路，每步用 `qp.adjoint(op)` 更新 ket 與 bras。`pennylane/devices/qubit/adjoint_jacobian.py:110-149`；數學推導 `pennylane/devices/qubit/adjoint_jacobian.md:3-92`

**適用條件（全部【原始碼確認】）**

| 限制 | 證據 |
|---|---|
| 觀測量必須有矩陣 | `pennylane/devices/default_qubit.py:295-297`（`obs.has_matrix`） |
| 算子須 `num_params == 0`，或無可訓練參數，或（`num_params == 1` 且 `has_generator`） | `:286-292` |
| 不可 `Conditional` / `MidMeasure` | `:288` |
| 不可對觀測量、不可對 state-prep 微分 | `adjoint_jacobian.py:84-92`（note） |
| 量測須「全 expval」或「全無 observable」；混合直接 raise | `default_qubit.py:256-262` |
| 必須 analytic（`no_sampling` transform 擋掉有限 shots） | `:336` |
| 可行性檢查＝跑一遍 compile pipeline，失敗回 False | `:300-312` |
| （legacy 路徑）不支援參數廣播 | `pennylane/devices/_qubit_device.py:1641` |

**導數從哪來**：`operation_derivative` 用 generator 矩陣 `G` 直接算 `∂U/∂x = 1j · G · U`
→ `pennylane/operation.py:40-61`。RX 的 `G = −0.5·X`（`pennylane/ops/qubit/parametric_ops_single_qubit.py:111-112`）、RZ 的 `G = −0.5·Z`（`:543-544`）。

**複雜度**：一次前向 + 一次反向；每個可訓練參數多做一次「`∂U` 作用 + `n_obs` 次內積」，**不是** 2P 次電路重跑。全 jacobian 形狀 `(n_obs, n_params)`。`adjoint_jacobian.py:117-137`；JVP 版 `:153-223`

**參數平移（對照）**

| 項目 | 內容 | 證據 |
|---|---|---|
| recipe 優先序 | 自訂 `grad_recipe` → `parameter_frequencies` → **generator 特徵值** | `pennylane/gradients/parameter_shift.py:250-292` |
| 頻率怎麼來 | 由 generator 特徵值取「正且唯一的兩兩差集」 | `gradients/general_shift_rules.py:92-109`；dispatch `parameter_shift.py:1311-1331` |
| 位移表 | `generate_shift_rule(frequencies, shifts, order)` → `(c_i, s_i)` 表 | `general_shift_rules.py:250-329` |
| RZ 標準式（可直接抄） | `d/dφ f = 1/2[f(φ+π/2) − f(φ−π/2)]` | `ops/qubit/parametric_ops_single_qubit.py:514` |
| 成本 | 每個參數 2 次電路（單頻率） | `:322-323` + 上列 |
| 方法優先序 | `device` > `backprop` > `parameter-shift` | `pennylane/workflow/get_best_diff_method.py:33-44` |

**對 q01 的決策**【推論】QBN 是 5 qubit + `expval(Z⊗Z)` → adjoint 條件全部滿足；但 adjoint 的優勢前提是「同電路、多參數、analytical」。SPEC §5.4 要求 v1 用參數平移對帳 CUDA-Q（1e-12），所以：
**v1 只做參數平移；adjoint 列為 v2 選配，且必須先量到交叉點再決定。**【待實測】

## 4. 可微分整合（autograd / torch）

| 機制 | 內容 | 證據 |
|---|---|---|
| **backprop 為何免費** | 整條 state 更新走 `pennylane.math.*`，那是 **autoray** 包裝的框架無關分派（numpy/autograd/torch/jax），「end-to-end differentiation is preserved」 | `pennylane/math/__init__.py:14-35`；`math/multi_dispatch.py:411`（tensordot）、`:541`（einsum） |
| 非 backprop 才需自訂層（torch） | `class ExecuteTapes(torch.autograd.Function)`：forward 只呼叫 `execute_fn`；backward 轉共軛約定後 `jpc.compute_vjp` | `pennylane/workflow/interfaces/torch.py:118-197`（關鍵 `:161`、`:186-189`） |
| torch 的樹狀結果限制 | `pytreeify` 把 forward 攤平、在 backward 還原 | `workflow/interfaces/torch.py:77-109` |
| 非 backprop 才需自訂層（autograd） | `@autograd.extend.primitive` + `autograd.extend.defvjp` | `workflow/interfaces/autograd.py:18-39`、`:162-168` |
| autograd 的已知坑 | autograd 會**每個輸出行各叫一次** grad_fn；因此內部快取整個 jacobian（其他介面不需要） | `workflow/interfaces/autograd.py:55-81` |
| backprop 支援條件 | `gradient_method in {"backprop","best"}` 且 `max_workers is None` 且**無 shots** 且觀測量非 `SparseHamiltonian` | `pennylane/devices/default_qubit.py:595-604` |

**對 q01 §5.4 的決策**【推論】
- torch 路線：把 sim 寫成**純 torch 張量運算**（`torch.tensordot` / `torch.stack` / `torch.roll`）即可免費拿到 autograd。**不要**模仿 PL 去寫 `torch.autograd.Function`——那層存在是為了「後端不可微卻要梯度」。
- 只有當 q01 要「同一份 NumPy 後端 + 硬接 torch 梯度」時才需要那層，而 SPEC §2.2 已排除靜默降級。
- 【待實測】torch `complex128` autograd 梯度 vs 參數平移，判準 `max|Δg| ≤ 1e-9`（SPEC §5.4）。

## 5. shots / 統計

| 項目 | 內容 | 證據 |
|---|---|---|
| analytic ↔ shots 切換點 | `if not circuit.shots:` → 解析 `measure`；否則 `rng = default_rng(rng)` + `measure_with_samples` | `pennylane/devices/qubit/simulate.py:275-284`／`:286-296` |
| 抽樣策略 | **先算精確機率再一次抽完 shots**，非逐 shot 重跑 | `sampling.py:439-476`（`flatten_state` → `qp.probs`）→ `:479-497` |
| **有 `np.random.Generator` 路徑** | `rng = np.random.default_rng(rng)`；`rng.choice(basis_states, shots, p=probs)` | `sampling.py:513`、`:527` |
| 位元展開 | `1 << np.arange(num_wires)[::-1]` → **wire 0 是最高位**（q0 在最左） | `sampling.py:529-531` |
| 正規化檢查 | 容差 1e-6，超過 raise `"probabilities do not sum to 1"` | `sampling.py:32-33`、`:514-519` |
| MCM（線路中量測）native | **每 shot 重跑整個電路**：`aux_circ = circuit.copy(shots=[1])`，`for i in range(total_shots): simulate_one_shot_native_mcm(...)` | `simulate.py:361-382` |
| MCM + JAX | 改用 `jax.vmap` 展開同一件事 | `simulate.py:364-374` |
| tree-traversal（另一條路） | 只在 `default.qubit` / `lightning.qubit` 有 | `simulate.py:356-359`；`workflow/qnode.py:289-294` |
| 種子管理 | 裝置層 `np.random.default_rng(seed)`，可 `reset_prng_key` | `default_qubit.py:562-570`、`:537-539` |
| shots 分桶 / 量測分組 | `shots.bins()` 逐區間處理；Pauli 量測分組避免重複對角化 | `sampling.py:328-333`、`:46-53` |

**與 SPEC §4.5 對帳**
- 「尾端 `mz` 用精確機率做多項式抽樣（O(shots)）」→ **與 PL 完全一致**（`sampling.py:527`）。
- 「線路中量測每 shot 重跑軌跡」→ **與 PL native MCM 一致**（`simulate.py:376-381`）。
- ⚠️ `sampling.py:529-531` 的 big-endian **是 PL 的約定，不是 CUDA-Q 的**；只能當「另一套獨立實作也這樣排」的旁證，**不能**當 SPEC §4.1 的證據。

## 6. 值得抄 3 件事

| # | 抄什麼 | 為什麼 | 位置 |
|---|---|---|---|
| 1 | **單閘特化 kernel + 量化門檻常數**：X 用 `roll`、Z/T/S 用半片乘常數、RX/RY/RZ 在 n<13 時不建矩陣、CNOT 用 `roll` 半片 | q01 的教學區間（n≤12）正是 PL 選 einsum 的區間；這些 kernel 是 O(2^n) 純搬移，避開矩陣乘法與 einsum 字串 parse 成本 | `apply_operation.py:29-30`、`:341-351`、`:521-544`、`:645-723`、`:762-778` |
| 2 | **`np.random.default_rng(seed)` + `rng.choice(..., p=probs)` 一次抽完** | 直接滿足 SPEC §4.5 的 O(shots)，且天然支援 §4.6 的 `set_random_seed` 完全可重現；無需自己寫 alias method | `sampling.py:500-531` |
| 3 | **解析導數用 generator**：`∂U/∂x = 1j·G·U`，位移規則由 `G` 的特徵值差集自動推導 | q01 要用參數平移對帳 CUDA-Q 到 1e-12；用 generator 可一次覆蓋 RX/RY/RZ 與 `.ctrl()` 變體，不必手寫每顆閘的 recipe。RZ 的 `1/2[f(φ+π/2)−f(φ−π/2)]` 可直接當單元測試 | `operation.py:40-61`；`general_shift_rules.py:92-109`、`:250-329`；`parametric_ops_single_qubit.py:514` |

> 補充：`@debug_logger` 的「未開啟時只多一次 `isEnabledFor`」也值得抄（`logging/decorators.py:50-61`）——是可安心留在熱路徑的觀測點。

## 7. 不值得抄 3 件事

| # | 不抄什麼 | 為什麼 | 位置 |
|---|---|---|---|
| 1 | **不要在訓練路徑上做單閘融合** | 官方自標「不可微處處、奇異點梯度 NaN」，且 `requires_grad` 時會跳過快速路徑 → 訓練時更貴又污染梯度。q01 若要做，只能掛在推論／取樣路徑 | `single_qubit_fusion.py:77-80`；`optimization_utils.py:96-99`、`:112-119` |
| 2 | **不要抄 `torch.autograd.Function` / `defvjp` 那層** | 那些是為「後端本身不可微卻要梯度」而存在。q01 的 torch 模式若整條 sim 走 torch 張量，autograd 是免費的；多寫一層只增加與 NumPy 路徑分歧的風險 | `workflow/interfaces/torch.py:25-26`、`:118-197`；`workflow/interfaces/autograd.py:18-53` |
| 3 | **不要把 `singledispatch` + `math.*`（autoray）抽象當 v1 主體** | （i）n≤12 時多一層分派是純開銷，而 q01 已在 0.6 ms 級、SPEC §6.4 只要 ≤5 ms，抽象買不到效能；（ii）抽象化會讓「與 CUDA-Q 逐元素相同（`max|dψ| = 0`）」的舉證變難。**【推論】** | `apply_operation.py:258-259`；`math/__init__.py:35` |

## 8. 待實測 / 缺口

| # | 項目 | 怎麼補 |
|---|---|---|
| 1 | lightning 的 Kokkos/OpenMP kernel、它自己的 gate fusion、thread 切分 | 需另 clone `PennyLaneAI/pennylane-lightning`（本 repo 無此目錄） |
| 2 | `EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = 13` 在**本機 numpy** 是否仍是最佳交叉點 | 用本機 numpy 複刻 einsum vs tensordot 於 n=5/8/10/12/14 對打 |
| 3 | torch `complex128` autograd 梯度精度 | SPEC §5.4 判準 1：`max|Δg| ≤ 1e-9` |
| 4 | adjoint vs 參數平移在 5–12 qubit QBN 的 wall-time 交叉點 | 需先有 q01 torch/NumPy 兩條後端 |

---

### 附：對 q01 的直接決策清單

| 決策 | 結論 | 依據 |
|---|---|---|
| 小 n 閘作用用 einsum 還是 tensordot？ | **einsum（n≤12）**，並把門檻做成常數 | §2（`apply_operation.py:29-30`） |
| 要不要 gate fusion？ | v1 **不要**；若要，只掛在取樣／推論路徑 | §7-#1 |
| 單閘要不要特化？ | **要**：X 用 roll、Z/T/S 用半面乘法、CNOT 用 roll 半面 | §6-#1 |
| 梯度走哪條？ | v1 **參數平移**（用 generator 推導 recipe）；torch 模式走純 torch 張量則免費 autograd；adjoint 列 v2 | §3、§4 |
| 抽樣怎麼寫？ | `np.random.default_rng(seed).choice(2**n, shots, p=probs)` | §6-#2 |
| 線路中量測？ | 每 shot 重跑一次軌跡 | §5（SPEC §4.5 已定，PL 做法相同） |
| 位元順序？ | PL 的 big-endian **不是**我們的 oracle；一律以 CUDA-Q（little-endian）為準，轉換只留在 `bitorder.py` | §5 末 |
