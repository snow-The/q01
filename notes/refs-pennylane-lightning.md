# PennyLane-**Lightning** C++ 核心考察（補主 repo 筆記缺的部分）

- 考察對象：`C:\Users\snow\source\repos\QBN\.refs\pennylane-lightning`（main @ `ce584b9`，版本 `pennylane_lightning/core/_version.py:19` → `0.46.0-dev27`）
- **下文路徑一律相對於 `.refs/pennylane-lightning/`**；引用主 repo 時寫 `../pennylane/`
- 標記：【確認】= 讀到原始碼該行；【推論】= 由原始碼導出未實測；【待測】= 需跑數字
- 讀者請先讀 `refs-pennylane.md`（那只答了 Python 側；本文專補 C++ 核心）

## 1. 架構（Q1）

### 1.1 類別層級與 dispatch
- `StateVectorBase<PrecisionT,Derived>` → CRTP `StateVectorLQubit<PrecisionT,Derived>` → `StateVectorLQubitManaged`（擁有資料）／`StateVectorLQubitRaw`（外部指標）。【確認】`core/simulators/lightning_qubit/StateVectorLQubit.hpp:57-58`
- 資料 = 單一 `std::vector<std::complex<T>, AlignedAllocator<T>>`，`data_[0]={1,0}`。【確認】`StateVectorLQubitManaged.hpp:65,75-82`
- 建構時填 **8 張** `unordered_map<Op,KernelType>` kernel 表（gate/generator/matrix/sparse × 有無控制）。【確認】`StateVectorLQubit.hpp:69-91,101-128`
- dispatch key = `(threading << 8) | memory_model`；**每顆閘作用 = 一次 `unordered_map::at()` 查表 + dispatcher 呼叫，無任何快取**（該檔 grep `cache` 0 命中）。【確認】`utils/Threading.hpp:51-56`、`StateVectorLQubit.hpp:359-390`

### 1.2 kernel 家族與特化軸
| 家族 | 內容 | 證據 |
|---|---|---|
| LM（預設基準） | 純量；**不預建 index 表**，用 bitwise 即時算 | `gates/cpu_kernels/GateImplementationsLM.hpp:49-52` |
| AVX2 / AVX512 | 同一組 `Apply*` 模板 × `packed_size`（register 內 double 數） | `gates/KernelType.hpp:27`、`avx_common/AVXUtil.hpp:214-218` |
| 生效區間 | **AVX 只在 `num_qubits >= 4` 被指派**（常數名叫 `leq_four`，實為 `larger_than_equal_to(4)`，[4,∞)） | `AssignKernelMap_AVX2.cpp:38,43-44`、`AssignKernelMap_AVX512.cpp:39,44-45`、`utils/IntegerInterval.hpp:62-67` |
| 優先序 | LM 覆蓋 full_domain 且 priority 0；AVX 用 priority 1 蓋掉 | `AssignKernelMap_Default.cpp:39,46-47`、`gates/KernelMap.hpp:311-364` |
- 【推論】q01 唯一需要的特化軸就是這一條：n<4 純量、n≥4 才值得談向量化。

### 1.3 單閘／雙閘 kernel 策略
- 單閘無控制熱路徑：`rev_wire = n-w-1` → `revWireParity` → 迴圈 `2^(n-1)` 次，λ 就地改兩個振幅。【確認】`GateImplementationsLM.hpp:689-702`
- **特化閘完全不建矩陣**：X=`std::swap`（`:713-716`）、Y=手寫 re/im 交換（`:769-775`）、H=±isqrt2 加減（`:866-872`）、CNOT=`swap(arr[i10],arr[i11])`（`:741-746`）。
- 通用路徑 `applyNCSingleQubitOp`/`applyNCTwoQubitOp`：**每次呼叫都把 matrix 複製進一個 `std::vector`**，再手寫展開 4/16 項乘加（無 BLAS）。【確認】`:286,294-303`、`:345,353-371`
- k≥3 閘：naive 三重迴圈 + dim 大小的 `coeffs` scratch。【確認】`:471-487`
- AVX 側再依閘落點二分／四分：單閘 `applyInternal`(register 內)／`applyExternal`(跨 register)；雙閘 internal-internal／internal-external／external-internal／external-external。`avx_common/README.md:5-7`、`SingleQubitGateHelper.hpp:164-181`（`2^n < packed_size/2` 時退回純量）

### 1.4 std::complex 與 AoS vs SoA ★
- **是 `std::complex`、是 AoS（交錯 `[re0,im0,re1,im1,…]`），不是 SoA**。【確認】`StateVectorLQubitManaged.hpp:65`、`StateVectorLQubit.hpp:60`
- SIMD **不去交錯**，而是用 **register 內 lane permutation 假裝去交錯**：`imagFactor` = `{-1,+1,-1,+1,…}` 符號遮罩（`avx_common/AVXUtil.hpp:171-174`），`swapRealImag`/`flip` 置換（`avx_common/ApplySingleQubitOp.hpp:76-96`）。
- 【推論】這是 lightning 效能的真正來源，但建立在「AoS + 對齊 + packed_size 為偶數」上；PyTorch/NumPy 沒有這個自由度。

### 1.5 Kokkos 是「另一條類別線」，不是同模板變體
- `StateVectorKokkos` 直接繼承 `StateVectorBase`（不經 `StateVectorLQubit`），資料 `Kokkos::View<ComplexT*>`，`ComplexT = Kokkos::complex<fp_t>`（**不是 std::complex**）。【確認】`lightning_kokkos/StateVectorKokkos.hpp:66,73,78`
- `applyMultiQubitOp`：先 `switch(wires.size())` 1/2/3/4 走專門 functor，`default` 才用 `TeamPolicy(two2N, AUTO, dim)` + per-team scratch。【確認】`:427-467`
- **「L7／L3」在整個 repo grep 0 命中**（`L7`、`L3` 皆無）。最接近的是 Kokkos 的 **L0/L1 scratch level** TODO：【確認】`StateVectorKokkos.hpp:449,575`、`gates/BasicGateFunctors.hpp:83`、`measurements/MeasurementsKokkos.hpp:186`。【推論】問題中的 L7/L3 應是此處的記憶偏差；真實特化軸 = §1.2 的 LM/AVX2/AVX512 × §1.3 的 internal/external。

## 2. 多執行緒（Q2）

### 2.1 三個編譯開關
| 開關 | 預設 | 作用 | 證據 |
|---|---|---|---|
| `ENABLE_OPENMP` | ON（Linux/macOS） | 尋找並連結 OpenMP；**Windows 被硬編 OFF** | `CMakeLists.txt:52-60`、`setup.py:135-136` |
| `LQ_ENABLE_KERNEL_OMP` | **OFF** | 決定 gate kernel 的 `PL_LOOP_PARALLEL` 是否變成真 pragma | `core/simulators/lightning_qubit/CMakeLists.txt:23,51-57`、`gates/cpu_kernels/GatePragmas.hpp:25-34` |
| `LQ_ENABLE_KERNEL_AVX_STREAMING` | OFF | 非時序 store | `core/simulators/lightning_qubit/CMakeLists.txt:24,59-67` |
- 原始碼預設 OFF，但**發佈的 Linux/macOS wheel 有開**（官方文件明說；Windows 未列）。【確認】`doc/lightning_qubit/development/kernel_tuning.rst:35-37,69-75`
- **Windows 是唯一被硬編關掉的平台**：`setup.py:135-136` → `["-DENABLE_OPENMP=OFF","-DENABLE_BLAS=OFF"]`；且 AVX2/AVX512 只在 x86-64 UNIX 編譯（`kernel_tuning.rst:61-64`）。
- ⇒【確認】原生 Windows 的 lightning.qubit = **單執行緒 + 無 BLAS + 無 AVX 特化**。這對 q01 是最有用的一條。

### 2.2 OpenMP 實際用在哪一層
| 位置 | 平行維度 | 觸發條件 | 證據 |
|---|---|---|---|
| adjoint `applyObservables` | **觀測量** | `num_observables > 1` 才進平行區；**==1 走序列 else** | `algorithms/AdjointJacobianLQubit.hpp:113-155` |
| adjoint `applyOperationsAdj` | 觀測量(states) | 恆有 parallel，但 `num_states==1` 時退化 | `:176-205` |
| adjoint jac 內層 | 觀測量 | 只存在於多觀測量版本 | `:467-477` |
| Hamiltonian `applyInPlace` | Hamiltonian 的**項** | `terms.size() > 1` | `observables/ObservablesLQubit.hpp:200-251` |
| `probs()`／`probs(wires)` | 2^n／2^k 個振幅 | 需 `PL_LQ_KERNEL_OMP` | `measurements/MeasurementsLQubit.hpp:95-100,153-161` |
| gate kernel 主迴圈 | gate 的 block 迴圈 | 需 `PL_LQ_KERNEL_OMP` | `GateImplementationsLM.hpp:435,614,684,695`（共 14 處） |
- **shots 完全沒有平行**：先算一次精確 probs，再用 alias method 一個序列迴圈抽 num_samples。【確認】`MeasurementsLQubit.hpp:640-679`、`MeasurementKernels.hpp:304-345`
- **多電路也沒有平行**：Python `execute` 是 `for circuit in circuits:` 序列。【確認】`lightning_qubit/lightning_qubit.py:377-391`
- `Threading::MultiThread` **全 repo 僅 3 處引用**（enum、`bestThreading()`、一個測試），而 `bestThreading()` **零呼叫者** ⇒ 生產路徑永遠 `SingleThread`。【確認】`utils/Threading.hpp:38-43,63-73`、`StateVectorLQubitManaged.hpp:76`

### 2.3 門檻常數（決定「小 n 值不值得平行」）
| 常數 | 值 | 語意 | 證據 |
|---|---|---|---|
| `STD_CROSSOVER`（innerProdC） | `1<<18 = 262144` | 內積 < 此值走 `std::inner_product` 序列；≥ 才 OpenMP | `utils/LinearAlgebra.hpp:122-143` |
| `STD_CROSSOVER`（omp_scaleAndAdd） | `1<<12 = 4096` | `y+=a*x` 的序列／平行切點 | `utils/LinearAlgebra.hpp:319-336` |
| `nthreads`（omp_innerProdC） | `data_size / (1<<17)` | 由資料量自算，**覆蓋 OMP_NUM_THREADS** | `utils/LinearAlgebra.hpp:96-104` |
- 5 qubit = 32 個振幅，兩個門檻都差 7～13 個數量級 ⇒ **即使 kernel-OMP 開啟也一行都不平行**。【推論，由常數直接判定】
- 單閘 kernel 平行迴圈上界 = `2^(n-1)`（n=5→16）；多閘 = `2^(n-nw_tot)`（2 閘→8）。【確認】`GateImplementationsLM.hpp:436,685-688,696`。【推論】16 次迭代的 `omp parallel for` 是純開銷。

### 2.4 `num_threads` 的唯一來源
- **只有 `OMP_NUM_THREADS` 環境變數**（OpenMP runtime 預設 = 實體核數）。CPU 後端**沒有任何** `omp_set_num_threads`（GPU 後端才在 `lightning_gpu/algorithms/AdjointJacobianGPU.hpp:161` 硬設 1）。【確認】全 repo grep
- Python 側唯一讀取處：`lightning_qubit/_adjoint_jacobian.py:93`，且**只在 `batch_obs=True` 且 measurement 數 > 1** 時用來切 linear combination（`:87-95`、`lightning_qubit.py:128`）。
- 官方文件同調。`doc/lightning_qubit/device.rst:113-142`、`kernel_tuning.rst:77-79`
- Kokkos 後端不同：`kokkos_args=InitializationSettings().set_num_threads(N)`，預設 0=自動。`lightning_kokkos/bindings/LKokkosBindings.hpp:240-257`

**Q2 結論（你的預期正確，且比預期更強）**：【確認+推論】對「5 qubit、單一電路、單一 observable」，lightning 的 OpenMP **在任何配置下都不會啟動平行**：(1) Windows 編譯根本沒有 OpenMP；(2) 即使 Linux wheel 有，gate kernel 從未被指派到 MultiThread dispatch key，adjoint 兩個 `parallel for` 在 `num_observables==1` 時都退化；(3) 內積／scaleAdd 門檻在 2^18／2^12。唯一真能平行的情境是「多 observable × `batch_obs=True` × `OMP_NUM_THREADS`」。

## 3. gate fusion / matmul（Q3）

### 3.1 沒有任何門融合
- **全 repo（含 doc/tests）grep `fusion|fuse|Fusion|Fuse` = 0 命中**。【確認】
- Python 側只跳過 identity（`continue`），不做代數化簡。`lightning_qubit/_state_vector.py:238-240`
- 【推論】lightning 的速度來自「C++ 小迴圈 + 零配置」，不是化簡 ⇒ 對 q01，**融合不是效能門檻，v1 不必做**（與 `refs-pennylane.md` §7-#1 一致）。

### 3.2 `applyMatrix` 的路徑與門檻
| 層 | 行為 | 證據 |
|---|---|---|
| Python | 先 `getattr(state, name)` 找專用方法，找不到才走 matrix | `lightning_qubit/_state_vector.py:248-249,283-296` |
| dispatcher | 只看 `wires.size()`：1→SingleQubitOp、2→TwoQubitOp、其餘→MultiQubitOp | `gates/DynamicDispatcher.hpp:765-780` |
| LM 1q/2q | 手寫展開 2×2／4×4；**matrix 每次呼叫都複製一份** | `GateImplementationsLM.hpp:286-313,345-381` |
| LM k-qubit | naive 三重迴圈 + dim 大小 scratch | `:471-489` |
| sparse | CSR（row_map/col_idx/values）走獨立 kernel | `:516-571`、`DynamicDispatcher.hpp:864-877` |
| AVX | internal/external 二分；`2^n < packed_size/2` 退回純量 | `SingleQubitGateHelper.hpp:164-181` |
- **沒有「applyMatrix 門檻常數」**：換路只由 `wires.size()` 與 `num_qubits`（AVX 需 ≥4）決定，沒有「矩陣大於 X 就換路」。【確認】`DynamicDispatcher.hpp:765-774`、`AssignKernelMap_AVX2.cpp:38`
- 真正量化的門檻只有 §2.3 兩個 `STD_CROSSOVER` 與 `packed_size/2`。【確認】

### 3.3 記憶體佈局考量
- 單一 flat vector；alignment 由 `CPUMemoryModel` 決定（Aligned512=64B／Aligned256=32B／Unaligned=`alignof(T)`）。`core/utils/CPUMemoryModel.hpp:41-47,98-108`
- `bestCPUMemoryModel()` 只在 **x86-64 且 Linux/macOS** 且有 AVX512F／AVX2+FMA 時回對齊模型，**其餘（含 Windows）一律 Unaligned**。`CPUMemoryModel.hpp:76-91`
- Kokkos 側是 `Kokkos::View` + scratch view。`StateVectorKokkos.hpp:78-101`

## 4. adjoint / backprop（Q4）

### 4.1 演算法
- C++ 實作，跟隨 arXiv:2009.02823；一次前向建 λ，之後反向逐 op 同步更新 λ 與每個 observable 一個 H_λ。【確認】`algorithms/AdjointJacobianLQubit.hpp:44,261-264,269-314,378-389`
- 導數：`jac = -2 * scaling * imag(⟨H_λ|λ⟩)`（`:74-77,90-93`）。`scaling` 由 `applyGenerator` 回傳，而 **generator 不是矩陣，是「作用一顆 Pauli + 回傳純量」**：RX/RY/RZ 分別作用 X/Y/Z 並回 `-0.5`。【確認】`:290-302`、`gates/cpu_kernels/PauliGenerator.hpp:33-59`
- 單 observable 有專用快路徑 `adjointJacobianSingleObservable`，`num_observables==1` 時自動走。【確認】`:225,339-341,385-388`
- 另有 device VJP（`algorithms/VectorJacobianProduct.hpp`；`lightning_qubit.py:370` 的 `use_device_jacobian_product`）。

### 4.2 條件（C++ 與 Python 兩層）
| 限制 | 證據 |
|---|---|
| 每顆參數化閘只能 1 個參數 | `AdjointJacobianLQubit.hpp:271-273` |
| StatePrep／BasisState 跳過（不可微） | `:274-277` |
| `jac.size() == tp_size * num_observables` | `:366-370` |
| 不可有 shots（必須 analytic） | `lightning_base/_adjoint_jacobian.py:194-199` |
| 不可量 StateMP | `:208-211,218-221` |
| **所有** measurement 必須是 ExpectationMP（混合即 raise） | `:223-227` |
| 裝置只在 `gradient_method in {adjoint,best}` 宣稱支援 | `lightning_qubit/lightning_qubit.py:410-416` |
| 多 observable + `batch_obs=True` 才依 OMP_NUM_THREADS 切分 | `_adjoint_jacobian.py:87-95` |
- 【推論】QBN 的形狀（5 qubit、`expval(Z⊗Z)`、analytic、單觀測量、單參數閘）**完全落在可支援集內**，且正好走 single-observable 快路徑。

### 4.3 與主 repo `default.qubit` 的差異
| 面向 | `default.qubit`（`../pennylane/`） | lightning（本 repo） |
|---|---|---|
| 語言 | **Python** `pennylane/devices/qubit/adjoint_jacobian.py:110-149` | **C++** `AdjointJacobianLQubit.hpp` |
| 導數來源 | generator **矩陣**：`∂U/∂x = 1j·G·U`（`pennylane/operation.py:40-61`） | 作用 Pauli 閘 + 回傳 `-0.5`，**不建矩陣**（`PauliGenerator.hpp:38-40`） |
| 觀測量平行 | 單電路內無；靠 PL 的 `max_workers` 做多電路（`pennylane/devices/default_qubit.py:793-824`） | **C++ OpenMP 直接在觀測量維度**平行（`AdjointJacobianLQubit.hpp:113-146`） |
| 條件檢查 | 跑一遍 compile pipeline（`default_qubit.py:300-312`） | C++ `PL_ABORT_IF` + Python `_handle_raises` 兩層 |
| 參數平移 | 完整 generator→frequencies→shift rule | **本 repo grep `parameter_shift` = 0 命中**，完全靠 PL core |
| backprop | 支援（`default_qubit.py:595-604`） | **完全不支援**（grep backprop/autograd/torch/jax = 0 個實作命中；`lightning_base/lightning_base.py:91` 只是留給 VJP 的中間態字典） |
- 【推論】lightning adjoint 的優勢不是數學不同，而是 (a) 不建 generator 矩陣、(b) C++ 迴圈、(c) 多觀測量時真平行。5 qubit 單觀測量下 (c) 不適用 ⇒ SPEC §3「v1 只做參數平移、adjoint 列 v2」**維持不變**。

## 5. 可借用的 3 件事

| # | 借什麼 | 為什麼 | 證據 |
|---|---|---|---|
| 1 | **X/Y/Z/H/S/T/CNOT 特化 kernel 完全不建矩陣** | 與主 repo 筆記 §6-#1 同結論，但這裡有 C++ 成本對照：通用路徑每次呼叫都複製 matrix 進 `std::vector`，特化路徑一次都沒有 | `GateImplementationsLM.hpp:713-716,741-746,769-775,866-872`；反例 `:286,345,463` |
| 2 | **alias method 抽樣**（Walker alias table） | 建表 O(2^n) 後**每 shot O(1)**；`Generator.choice(p=…)` 是 cumsum 表 + 每 shot `searchsorted`（O(n) 次比較）。與 SPEC §4.5「精確機率、O(shots)」一致，且最能保證 §6.6 的 p99 <5 ms | `measurements/MeasurementsLQubit.hpp:640-679`、`measurements/MeasurementKernels.hpp:304-345` |
| 3 | **具名門檻常數 + 明確 fallback** 的形式 | `STD_CROSSOVER` 2^18／2^12、`internal_wires=log2(packed_size/2)`、`2^n<packed_size/2` 退回純量、AVX 需 n≥4。q01 的 einsum↔tensordot 門檻應照此寫成具名常數 + 一個測試，而不是散在 if 裡 | `utils/LinearAlgebra.hpp:122-124,319-321`、`SingleQubitGateHelper.hpp:164-173`、`AssignKernelMap_AVX2.cpp:38` |

## 6. 不值得抄的 3 件事

| # | 不抄什麼 | 為什麼 | 證據 |
|---|---|---|---|
| 1 | **多 kernel 家族 + runtime kernel map + dynamic dispatcher** | 8 張表 × threading × memory model 換到的是「適應不同 CPU／對齊」；q01 是 Python、n≤20，numpy 自己就會選 SIMD。引入只會讓「與 CUDA-Q 逐元素相同（`|dψ|max=0`）」的舉證變難 | `StateVectorLQubit.hpp:69-91,101-128`、`gates/KernelMap.hpp:270-293`、`gates/DynamicDispatcher.hpp` |
| 2 | **`std::complex` AoS + lane permutation 的 SIMD 技巧** | 那是「AoS 不可改」下的變通；PyTorch/NumPy 的 complex 張量沒有這個自由度，硬仿會做出不能測的 kernel | `avx_common/AVXUtil.hpp:171-174`、`avx_common/ApplySingleQubitOp.hpp:76-96` |
| 3 | **`Threading` dispatch key 這套「假執行緒選項」** | `Threading::MultiThread` 從未被生產路徑選中、`bestThreading()` 零呼叫者 ⇒ 只增加狀態空間不改變行為。平行要放在 shots／參數掃描／多電路三個**真實**維度（SPEC §8 已如此寫） | `utils/Threading.hpp:38-43,63-73`；全 repo `MultiThread` 僅 3 命中 |

## 7. 對 q01 的決策影響

| 決策 | 結論 | 依據 |
|---|---|---|
| 做 SIMD／多 kernel 家族？ | **不做**。我們的目標平台 Windows 上 lightning 自己都不做 | §1.2、§2.1、§3.3 |
| 做 gate fusion？ | **不做**（lightning 也沒有，0 命中） | §3.1 |
| 多執行緒放哪裡？ | 只放 shots／參數掃描／多電路；**單電路內一律序列** | §2.2、§2.4 |
| 抽樣用什麼？ | alias method（或先對打 `Generator.choice`） | §5-#2 |
| 梯度路線 | v1 參數平移不變；adjoint 列 v2，且**不要期待 5 qubit 有平行紅利** | §4.2、§4.3 |
| 門檻常數 | 學**形式**（具名常數 + 測試），不抄它的值 | §5-#3 |

### 待實測 / 缺口
1. 【待測】alias method vs `np.random.Generator.choice(p=…)` 在 5 qubit、10k–100k shots 的 wall-time 與 p99（SPEC §6.4/§6.6）。
2. 【待測】einsum vs tensordot 在本機 numpy 於 n=5/8/10/12/14 對打（承 `refs-pennylane.md` §8-#2，仍是最高價值的未驗證項）。
3. 【待測】adjoint vs 參數平移在 5–12 qubit 的 wall-time 交叉點。
4. 【缺口】未查 lightning 的 Windows wheel 是否真的不存在（本調查只讀原始碼，未安裝）。
