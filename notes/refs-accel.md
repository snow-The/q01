# 加速候選評估：Numba / Numexpr / Dask / mpmath

> q01 SPEC §8、附錄 B 的研究產出（只涵蓋這四個候選；Qiskit/PennyLane/tket 另檔）。
> 方法：**只讀原始碼與文件**（`.refs/` 淺克隆），未安裝任何套件、未執行任何程式、未 build。
> 基準：`numba 4b3e8fb`（0.68.0-dev，最新 notes 0.68.0）、`numexpr 7031844`（2.14.3.dev0）、`dask 2026.8.0`、`mpmath 7783b37`。
> **證據分級**：【原始碼確認】= 可當場查證的檔案:行；【推論】= 我從證據推的，可能是錯的；【待實測】= 必須在 i7-11800H 上量。

## 0. 被評估的 workload（不是泛泛而談）

| 項目 | n=5 | n=12 | n=16 | n=20 | 來源 |
|---|---|---|---|---|---|
| 狀態向量 | 32 振幅 / 512 B | 4,096 / 64 KB | 65,536 / 1 MB | 1,048,576 / 16.8 MB | 2^n x complex128 |
| 單次電路（H^n + 環形 CX，best-of-300 min） | 0.208 ms | 1.037 ms | 23.5 ms | 857 ms | `.remote/SPEC-q01.md:235-238`（n=5,12）、`:246-249`（n=16,20） |
| 完整 QBN 電路（n=5） | 0.593 ms | — | — | — | `.remote/SPEC-q01.md:226` |
| 取樣 100k shots | 4.22 ms | — | — | — | `.remote/SPEC-q01.md:229` |

內層迴圈兩形狀：（a）單閘 = `reshape((2,)*n) + moveaxis + tensordot`；（b）受控閘 = 布林遮罩半邊更新。
訓練 = 數千次前向 + 參數平移梯度；參數掃描 = embarrassingly parallel（每 task 一個純量損失）。

## 1. 先做成本分解（這決定 numba 值不值得）【推論，由 SPEC §6.2 兩點反推】

t(n) = a + b·(2^n · 2n)，2n = 閘數；n=5 → 2^n·2n = 320，n=12 → 98,304。
b = (1.037 − 0.208) ms / (98,304 − 320) = **8.5e-6 ms ≈ 8.5 ns／（振幅·閘）**；a = 0.208 − 320·8.5e-6 ≈ **0.205 ms 固定成本**。
⇒ n=5 有 **~98% 的時間與振幅無關**（Python/NumPy 呼叫 + 配置），n=12 固定成本仍佔 ~20%，n=16 預測只佔 ~1%
（模型在 n≥16 低估：實測 23.5 ms vs 模型 17.9 ms，表示大 n 另受記憶體/複製支配）。**JIT 能省的上限就是這個 a。**
> 【待實測】只有兩點，且 SPEC §6.3 註記 n≥14 為未暖機 v2 量測；回來後用 v4 暖機曲線至少 5 點重做。

## 2. Numba

**2.1 complex128 支援現況【原始碼確認】** numba 沿用 NumPy 的 loop-type 字元：`D`=complex128、`F`=complex64。

| 我們要用的 ufunc | 證據 |
|---|---|
| `np.add` / `np.multiply`（`'DD->D'`）；`np.absolute`；`np.conjugate` | `numba/np/ufunc_db.py:133, 169, 98, 352` |
| `np.exp` `np.sqrt` `np.sin` `np.cos`（`'D->D'`，皆有 nopython 實作） | `numba/np/ufunc_db.py:369, 418, 464, 471` |
| dtype 對映 / Complex 型別 / BLAS 的 complex（complex128→`'z'`） | `numba/np/numpy_support.py:34-35`；`numba/core/types/scalars.py:133`；`numba/np/linalg.py:101-102` |

不支援 extended precision（`clongdouble`）——`docs/source/reference/numpysupported.rst:218-222`；我們不用，無妨。

**2.2 能不能 JIT「閘作用 + 電路執行」核心**

- 【原始碼確認】**`np.tensordot` 與 `np.einsum` 在 numba 裡沒有實作**：整個 `.refs/numba` 對 `einsum` **0 命中**；
  `tensordot` 只有一處，是 Version 0.6 的舊公告 `docs/source/release-notes.rst:6156`。
  ⇒ **不能把現行 `tensordot` 寫法搬進 `@njit`**，這是本次評估最硬的技術限制。
- 【原始碼確認】`np.moveaxis` 自 0.64 起支援（`numba/np/arrayobj.py:7329-7330`、`docs/source/release/0.64.0-notes.rst:28`）；
  `np.dot`/`@` 支援（`numba/np/linalg.py:585-598`）。
- 【推論】**可行設計**：電路壓成扁平陣列（opcode / target / control-mask / theta），一個 `@njit` 內用 `if opcode == k`
  分派，振幅更新改成**索引算術**（`lo = i & ~bit`、`hi = lo | bit`）的直接半邊更新——不需要 reshape/tensordot，
  順便消掉 `moveaxis` 的隱藏複製。雷：`@njit` 吃不下 Python 的閘物件清單（必須先壓平，或 `numba.typed.List`）。

**2.3 對 n ≤ 12 的 Python 開銷能帶來多少**

- 【原始碼確認】numba 文件自己的量測 `docs/source/user/5minguide.rst:127-153`：同一個 `@jit(nopython=True)` 函式
  （10 次 `tanh` + 100 元素加法）首呼 **0.330 s**（含編譯），第二次 **6.68e-6 s = 6.7 µs**。
- 【推論】6.7 µs 是「一次 njit 呼叫 + 少量元素工作」的量級下限。把 §1 的 a≈0.205 ms 換成「一次呼叫跑完整條電路」，
  n=5 電路的合理目標是 **10–30 µs ≈ 7–20×**；n=12 只能省掉 a（1.037 → ~0.84 ms，**~1.2×**）。
  **n 越大 JIT 相對收益越小**——SPEC §8 假設的「10–50×」只在 n≤8 站得住。前提：一次呼叫跑完整條電路；
  若每閘一次 Python↔JIT 邊界，省下的錢會被邊界成本吃回去。
- 【待實測】①njit 空呼叫 overhead ②n=5/8/10/12 JIT vs NumPy ③含 bitorder perm 的端到端（JIT 也可能順手吸收 bit-reversal，對 §4.1 契約是加分）。

**2.4 限制與雷**

| 雷 | 證據 | 對 q01 的處置 |
|---|---|---|
| **object mode 已經沒有了**（0.59 移除 fallback、nopython 預設 True） | `numba/docs/source/release/0.59.0-notes.rst:407-414` | 不會靜默掉回慢速解譯，但**編譯失敗＝硬失敗**（TypingError）⇒ 必須有退回純 NumPy 的 fallback 且要有測試 |
| 編譯 ~0.1–1 s/函式，每個 process 重啟都要重編；`cache=True` 寫 `__pycache__` 或 `$HOME/.cache/numba` | `5minguide.rst:152`；`docs/source/reference/jit-compilation.rst:67-73` | 一律 `cache=True` |
| `cache=True` 限制：函式必須能被 locator 找到（不能 REPL/exec/動態定義） | `numba/core/caching.py:445-453`（"cannot cache function %r: no locator available"） | JIT 核心定義在**檔案模組層級** |
| **版本封印**：0.68.0 支援 Python 3.10–3.15、NumPy `1.22<=v<1.27` **或** `2.0<=v<2.6`；llvmlite 綁死（文件列 0.50.x；HEAD setup.py 已是下一版 0.51.x-dev） | `numba/docs/source/user/installing.rst:265-266`；`numba/setup.py:415-417` | 參考機 numpy 2.5.3 落在 2.0–2.6 內 ✅ |
| **執行期只檢查下界**（"needs NumPy ... or greater"），上界只是「測過」 | `numba/__init__.py:42` | numpy 2.6 一發布就是未測區；SPEC §9 的 `numpy>=2` 無上界 ⇒ numba 只能是 **optional extra `q01[fast]`** 且 pin `numpy<2.6`，**絕不進 runtime 依賴** |
| Windows 多執行緒 layer：優先序 `['tbb','omp','workqueue']`，但**只有 workqueue 保證存在**（TBB 要另外 `pip install tbb`） | `numba/core/config.py:408-411`；`docs/source/user/threading-layer.rst:28-33` | 不要用 `parallel=True` |
| 真正多核路徑：`nogil=True` 釋放 GIL、`prange`、`set_num_threads` | `jit-compilation.rst:60`、`docs/source/user/parallel.rst:101-108`、`numba/np/ufunc/parallel.py:586` | 用 `nogil=True` + 自開 `ThreadPoolExecutor`（共用同一狀態向量、零 pickling），或 `prange` 切半邊更新 |

## 3. Numexpr

**3.1 它最佳化的運算式型態【原始碼確認】** 是**逐元素虛擬機**：`docs/intro.rst:14-16` "perform the operation
element-wise. The virtual machine uses 'vector registers': each register is many elements wide (by default 4096 elements)"。
動機是**暫存陣列與快取**而非 FLOPs（`docs/intro.rst:18-23`：NumPy 的 `2*a+3*b` 要三個同樣大的暫存陣列）；
適用門檻 `docs/intro.rst:61-67`："speed-ups ... 0.95x and 20x, being 2x, 3x or 4x typical values ...
you will need to operate with **large arrays (typically larger than the cache size of your CPU)**"。
機制：opcode 直譯器 `numexpr/interp_body.cpp` + `numexpr/opcodes.hpp`；區塊 `BLOCK_SIZE1 = 1024`（`numexpr/numexpr_config.hpp:16-20`）；
暫存配置最佳化 `numexpr/necompiler.py:532 optimizeTemporariesAllocation`。

**3.2 與我們的張量縮併（BLAS 路徑）有沒有交集：沒有【原始碼確認】**

- 全 `numexpr` 目錄對 `tensordot`/`matmul`/`einsum`/`contraction` **0 命中**；`reshape` 只出現在
  `numexpr/tests/test_numexpr.py:125` 等處，那是 NumPy 的 reshape 用來造測試資料。
- 沒有索引/gather：只有逐元素 `where(bool, a, b)`（`docs/user_guide.rst:202`）；變數只能是 ndarray 或純量
  （`numexpr/expressions.py:32-34, 484-489`）。
- ⇒ 受控閘的 `sv[hi] = a*sv[hi] + b*sv[lo]`（位移索引，非同索引逐元素）**無法表達**；單閘 tensordot 也不行。
  唯一理論接點是機率 `abs(sv)**2` 的 1–2 趟融合，但要 n≥18（16.8 MB）才可能贏，而 §6.1 的取樣 4.2 ms 早已達標。
  ⇒ **對 q01 沒有決策價值。**

**3.3 多執行緒與 VML 的實際機制【原始碼確認】**

| 主題 | 事實 | 位置 |
|---|---|---|
| VML 預設**關閉** | `USE_VML` 只在「自己編譯且放 site.cfg 指向 oneAPI MKL」時定義；原始碼那段 `#define USE_VML` 是**被註解掉的** | `numexpr/numexpr_config.hpp:10-12`；`numexpr/setup.py:64-89` |
| 執行期查詢 | `numexpr.use_vml` 由 C 模組回報（PyPI wheel 通常 False） | `numexpr/__init__.py:24` |
| VML 兩個旋鈕 | `set_vml_accuracy_mode`→`vmlSetMode`；`set_vml_num_threads`→`mkl_domain_set_num_threads` | `numexpr/utils.py:40-100` |
| 多執行緒 | 匯入時建 pthread 池（`module.cpp:238 pthread_create`、`:312 pthread_join`）；Windows 用自帶模擬層 | `numexpr/win32/pthread.{c,h}`、`numexpr_config.hpp:30-37` |
| 執行緒數 | `set_num_threads`；未設 `NUMEXPR_MAX_THREADS` 時**安全上限 16**（`DEFAULT_MAX_THREADS=64`） | `numexpr/utils.py:103, 139-151`；`numexpr_config.hpp:24` |
| 平行粒度 | 只在 block 迴圈層：`taskfactor = 16*BLOCK_SIZE1*gs.nthreads`；`nthreads==1` 強制序列 | `numexpr/interpreter.cpp:793, 910` |

## 4. Dask

**4.1 額外成本：官方自己給了數字【原始碼確認】**

| 主張 | 數字 | 位置 |
|---|---|---|
| 每個 task 的排程開銷 | **200 µs – 1 ms** | `dask/docs/source/best-practices.rst:101` |
| delayed task 開銷 | "a few hundred microseconds" | `dask/docs/source/delayed-best-practices.rst:173` |
| FAQ 的 back-of-the-envelope | 每 task ~200 µs | `dask/docs/source/faq.rst:305-310` |

機制上本機 threads 排程器 = `ThreadPoolExecutor` + 一層簿記：`dask/threaded.py:16,20` → `dask/local.py:144 start_state_from_dask`
→ `:246 execute_task`（每 task 一次）；圖先過 `dask/optimization.py:86 fuse_linear` 等融合。

**4.2 對「參數掃描／多 shots／多電路」的淨值【推論】**

| 面向 | 判斷 |
|---|---|
| 收益 | 跨機器、dashboard、retry/annotation、`cache.py` 快取——單機 16 執行緒的參數掃描**一項都用不到** |
| 成本 | 直接依賴 7 套（click/cloudpickle/fsspec/packaging/partd/pyyaml/toolz，`dask/pyproject.toml:35-50`）；`distributed` 還是**另一個 extra**（`dask/dask/distributed.py:3-16`，沒裝就 raise） |
| 門檻 | 每 task 必須 **≫ 1 ms**。我們 n=5..12 每 task 只有 0.2–1.0 ms ⇒ 排程開銷 ≥ 計算量；n=16/20（23 ms/857 ms）才划算，但那用 `concurrent.futures`/joblib 一樣做得到 |
| GIL | 【推論】numpy 只在 C 迴圈/BLAS 內放 GIL，n≤12 的成本主要在 Python 呼叫（§1 的 a）⇒ 執行緒拿不到加速，小 n 要 process 或 numba `nogil`。【待實測】用真實掃描比 threads/processes |

**決策**：v1 不引入；v2 只在「每 task ≥ 10 ms」且「真的需要跨機器」時重評。（本機無 joblib，未做三方實測。）

## 5. mpmath（第三個 oracle）

**5.1 它能補上 SPEC §7 沒有的東西【推論】** SPEC §7 已有 CUDA-Q（L1–L4）與 Qiskit（L0/L1），但兩者與我們
**共用 float64 算術假設**；mpmath 提供**不同算術**（任意精度、純 Python），能抓「兩邊犯了同一個雙精度錯誤」的 bug，
且**不需要任何量子 SDK**（正合 SPEC §7.2「學生不需 SDK 也能自我檢查」）。

**5.2 API 與可行性【原始碼確認】**

- 上下文 `mp.dps` 的 `mp`（`mpmath/ctx_mp.py`）、`mpmathify`（`ctx_mp.py:625`）。沒有原生向量型別 ⇒ 用
  `mp.matrix`（`mpmath/matrices/matrices.py:285 __init__`、`:632 __mul__`、`:773 transpose`、`:785 transpose_conj`）或 list of `mpc`。
- **PSLQ 是最高價值的功能**：`pslq(x, tol=None, maxcoeff=1000, maxsteps=100)` 反解整數關係 `c·x ≈ 0`
  （`mpmath/identification.py:18-30`；範例 `pslq([-1, pi], tol=0.01) → [22, 7]`；另見 `:315 findpoly`）
  ⇒ 可把「去相位 0.0502」「ECE 2.78e-17」這類實測值**反查成閉式**，不只是比數字。
- 更強的 oracle 形式：區間算術 `mp.iv`（`mpmath/ctx_iv.py`；`docs/contexts.rst:132` "Interval arithmetic provides
  rigorous error tracking"）⇒ 給**誤差上界**而非 30–50 位浮點。
- **numpy 互操作是明確成本**：只吃 numpy *純量*（`mpmath/ctx_mp_python.py:887-897 npconvert`）；
  `mpmathify(np.array([1]))` 直接 `TypeError`（`mpmath/tests/test_convert.py:289`）⇒ 要自己寫逐元素轉換（list comprehension）。

**5.3 效能量級【推論，待實測】** 預設是**純 Python 大整數**：`mpmath/libmp/backend.py:20-38`（`BACKEND='python'`、`MPZ=int`），
只有裝 gmpy2 才切 `'gmpy'`，而文件說它「much faster, especially at high precision (approximately above 100 digits)」
（`docs/setup.rst:76-79`）——**30–50 位不一定吃得到這個紅利**。矩陣是 Python 迴圈：`docs/matrices.rst:311`
"If you need more speed, use NumPy"。量級估計（1 次 `mpc` 乘法 1–3 µs）：n=5（~640 次）→ 10 ms 級可用；
n=12（~8 萬次）→ 0.1–1 s 勉強；n≥16 不實用。**只能當小 n oracle**。
【原始碼確認】成本幾乎為零：`mpmath/pyproject.toml` **沒有 `dependencies` 欄**（只有 `requires-python>=3.10` 與 optional gmpy2）。
**建議**：納入 v1 **測試期** optional extra（`q01[oracle]`），不進 runtime。

## 6. 決策表

| 候選 | 預期收益（對我們的 workload） | 成本 | 納入 q01 | 定案需要的實測 |
|---|---|---|---|---|
| **Numba（JIT 電路核心）** | n≤8：7–20×（省掉 §1 的 a）；n=12：~1.2×；n≥16：<1% | optional extra；llvmlite 原生輪子 + **numpy 上界 2.6**；編譯 0.1–1 s（需 cache=True）；必須改寫成索引算術（無 tensordot） | **v2 候選**（v1 先留可替換的 `backends/numpy.py` 介面） | ①njit 空呼叫 overhead ②n=5/8/10/12 JIT vs NumPy ③冷啟+快取時間 |
| **Numba（nogil + 執行緒／prange）** | n≥16 的多核（numpy elementwise 單執行緒）；省 pickling | 無額外依賴，但 Windows 只有 workqueue 保證存在 | **v2 候選** | 16 執行緒下 n=16/20 單閘吞吐 vs 單執行緒 |
| **Numexpr** | ≈0：無縮併、無索引；只有機率向量的 1–2 趟融合 | 依賴少（只有 numpy）但換不到東西 | **永不**（除非未來出現 n≥18 的 elementwise 熱點） | 若真要：n=18/20 的 `abs(sv)**2` numpy vs numexpr |
| **Dask** | 0（單機 16 執行緒）。每 task 開銷 200 µs–1 ms ≥ 我們 n≤12 的計算量 | 7 個直接依賴 + distributed 另一個 extra | **v1/v2 都不納入**（只在跨機器時評估） | 參數掃描：`concurrent.futures`(threads/processes) vs `dask.delayed` 每 task 實測 |
| **mpmath（第三 oracle）** | 高（正確性，非速度）：PSLQ 反查閉式、n≤5 高精度獨立實作、`mp.iv` 誤差上界 | 零依賴、純 Python；numpy 只能逐元素轉換 | **納入 v1 測試期**（`q01[oracle]`，非 runtime） | ①n=5 高精度機率 vs NumPy ②PSLQ 對 0.0502 是否解出閉式 ③50 dps 下 n=5/12 的秒數 |

**跨候選一句話**：n≤12 的瓶頸是 **Python 呼叫開銷**（只有 numba 打得到）；n≥16 是**單執行緒記憶體頻寬**（只有多核打得到）。
numexpr 打的是 elementwise 記憶體流量（我們不缺），Dask 打的是分散式排程（我們不需要）。
真正缺的不是加速器，是**第三個獨立算術**（mpmath）。

## 7. 我沒能查證、不要當事實的事

- 本檔**未執行任何程式碼**；所有時間預測都是推論（含 §1 的 a、b 反推與 numba 的 7–20×）。
- numba 的 6.7 µs / 0.330 s 是上游文件在**他們的機器**上的數字（`5minguide.rst:152-153`），不是我們的機器。
- joblib / `concurrent.futures` 未實測；本機未安裝任何候選套件（numba/numexpr/dask 於本機 site-packages 探測 0 命中；mpmath 只存在於無關的 venv）。
- numba 0.68.0 的 release 日期在支援表上仍是 `TBD`（`installing.rst:265`）⇒ 版本號與 wheel 可用性待發布確認。
