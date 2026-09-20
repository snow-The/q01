# NVIDIA 生態圈考察：cudaq-algorithms 與 cuda-q-academic 對 QBN 的價值

- **考察對象**：
  - `CA/` = `.refs/cudaq-algorithms`，git HEAD `2dfa0fce8cb1955d391fe7168c0a65475339e80c`（2026-09-15，"Add unary-iteration and QROM primitives (#40)"）。**shallow clone（`git rev-list --count HEAD` = 1）**，行號只對本 checkout 有效。
  - `AA/` = `.refs/cuda-q-academic`，269 MB（絕大部分是圖檔，見 B1）。
- **標記**：【檔案確認】= 我在該行讀到的事實；【推論】= 由事實推得但檔案未直述；【待實測】= 需在參考機驗證。
- **前提（來自 `refs-cudaq.md`，不重複）**：q01 存在的理由是原生 Windows 裝不起 CUDA-Q；`ParameterShift` 在 CUDA-Q 本體是 `parameter_shift.h:17` ±π/2、除以 2。

---

## A. cudaq-algorithms —— 官方 FTQC 演算法庫，**與 NISQ／QML 完全無關**

### A1. 它提供哪些演算法／primitive【檔案確認】

定位在 `CA/README.md:3-7`：「primitive-first algorithms library … focused on fault-tolerant quantum computing (FTQC) primitives: block encodings, qubitization, quantum singular value transformation, product formulas」。**明確排除我們的領域**（`CA/README.md:9-10`）：

> NISQ-era application workflows such as VQE, ADAPT-VQE, QAOA, GQE, and optimizer loops are intentionally out of scope.

| 類別 | 內容 | 證據（`CA/` 相對路徑:行號） |
|---|---|---|
| fermion→qubit | Jordan-Wigner、Bravyi-Kitaev | `python/cudaq_algorithms/fermion/_compilers.py:310`、`:329` |
| state prep | Hartree-Fock、單/雙激發、UCCSD/UCCGSD/UPCCGSD/CEO | `stateprep/_kernels.py:36`、`:60`、`:392`、`:446`、`:529`、`:542`、`:555`、`:568` |
| 算子池（host） | UCCSD/UCCGSD/UPCCGSD/CEO 池、參數量、Pauli lists | `stateprep/_pools.py:69`、`:186`、`:253`、`:273`、`:342`、`:379` |
| Givens/Slater | 行列式態準備 | `stateprep/_givens.py` |
| Block encoding | `BlockEncoding` 協定、`PauliLCU` | `block_encoding.py:53`、`pauli_lcu.py:455` |
| LCU 裝置核心 | prepare/select/apply/walk/reflect/adjoint/phase-seq | `pauli_lcu.py:66`、`:121`、`:200`、`:210`、`:227`、`:235`、`:281` |
| Qubitization | `Walk`、`reflection_observable` | `qubitization.py:57`、`:41` |
| QSVT | `QSVT`、`PhaseSequence`、`recover_real_time_evolution` | `qsvt.py:153`、`:68`、`:304` |
| Trotter | `Trotter`、`TrotterOrdering`、資源估計（order 1/2/4） | `trotter.py:324`、`:236`、`:309`、`:503`、`:248` |
| 電路 primitive | `QROM`、unary iteration、Toffoli/select-swap 計數 | `primitives/_qrom.py:158`、`:149`；`primitives/_unary_iteration.py:253`、`:490`、`:311` |
| 化學橋接 | `from_pyscf` / `from_psi4` / `from_fcidump` | `chemistry.py:128`、`:170`、`:225` |
| 雙因子化（古典） | X-DF / C-DF / RC-DF，NumPy/SciPy（可選 CuPy） | `double_factorization/_factorization.py` |
| 模擬用 helper | `good_subspace` / `action` / `transform` / `evolve` | `sim_utils.py:34`、`:53`、`:62`、`:77` |

**沒有任何 gradient / ParameterShift / VQE 迴圈 / 分類器**：全庫對 `gradient` 的命中只有雙因子化的「共軛梯度」（`double_factorization/_factorization.py:288`、`:542`、`:606`）；`ParameterShift` 零命中。【檔案確認】

### A2. 授權與關係【檔案確認】

| 問題 | 答案 | 證據 |
|---|---|---|
| 授權 | **Apache-2.0** | `CA/LICENSE`（11636 B）、`CA/NOTICE:3`、`CA/pyproject.toml:13` `license = "Apache-2.0"`、`CA/README.md:53` |
| 是獨立套件嗎 | 是，獨立 repo、獨立 PyPI 名 `cudaq-algorithms` | `CA/pyproject.toml:6` |
| 要另裝嗎 | **要**。依賴 `cudaq >= 0.15.0, < 0.16` | `CA/pyproject.toml:14-20` |
| 裝法 | `pip install cudaq-algorithms`；但**尚未正式上 PyPI** | `CA/docs/sphinx/getting_started/index.rst:45`、`:49-50`（"Until the first PyPI release, install from a release wheel …"） |
| 與 CUDA-Q 版本綁定 | 釘住 CUDA-Q commit `b28cf0f6f2d9387f12ca81b6824b8833a38530af` | `CA/.cudaq_version:1-5` |
| 只有純 Python | 是，wheel 是 `py3-none-any` | `CA/.github/workflows/wheel_algorithms.yaml:94`、`:121` |
| 引用要求 | 「If you use CUDA-Q Algorithms in your work, please cite it as well as CUDA-Q」 | `CA/CITATION.cff:3-5` |

【檔案確認，最關鍵】每個模組都 `import cudaq`（例：`CA/python/cudaq_algorithms/pauli_lcu.py:9`、`sim_utils.py:18`）⇒ **q01 不能當它的後端**；它需要一個真的 CUDA-Q 執行環境。

### A3. 對我們的具體價值

**(a) 論文 Related Work 要不要引用它？** 【推論，建議】**不要當 baseline 引用**——它刻意排除 VQE/QAOA/QML（`README.md:9-10`），與 QBN 分類器無可比性，硬引會被審稿人視為 filler。**但值得引用一句作為 scope 正當性**：NVIDIA 官方演算法庫把 NISQ 變分工作流列為 out of scope，佐證「CUDA-Q 生態的官方重心不在 5-qubit 變分分類器」，正好支撐我們「不主張量子優勢、定位為教學與可重現性」的論述。若最終完全沒用到它的程式碼，`CITATION.cff:3-5` 的引用義務不觸發。

**(b) 手寫 ParameterShift／QBN 電路有官方對應物可對照嗎？** 【檔案確認】**這個 repo 沒有**（見 A1 末）。可用的官方對照物在別處：
- `ParameterShift` 本體 → CUDA-Q 核心（`refs-cudaq.md` §5 已記）。
- **官方示範用法** → 學術 repo `AA/utilities/vqe.py:44` `gradients.ParameterShift`、`:196` 預設 `gradient="parameter_shift"`、`:142` `gradient.compute(parameters, function_at, value)`。這是「官方怎麼把梯度接到優化器」的最小範例，可與 q01 `gradients.py` 的呼叫端契約對照。【推論】
- 唯一像 QBN 的形狀：`AA/quantum-machine-learning-and-data-analysis/01_*.ipynb:697-719` 的 PyTorch autograd bridge（見 B2）。

**(c) 有可以直接拿來當「獨立第三實作」的部件嗎？** 【檔案確認 + 推論】**電路層面：沒有**。**測試／約定層面：有兩個高價值可搬物件**：
1. `CA/tests/python/dense_references.py:8-26` `dense_matrix(terms, num_qubits)` —— 零依賴的稠密 Pauli-sum 建構器，且**明確 little-endian**（`:17` `bit = (column >> qubit) & 1`），`:29 random_ket(seed)` 造隨機態。這可以當 q01 `spin`／`observe` 的**第四個獨立 oracle**（目前是 CUDA-Q golden ＋ 手算），且不需安裝 cudaq。【推論：可移植，待實測】
2. `CA/docs/sphinx/conventions.rst` —— 這頁自己就是「約定即規格」的範本，直接可引用（見下）。

### A4. Windows 支援的證據：**沒有，且 CI 全是 Linux**【檔案確認】

- 全 repo 文字檔（`*.md,*.rst,*.py,*.toml,*.yaml,*.yml,*.txt,*.cfg`）對 `windows|win32|msvc|powershell` **零命中**。
- 所有 `runs-on` 只有 `ubuntu-latest` 或 NVIDIA 自架 `linux-*-cpuN`：`CA/.github/workflows/build_cudaq_wheels.yaml:43`、`build_cudaq.yaml:38`、`lib_algorithms.yaml:66`、`wheel_algorithms.yaml:99`、`docs.yaml:34`、`:67`、`pr_*.yaml`。
- 容器全是 Linux：`ghcr.io/nvidia/cuda-quantum-devcontainer:...-gcc12-main`（`build_cudaq.yaml:39`、`lib_algorithms.yaml:67`）、`container: python:<matrix-python>`（`wheel_algorithms.yaml:102`，實為 ubuntu）。
- 【推論】但交付物是 `py3-none-any` 純 Python wheel（`wheel_algorithms.yaml:94`、`:121`）⇒ 在 Windows 上「裝得起來」；真正的阻礙不是它，而是它硬依賴的 `cudaq` 本體在原生 Windows 的可得性。【待實測】參考機上 `pip install cudaq-algorithms` 能否配合 Windows 版 cudaq，**本專案未測，且不建議為此投入**。

---

## B. cuda-q-academic —— 教學 notebook 集合，269 MB 但約九成是圖

### B1. 到底裝了什麼【檔案確認】

**結論：這不是資料集，是一組 Jupyter 教學模組＋插圖＋互動 widget。** 各目錄實測大小／檔案數：

| 目錄 | 大小 | 檔案 | 用途 |
|---|---|---|---|
| `qec101/` | 106.1 MB | 118 | 量子糾錯 9 課（**幾乎全部體積是 decoder 插圖**） |
| `images/` | 14.5 MB | 47 | 全站共用插圖 |
| `chemistry-simulations/` | 13.3 MB | 59 | VQE / ADAPT-VQE / QPE / Krylov / QM-MM |
| `calibration/` | 10.2 MB | 39 | 超導量子位元校準資源＋Ising Calibration demo |
| `quantum-machine-learning-and-data-analysis/` | 8.9 MB | 27 | **QML 4 課（與我們最相關）** |
| `quantum-applications-to-finance/` | 3.9 MB | 24 | 量子漫步、投資組合 |
| `ai-for-quantum/` | 3.7 MB | 16 | AI 輔助量子（么正編譯、解碼器） |
| `qaoa-for-max-cut/` | 3.3 MB | 22 | QAOA＋電路切割 |
| `simulation/` | 2.8 MB | 10 | **模擬器選型 101** |
| `hybrid-workflows/` | 2.4 MB | 13 | 混合工作流 |
| `dynamics101/` | 1.4 MB | 9 | Jaynes-Cummings／Lindblad |
| `qis-examples/` | 1.3 MB | 7 課 | 基礎演算法（teleportation…Shor） |
| `quick-start-to-quantum/` | 1.0 MB | 13 | **入門 4 課（唯一 CPU 可跑）** |
| `quantum-ai-project-template/` | 0.1 MB | 8 | 教師用分組專題模板 |
| `utilities/` | 小於 0.1 MB | 2 | `vqe.py`、`qaoa_helper.py` |

體積集中在少數大圖：`qec101/Images/decoder/pauliframes.png` 16.7 MB、`decoding.png` 12.8 MB、`bposd.png` 11.3 MB、`qec101/Images/noisy/channels.png` 7.0 MB、`images/accelerating/maxcut_ani.gif` 6.1 MB、`calibration/images/calibration_images.zip` 4.6 MB、`quantum-machine-learning-and-data-analysis/images/nn_math.png` 4.2 MB。⇒ **盲目全讀是浪費；只讀 notebook 與 README 即可。**【檔案確認】

**機器可讀目錄**：`AA/curriculum.json` 是唯一內容真相來源（`AA/llms.txt:3`、`AA/AGENTS.md`），實測：13 條 track、**43 個 lesson**、44 個 widget，`metadata.last_updated = "2026-09-11"`。每模組固定結構：學生 notebook（練習處留 `##TODO##`）＋ `README.md` ＋ `solutions/`（完整解答）＋ `images/`（`AA/AGENTS.md`）。

### B2. 可直接對應我們書的自學教材【檔案確認】

**唯一「CPU 就能跑」的變分教材**：`quick-start-to-quantum/03_quick_start_to_quantum.ipynb`（*Lab 3 - Add a Bit of Variation: Write your first variational program*）與 `04`（*Lab 4 - Converge on a Solution*）。兩檔**沒有 `GPU Required` callout**（全檔 grep 零命中）⇒ 與我們「學生在自己筆電上跑」的前提一致。其餘全部需要 GPU（見下）。

**QML track（`curriculum.json` track `quantum-machine-learning-and-data-analysis`，難度 intermediate–advanced，4 課）**：

| 課 | 對我們的用處 | 證據 |
|---|---|---|
| `01_an_introduction_to_hybrid_quantum_neural_networks.ipynb` | **最有價值**：手寫 parameter-shift 的 PyTorch autograd bridge；shift=π/2 的理由；結尾自我限定 | `:59`（key term "Parameter-shift rule"）、`:614`（"The shift of δ = π/2 is not arbitrary"）、`:697-719`（`backward` 實作，`:715` `gradients[:, i] = (exp_vals_plus - exp_vals_minus) / 2.0`）、`:728-737`（"Check the parameter-shift gradient before training"：誤用 sign/shift 會產生「看起來合理」的梯度）、`:516`（ansatz／barren plateau）、`:80`（GPU Required）、`:1138`（**"This small, simulator-based workflow demonstrates how to construct and train an HQNN, not evidence of a practical quantum advantage."**）、`:39`（與中原大學 CYCU 共同開發，動機論文 IEEE Access 太陽輻照預測） |
| `02_advanced_hqnns.ipynb` | SPSA（P 個角度只要 2 次電路，對照我們 2P 次的成本論述）、MQPU、cuDNN | `:38`、`:52`、`:906-908`（"parameter-shift … requires 2P shifted logical circuits … SPSA estimates every angle derivative from only two simultaneous perturbations"）、`:929`（Spall 原始文獻）、`:74`（GPU Required） |
| `03_quantum_svm.ipynb` | **與論文立場同一句話**：官方教材自己要求區分「後端擴展示範」與「量子優勢」 | `:39`（"Distinguish a backend-scaling demonstration from evidence of model generalization or quantum advantage"）、`:1211`（"Is the accuracy of this kernel evidence of quantum advantage? Why not?"）、`:708`、`:79`（GPU Required） |
| `04_quantum_pagerank.ipynb` | Lindblad／密度矩陣；與我們相關性最低 | `:82`（GPU Required；"The classical exercises run on a CPU"） |

**量測與雜訊**：`qec101/03_QEC_Noisy_Simulation.ipynb`（*Simulating Quantum Noise*）—— 密度矩陣 vs trajectory、Kraus 算子、ZNE 誤差緩解、Stim 跑 Steane code、由超導 transmon 動力學建 noise model；`:51-58` 列出 noise channel / density matrix / trajectory simulation / Kraus operator / zero noise extrapolation / circuit-level noise；`:89` 標示 GPU Required。【檔案確認】這是書中「雜訊與量測」章最現成的官方補充讀物。

**位元順序**：`simulation/01_simulation101.ipynb` 明確把 endianness 列為學習目標與實作註記 —— `:39`（"Analyze how endianness, contraction order, bond dimension, and branching affect simulator performance"）、`:116`（`|q0 q1 q2>` 的基底排序聲明）、`:209`（程式碼註解 "Endianness note: we list basis states as |q0 q1 q2>"）、`:72`（GPU Required）。【檔案確認】⇒ 我們 SPEC §4.1「位元順序是本書最危險的坑」有了官方教材的旁證。

**其他**：`AA/Instructor-Guide.md:29` 學習路徑 builder（自選模組產生單一分享連結）、`:65-87` 五條示範路徑；`AA/notebook_template.ipynb` 定義 notebook 前導 schema（**What You Will Do / Prerequisites / Key Terminology / CUDA-Q Syntax / Solutions**）—— 這個 schema 值得我們書的每章開頭照抄。【推論】

### B3. 學術申請管道【檔案確認：**repo 內沒有「申請表」這種東西**】

我全 repo 搜 `academic grant|GPU credit|apply|application form|partnership|DLI|teaching kit`，**唯一相關的是一段給教師的提醒**（`AA/quantum-ai-project-template/Teaching-Guide.md:74`）：

> **GPU acceleration** — if you are provisioning Brev access, GPU cloud credits, or a university HPC allocation for Milestone 3, tell students this before they start and include access instructions; if GPU access is not available, tell them CPU simulation is sufficient…

也就是說：**本 repo 沒有 GPU 額度／教學資源的申請條件與表單**；它給的是「怎麼拿到算力」的既有入口：

| 管道 | 內容 | 連結／證據 |
|---|---|---|
| **Brev**（一鍵啟動） | 預裝 CUDA-Q 的 CPU 或 GPU 實例 | `AA/README.md:84-88`（launchable URL `https://brev.nvidia.com/launchable/deploy/now?launchableID=env-39dN1v7RucHHgj97LILUlnXjnk5`）、`AA/quantum-ai-project-template/Teaching-Guide.md:122` |
| **qBraid** | 預裝 CUDA-Q 的託管 Jupyter，支援課堂協作（建議 M1–2） | `AA/README.md:92`、`AA/quantum-ai-project-template/Teaching-Guide.md:120` |
| **Amazon Braket / Colab** | 替代入口；Colab 需取消註解安裝 cell | `AA/README.md:93-94` |
| 真實 QPU | 經 CUDA-Q hardware backend、qBraid 或 Braket | `AA/quantum-ai-project-template/Teaching-Guide.md:75` |
| 算力分級表 | Prototype `qpp-cpu`（無需 GPU）→ GPU `nvidia`/`custatevec-fp32` → 可選真 QPU | `AA/quantum-ai-project-template/quantum-group-project-guide.md:19-21` |
| **聯絡窗口** | `cuda-quantum-academic@nvidia.com`；教師回饋可能被收錄為官方建議題目（附姓名） | `AA/Instructor-Guide.md:100`、`AA/quantum-ai-project-template/Teaching-Guide.md:233`、`:241` |
| 電子報 | 教學／發布／活動更新 | `AA/README.md:8`、`:24` |
| 課程模板 | 角色分工（Lead／最佳化／QA／技術行銷）＋AI agent 設定＋評分規準 | `AA/quantum-ai-project-template/README.md`、`quantum-group-project-guide.md:145` |

【推論】若我們想要正式 GPU 額度，得走 repo 外的 NVIDIA Academic Grant／大學合作管道——**本 repo 無法回答，不要在這份筆記裡斷言**。【待實測／待查】

### B4. 授權與可否再利用【檔案確認】

- **repo `LICENSE` 是 CC BY-NC 4.0**：`AA/LICENSE:1`（"Attribution-NonCommercial 4.0 International"）、`:57`、`:116`（"NonCommercial means not primarily intended for or directed towards commercial advantage"）、`:152`／`:155`（僅授予 NonCommercial 用途）、`:222-227`（署名義務），共 584 行。
- **檔案層級 SPDX 是雙授權**：notebook 首個 code cell 為 `Apache-2.0 AND CC-BY-NC-4.0`（例：`AA/quantum-machine-learning-and-data-analysis/01_*.ipynb:10`、`AA/qec101/03_QEC_Noisy_Simulation.ipynb:10`、`AA/simulation/01_simulation101.ipynb:9`）；純程式檔只有 Apache-2.0（`AA/utilities/vqe.py:1`）。
- `AA/README.md:14`、`:107`：「freely available … Materials are free to use for educational purposes under Apache-2.0 and CC-BY-NC-4.0」；`:107` 另提醒執行期會下載第三方軟體，需自行確認其授權。
- 【推論，重要】**我們的書若免費公開（教學用途）→ 可引用／改編，需署名**；**若未來要販售或放進付費牆 → NC 條款擋住**，屆時只能引用 Apache-2.0 的 `.py` 部分、不能重製 notebook 內文與圖。書的參考資料清單應逐條標明「CC BY-NC 4.0（僅非商業）」。圖檔（尤其 qec101）體積巨大且屬 NC，**不要直接打包進我們的 repo**。

---

## C. 一句話結論

- **cudaq-algorithms：不值得為書或論文花時間**（FTQC primitives、需真 CUDA-Q、與 QBN 零重疊）；**唯一要撿的兩樣是 `tests/python/dense_references.py` 的 little-endian 稠密參考，與 `conventions.rst` 的精度約定頁**——後者可直接支撐論文 §5／§7.3。
- **cuda-q-academic：值得，但只值得讀 4 個檔＋2 頁文件**——`quick-start 03/04`（唯一 CPU 可跑的變分入門）、`QML/01`（官方 parameter-shift 實作與驗證練習）、`QML/03`（官方自己要求區分 scaling demo 與 quantum advantage）、`simulation/01`（endianness）、`qec101/03`（雜訊／Kraus／ZNE）。授權允許教學再利用，但 **NC 條款要在書的參考清單寫清楚**。

### 落地的具體建議

1. **書（教學）**：不引入 `cudaq-algorithms` 依賴。新增「延伸閱讀」區塊指向 `AA/quick-start-to-quantum/03,04`（標「CPU 可跑」）與 `AA/quantum-machine-learning-and-data-analysis/01`（標「需 GPU」）；雜訊章指向 `AA/qec101/03`。每章開頭照抄 `AA/notebook_template.ipynb` 的五段 schema。
2. **論文（Related Work）**：引用 `CA/README.md:9-10` 作為「官方演算法庫刻意排除 NISQ 變分工作流」的 scope 佐證（**不**作為 baseline）；引用 `CA/docs/sphinx/conventions.rst:152-155`（"The default CUDA-Q target is fp32 … All library tests pin qpp-cpu (fp64) and assert at 1e-10..1e-12"）＋ `CA/tests/python/conftest.py:13` 作為 **NVIDIA 自己對精度／後端約定的獨立陳述**，強化我們 §5 精度規格與 §7.3 可重現性聲明；引用 `AA/quantum-machine-learning-and-data-analysis/01_*.ipynb:1138`（"…not evidence of a practical quantum advantage"）與 `03_*.ipynb:39` 支持我們「不主張量子優勢」的立場。
3. **q01（工程）**：把 `CA/tests/python/dense_references.py:8-29` 的 `dense_matrix` / `random_ket` 移植為第四個獨立 oracle（放 `tests/`，不進 `src/`），用它交叉驗證 `bitorder.py` 的 little-endian 轉換與 `spin`／`observe` 的期望值；在 `bitorder.py` 註解補上 `conventions.rst:23` 的 `functools.reduce(np.kron, ops[::-1])` 反向 kron 寫法當外部依據。**不要**嘗試在 q01 上裝 `cudaq-algorithms`。
4. **不要做**：不要下載／打包 `AA` 的圖（NC ＋ 體積）；不要為了 Windows 支援去測 `cudaq-algorithms`（CI 全 Linux、且它需要 cudaq 本體，與 q01 路線無關）。
