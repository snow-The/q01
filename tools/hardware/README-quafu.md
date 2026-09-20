# 真機路線：Quafu（BAQIS 超導雲）

> 決策（2026-09-19，作者）：真機走 **Quafu**。理由：免費、只綁信箱、每月 1000 次啟動額度、
> 且能一次使用上百個真實 qubit。SDK：`pyquafu`（https://github.com/ScQ-Cloud/pyquafu）。

## 一、環境現況（已驗證）

| 項目 | 值 |
|---|---|
| 套件 | `pyquafu 0.4.5`（Apache-2.0） |
| **import 名** | **`quafu`**（⚠️ 發行名是 `pyquafu`，**import 要用 `quafu`**） |
| 安裝位置 | `/root/quafu/.venv`（**獨立 venv**） |
| Python / numpy | 3.12.3 / **1.26.4** |
| 為什麼獨立 | `pyquafu` 釘 **`numpy<2.0.0`**，與 q01（numpy 2.5.3）**衝突** |
| 輪子 | 有 cp39–cp313、win_amd64／manylinux／macOS ✓ |

## 二、API 完備度（實測）

| 能力 | 物件 | 對我們的用途 |
|---|---|---|
| **OpenQASM 匯入** | `QuantumCircuit.from_openqasm` | ★ 交換格式：q01 端匯出 QASM，不必在 q01 裝 pyquafu |
| **本地模擬** | `quafu.simulate` | ★ **先用本地驗證，不浪費真機額度** |
| **轉譯** | `quafu.transpiler` | 映射到目標晶片的耦合圖 |
| 閘集 | `h cx cy cz cp cs ct iswap fredkin mcx mcy mcz dagger add_controls`… | 我們的 ansatz（RY/RZ/CX）完整覆蓋 |
| 提交 | `Task`、`User`、`ExecResult` | 真機任務與結果 |
| 其他子模組 | `qfasm, synthesis, visualisation, dagcircuits, algorithms` | 備用 |

## 三、整合架構（沿用既有的 JSON 交換模式）

```
q01（numpy≥2，測過的 IR）
        │  匯出 QASM／JSON（純文字，無新依賴）
        ▼
/root/quafu/.venv（Python 3.12 + numpy 1.26）
        ├─ QuantumCircuit.from_openqasm
        ├─ quafu.simulate          ← 先在本地驗證（免費）
        ├─ quafu.transpiler        ← 映射到目標晶片
        └─ Task → 真機             ← 最後才花額度
                    │ counts.json
                    ▼
        我們的 ECE／溫度縮放／盲點分析管線（專案最強的一段）
```

## 四、論文影響（RQ3）

現在論文最弱的一點是「全部結果都可古典模擬」。有真機之後應改為：

**RQ3：不確定性與校準分析是否轉移到真機？**

- 用 QBN 輸出層（4–8 qubit）在真機上跑分類，取 counts
- 套上既有的 ECE／溫度縮放／盲點分析（**這是專案最獨特的一段**）
- 與**黃金向量**（CUDA-Q 精確模擬）逐項對照 ⇒ 直接量出「真機 vs 模擬」
- 加 readout error mitigation（真機論文標準配備）

⚠️ 仍**不主張量子優勢**（4–8 qubit 依然可古典模擬）；
但貢獻從「模擬研究」升級為「**在真機上端到端驗證的管線 ＋ 雜訊特性刻畫**」。

## 五、待辦（P5）

1. **重置 Quafu token**（舊的已被服務端拒絕；新 token 寫進 `~/.dsh/quafu.token`，**不進聊天、不進 git**）
2. 用 token 列後端 → 記錄晶片規格（qubit 數、耦合圖、basis gates、讀出錯誤率）
3. q01 端：**IR → QASM 匯出**（純 Python，零新依賴）
4. QASM → quafu → **本地 simulate** → 與黃金向量對帳（**先不碰真機**）
5. 用 `transpiler` 映射到目標晶片，再本地模擬確認等價
6. 真機試跑 1–2 個任務（驗證 token、shots 參數、結果解析）
7. 真機結果接進校準分析管線 → RQ3

## 六、安全約定

- token 只從 `~/.dsh/quafu.token` 讀取，**永不回顯**（只印長度／遮罩）
- 不寫進任何 git 追蹤的檔案、不寫進 settings.yaml、不貼進聊天

---

## 七、不需要 token 的部分：已完成並驗證（2026-09-19）

### 7.1 q01 端：IR → OpenQASM 2.0

`packages/q01/tools/hardware/qasm_export.py`（純 Python、零新依賴）。
閘集刻意限定在硬體實驗所需：`h x y z s sdg t tdg sx rx ry rz`、單控制 X（`cx`）、
雙控制 X（`ccx`）；**多控制旋轉不匯出**（QASM 2.0 沒有標準閘）→ 明確報錯，不近似。

### 7.2 交換格式先用第三方驗證（不碰 quafu）

`verify_qasm_via_qiskit.py`：IR → QASM → `qiskit.qasm2.loads` → `Statevector` → 比黃金向量。

| 案例 | QASM 行數 | max\|ΔP\| |
|---|---|---|
| `bell` | 5 | **0.000e+00** |
| `qbn5_book` | 23 | **3.469e-17** |
| `qbn5_depth2` | 38 | **1.110e-16** |
| `entangle_then_rotate` | 6 | **0.000e+00** |
| `mcry3` | — | 正確 SKIP（多控制 RY 非標準 QASM 2.0） |

**這步的價值**：把「QASM 寫錯了」與「真機 SDK 收不收」兩個問題分開；
之後 quafu 端若有問題，一定是 SDK 側。

### 7.3 quafu 端：同一份 QASM 餵 `quafu.simulate`（本地、免費、不耗額度）

`verify_qasm_via_quafu.py`（跑在 `/root/quafu/.venv`）。API 實測：

- `QuantumCircuit(qnum, cnum=None)`；**`from_openqasm(text)` 是實例方法**（不是 classmethod）
- `simulate(qc, psi=..., simulator='statevector', shots=0, use_gpu=False, use_custatevec=False) -> SimuResult`
  ⇒ `shots=0` + `simulator='statevector'` 取得精確振幅

### 7.4 ★ L0 公約：**quafu 是 big-endian（q0 = MSB）**

| 案例 | 直接比對 | 位元反轉後 |
|---|---|---|
| `bell` | 0.000e+00 | 0.000e+00（對稱，判不出來） |
| `qbn5_book` | 5.575e-02 | **4.441e-16** ✅ |
| `qbn5_depth2` | 1.306e-01 | **2.776e-16** ✅ |
| `entangle_then_rotate` | 1.097e-01 | **0.000e+00** ✅ |

**⇒ quafu 的狀態索引是 big-endian，與 CUDA-Q／Qiskit／PennyLane（little-endian）相反。**
反轉後最差 **4.44e-16**（判準 1e-10）。

> 有趣的巧合：quafu 的順序與 **q01 的內部表示**一致（q01 對外才轉成 little-endian）。
> 真機結果接回來時，**位元順序轉換一律走 `bitorder.py`**（全專案唯一置換處）。

### 7.5 現在的狀態

| 步驟 | 狀態 |
|---|---|
| q01 端 IR → QASM | ✅ 完成並驗證（經 Qiskit，≤1.1e-16） |
| quafu 端收 QASM + 本地模擬 | ✅ 完成並驗證（反轉後 ≤4.4e-16） |
| L0 公約 | ✅ 定案（big-endian） |
| 列後端／晶片規格 | ⏸ **需要 token** |
| 真機試跑 | ⏸ 需要 token（先用 1–2 個任務驗證流程） |

**⇒ token 一到就能直接送：整條路（QASM → quafu → 模擬對帳）已經在本地跑通。**

---

## 八、後端清單實測（token 生效，2026-09-19）

`User(api_token=...)` → `get_available_backends()`，**共 16 個後端**：

| 名稱 | qubits | 狀態 | 備註 |
|---|---|---|---|
| **ScQ-P5** | **5** | **Online** | 我們的主電路規模（5 qubit） |
| **Baihua（百花）** | **119** | **Online** | ★ 真機、上百 qubit |
| ScQ-Sim10 | 10 | Online | 官方模擬器 |
| ScQ-P10 | 10 | Offline | |
| ScQ-P21 | 11 | Offline | |
| ScQ-P102 | 102 | Obsolete | |
| Baiwang | 136 | Obsolete | |
| Miaofeng | 108 | Obsolete | |
| Dongling | 105 | Offline | |
| Haituo | 105 | Offline | |
| Yunmeng | 156 | Obsolete | |
| Xiang | 35 | Obsolete | |
| ScQ-P3 / ScQ-TEST | 3 / 3 | Offline | |
| ScQ-Sim | 2 | Obsolete | |

**⇒ 之前「Quafu 只有 5 qubit」的判斷不完整：`Baihua` 是 119 qubit 且在線。**
（「Obsolete」多半是退役機，但代表歷史上有過；離線不代表永久不可用。）

### 8.1 pyquafu 正確用法（踩過的三個坑）

```python
from quafu import User, Task
u = User(api_token=token)        # ★ 直接帶入即可；不要用 User()（它會去讀 ~/.quafu/api）
b = u.get_available_backends()   # 回 {名稱: Backend}
```

- ❌ `User()` 無參 → `UserError: Please first save api token`（它不會自己找我們的 token 檔）
- ❌ `User.save_apitoken(token)` → `AttributeError: 'str' object has no attribute 'token_dir'`（它是實例方法）
- ❌ `Task(api_token=...)` → `Task` 要的是 **`user=`**
- ✅ `User(api_token=token)` **不落地任何檔案**（對保密更好）

### 8.2 Backend 物件可用的資訊

`name`、`qubit_num`、`qv`、`status`、`system_id`、`task_in_queue`、
**`get_chip_info()`**、**`get_valid_gates()`** ← 下一步要用這兩個取耦合圖與 basis gates。

### 8.3 額度（作者確認）

**1000 次／月，計費單位是「任務數」不是 shots** ⇒ 每次提交要盡量把 shots 用滿，
而且**本地模擬先行**（`quafu.simulate`）是必須的，別把額度浪費在 debug 上。

作者另有 N 個信箱與 IP ⇒ 額度可橫向擴充（但仍以省著用為原則）。

### 8.4 閘集與可用性實測（2026-09-19）

| 後端 | status | 佇列 | 有效閘（節錄） |
|---|---|---|---|
| ScQ-P5 | Online | 721 | `cx cz rx ry rz x y z h sx sy swap cy cnot id barrier` |
| Baihua | Online | 475 | `cx cz rx ry rz x y z h delay barrier`（**無 `sx/sy/swap`**）—— ⛔ **無權限，送不出去** |
| ScQ-Sim10 | Online | 0 | 官方模擬器（可送，見 §8.6） |
| Baiwang / ScQ-P102 / Yunmeng / Xiang | Obsolete | 有數字 | 仍可查詢，但**不應視為可用** |

**⇒ 我們的 ansatz 只需要 `ry/rz/cx`；但「閘集支援」不等於「有權限」。**

★ **實測（2026-09-19）：Baihua 送不出去。** 閘集查詢、狀態查詢都正常（`Online`、119 qubit），
但 `send()` 被伺服器擋下：

```
quafu.exceptions.user_error.UserError: 'Sorry, you do not have permission to use this chip.'
```

⇒ **API 列得出來 ≠ 能送任務**（這正是使用者「網頁上沒看到 Baihua」的原因）。
這個拒絕**不消耗額度**（伺服器端就擋掉），所以「送一次看錯誤訊息」是**免費**的權限探測法。
**目前確認唯一可用的真機是 ScQ-P5。**

### 8.5 API 事實（pyquafu 0.4.5 實測）

- `User(api_token=tok)` → `get_available_backends()`：回 `{名稱: Backend}`
- `Backend`：`name` / `qubit_num` / `qv` / `status` / `task_in_queue` / `get_valid_gates()`
  （`get_chip_info()` 是本地方法，內部再建 `User()` 所以會失敗；耦合圖要用別的路徑）
- `Task(user=u)` → `.config(backend=..., shots=..., compile=True)` → `.run(qc)`
- ★ **`run()` 是同步的**：原始碼 `run(qc)` = `send(qc, wait=True)`，回傳的 `ExecResult`
  **已經帶著結果**（`.counts` / `.probabilities` / `.logicalq_res`），不需要再輪詢。
- ★ **task id 在 `ExecResult.taskid`，不在 `Task` 上**：`Task` 實例**沒有** `taskid`
  屬性（`t.taskid` 取不到）；正確位置是 `ExecResult.taskid`（例：`8E04BDA01432EDCF`）。
  用錯位置會變成 `retrieve(None)` → 伺服器回錯誤 JSON → `ExecResult` 建構時
  `KeyError: 'task_id'`，看起來像「任務沒成立」，其實結果早就在手上。
- **長時間排隊的正確做法：非同步送單**
  `res = t.send(qc, wait=False)` → 立刻拿到 `res.taskid`（**先存檔再等待**）→
  之後 `t.retrieve(tid)` 輪詢。就算連線斷了，任務也不會丟。
- `Task.retrieve(taskid: str)` 是**查詢**（**不耗額度**）；`Task.get_history()` 目前回 `{}`
- **權限不足會拋 `UserError`，在 `send()` 階段就被伺服器擋下，且不消耗額度**；
  額度只在**任務真的成立**時才計。（Baihua 就是這樣被擋的，見 §8.4）
- `compile=True` 會做映射編譯：`transpiled_openqasm` 可能配置**比邏輯位更多的實體位**
  （`qbn5_book` 的 5 個邏輯位在 Sim10 上被放到 `qreg q[10]`），並附
  `measures = {實體位: clbit}`；回傳的 `counts` 鍵是 **creg 順序**的字串。

### 8.6 ScQ-Sim10：第一個成功送出的任務（2026-09-19）

| 項目 | 值 |
|---|---|
| taskid | `8E04BDA01432EDCF`（2 000 shots）、`8E05353008B850AF`（20 000 shots） |
| 後端 / shots | ScQ-Sim10 / 2 000 與 20 000 |
| 電路 | `qbn5_book`（5 qubit、20 閘） |
| 回傳鍵數 | 25（2 000 shots）／29（20 000 shots） |

**對帳黃金向量**（`max|ΔP|`）：

| 位元序處理 | 2 000 shots | 20 000 shots |
|---|---|---|
| **反轉一次（映到我們的索引）** | **6.04e-03** | **1.76e-03** ✅ |
| 不反轉 | 5.78e-02 | 5.71e-02 |

- top-3 狀態與黃金向量**完全一致**：golden `[0, 17, 29]` ↔ quafu `[0, 17, 29]`。
- 這同時**再確認了大端序公約**（§7.4）：只有「反轉一次」的映射才對得上。

#### 8.6.1 一個被實驗否證的假設（方法論示範）

2 000 shots 那次跑完，$\chi^2 = 55.5/\mathrm{df}{=}31$（$p \approx 7\times10^{-4}$）、
32 個態裡有 2 個 $|z| > 3$——**單看這個結果，不能宣稱「只是取樣漲落」**。
其中 33.7 的 $\chi^2$ 來自 3 個期望值 $< 1.5$ 的罕見態（`idx14` 期望 0.33、觀測 3），
這個模式（罕見態系統性偏多）看起來像**噪聲底**。

於是提出可否證的預測，並用 10 倍 shots 檢驗：

| shots | max\|ΔP\| | max\|z\| | $n(\|z\|>3)$ | $\chi^2/\mathrm{df}$ |
|---|---|---|---|---|
| 2 000 | 6.04e-03 | 4.66 | 2 | 55.5 / 31 |
| **20 000** | **1.76e-03** | **2.35** | **0** | **33.5 / 31** |

- 若是**噪聲底**（系統性）：偏差量固定，$z$ 應隨 $\sqrt{N}$ **變大**（約 3.2 倍）。
- 若是**取樣漲落**（統計性）：偏差量應隨 $1/\sqrt{N}$ **變小**（約 3.2 倍）。
- 實測：$6.04\times10^{-3} \to 1.76\times10^{-3}$（縮小 **3.4 倍**，$\sqrt{10} = 3.16$），
  $|z|_{\max}$ 由 4.66 降到 **2.35**（32 個標準常態的期望最大值約 2.2）。

**⇒ 假設被否證：ScQ-Sim10 是理想取樣器，2 000 shots 那次是罕見但正常的統計漲落。**

#### 8.6.2 轉譯也被獨立驗證過

把雲端回傳的 `transpiled_openqasm`（10 個實體位、63 閘、含 SWAP 分解）拿回來，
用 Qiskit 態向量**無噪聲**重算，再依 `measures = {6:0, 9:1, 5:2, 8:3, 7:4}` 邊際化：

| 比對 | max\|ΔP\| |
|---|---|
| 轉譯後電路（無噪聲）vs 黃金向量 | **8.88e-16** ✅ |
| 同一結果不反轉 | 5.58e-02 |

⇒ `compile=True` 的映射編譯是**忠實**的；§8.6.1 的偏差不可能來自轉譯。

- 證據檔：`hardware/runs/sim10_qbn5_book_2000.json`、`hardware/runs/sim10_qbn5_book_20000.json`

### 8.7 目前狀態

- ✅ 本地鏈路：IR → QASM → quafu 模擬 → 對帳（≤4.4e-16）
- ✅ token 生效、16 個後端可列、閘集可查
- ✅ **第一個任務成功送出並回收**（ScQ-Sim10，見 §8.6）
- 🔄 ScQ-P5 真機已送出（taskid 存於 `hardware/runs/scqp5_qbn5_book.taskid`），等待佇列
