"""張量網路示範：把古典貝氏網路寫成具名指標的張量網路，並逐步縮併。

對應章節：docs/02-QBN理論/張量網路與電路對應.md
執行方式：uv run python dev/tensor_network_demo.py

設計原則（本書的實作紀律）：
  1. 用「具名指標」而不是軸的位置來思考——這是避免縮併錯誤的唯一可靠方法。
  2. 每一步都印出形狀，讓讀者能對照數學式。
  3. 一定要有「暴力法」對照組，因為交叉驗證勝過自信。
"""

from __future__ import annotations

import itertools
from itertools import product

import numpy as np

# ---------------------------------------------------------------------------
# 一、帶具名指標的張量
# ---------------------------------------------------------------------------


class Tensor:
    """帶有具名指標的張量。

    指標（indices）是字串元組，長度必須等於資料的維度。
    例如 P(x2 | x1) 是 Tensor(data, ("x1", "x2"))。
    """

    __slots__ = ("data", "indices")

    def __init__(self, data: np.ndarray, indices: tuple[str, ...]):
        data = np.asarray(data, dtype=float)
        if data.ndim != len(indices):
            raise ValueError(f"維度不符：data.ndim={data.ndim}, indices={indices}")
        self.data = data
        self.indices = tuple(indices)

    def rename(self, mapping: dict[str, str]) -> "Tensor":
        """把指標改名（不改變資料），用於整理縮併後的指標名稱。"""
        return Tensor(self.data, tuple(mapping.get(i, i) for i in self.indices))

    def normalized(self) -> "Tensor":
        """去掉指標名稱中的 '#k' 後綴，讓同一條超邊的多個副本同名。"""
        return self.rename({i: i.split("#")[0] for i in self.indices})

    def __repr__(self) -> str:
        return f"Tensor(indices={self.indices}, shape={self.data.shape})"

    # -- 內部：把指標名稱映射成 einsum 的字母 --------------------------------
    def _letters(self, union: list[str]) -> dict[str, str]:
        return {name: chr(ord("a") + i) for i, name in enumerate(union)}

    def contract(self, other: "Tensor", keep: tuple[str, ...] = ()) -> "Tensor":
        """縮併兩個張量的共同指標（愛因斯坦求和約定），回傳新張量。

        參數
        ----
        keep : 指定要**保留（不求和不縮併）**的共同指標。用於需要保留聯合分布的場合
               （例如求 P(x1,x2,x3,x4) 時，x1_a 與 x1_b 要留著才能取對角）。

        預設行為：所有共同指標都被求和掉（標準張量縮併）。
        回傳的指標順序為「self 獨有 + other 獨有 + keep」，不保證有語意。
        """
        actual_keep = tuple(i for i in keep if i in self.indices and i in other.indices)
        contract_idx = [i for i in self.indices if i in other.indices and i not in actual_keep]
        only_self = [i for i in self.indices if i not in other.indices]
        only_other = [i for i in other.indices if i not in self.indices]
        if not contract_idx and not actual_keep:
            raise ValueError(
                f"沒有共同指標可縮併：{self.indices} vs {other.indices}"
            )

        union: list[str] = []
        for name in list(self.indices) + list(other.indices):
            if name not in union:
                union.append(name)
        letters = {name: chr(ord("a") + i) for i, name in enumerate(union)}

        lhs_a = "".join(letters[i] for i in self.indices)
        lhs_b = "".join(letters[i] for i in other.indices)
        rhs = "".join(letters[i] for i in only_self + only_other + list(actual_keep))

        result = np.einsum(f"{lhs_a},{lhs_b}->{rhs}", self.data, other.data,
                           optimize=True)
        return Tensor(result, tuple(only_self + only_other + list(actual_keep)))

    def marginalize(self, keep: tuple[str, ...]) -> "Tensor":
        """對不在 keep 中的指標求和（＝邊際化 / 部分跡的古典對應）。"""
        for name in keep:
            if name not in self.indices:
                raise ValueError(f"指標 {name} 不存在於 {self.indices}")
        union = list(self.indices)
        letters = self._letters(union)
        lhs = "".join(letters[i] for i in self.indices)
        rhs = "".join(letters[i] for i in keep)
        return Tensor(np.einsum(f"{lhs}->{rhs}", self.data, optimize=True), keep)


def copy_tensor(
    name: str,
    child_names: tuple[str, ...] | None = None,
    dim: int = 2,
) -> Tensor:
    """建立 Copy 張量（對角張量）：所有下標必須相等。

    參數
    ----
    name        : 父節點側的指標名稱（例如 "x1"）
    child_names : 子節點側的指標名稱。**這些名字必須與子節點張量裡的指標一致**
                  （例如 B 用 "x1_a"、C 用 "x1_b"）。預設為 name#0, name#1, ...
    dim         : 每個軸的維度

    為什麼子節點側要另取名字？因為 einsum 的規則是「同一張量裡一個字母只能用一次」。
    若三個軸都叫 x1，np.einsum('a,aaa->') 會把整個張量縮成純量，語意就毀了。
    Copy 在本質上是**超邊（hyperedge）**：一條邊連接三個以上的端點，
    而 einsum 只能表達普通的邊，所以我們用「父側一個名字 + 子側各自的名字」來模擬，
    再靠這個對角張量把「所有子側的值必須等於父側」這個約束編碼進去。
    """
    if child_names is None:
        child_names = tuple(f"{name}#{k}" for k in range(len(child_names or ())))
    n_copies = len(child_names)
    shape = (dim,) * (n_copies + 1)
    data = np.zeros(shape)
    for v in range(dim):
        data[(v,) * (n_copies + 1)] = 1.0
    return Tensor(data, (name,) + tuple(child_names))


def random_cpt(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """產生合法的條件機率表：對最後一個軸（own index）歸一化。"""
    t = rng.random(shape) + 0.1  # 避免 0 造成 log(0)
    return t / t.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 二、4 節點圖：X1 → X2, X1 → X3, (X2,X3) → X4
# ---------------------------------------------------------------------------


def build_network(seed: int = 42):
    rng = np.random.default_rng(seed)
    A = Tensor(random_cpt((2,), rng), ("x1",))                # P(x1)
    B = Tensor(random_cpt((2, 2), rng), ("x1_a", "x2"))       # P(x2 | x1)
    C = Tensor(random_cpt((2, 2), rng), ("x1_b", "x3"))       # P(x3 | x1)
    D = Tensor(random_cpt((2, 2, 2), rng), ("x2", "x3", "x4"))  # P(x4 | x2, x3)
    return A, B, C, D


def joint_by_contraction(A, B, C, D, use_copy: bool = True) -> Tensor:
    """逐步縮併求聯合分布 P(x1, x2, x3, x4)。

    use_copy=True  → 插入 Copy 張量（正確做法）
    use_copy=False → 讓 x1 重複出現（錯誤示範，用於對照實驗）
    """
    if use_copy:
        # Copy 把 x1 的值送給 B（用 x1_a）與 C（用 x1_b）兩個子節點
        cp = copy_tensor("x1", ("x1_a", "x1_b"))
        t = A.contract(cp)          # [x1_a, x1_b]
        t = t.contract(B)           # 縮掉 x1_a → [x1_b, x2]
        t = t.contract(C)           # 縮掉 x1_b → [x2, x3]
        return t.contract(D)        # 縮掉 x2, x3 → [x4]
    # 錯誤示範：直接串接會讓 x1 被求和掉（見章節的對照實驗）
    t = A.contract(B)               # x1 被求和 → 失去與 C 的關聯
    t = t.contract(C)               # C 的 x1_b 找不到對應指標 → 失敗
    return t.contract(D)


def joint_bruteforce(A, B, C, D) -> np.ndarray:
    """暴力法：對 16 種組合逐一計算乘積。作為交叉驗證的黃金標準。"""
    out = np.zeros((2, 2, 2, 2))
    for x1, x2, x3, x4 in product(range(2), repeat=4):
        out[x1, x2, x3, x4] = (
            A.data[x1] * B.data[x1, x2] * C.data[x1, x3] * D.data[x2, x3, x4]
        )
    return out


def main() -> None:
    print("=" * 72)
    print("張量網路示範：4 節點貝氏網路 X1 → X2, X1 → X3, (X2,X3) → X4")
    print("=" * 72)

    A, B, C, D = build_network()
    print("\n--- 各節點的條件機率表 ---")
    print(f"A = P(x1)        : {np.round(A.data, 4)}")
    print(f"B = P(x2|x1)     :\n{np.round(B.data, 4)}")
    print(f"C = P(x3|x1)     :\n{np.round(C.data, 4)}")
    print(f"D = P(x4|x2,x3)  :\n{np.round(D.data.reshape(4, 2), 4)}  （列索引 = 2*x2 + x3）")

    # --- 逐步縮併 ---
    print("\n--- 逐步縮併（含 Copy 張量）---")
    cp = copy_tensor("x1", ("x1_a", "x1_b"))
    print(f"Copy 張量        : {cp}，非零元素 {int((cp.data != 0).sum())} 個")
    t = A.contract(cp)
    print(f"A ∘ Copy         : {t}")
    t = t.contract(B)
    print(f"(A∘Copy) ∘ B     : {t}")
    t = t.contract(C)
    print(f"... ∘ C          : {t}")
    joint = t.contract(D)
    print(f"... ∘ D          : {joint}  ← P(x4)（x1,x2,x3 已被縮併掉）")

    # --- 聯合分布：用暴力法（已由上面的縮併路徑交叉驗證 P(x4)） ---
    # 為什麼不繼續用張量縮併算聯合分布？
    #   einsum 的規則是「同一張量裡一個字母只能用一次」。要保留 Copy 的兩條同名軸，
    #   需要「同一軸出現兩次」的語法，而 einsum 只在最後取對角時支援
    #   （np.einsum(t, [0,0,1,2,3], [1,2,3,4])），中間過程無法保留同名軸。
    #   這正是 Copy 是**超邊（hyperedge）**而非普通邊的具體表現。
    #   實務結論：逐步縮併用來驗證機制，聯合分布用暴力法（$2^4=16$ 種組合）最可靠。
    print("\n--- 聯合分布（暴力法）---")
    jdata = joint_bruteforce(A, B, C, D)
    print(f"jdata shape = {jdata.shape}（軸順序：x1, x2, x3, x4）")

    # --- 交叉驗證：縮併路徑與暴力法必須一致 ---
    print("\n--- 交叉驗證（縮併 vs 暴力法）---")
    print(f"聯合分布總和（應為 1）: {jdata.sum():.12f}")
    # 逐步縮併得到的 P(x4) 必須等於聯合分布對 x1,x2,x3 的邊際
    delta = np.abs(joint.data - jdata.sum(axis=(0, 1, 2))).max()
    print(f"P(x4) 逐步縮併 vs 聯合分布邊際，最大絕對誤差: {delta:.3e}")
    assert delta < 1e-12, "縮併路徑與暴力法不一致！"

    # --- 邊際化 ---
    print("\n--- 邊際化（＝古典的求和）---")
    P_x4 = jdata.sum(axis=(0, 1, 2))
    P_x1 = jdata.sum(axis=(1, 2, 3))
    print(f"P(x4) = {np.round(P_x4, 6)}  總和 {P_x4.sum():.12f}")
    print(f"P(x1) = {np.round(P_x1, 6)}  總和 {P_x1.sum():.12f}")
    print(f"對照：A = {np.round(A.data, 6)}（P(x1) 應等於 A，因為 x1 是根節點）")
    assert np.abs(P_x1 - A.data).max() < 1e-12

    # --- 錯誤示範：忘記 Copy 張量 ---
    print("\n--- 對照實驗：忘記 Copy 張量會發生什麼 ---")
    try:
        wrong = joint_by_contraction(A, B, C, D, use_copy=False)
        print(f"結果指標 = {wrong.indices}")
        print("→ 注意：x1 已經被求和掉，而且得到的是錯誤的張量結構。")
        print("   這正是為什麼古典圖模型『必須』有 Copy 張量：")
        print("   沒有它，你表達的是『x1 被邊際化掉』而不是『x1 同時影響兩個子節點』。")
    except ValueError as exc:
        print(f"縮併直接失敗：{exc}")
        print("→ 這也是重點：少一個 Copy 張量，你的張量網路在維度上就不一致了。")

    # --- 條件分布：從聯合分布反推 CPT ---
    print("\n--- 從聯合分布反推條件分布 P(x2 | x1) ---")
    P_x1x2 = jdata.sum(axis=(2, 3))
    cond = P_x1x2 / P_x1x2.sum(axis=1, keepdims=True)
    print(f"從聯合分布算出的 P(x2|x1):\n{np.round(cond, 6)}")
    print(f"原始 CPT B              :\n{np.round(B.data, 6)}")
    print(f"最大絕對誤差            : {np.abs(cond - B.data).max():.3e}")

    print("\n" + "=" * 72)
    print("結論：古典貝氏網路 = 一張張量網路；Copy 張量是多子節點的必要條件。")
    print("      量子版本無法使用 Copy 張量（不可複製定理），必須改用 splitter。")
    print("=" * 72)


if __name__ == "__main__":
    main()
