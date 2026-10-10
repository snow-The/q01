"""架構守衛：套件的**模組層**不得 import 第三方套件（numpy 除外）。

q01 的核心承諾是「只裝 numpy 就能跑」（README：不需要 WSL、不需要 CUDA、
不需要安裝 CUDA-Q）。它靠兩件事維持：

  1. 模組層只 import numpy 與標準函式庫；
  2. qiskit / cirq / pennylane / torch 一律在**函式內延遲 import**，
     並在缺套件時丟出附有修法的錯誤（見 `hardware._provider`）。

第 2 點過去只是慣例、沒有守衛：任何一次把延遲 import 提到模組層的編輯，
都會讓 `import q01` 直接失敗——而且是在**只有 numpy 的環境**才看得出來，
開發機通常裝了全部 extras，所以不會發現。本測試把這個慣例變成契約。
"""
from __future__ import annotations

import ast
import pathlib
import sys

PKG = pathlib.Path(__file__).resolve().parent.parent / "src" / "q01"

#: 允許出現在模組層的第三方套件
ALLOWED_THIRD_PARTY = {"numpy"}

#: 套件自己的名字（絕對自 import 不算是外部依賴）
SELF = {"q01"}

STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def _module_level_imports(tree: ast.Module):
    """只走訪 ast.Module.body —— 函式內的 import 不會出現在這裡。"""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相對 import（from . import x）＝套件內部
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


def test_no_module_level_third_party_imports():
    """掃描每一個模組的模組層 import，回報所有違規（不是只報第一個）。"""
    offenders: list[str] = []
    scanned = 0

    for path in sorted(PKG.rglob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for top, lineno in _module_level_imports(tree):
            if top in ALLOWED_THIRD_PARTY or top in SELF or top in STDLIB:
                continue
            rel = path.relative_to(PKG)
            offenders.append(f"{rel}:{lineno} 模組層 import 了 {top!r}")

    assert scanned > 0, f"掃不到任何模組，路徑錯了：{PKG}"
    assert not offenders, (
        "模組層出現第三方 import —— 這會讓「只裝 numpy 就能跑」的承諾失效。\n"
        "請把它移進函式內，並在缺套件時丟出附有修法說明的錯誤：\n  "
        + "\n  ".join(offenders)
    )


def test_q01_imports_without_optional_extras():
    """`import q01` 本身不得觸發任何選配依賴。

    這裡不模擬「沒裝 qiskit」的環境（開發機可能裝了），而是直接斷言
    import 後 q01 已載入的子模組裡，沒有把選配套件拉進 sys.modules 的模組層引用
    ——真正的保證由上一條測試提供，本條確保 import 路徑本身是乾淨的。
    """
    import importlib

    for name in [m for m in list(sys.modules) if m == "q01" or m.startswith("q01.")]:
        del sys.modules[name]

    importlib.import_module("q01")

    leaky = [
        name
        for name in ("qiskit", "cirq", "pennylane", "torch", "qiskit_quantuminspire")
        if name in sys.modules
    ]
    assert not leaky, (
        "`import q01` 把選配套件拉進了 sys.modules："
        + ", ".join(leaky)
        + "\n選配依賴必須在真正用到時才 import。"
    )
