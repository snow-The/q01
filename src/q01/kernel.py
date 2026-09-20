"""`@kernel` 裝飾器、追蹤式執行、量測、adjoint／control。

與 CUDA-Q 的刻意對齊（SPEC §4.4）：
- kernel **必須寫在檔案裡**（因為要取原始碼；CUDA-Q 的 AST 編譯器有同樣限制）。
- 閘名（`h`、`ry`…）由 q01 注入 kernel 的名稱空間，學生不必 import——與 CUDA-Q 相同。
- **動態 qubit 數允許**（`qvector(n)` 的 n 可以是執行期參數）。
- **不加** `strict` 旋鈕（真 `@cudaq.kernel` 沒有這個參數）；靜態檢查用獨立的 `check_kernel()`。
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np

from .circuit import NoTraceError, Op, QubitRef, QVector, Trace, current_trace, trace_context
from .gates import GATES, split_qubits
from .simulator import Simulator

__all__ = ["Kernel", "kernel", "qvector", "qubit", "mz", "mx", "my", "adjoint", "control",
           "check_kernel", "KernelCompileError", "FORBIDDEN_NODES"]


class KernelCompileError(RuntimeError):
    """kernel 取不到原始碼或語法不合法。"""


# --------------------------------------------------------------------------
# qubit 配置
# --------------------------------------------------------------------------
def qvector(n: int) -> QVector:
    """`cudaq.qvector(n)` —— 配置 n 個 qubit（n 可以是執行期參數）。"""
    return current_trace().allocate(int(n))


def qubit() -> QubitRef:
    """`cudaq.qubit()` —— 配置單一 qubit，直接當 target 用（`ry(theta, q)`）。"""
    return current_trace().allocate(1)[0]


# --------------------------------------------------------------------------
# 量測
# --------------------------------------------------------------------------
def _measure(target, basis: str):
    tr = current_trace()
    qubits = split_qubits(target)
    if basis == "x":
        for q in qubits:
            tr.record("h", (), (q,), ())
    elif basis == "y":
        for q in qubits:
            tr.record("sdg", (), (q,), ())
            tr.record("h", (), (q,), ())
    tr.mark_measured(qubits)
    if tr.live is None:                      # pragma: no cover - 內部一定會用 live trace
        raise RuntimeError("q01 內部的量測需要 live 追蹤模式")
    bits = tr.live.measure_collapse(qubits)
    tr.outcomes.append(tuple(int(b) for b in bits))
    return bool(bits[0]) if len(qubits) == 1 else [bool(b) for b in bits]


def mz(target):
    """計算基底量測（Z 基底）：單一 qubit 回 bool，多個回 list[bool]。"""
    return _measure(target, "z")


def mx(target):
    """X 基量測（先 H）。"""
    return _measure(target, "x")


def my(target):
    """Y 基量測（先 S†H）。"""
    return _measure(target, "y")


# --------------------------------------------------------------------------
# adjoint / control（只支援自訂 kernel，與 CUDA-Q 相同）
# --------------------------------------------------------------------------
_SELF_INVERSE = {"h", "x", "y", "z"}


def _adjoint_op(op: Op) -> Op:
    if op.name in ("rx", "ry", "rz"):
        return Op(op.name, (-op.params[0],), op.qubits, op.controls)
    inverse = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
    return Op(inverse.get(op.name, op.name), op.params, op.qubits, op.controls)


def adjoint(sub, *args) -> None:
    """`cudaq.adjoint(sub, q, theta)` —— 施加子電路的反么正（反序 + 共軛）。"""
    tr = current_trace()
    temp = Trace()
    with trace_context(temp):
        sub(*args)
    for op in reversed(temp.ops):
        a = _adjoint_op(op)
        tr.record(a.name, a.params, a.qubits, a.controls)


def control(sub, ctrls, *args) -> None:
    """`cudaq.control(sub, [c0, c1], q, theta)` —— 受控版本的自訂子電路。"""
    tr = current_trace()
    temp = Trace()
    with trace_context(temp):
        sub(*args)
    cs = split_qubits(ctrls)
    for op in temp.ops:
        tr.record(op.name, op.params, op.qubits, tuple(op.controls) + cs)


# --------------------------------------------------------------------------
# 靜態檢查（照 CUDA-Q 的 AST 白名單：`ast_bridge.py:1913-1920` 是「一律報錯」的反向機制）
# --------------------------------------------------------------------------
FORBIDDEN_NODES: dict[type, str] = {
    ast.Assert: "assert",
    ast.With: "with",
    ast.AsyncWith: "async with",
    ast.Try: "try/except",
    ast.Raise: "raise",
    ast.ClassDef: "class",
    ast.Dict: "dict 字面值",
    ast.Set: "set 字面值",
    ast.JoinedStr: "f-string",
    ast.FormattedValue: "f-string",
    ast.NamedExpr: "walrus（:=）",
    ast.Global: "global",
    ast.Nonlocal: "nonlocal",
    ast.Import: "import（請寫在檔案最上方）",
    ast.ImportFrom: "import（請寫在檔案最上方）",
    ast.Lambda: "lambda",
    ast.Yield: "yield",
    ast.YieldFrom: "yield from",
    ast.Await: "await",
    ast.Delete: "del",
    ast.Starred: "星號展開",
    ast.AsyncFunctionDef: "async def",
}


def check_kernel(fn) -> list[str]:
    """靜態檢查 kernel 是否落在 q01／CUDA-Q 支援的 Python 子集內。

    回傳問題清單（空 = 通過）。這是**獨立函式**，不是 `@kernel` 的參數（SPEC §4.4）。
    """
    target = fn.fn if isinstance(fn, Kernel) else fn
    try:
        src = textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError) as exc:      # pragma: no cover
        return [f"取不到原始碼（kernel 必須寫在檔案裡）：{exc}"]
    tree = ast.parse(src)
    problems: list[str] = []
    for node in ast.walk(tree):
        key = type(node)
        if key in FORBIDDEN_NODES:
            problems.append(f"第 {getattr(node, 'lineno', '?')} 行：不支援 {FORBIDDEN_NODES[key]}")
    return problems


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------
def _kernel_namespace() -> dict:
    ns = {name: g for name, g in GATES.items()}
    ns.update({"qvector": qvector, "qubit": qubit, "mz": mz, "mx": mx, "my": my,
               "adjoint": adjoint, "control": control, "kernel": kernel})
    return ns


class Kernel:
    """包住使用者函式的可呼叫物件；呼叫時進行追蹤（trace）。"""

    def __init__(self, fn) -> None:
        if not callable(fn):
            raise TypeError("@kernel 只能修飾函式")
        self.fn = fn
        self.name = getattr(fn, "__name__", "kernel")
        self.__name__ = self.name
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__wrapped__ = fn
        self._compiled = None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<q01.Kernel {self.name}>"

    def __getattr__(self, item):
        if item in ("adjoint", "ctrl", "control"):
            raise AttributeError(
                f"q01／CUDA-Q 不支援 {self.name}.{item}(...) 這種**方法**寫法。\n"
                f"  真 CUDA-Q 的錯誤是：unhandled function call - {item}\n"
                f"  請改用函式形式：cudaq.{item}({self.name}, q, theta)"
                + ("（受控版本：cudaq.control(sub, [c0, c1], q, theta)）" if item == "control" else "")
            )
        raise AttributeError(f"{self.name!r} 沒有屬性 {item!r}")

    # ---------------------------------------------------------------- 編譯
    def _compile(self):
        if self._compiled is not None:
            return self._compiled
        try:
            src = textwrap.dedent(inspect.getsource(self.fn))
        except (OSError, TypeError) as exc:
            raise KernelCompileError(
                f"@q01.kernel 取不到 {self.name} 的原始碼：kernel 必須寫在檔案裡"
                "（與 CUDA-Q 相同：不能定義在互動式環境或 exec 的字串中）。"
            ) from exc

        tree = ast.parse(src)
        node = next((n for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == self.name), None)
        if node is None:                      # pragma: no cover
            raise KernelCompileError(f"在原始碼中找不到函式 {self.name}")
        node.decorator_list = []
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace = dict(self.fn.__globals__)   # 保留使用者的 import（例如 import q01 as cudaq）
        namespace.update(_kernel_namespace())   # 注入閘名（與 CUDA-Q 相同）
        filename = inspect.getsourcefile(self.fn) or "<q01-kernel>"
        exec(compile(module, filename, "exec"), namespace)
        self._compiled = namespace[self.name]
        return self._compiled

    # ---------------------------------------------------------------- 執行
    def _execute(self, args, kwargs, *, live: bool, rng) -> tuple[Trace, object]:
        fn = self._compile()
        tr = Trace()
        tr.live = Simulator(0, rng) if live else None
        with trace_context(tr):
            result = fn(*args, **kwargs)
        return tr, result

    def __call__(self, *args, **kwargs):
        """在 kernel 內呼叫 → 內聯其閘；在 kernel 外呼叫 → 單獨追蹤並回傳 `Trace`。"""
        try:
            current_trace()
        except NoTraceError:
            tr, _ = self._execute(args, kwargs, live=True, rng=np.random.default_rng())
            return tr
        fn = self._compile()
        return fn(*args, **kwargs)

    def trace(self, *args, live: bool = True, rng=None, **kwargs) -> Trace:
        tr, _ = self._execute(args, kwargs, live=live,
                              rng=rng if rng is not None else np.random.default_rng())
        return tr


def kernel(fn=None, **_ignored):
    """`@cudaq.kernel` 的對應物。注意：**沒有** `strict` 參數（真 CUDA-Q 也沒有）。"""
    if fn is None:
        return lambda f: Kernel(f)
    return Kernel(fn)
