"""Pauli 代數與 `observe`（期望值）。

契約（SPEC §3.2）：**預設是決定性的**——直接算 ⟨ψ|P|ψ⟩，不是取樣估計。
只有 `shots_count>0` 才有 shot noise，而 v1 只支援對角的 Z 乘積。
"""

from __future__ import annotations

import numpy as np

from .simulator import Simulator
from .targets import make_rng
from . import bitorder

__all__ = ["SpinOperator", "SpinOps", "spin", "observe", "ObserveResult"]

_MUL = {
    ("x", "y"): (1j, "z"), ("y", "x"): (-1j, "z"),
    ("y", "z"): (1j, "x"), ("z", "y"): (-1j, "x"),
    ("z", "x"): (1j, "y"), ("x", "z"): (-1j, "y"),
}


class SpinOperator:
    """Pauli 字串的線性和；`spin.z(0) * spin.z(1)` 回傳乘積。"""

    def __init__(self, terms: dict[tuple[tuple[int, str], ...], complex] | None = None) -> None:
        self.terms: dict[tuple[tuple[int, str], ...], complex] = dict(terms or {})

    @staticmethod
    def single(qubit: int, which: str) -> "SpinOperator":
        return SpinOperator({((int(qubit), which),): 1.0 + 0.0j})

    def _label(self) -> str:
        parts = []
        for term, coef in self.terms.items():
            ops = "".join(f"{which.upper()}{q}" for q, which in sorted(term))
            parts.append(f"({coef}) * {ops or 'I'}")
        return " + ".join(parts) if parts else "(0j) * I"

    def __repr__(self) -> str:
        return self._label()

    def __mul__(self, other: "SpinOperator") -> "SpinOperator":
        out: dict[tuple[tuple[int, str], ...], complex] = {}
        for ta, ca in self.terms.items():
            for tb, cb in other.terms.items():
                coef = ca * cb
                merged = dict(ta)
                identity = False
                for q, which in tb:
                    if q in merged:
                        old = merged[q]
                        if old == which:
                            del merged[q]                 # P·P = I
                            continue
                        factor, new = _MUL[(old, which)]
                        coef *= factor
                        merged[q] = new
                    else:
                        merged[q] = which
                key = tuple(sorted(merged.items()))
                out[key] = out.get(key, 0.0) + coef
        return SpinOperator({k: v for k, v in out.items() if v != 0})

    def is_diagonal_z(self) -> bool:
        return all(all(which == "z" for _, which in term) for term in self.terms)


class SpinOps:
    """`cudaq.spin` —— `spin.z(0)`、`spin.x(i)`、`spin.y(i)`。"""

    def z(self, qubit: int) -> SpinOperator:
        return SpinOperator.single(qubit, "z")

    def x(self, qubit: int) -> SpinOperator:
        return SpinOperator.single(qubit, "x")

    def y(self, qubit: int) -> SpinOperator:
        return SpinOperator.single(qubit, "y")


spin = SpinOps()


class ObserveResult:
    def __init__(self, value: complex, shots: int | None = None) -> None:
        self._value = complex(value)
        self.shots = shots

    def expectation(self, *args, **kwargs) -> complex:
        return self._value

    def __repr__(self) -> str:  # pragma: no cover
        return f"<q01.ObserveResult expectation={self._value}>"


def observe(kernel, spin_op: SpinOperator, *args, shots_count: int | None = None,
            seed: int | None = None, **kwargs) -> ObserveResult:
    if spin_op is None:
        raise TypeError("observe 需要一個 spin 運算子，例如 cudaq.spin.z(0)")
    tr = kernel.trace(*args, live=True, rng=make_rng(seed), **kwargs)
    n = tr.n_qubits
    psi = np.asarray(tr.live.psi)

    if shots_count not in (None, -1, 0):
        shots = int(shots_count)
        if not spin_op.is_diagonal_z():
            raise NotImplementedError(
                "q01 v1 的 shots_count>0 只支援對角的 Z 乘積（例如 spin.z(0)*spin.z(1)）。"
                "含 X／Y 的取樣期望值請用真 CUDA-Q；或不給 shots_count（預設為精確值）。"
            )
        probs = np.abs(psi) ** 2
        draws = make_rng(seed).choice(n and (1 << n), size=shots, p=probs / probs.sum())
        total = 0.0 + 0.0j
        for term, coef in spin_op.terms.items():
            signs = np.ones(1 << n)
            idx = np.arange(1 << n)
            for q, _ in term:
                signs *= 1 - 2 * ((idx >> (n - 1 - q)) & 1)
            total += coef * float(signs[draws].mean())
        return ObserveResult(total, shots)

    total = 0.0 + 0.0j
    for term, coef in spin_op.terms.items():
        sim = Simulator(n)
        sim.psi = psi.copy()
        total += coef * sim.expectation_pauli(term)
    return ObserveResult(total, None)
