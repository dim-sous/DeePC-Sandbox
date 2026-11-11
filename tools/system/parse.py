"""Typed rows to the functions of a system x' = f(x, u, d, θ), y = h(x, θ).

One row per state equation (row i is x_i') and one per output equation (row j is y_j), each an
expression in the states ``x1 … xn``, the inputs ``u1 … um``, the parameters ``theta1 … thetaq``
and the functions in ``FUNCTIONS``; a state row may also name the disturbance inputs ``d1 … dr``,
where the unmodelled disturbance enters (the developer, 2026-11-11); θ's values come with the
rows. Each row is checked token by
token before SymPy reads it, because ``sympy.parsing.sympy_parser.parse_expr`` evaluates Python:
a name outside the list, a ``.`` outside a number, or any other character is refused in words and
never reaches the parser. Every number is read as a float, so SymPy computes nothing exactly (an
exact ``9^9^9`` would not finish), and a row holding a constant too large for a float, or not
real (``sqrt(-1)``), is refused. The sizes: n = the number of state rows, p = the number of output
rows, q = the number of θ values, m = the highest u index typed, r = the highest d index typed
(0 for none); every u1…um and d1…dr must act in f, and every θ value in some row, as SymPy reads
them (``0*u1`` reads as nothing, so it does not act; a θ no equation uses is a setting the system
would not use).
"""

from __future__ import annotations

import dataclasses
import math
import re
import sys
import tokenize
from typing import Callable, Sequence

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

__all__ = ["FUNCTIONS", "Parsed", "parse"]

#: The function names (and the constant pi) a typed expression may use, as SymPy objects.
FUNCTIONS: dict[str, object] = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
                                "log": sp.log, "sqrt": sp.sqrt, "tanh": sp.tanh, "abs": sp.Abs,
                                "pi": sp.pi}
_CHARS = re.compile(r"[0-9A-Za-z_+\-*/^().\s]*")
# a number not glued to a name: 2, 0.5, .5, 5., 1e-3
_NUMBER = re.compile(r"(?<![A-Za-z_\d.])(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
_NAME = re.compile(r"[A-Za-z_]\w*")
_VARIABLE = re.compile(r"(x|u|d|theta)([1-9]\d*)")
# what parse_expr's transformations write (auto_number, auto_symbol); no builtins
_GLOBALS = {"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
            "Symbol": sp.Symbol, "__builtins__": {}}
_TRANSFORMS = standard_transformations + (convert_xor,)      # ^ is a power, as typed maths reads


@dataclasses.dataclass(frozen=True)
class Parsed:
    """A parsed system: its sizes, its rows as SymPy reads them (θ kept as symbols), its θ
    values, and NumPy functions ``f(x, u, d) -> (n,)`` and ``h(x) -> (p,)`` at those values; ``d``
    is (r,), empty for a system that names no disturbance."""

    n: int
    m: int
    p: int
    q: int
    r: int
    f_read: tuple[str, ...]
    h_read: tuple[str, ...]
    theta: tuple[float, ...]
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]


def _rows(rows: Sequence[str], label: str) -> list[str]:
    out = [row.strip() for row in rows]
    for i, row in enumerate(out, 1):
        if not row:
            raise ValueError(f"{label.format(i)} is empty")
    return out


def _checked(text: str, where: str) -> set[str]:
    """The variable names row ``where`` uses, or a refusal in words."""
    where = f"{where} = {text!r}"
    if not _CHARS.fullmatch(text):
        bad = sorted(set(text) - set("0123456789+-*/^(). _") - {c for c in text if c.isalnum()})
        raise ValueError(f"{where} holds {bad}: only numbers, x1…, u1…, d1…, theta1…, the "
                         f"functions {sorted(FUNCTIONS)}, + - * / ^ ** and parentheses are read")
    bare = _NUMBER.sub(" ", text)
    if "." in bare:
        raise ValueError(f"{where} has a '.' outside a number")
    names = set(_NAME.findall(bare))
    unknown = sorted(n for n in names if n not in FUNCTIONS and not _VARIABLE.fullmatch(n))
    if unknown:
        raise ValueError(f"{where} names {unknown}: the names read are x1…, u1…, d1…, theta1… "
                         f"and {sorted(FUNCTIONS)}")
    return {n for n in names if _VARIABLE.fullmatch(n)}


def _as_floats(text: str, where: str) -> str:
    """``text`` with every number written as a float literal."""
    def literal(match: re.Match) -> str:
        value = float(match.group(0))
        if not math.isfinite(value):
            raise ValueError(f"{where} holds the number {match.group(0)}, which is not finite")
        return repr(value)
    return _NUMBER.sub(literal, text)


def _indices(names: set[str], kind: str) -> set[int]:
    return {int(m.group(2)) for n in names if (m := _VARIABLE.fullmatch(n)) and m.group(1) == kind}


def _union(sets: list[set[str]], kind: str) -> set[int]:
    return set().union(*(_indices(s, kind) for s in sets))


def parse(f_rows: Sequence[str], h_rows: Sequence[str], theta: Sequence[float]) -> Parsed:
    """The state rows ``f_rows`` (x1', x2', …), the output rows ``h_rows`` (y1, y2, …) and the
    values ``theta`` (theta1, theta2, …) to a :class:`Parsed`; anything the module refuses raises
    ``ValueError`` naming the row and what is wrong with it."""
    f_parts, h_parts = _rows(f_rows, "x{}'"), _rows(h_rows, "y{}")
    if not f_parts:
        raise ValueError("no state row: x1' = … is needed")
    if not h_parts:
        raise ValueError("no output row: y1 = … is needed")
    theta = tuple(float(v) for v in theta)
    for k, value in enumerate(theta, 1):
        if not math.isfinite(value):
            raise ValueError(f"θ{k} = {value} is not finite")
    f_names = [_checked(t, f"x{i}'") for i, t in enumerate(f_parts, 1)]
    h_names = [_checked(t, f"y{j}") for j, t in enumerate(h_parts, 1)]
    n, p, q = len(f_parts), len(h_parts), len(theta)
    used_x = _union(f_names + h_names, "x")
    if used_x and max(used_x) > n:
        raise ValueError(f"x{max(used_x)} is named, but there are {n} state row(s), so the states "
                         f"are x1…x{n}")
    used_theta = _union(f_names + h_names, "theta")
    if used_theta and max(used_theta) > q:
        raise ValueError(f"theta{max(used_theta)} is named, but θ holds {q} value(s)")
    if _union(h_names, "u"):
        raise ValueError("an output row names an input u: the outputs are y = h(x, θ)")
    if _union(h_names, "d"):
        raise ValueError("an output row names a disturbance d: the outputs are y = h(x, θ); a "
                         "disturbance enters through a state row")
    typed_u = _union(f_names, "u")
    if not typed_u:
        raise ValueError("no state row names an input u1…: the system has nothing to control it by")
    m = max(typed_u)
    r = max(_union(f_names, "d"), default=0)
    xs, us = sp.symbols(f"x1:{n + 1}"), sp.symbols(f"u1:{m + 1}")
    ds = sp.symbols(f"d1:{r + 1}") if r else ()
    thetas = sp.symbols(f"theta1:{q + 1}") if q else ()
    local = {**FUNCTIONS, **{str(s): s for s in (*xs, *us, *ds, *thetas)}}

    def read(text: str, where: str) -> sp.Expr:
        where = f"{where} = {text!r}"
        try:
            expr = parse_expr(_as_floats(text, where), local_dict=dict(local),
                              global_dict=dict(_GLOBALS), transformations=_TRANSFORMS)
        # what parse_expr raises on text that is not an expression: unbalanced parentheses
        # (TokenError), a misplaced operator (SyntaxError), a function called wrongly (TypeError),
        # a division by a zero number (ZeroDivisionError)
        except (tokenize.TokenError, SyntaxError, TypeError, ZeroDivisionError) as exc:
            raise ValueError(f"{where} is not an expression ({type(exc).__name__}: {exc})") from exc
        if not isinstance(expr, sp.Expr):          # "()" reads as a Python tuple
            raise ValueError(f"{where} is not an expression (it reads as {expr!r})")
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError(f"{where} reads as {expr}, which is not finite")
        if expr.has(sp.I):
            raise ValueError(f"{where} reads as {expr}, which is not real")
        if any(abs(c) > sys.float_info.max for c in expr.atoms(sp.Float)):
            raise ValueError(f"{where} holds a constant too large for a float")
        return expr

    f_exprs = [read(t, f"x{i}'") for i, t in enumerate(f_parts, 1)]
    h_exprs = [read(t, f"y{j}") for j, t in enumerate(h_parts, 1)]
    read_names = {str(s) for e in f_exprs + h_exprs for s in e.free_symbols}
    unused = sorted(set(range(1, q + 1)) - _indices(read_names, "theta"))
    if unused:
        raise ValueError(f"θ{unused[0]} = {theta[unused[0] - 1]} acts in no row as read: a value "
                         f"no equation uses")
    in_f = {str(s) for e in f_exprs for s in e.free_symbols}
    for kind, size, what in (("u", m, "inputs"), ("d", r, "disturbances")):
        silent = sorted(set(range(1, size + 1)) - _indices(in_f, kind))
        if silent:
            raise ValueError(f"{kind}{silent[0]} acts in no state row as read "
                             f"({'; '.join(map(str, f_exprs))}), but {kind}{size} is typed: the "
                             f"{what} are {kind}1…{kind}{size}, each acting through a state row")
    f_raw = sp.lambdify((xs, us, ds, thetas), f_exprs, "numpy")
    h_raw = sp.lambdify((xs, thetas), h_exprs, "numpy")
    return Parsed(n=n, m=m, p=p, q=q, r=r, f_read=tuple(map(str, f_exprs)),
                  h_read=tuple(map(str, h_exprs)), theta=theta,
                  f=lambda x, u, d: np.array(f_raw(x, u, d, theta), dtype=float),
                  h=lambda x: np.array(h_raw(x, theta), dtype=float))
