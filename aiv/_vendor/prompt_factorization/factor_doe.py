"""theta_full (YAML)  ->  space-filling design over prompt factors.

Consumes the YAML `theta_full` emitted by the prompt-factorization instructor
and produces a space-filling design over its factors. The generator is the phi_p
optimizer of Sudjianto & Zhang, "GPU-Accelerated Gradient-Based Space-Filling
Design" (JASA), vendored in `frontier_discovery.spacefilling`. Self-contained:
numpy + `frontier_discovery.spacefilling` (torch only for `sobol+refine`), and
pyyaml only for `design_from_yaml`. No knowlytix dependency.

This module lives at the DOE stage of the pipeline, downstream of factorization:

    P -> theta_full (factorizer)  ->  theta_active (designer)  ->  THIS -> runs

The factorizer does not select or design; this module does. It adds the three
things the paper's [0,1]^m formulation does not itself provide for a *prompt*
factor space:

  1. Type-aware decoding. Ordered factors (BINARY / ORDINAL / NESTED_ORDINAL /
     CONTINUOUS) map to the axis monotonically, so phi_p spread is meaningful.
     Unordered factors (CATEGORICAL / STRUCTURED) get a seeded relabeling of the
     axis. This randomizes nominal encoding; it does not remove artificial axis
     distances from the refinement objective.
  2. Full-factor cardinality weighting. The paper's w_j = |L_j| / mean(|L|)
     improves coverage of high-cardinality factors. The vendored `_dim_weights`
     applies it only when every factor is categorical; a mixed prompt space
     silently drops it. We compute the weights across all factors here.
  3. Constraint enforcement. The factorizer emits REQUIRES / EXCLUDES / ACTIVATES
     constraints; the paper's design is unconstrained in the box. We enforce them
     by masking (ACTIVATES -> NOT_APPLICABLE) and rejection (REQUIRES / EXCLUDES).
     Rows are unique by default; backfill replaces rejected or duplicate decoded
     rows. Explicit replication is available for repeated measurements.

Scope: rejection + prefix-trim is not a constrained space-filling optimizer;
where the feasible region is a small fraction of the box, design directly over
it instead (open).
"""

from __future__ import annotations

import json
import math
import re
import warnings
from itertools import product
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .spacefilling import optimize as _phip_optimize

NOT_APPLICABLE = "NOT_APPLICABLE"

# Prompt-factorization type -> ordered-on-the-axis?  (gap 1)
_ORDERED_TYPES = {"BINARY", "ORDINAL", "NESTED_ORDINAL", "CONTINUOUS"}
_UNORDERED_TYPES = {"CATEGORICAL", "STRUCTURED"}


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
@dataclass
class EncodedFactor:
    name: str
    ptype: str                       # the factorizer's declared type
    ordered: bool
    kind: str                        # "discrete" | "continuous"
    levels: list[Any] = field(default_factory=list)   # canonical level order (discrete)
    low: float | None = None
    high: float | None = None
    cardinality: int = 2

    def decode(self, u: float, rng_perm: np.ndarray | None) -> Any:
        if self.kind == "continuous":
            return float(u * (self.high - self.low) + self.low)
        k = self.cardinality
        idx = min(int(u * k), k - 1)
        if not self.ordered and rng_perm is not None:
            idx = int(rng_perm[idx])           # randomize nominal label placement
        return self.levels[idx]


def _levels_in_order(levels_field: Any) -> list[Any]:
    """Return level identifiers in canonical order.

    The factorizer emits `levels` as a mapping {id: description}. For ORDINAL /
    NESTED_ORDINAL the declaration defines the order; for CATEGORICAL the keys
    have no order. We preserve dict insertion / list order.
    """
    if isinstance(levels_field, dict):
        return list(levels_field.keys())
    if isinstance(levels_field, list):
        return list(levels_field)
    raise ValueError(f"unsupported levels field: {levels_field!r}")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def encode_factors(
    factor_inventory: list[dict],
    *,
    continuous_bounds: dict[str, tuple[float, float]] | None = None,
) -> list[EncodedFactor]:
    """Validate a minimal inventory and encode it for design.

    Continuous bounds come from ``bounds: [low, high]`` or an explicit override.
    Ordered levels use declaration order; comparisons refer to level IDs.
    Single-level non-binary factors are permitted for fixed design dimensions.
    """
    continuous_bounds = continuous_bounds or {}
    if not isinstance(factor_inventory, list) or not factor_inventory:
        raise ValueError("empty or malformed factor_inventory: nothing to design over")
    out = []
    seen = set()
    for f in factor_inventory:
        if not isinstance(f, dict) or "id" not in f or "type" not in f:
            raise ValueError(f"factor missing id or type: {f!r}")
        fid, ptype = f["id"], f["type"]
        if not isinstance(fid, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", fid):
            raise ValueError(f"invalid factor id {fid!r}")
        if fid in seen:
            raise ValueError(f"duplicate factor id {fid!r}")
        seen.add(fid)
        if not isinstance(ptype, str) or ptype not in _ORDERED_TYPES | _UNORDERED_TYPES:
            raise ValueError(f"{fid}: unknown type {ptype!r}")
        if ptype == "CONTINUOUS":
            bounds = continuous_bounds.get(fid, f.get("bounds"))
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"{fid}: CONTINUOUS requires bounds [low, high]")
            try:
                low, high = map(float, bounds)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{fid}: invalid continuous bounds") from exc
            if any(isinstance(v, bool) for v in bounds) or not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f"{fid}: bounds must be finite with low < high")
            out.append(EncodedFactor(fid, ptype, True, "continuous", low=low, high=high))
            continue
        if "levels" not in f:
            raise ValueError(f"{fid}: non-continuous factor has no levels")
        levels = _levels_in_order(f["levels"])
        if not levels or any(not isinstance(v, (str, int, float, bool)) or
                             (isinstance(v, float) and not math.isfinite(v)) or
                             v == NOT_APPLICABLE for v in levels):
            raise ValueError(f"{fid}: levels must be nonempty finite scalar identifiers; NOT_APPLICABLE is reserved")
        if len(set(levels)) != len(levels) or len({str(v) for v in levels}) != len(levels):
            raise ValueError(f"{fid}: duplicate or ambiguous level identifiers")
        if ptype == "BINARY" and len(levels) != 2:
            raise ValueError(f"{fid}: BINARY requires exactly two levels")
        out.append(EncodedFactor(fid, ptype, ptype in _ORDERED_TYPES, "discrete",
                                 levels=levels, cardinality=len(levels)))
    unknown = set(continuous_bounds) - {f.name for f in out if f.kind == "continuous"}
    if unknown:
        raise ValueError(f"bounds overrides reference non-continuous or unknown factors: {sorted(unknown)}")
    return out


def build_dim_weights(
    factors: list[EncodedFactor],
    *,
    continuous_card: int | None = None,
) -> np.ndarray:
    """Cardinality weights w_j = |L_j| / mean(|L|), across ALL factors (gap 2).

    Continuous factors get an effective cardinality (`continuous_card`, default =
    max discrete cardinality) so they are treated as fine-resolution, matching
    the paper's intent that finer factors get more spread.
    """
    disc = [f.cardinality for f in factors if f.kind == "discrete"]
    if continuous_card is None:
        continuous_card = max(disc) if disc else 2
    _positive_int(continuous_card, "continuous_card")
    cards = np.array(
        [f.cardinality if f.kind == "discrete" else continuous_card for f in factors],
        dtype=np.float64,
    )
    return cards / cards.mean()


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Predicate:
    """Resolved typed comparison. Inactive values satisfy no ordinary comparison."""
    factor: EncodedFactor
    op: str
    value: Any

    @property
    def dependencies(self):
        return {self.factor.name}

    def __call__(self, row: dict) -> bool:
        raw = row.get(self.factor.name, NOT_APPLICABLE)
        if self.value == NOT_APPLICABLE:
            return (raw == NOT_APPLICABLE) if self.op == "==" else (raw != NOT_APPLICABLE)
        if raw == NOT_APPLICABLE:
            return False
        if self.op == "==":
            return raw == self.value
        if self.op == "!=":
            return raw != self.value
        lhs, rhs = raw, self.value
        if self.factor.kind == "discrete":
            lhs, rhs = self.factor.levels.index(raw), self.factor.levels.index(self.value)
        return {">=": lhs >= rhs, "<=": lhs <= rhs, ">": lhs > rhs, "<": lhs < rhs}[self.op]


@dataclass(frozen=True)
class Condition:
    operator: str
    terms: tuple

    @property
    def dependencies(self):
        return set().union(*(t.dependencies for t in self.terms))

    def __call__(self, row):
        if self.operator == "all":
            return all(t(row) for t in self.terms)
        if self.operator == "any":
            return any(t(row) for t in self.terms)
        return not self.terms[0](row)


_ATOM = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$")


def _condition(expr, by):
    if isinstance(expr, str):
        # Text is deliberately a small grammar. Rich conditions use typed nodes.
        if " and " in expr:
            return Condition("all", tuple(_condition(x, by) for x in expr.split(" and ")))
        match = _ATOM.fullmatch(expr)
        if not match:
            raise ValueError(f"unsupported condition {expr!r}")
        name, op, value = match.groups()
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ValueError(f"invalid quoted level in {expr!r}") from exc
        elif not re.fullmatch(r"[A-Za-z0-9_.+\-]+", value):
            raise ValueError(f"unsupported level syntax {value!r}; use a typed condition")
        expr = {"factor": name, "op": op, "value": value}
    if not isinstance(expr, dict):
        raise ValueError("condition must be text or a typed object")
    for operator in ("all", "any", "not"):
        if operator in expr:
            terms = [expr[operator]] if operator == "not" else expr[operator]
            if set(expr) != {operator} or not isinstance(terms, list) or not terms:
                raise ValueError(f"invalid {operator} condition")
            return Condition(operator, tuple(_condition(t, by) for t in terms))
    if set(expr) != {"factor", "op", "value"}:
        raise ValueError("atomic condition requires factor, op and value")
    name, op, value = expr["factor"], expr["op"], expr["value"]
    if not isinstance(name, str) or name not in by:
        raise ValueError(f"unknown constraint factor {name!r}")
    f = by[name]
    if op not in ("==", "!=", ">=", "<=", ">", "<"):
        raise ValueError(f"unsupported comparison {op!r}")
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{name}: constraint value must be a scalar")
    if value == NOT_APPLICABLE:
        if op not in ("==", "!="):
            raise ValueError("NOT_APPLICABLE supports only == and !=")
    elif f.kind == "continuous":
        try:
            if isinstance(value, bool):
                raise ValueError()
            value = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}: comparison requires a number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name}: comparison requires a finite number")
    else:
        values = {str(v): v for v in f.levels}
        if str(value) not in values:
            raise ValueError(f"{name}: unknown constraint level {value!r}")
        value = values[str(value)]
        if not f.ordered and op not in ("==", "!="):
            raise ValueError(f"{name}: unordered factors support only == and !=")
    return Predicate(f, op, value)


@dataclass(frozen=True)
class Activation:
    child: str
    when: Any


def _order_activations(activations):
    grouped = {}
    for a in activations:
        grouped.setdefault(a.child, []).append(a.when)
    ordered, visiting, done = [], [], set()

    def visit(child):
        if child in visiting:
            raise ValueError("activation cycle: " + " -> ".join(visiting + [child]))
        if child in done:
            return
        visiting.append(child)
        terms = grouped[child]
        for parent in sorted(set().union(*(t.dependencies for t in terms))):
            if parent in grouped:
                visit(parent)
        visiting.pop()
        done.add(child)
        # Multiple activation rules on a child are conjunctive.
        ordered.append(Activation(child, Condition("all", tuple(terms))))

    for child in sorted(grouped):
        visit(child)
    return ordered


def _order_pair(c, by):
    if "expression" in c:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.]*)\s+precedes\s+([A-Za-z_][A-Za-z0-9_.]*)", c["expression"])
        if not match:
            raise ValueError("ORDERS requires 'factor precedes factor'")
        before, after = match.groups()
    else:
        before, after = c.get("before"), c.get("after")
    if not isinstance(before, str) or not isinstance(after, str) or before not in by or after not in by or before == after:
        raise ValueError("ORDERS requires two distinct declared factors")
    return before, after


def compile_constraints(constraints: list[dict], factors: list[EncodedFactor], *,
                        allow_unparsed: bool = False):
    """Resolve constraints before sampling; reject unsupported input by default.

    Text: REQUIRES uses an atom, conjunction, or A => B; EXCLUDES forbids an
    atom/conjunction (implications are rejected as ambiguous); ACTIVATES uses
    A => child applicable. Typed equivalents use if/then, when, and child.
    ORDERS is validated here and enforced by the full-document renderer.
    Exploratory allow_unparsed returns rejected expressions and emits a warning.
    Activation cycles always fail, including in exploratory mode.
    """
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list")
    by = {f.name: f for f in factors}
    feasibility, activations, unparsed = [], [], []
    for c in constraints:
        try:
            if not isinstance(c, dict) or not isinstance(c.get("type"), str):
                raise ValueError("constraint requires type")
            kind = c["type"].upper()
            if "expression" in c and not isinstance(c["expression"], str):
                raise ValueError("constraint expression must be text")
            if kind == "ORDERS":
                _order_pair(c, by)
                continue
            if kind not in ("REQUIRES", "EXCLUDES", "ACTIVATES"):
                raise ValueError(f"unknown constraint type {kind!r}")
            expr = c.get("expression")
            if kind == "ACTIVATES":
                if expr is not None:
                    match = re.fullmatch(r"(.*?)=>\s*([A-Za-z_][A-Za-z0-9_.]*)\s+applicable", expr.strip())
                    if not match:
                        raise ValueError("unsupported activation expression")
                    when, child = match.groups()
                else:
                    when, child = c.get("when"), c.get("child")
                if not isinstance(child, str) or child not in by:
                    raise ValueError(f"unknown activation child {child!r}")
                activations.append(Activation(child, _condition(when, by)))
            elif kind == "REQUIRES" and ((isinstance(expr, str) and "=>" in expr) or "if" in c):
                lhs, rhs = expr.split("=>", 1) if expr is not None else (c["if"], c.get("then"))
                a, b = _condition(lhs, by), _condition(rhs, by)
                feasibility.append(lambda row, a=a, b=b: not a(row) or b(row))
            else:
                predicate = _condition(expr if expr is not None else c.get("when"), by)
                feasibility.append((lambda row, p=predicate: not p(row)) if kind == "EXCLUDES" else predicate)
        except (ValueError, TypeError) as exc:
            if not allow_unparsed:
                raise ValueError(f"invalid constraint {c!r}: {exc}") from exc
            unparsed.append(str(c.get("expression", c)) if isinstance(c, dict) else str(c))
    ordered = _order_activations(activations)
    if unparsed:
        warnings.warn(f"constraints NOT enforced: {unparsed}", stacklevel=2)
    return feasibility, ordered, unparsed


def _apply_activations(row: dict, activations: list[Activation]) -> dict:
    for a in activations:
        if not a.when(row):
            row[a.child] = NOT_APPLICABLE
    return row


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
# Reasonable space-filling defaults for the phi_p refine: a gentle polish of the
# Sobol init, NOT pure maximin. Unconstrained phi_p maximin has its optimum on
# the box boundary, so full convergence (high p_final, many iterations) packs the
# points onto hypercube corners -- after discretization to few levels only the
# extreme levels appear. A low p_final with few iterations improves interior
# spacing while preserving Sobol's level balance. Push p_final/iters up (toward
# maximin) only when factors have many levels (fine discretization) or are
# continuous, where boundary clustering costs less than interior coverage.
REFINE_DEFAULTS: dict = {
    "p_start": 2,
    "p_final": 4,
    "n_stages": 3,
    "use_lbfgs": True,
    "iters_per_stage": 3,
}


def _sample_unit_cube(
    m: int, n: int, method: str, seed: int, dim_weights: np.ndarray | None,
    refine_params: dict | None = None,
):
    """Sobol base, optionally phi_p-refined via the paper's optimizer."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=m, scramble=True, seed=seed)
    base = sampler.random_base2(m=max(1, math.ceil(math.log2(max(2, n)))))[:n]
    if method == "sobol":
        return base
    if method == "sobol+refine":
        params = {**REFINE_DEFAULTS, **(refine_params or {})}
        refined, _, _ = _phip_optimize(
            n, m, seed=seed, X_init=base,
            dim_weights=(None if dim_weights is None else np.asarray(dim_weights)),
            **params,
        )
        return refined
    if method == "lhs":
        return qmc.LatinHypercube(d=m, seed=seed).random(n=n)
    raise ValueError(f"method must be sobol / sobol+refine / lhs, got {method!r}")


def _row_key(row, factors):
    return tuple(row[f.name] for f in factors)


def _validate_fixed(fixed, factors):
    fixed = dict(fixed or {})
    by = {f.name: f for f in factors}
    if set(fixed) - set(by):
        raise ValueError("fixed assignments reference unknown factors")
    for name, value in fixed.items():
        if value == NOT_APPLICABLE:
            continue  # applicability is checked on each complete row
        norm = _normalize_row({name: value}, [by[name]])
        if norm is None:
            raise ValueError(f"{name}: invalid fixed value {value!r}")
        fixed[name] = norm[name]
    return fixed


def _diagnostics(rows, factors, n, dropped_duplicate, dropped_infeasible, reason):
    unique = len({_row_key(row, factors) for row in rows})
    return dict(requested=n, returned=len(rows), unique_rows=unique,
                replicate_rows=len(rows) - unique, dropped_duplicate=dropped_duplicate,
                dropped_infeasible=dropped_infeasible, shortfall_reason=reason)


def design(
    factor_inventory: list[dict], constraints: list[dict] | None = None,
    n: int = 100, *, method: str = "sobol", seed: int = 0, oversample: int = 8,
    continuous_bounds: dict[str, tuple[float, float]] | None = None,
    continuous_card: int | None = None, refine_params: dict | None = None,
    extra_predicates: list[Callable[[dict], bool]] | None = None,
    allow_unparsed: bool = False, replicate: bool = False, max_rounds: int = 6,
    enumeration_limit: int = 10000, fixed: dict | None = None,
) -> dict:
    """Generate feasible configurations, unique unless replicate=True.

    The first unconstrained draw has n points. Subsequent draws replace rejected
    or duplicate decoded rows. If a small finite space remains underfilled, an
    exhaustive feasible pool supplies missing rows and detects exhaustion.
    Diagnostics describe the final configurations, not just continuous samples.
    """
    for name, value in (("n", n), ("oversample", oversample), ("max_rounds", max_rounds),
                        ("enumeration_limit", enumeration_limit)):
        _positive_int(value, name)
    if not isinstance(replicate, bool):
        raise ValueError("replicate must be boolean")
    if method not in ("sobol", "sobol+refine", "lhs"):
        raise ValueError(f"unsupported design method {method!r}")
    factors = encode_factors(factor_inventory, continuous_bounds=continuous_bounds)
    w = build_dim_weights(factors, continuous_card=continuous_card)
    feas, acts, unparsed = compile_constraints(constraints or [], factors, allow_unparsed=allow_unparsed)
    feas += list(extra_predicates or [])
    fixed = _validate_fixed(fixed, factors)
    varying = [f for f in factors if f.name not in fixed]
    varying_w = w[[i for i, f in enumerate(factors) if f.name not in fixed]]
    rng = np.random.default_rng(seed)
    perms = {f.name: rng.permutation(f.cardinality) if f.kind == "discrete" and not f.ordered else None
             for f in varying}
    kept, seen = [], set()
    dropped_duplicate = dropped_infeasible = 0

    def accept(raw):
        nonlocal dropped_duplicate, dropped_infeasible
        row = _normalize_row(raw, factors, acts)
        if row is None or not all(p(row) for p in feas):
            dropped_infeasible += 1
            return
        key = _row_key(row, factors)
        if not replicate and key in seen:
            dropped_duplicate += 1
            return
        seen.add(key)
        kept.append(row)

    for attempt in range(max_rounds):
        if not varying:
            for _ in range(n if replicate else 1):
                accept(fixed)
            break
        want = n if attempt == 0 and not feas else max((n - len(kept)) * oversample, 16)
        u = _sample_unit_cube(len(varying), want, method, seed + attempt * 101, varying_w, refine_params)
        for point in u:
            raw = {f.name: f.decode(float(point[i]), perms[f.name]) for i, f in enumerate(varying)}
            accept({**raw, **fixed})
            if len(kept) == n:
                break
        if len(kept) == n:
            break

    reason, feasible_count = None, None
    if len(kept) < n:
        finite = all(f.kind == "discrete" for f in varying)
        size = math.prod(f.cardinality for f in varying) if finite else enumeration_limit + 1
        if size <= enumeration_limit:
            pool = {}
            for values in product(*(f.levels for f in varying)):
                raw = {**dict(zip((f.name for f in varying), values)), **fixed}
                row = _normalize_row(raw, factors, acts)
                if row is not None and all(p(row) for p in feas):
                    pool[_row_key(row, factors)] = row
            feasible_count = len(pool)
            values = list(pool.values())
            for i in rng.permutation(len(values)):
                if len(kept) == n:
                    break
                accept(values[int(i)])
            if replicate and values:
                while len(kept) < n:
                    accept(values[int(rng.integers(len(values)))])
            if len(kept) < n:
                reason = "finite_space_exhausted" if pool else "no_feasible_rows"
        else:
            reason = "sampling_budget_exhausted"
    if len(kept) < n:
        warnings.warn(f"{reason}: returned {len(kept)}/{n} rows", stacklevel=2)
    return dict(rows=kept, factors=factors, dim_weights=w, method=method,
                unparsed_constraints=unparsed, feasible_unique_count=feasible_count,
                **_diagnostics(kept, factors, n, dropped_duplicate, dropped_infeasible, reason))


def design_from_yaml(path_or_text: str | Path, n: int = 100, *, validate: bool = True,
                     strict: bool = True, **kw) -> dict:
    """Load a `theta_full` YAML (path or text) and design over it (space-filling)."""
    doc = _load_yaml(path_or_text)
    if validate:
        validate_factorization(doc, strict=strict)
    return design(doc["factor_inventory"], doc.get("constraints", []), n, **kw)


# --------------------------------------------------------------------------
# Chain-of-thought design (LLM reasons over the parameterization)
# --------------------------------------------------------------------------
def _describe_factor(f: EncodedFactor) -> str:
    if f.kind == "continuous":
        return f"- {f.name} ({f.ptype}): any number in [{f.low}, {f.high}]"
    kind = "ordered" if f.ordered else "unordered"
    return f"- {f.name} ({f.ptype}, {kind}): one of {[str(v) for v in f.levels]}"


def _build_cot_prompt(factors: list[EncodedFactor], constraints: list[dict], n: int) -> str:
    lines = [_describe_factor(f) for f in factors]
    cons = ["- " + json.dumps(c)
            for c in constraints if str(c.get("type", "")).upper() != "ORDERS"]
    cons_block = ("\nHonor these constraints (do not emit configurations that violate them):\n"
                  + "\n".join(cons)) if cons else ""
    return (
        f"You are designing a test suite of {n} configurations over {len(factors)} factors. "
        "Spread the configurations across the joint factor "
        "space: cover each factor's levels and vary the combinations so the points fill the "
        "space rather than clustering.\n\n"
        "Factors and allowed values:\n" + "\n".join(lines) + cons_block + "\n\n"
        f"Output exactly {n} configurations as a single JSON array of {n} objects. Each object "
        "must contain every factor name as a key, with a value drawn only from that factor's "
        "allowed values (use the values verbatim as written above). An inactive child may use "
        "NOT_APPLICABLE; an active factor must use a declared value. Output ONLY the JSON array, "
        "no prose and no markdown fences."
    )


def _extract_json_array(text: str) -> list:
    """Decode JSON arrays without treating brackets inside strings as nesting."""
    for match in re.finditer(r"\[", text):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, list):
            return obj
    return []


def _normalize_row(row: dict, factors: list[EncodedFactor], activations=()) -> dict | None:
    """Canonicalize, mask in dependency order, then require valid active values.

    All keys are required, including inactive children. The explicit sentinel is
    permitted provisionally but accepted only when a child's rule masks it.
    """
    if not isinstance(row, dict) or set(row) != {f.name for f in factors}:
        return None
    out = {}
    for f in factors:
        v = row[f.name]
        if v == NOT_APPLICABLE:
            out[f.name] = v
            continue
        if f.kind == "continuous":
            try:
                if isinstance(v, bool):
                    return None
                x = float(v)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(x) or not f.low <= x <= f.high:
                return None
            out[f.name] = x
        else:
            by_str = {str(lvl): lvl for lvl in f.levels}
            if str(v) not in by_str:
                return None
            out[f.name] = by_str[str(v)]
    out = _apply_activations(out, activations)
    inactive = {a.child for a in activations if not a.when(out)}
    if any(v == NOT_APPLICABLE and name not in inactive for name, v in out.items()):
        return None
    return out


def cot_design(
    factor_inventory: list[dict],
    constraints: list[dict] | None = None,
    n: int = 100,
    *,
    llm: Callable[[str], str],
    max_rounds: int = 3,
    continuous_bounds: dict[str, tuple[float, float]] | None = None,
    extra_predicates: list[Callable[[dict], bool]] | None = None,
    allow_unparsed: bool = False, replicate: bool = False, fixed: dict | None = None,
) -> dict:
    """LLM produces the design by reasoning over the parameterization.

    `llm` is any callable `llm(prompt: str) -> str` (e.g. `frontier_discovery.lm.LM`).
    Rows are validated against the declared levels, masked/filtered by the same
    constraints as the space-filling path, and topped up over up to `max_rounds`
    prompts if the model returns too few valid feasible rows. Not deterministic.
    """
    _positive_int(n, "n")
    _positive_int(max_rounds, "max_rounds")
    if not isinstance(replicate, bool):
        raise ValueError("replicate must be boolean")
    if llm is None:
        raise ValueError("method='cot' requires an `llm` callable (e.g. frontier_discovery.lm.LM); got None")
    constraints = constraints or []
    factors = encode_factors(factor_inventory, continuous_bounds=continuous_bounds)
    feas, acts, unparsed = compile_constraints(constraints, factors, allow_unparsed=allow_unparsed)
    fixed = _validate_fixed(fixed, factors)
    feas = feas + list(extra_predicates or [])

    kept: list[dict] = []
    seen: set = set()
    raw: list[str] = []
    dropped_invalid = 0
    dropped_duplicate = 0
    dropped_infeasible = 0
    for _ in range(max_rounds):
        need = n - len(kept)
        if need <= 0:
            break
        prompt = _build_cot_prompt(factors, constraints, need)
        if fixed:
            prompt += "\nFixed assignments (must match): " + json.dumps(fixed)
        prompt += "\nReplication " + ("is allowed." if replicate else "is forbidden. Exclude these already accepted rows: " + json.dumps(kept))
        text = llm(prompt)
        raw.append(text)
        for row in _extract_json_array(text):
            if isinstance(row, dict) and any(str(row.get(k)) != str(v) for k, v in fixed.items()):
                dropped_invalid += 1
                continue
            norm = _normalize_row(row, factors, acts)
            if norm is None:
                dropped_invalid += 1
                continue
            if not all(p(norm) for p in feas):
                dropped_infeasible += 1
                continue
            key = tuple(norm[f.name] for f in factors)
            if not replicate and key in seen:
                dropped_duplicate += 1
                continue
            seen.add(key)
            kept.append(norm)
            if len(kept) >= n:
                break

    if len(kept) < n:
        warnings.warn(f"cot: returned {len(kept)}/{n} valid feasible rows after {max_rounds} round(s)")
    return {
        "rows": kept[:n],
        "factors": factors,
        "method": "cot",
        "unparsed_constraints": unparsed,
        "dropped_invalid": dropped_invalid,
        "dropped_infeasible": dropped_infeasible,
        "llm_responses": raw,
        "requested": n,
        **_diagnostics(kept, factors, n, dropped_duplicate, dropped_infeasible,
                       "llm_budget_exhausted" if len(kept) < n else None),
    }


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------
_METHODS = ("cot", "sobol", "sobol+refine")


def generate_design(
    factor_inventory: list[dict],
    constraints: list[dict] | None = None,
    n: int = 100,
    *,
    method: str = "sobol",
    llm: Callable[[str], str] | None = None,
    seed: int = 0,
    **kw,
) -> dict:
    """Generate an `n`-run design over the factorization by the chosen method.

    method (default "sobol"):
      "sobol"        scrambled Sobol, discretized (numpy+scipy, no torch). Best
                     measured discretized projection balance and 2-factor coverage
                     across N and cardinality -- the default.
      "sobol+refine" Sobol then phi_p-refined (Sudjianto & Zhang), needs torch.
                     Optimizes continuous fill distance / maximin (the JASA
                     criterion) at a cost to discretized projection balance; tune
                     via `refine_params`. Prefer when fill distance is the design
                     objective rather than marginal balance.
      "cot"          LLM reasons over the parameterization (requires `llm`).
    n is the number of runs (design array size). Returns a dict with `rows`,
    `returned`, `requested`, `method`, and method-specific extras.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if method == "cot":
        return cot_design(factor_inventory, constraints, n, llm=llm, **kw)
    res = design(factor_inventory, constraints, n, method=method, seed=seed, **kw)
    res["method"] = method
    return res


def generate_from_yaml(
    path_or_text: str | Path,
    n: int = 100,
    *,
    method: str = "sobol",
    llm: Callable[[str], str] | None = None,
    seed: int = 0,
    validate: bool = True,
    strict: bool = True,
    **kw,
) -> dict:
    """Load a `theta_full` YAML and generate a design by the chosen method."""
    doc = _load_yaml(path_or_text)
    if validate:
        validate_factorization(doc, strict=strict)
    return generate_design(doc["factor_inventory"], doc.get("constraints", []), n,
                           method=method, llm=llm, seed=seed, **kw)


def _load_yaml(path_or_text: str | Path) -> dict:
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                if key in result:
                    raise ValueError(f"duplicate YAML key {key!r}")
                result[key] = loader.construct_object(value_node, deep=deep)
            except TypeError as exc:
                raise ValueError("YAML keys must be scalar") from exc
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    if isinstance(path_or_text, Path):
        text = path_or_text.read_text()
    else:
        text = str(path_or_text)
        if "\n" not in text:
            try:
                path = Path(text)
                if path.is_file():
                    text = path.read_text()
            except OSError:
                pass  # long one-line YAML is not a path
    doc = yaml.load(text, Loader=UniqueLoader)
    if not isinstance(doc, dict):
        raise ValueError("factorization YAML must contain an object")
    return doc


def validate_factorization(doc: dict, *, strict: bool = False) -> list[str]:
    """Validate the full contract; non-strict mode reports completeness issues.

    Malformed inventories always fail. Minimal inventory callers should use
    encode_factors directly. Strict YAML entry points require schema version 1.
    """
    from frontier_discovery.factorization import validate_document
    return validate_document(doc, strict=strict)


# --------------------------------------------------------------------------
# Seed the frontier-discovery loop (inner array = initial candidate population)
# --------------------------------------------------------------------------
def slot_renderer(
    factor_inventory: list[dict],
    *,
    na_text: str = "",
) -> Callable[[dict], dict[str, str]]:
    """Build a render fn: one design row -> {component_name: text}.

    A minimal slot-structured renderer (one component per factor, §10 of the
    factorization skill): the component name is the factor id and its text is the
    chosen level's description (falling back to the level id). Every candidate
    carries the same component keys, so the population is a valid multi-candidate
    seed. NOT_APPLICABLE renders as `na_text`. Replace with the domain's real G
    when there is one; this is the default so the loop can be seeded immediately.
    """
    levels_by = {f["id"]: (f.get("levels") if isinstance(f.get("levels"), dict) else None)
                 for f in factor_inventory}

    def render(row: dict) -> dict[str, str]:
        out: dict[str, str] = {}
        for fid, levels in levels_by.items():
            if fid not in row:
                raise ValueError(f"missing factor {fid!r} in renderer row")
            v = row[fid]
            if v == NOT_APPLICABLE:
                out[fid] = na_text
            elif levels is not None and levels.get(v) not in (None, ""):
                out[fid] = str(levels.get(v))
            else:
                out[fid] = str(v)          # continuous value or bare level id
        return out

    return render


def build_seed_population(
    factor_inventory: list[dict],
    constraints: list[dict] | None = None,
    n: int = 32,
    *,
    render: Callable[[dict], dict[str, str]] | None = None,
    method: str = "sobol",
    llm: Callable[[str], str] | None = None,
    seed: int = 0,
    return_design: bool = False,
    **kw,
):
    """Generate a design and render it into a seed population for the loop.

    Returns a `list[{component: text}]` ready to pass as `seed_candidate=` to
    `frontier_discovery.discover` / `api.optimize` (a list initializes a
    multi-candidate initial population — the inner array). With
    `return_design=True`, returns `(population, design_result)` so the raw rows
    and design diagnostics are available.
    """
    res = generate_design(factor_inventory, constraints, n, method=method, llm=llm, seed=seed, **kw)
    r = render or slot_renderer(factor_inventory)
    population = [r(row) for row in res["rows"]]
    return (population, res) if return_design else population


class FactorDoeDesigner:
    """PopulationDesigner / ConditionDesigner over a `theta_full` factorization.

    Carries the full factor inventory + constraints (richer than `FactorSpec`),
    so `design()` ignores any `factors` passed by the caller and returns rows from
    `generate_design`. Conforms to the `frontier_discovery.design` protocols, so
    once the config-level designer seam is wired it drops in as init="factor_doe".
    Today, prefer `build_seed_population(...)` + `seed_candidate=`.
    """

    def __init__(
        self,
        factor_inventory: list[dict],
        constraints: list[dict] | None = None,
        *,
        method: str = "sobol",
        llm: Callable[[str], str] | None = None,
        seed: int = 0,
        **design_kw,
    ):
        self.factor_inventory = factor_inventory
        self.constraints = constraints or []
        self.method = method
        self.llm = llm
        self.seed = seed
        self.design_kw = design_kw

    def _rows(self, n: int, seed: int | None) -> list[dict]:
        return generate_design(
            self.factor_inventory, self.constraints, n,
            method=self.method, llm=self.llm,
            seed=self.seed if seed is None else seed, **self.design_kw,
        )["rows"]

    def design(self, factors=None, n: int = 100, *, seed: int | None = None) -> list[dict]:
        if factors is not None:
            warnings.warn("FactorDoeDesigner ignores `factors`; it uses its own factor_inventory")
        return self._rows(n, seed)

    def design_conditions(self, factors=None, n: int = 100, *, seed: int | None = None):
        from frontier_discovery.design import ConditionPoint
        return [ConditionPoint(factors=r, weight=1.0) for r in self._rows(n, seed)]
