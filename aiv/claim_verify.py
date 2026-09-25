"""Deterministic claim verification against a grounded evidence store.

A claim is a triple $(h, r, t)$. The store supports it when it asserts that
triple, contradicts it when the same head and relation carry an incompatible
tail, and leaves it Absent when it holds no fact for that head or relation. No
language model is on this path: the verdict follows from a lookup and a
comparison, and the comparison rule is recorded with the verdict.

Tails are compared numerically whenever the claim states a number, because the
formatting of a numeral carries no meaning: a claimed threshold of 0.70 and a
stored threshold of 0.7 are the same value, and comparing them as text would
report a contradiction where none exists. The stored number is read from the
exact numeric register when the store holds one for the relation and entity, and
otherwise parsed from the asserted tails. Non-numeric tails are compared as
case-folded text.

The store is supplied by the caller, so the verifier can be exercised against a
fixture as well as against a live store.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SUPPORTED = "Supported"
CONTRADICTED = "Contradicted"
ABSENT = "Absent"

# Relative tolerance for numeric tails. It absorbs the representation noise of a
# value that has been through a float sum, while keeping genuinely different
# reported figures apart: 0.7755 and 0.776 remain distinct.
REL_TOL = 1e-9


@dataclass
class Verdict:
    """A verdict with the evidence and comparison rule behind it."""

    label: str                       # Supported, Contradicted or Absent
    reason: str                      # why, in words
    stored: list = field(default_factory=list)   # the asserted tails consulted
    source: str | None = None        # report section the evidence came from
    compared: str | None = None      # "numeric", "text" or None when undecided

    def __bool__(self) -> bool:
        return self.label == SUPPORTED


def parse_number(value) -> float | None:
    """The numeric value of a claim tail, or None when it does not state one.

    Thousands separators are removed because they are formatting. Infinities and
    NaN are rejected: they parse as floats but cannot be compared as figures."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    try:
        v = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def numbers_agree(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0)


def provenance_index(provenance: dict, key: str = "section") -> dict:
    """Index a store's provenance.json by (head, relation) for span lookup.

    `key` selects which evidence field is reported: "section" gives the report
    heading, "span" the source text."""
    idx: dict = {}
    for entry in provenance.get("triples", []):
        h, r, t = entry["triple"]
        spans = [ev.get(key, "") for ev in entry.get("evidence", []) if ev.get(key)]
        if spans:
            idx.setdefault((h, r), []).append((str(t), spans[0]))
    return idx


class ClaimVerifier:
    """Supported / Contradicted / Absent for a claim triple against a store.

    `store` needs `fuzzy_match_entity`, `query_triples` and, to use the exact
    numeric register, `lookup_enm`."""

    def __init__(self, store, provenance: dict | None = None, evidence_key: str = "section"):
        self.store = store
        self.prov = provenance_index(provenance or {}, evidence_key)

    def source_for(self, head: str, relation: str, tail) -> str | None:
        """The evidence section for the exact tail, else any for (head, relation)."""
        entries = self.prov.get((head, relation), [])
        for stored_tail, span in entries:
            if stored_tail.lower() == str(tail).lower():
                return span
        return entries[0][1] if entries else None

    def _stored_number(self, head: str, relation: str, tails: list) -> float | None:
        """The stored figure: the exact numeric register first, then the tails."""
        lookup = getattr(self.store, "lookup_enm", None)
        if lookup is not None:
            try:
                entry = lookup(relation, head)
            except Exception:
                entry = None
            if isinstance(entry, dict):
                n = parse_number(entry.get("value"))
                if n is not None:
                    return n
        for t in tails:
            n = parse_number(t)
            if n is not None:
                return n
        return None

    def verify(self, head: str, relation: str, tail) -> Verdict:
        h, r, t = str(head).lower(), str(relation).lower(), str(tail).lower()

        matched = self.store.fuzzy_match_entity(h)
        if matched is None:
            return Verdict(ABSENT, "head not in store")

        asserted = self.store.query_triples(head=matched, relation=r)
        if not asserted:
            return Verdict(ABSENT, "no fact for (head, relation)")
        tails = sorted({str(x) for _, _, x in asserted})

        claimed = parse_number(t)
        if claimed is not None:
            stored = self._stored_number(matched, r, tails)
            if stored is not None:
                if numbers_agree(claimed, stored):
                    return Verdict(SUPPORTED, f"stored {stored!r}", tails,
                                   self.source_for(matched, r, t), "numeric")
                return Verdict(CONTRADICTED, f"stored {stored!r}", tails,
                               self.source_for(matched, r, tails[0]), "numeric")

        if any(t == x.lower() for x in tails):
            return Verdict(SUPPORTED, f"stored {tails}", tails,
                           self.source_for(matched, r, t), "text")
        return Verdict(CONTRADICTED, f"stored {tails}", tails,
                       self.source_for(matched, r, tails[0]), "text")
