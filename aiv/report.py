"""Schema-driven report assembly, section scoring and instruction optimization
for the reporting workflow of Chapter 9.

The chapter separates three responsibilities, and this module keeps them separate.
A query engine collects evidence from the store by required section and field, so
every statement a section may make is bound to a stored fact with its source span,
and a required field the store cannot answer becomes a recorded gap rather than
silence. A writing model turns a packet into prose; it is supplied by the caller
and is never consulted about whether its own output is supported. Verification
scores the returned section against the packet along the axes the chapter names:
faithfulness, completeness, qualification preservation, numerical fidelity and
provenance coverage. Every one of those is computed from the graph, so no language
model is on the measurement.

Instruction optimization treats the report-generation instruction as the candidate.
`aiv._vendor.prompt_factorization` renders a population of complete instructions
from a factorization document, each section is drafted under each instruction, and
the instructions are ranked by the scores above. The search varies how the fixed
evidence is expressed; it never varies the evidence.

The store is the only external system involved, through knowlytix.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

# A number token, matched maximally so that 0.7744 is never read as 0.7.
_NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?!\d)")

VALUE = "value"           # a figure or status the section must report
QUALIFIER = "qualifier"   # a condition that must travel with the figure

# Required sections and the fields each must carry. The fairness section is
# required by the schema and unsupported by the store, which is how an evidence
# gap reaches the report instead of being absent from it.
REPORT_SCHEMA: dict[str, list[tuple[str, str, str]]] = {
    "Discrimination": [
        ("auc", "has_value", VALUE),
        ("auc", "has_threshold", QUALIFIER),
        ("auc", "has_status", QUALIFIER),
        ("ks", "has_value", VALUE),
        ("ks", "has_threshold", QUALIFIER),
    ],
    "Calibration and loss": [
        ("log loss", "has_value", VALUE),
        ("log loss", "has_threshold", QUALIFIER),
        ("accuracy", "has_value", VALUE),
        ("accuracy", "has_threshold", QUALIFIER),
    ],
    "Risk drivers": [
        ("score", "has_main_effect_importance", VALUE),
        ("score", "has_monotonicity", QUALIFIER),
        ("dti", "has_main_effect_importance", VALUE),
        ("dti", "has_monotonicity", QUALIFIER),
    ],
    "Fairness": [
        ("fairness", "has_status", VALUE),
        ("fairness", "has_disparate_impact", VALUE),
    ],
}


@dataclass
class Fact:
    head: str
    relation: str
    tail: str
    role: str
    span: str | None = None

    @property
    def supported(self) -> bool:
        return self.tail is not None


@dataclass
class EvidencePacket:
    """Everything one section may assert, and everything it must not."""

    section: str
    facts: list[Fact] = field(default_factory=list)
    gaps: list[tuple[str, str]] = field(default_factory=list)

    @property
    def values(self) -> list[Fact]:
        return [f for f in self.facts if f.role == VALUE]

    @property
    def qualifiers(self) -> list[Fact]:
        return [f for f in self.facts if f.role == QUALIFIER]

    def as_prompt_evidence(self) -> str:
        lines = [f"- {f.head} {f.relation.replace('has_', '')}: {f.tail}"
                 for f in self.facts]
        lines += [f"- {h} {r.replace('has_', '')}: NOT IN THE EVIDENCE STORE"
                  for h, r in self.gaps]
        return "\n".join(lines)


def load_provenance(store_dir: str) -> dict:
    """Map each stored triple to the first source span asserting it."""
    path = os.path.join(store_dir, "provenance.json")
    if not os.path.isfile(path):
        return {}
    index = {}
    for entry in json.load(open(path, encoding="utf-8")).get("triples", []):
        head, relation, tail = entry["triple"]
        spans = [ev.get("span", "") for ev in entry.get("evidence", [])]
        index[(head, relation, tail)] = spans[0] if spans else None
    return index


def assemble_packets(store, provenance: dict | None = None,
                     schema: dict | None = None) -> list[EvidencePacket]:
    """Query the store section by section. A required field the store cannot
    answer is recorded as a gap, so the reporting stage can distinguish evidence
    that is missing from evidence that was never requested."""
    provenance = provenance or {}
    schema = schema or REPORT_SCHEMA
    packets = []
    for section, fields in schema.items():
        packet = EvidencePacket(section=section)
        for head, relation, role in fields:
            tail = _lookup(store, head, relation)
            if tail is None:
                packet.gaps.append((head, relation))
            else:
                packet.facts.append(
                    Fact(head, relation, tail, role,
                         provenance.get((head, relation, tail))))
        packets.append(packet)
    return packets


def _lookup(store, head: str, relation: str) -> str | None:
    """One (head, relation) read against the store, with no language model."""
    matched = store.fuzzy_match_entity(head.lower())
    if matched is None:
        return None
    triples = store.query_triples(head=matched, relation=relation.lower())
    if not triples:
        return None
    return str(triples[0][2])


def _states(text: str, value: str) -> bool:
    """Token-boundary match, so a reported 0.7 is not satisfied by 0.7744."""
    return re.search(rf"(?<!\w){re.escape(str(value).lower())}(?!\w)",
                     text.lower()) is not None


def score_section(text: str, packet: EvidencePacket) -> dict:
    """Score one drafted section against its packet.

    The axes are the ones \\cref{ch:report} names. Each is a ratio in [0, 1] with
    an empty denominator scored 1, and each is computed from the packet rather
    than from a judgement about the prose.
    """
    stated_values = [f for f in packet.values if _states(text, f.tail)]
    stated_quals = [f for f in packet.qualifiers if _states(text, f.tail)]
    stated = stated_values + stated_quals
    packet_numbers = {f.tail for f in packet.facts if _NUMBER.fullmatch(f.tail)}
    in_text = set(_NUMBER.findall(text))
    # A gap is faithfully handled when the section does not assert content for it.
    gap_claims = sum(1 for head, _ in packet.gaps if _gap_asserted(text, head))

    def ratio(num, den):
        return 1.0 if den == 0 else num / den

    scores = {
        "faithfulness": ratio(len(packet.gaps) - gap_claims, len(packet.gaps)),
        "completeness": ratio(len(stated_values), len(packet.values)),
        "qualification": ratio(len(stated_quals), len(packet.qualifiers)),
        "numerical_fidelity": ratio(len(in_text & packet_numbers), len(in_text)),
        "provenance": ratio(sum(1 for f in stated if f.span), len(stated)),
    }
    scores["overall"] = sum(scores.values()) / len(scores)
    return scores


# Phrases that declare evidence to be missing. Generic negation is deliberately
# excluded: "fairness testing showed no disparate impact" asserts a result the
# store does not hold, and reading its "no" as a gap declaration would score a
# fabrication as faithful.
_MISSING_EVIDENCE = ("no evidence", "not in the evidence", "no such", "not covered",
                     "not available", "not provided", "not reported", "not tested",
                     "not been", "absent", "missing", "gap", "unavailable",
                     "outstanding", "cannot be reported")


def _gap_asserted(text: str, head: str) -> bool:
    """True when the section says something about a field the store cannot support.

    Naming the field as unevidenced is the handled case. Not mentioning it at all
    is also not an assertion, so only a mention without missing-evidence wording
    counts against faithfulness.
    """
    low = text.lower()
    if head.lower() not in low:
        return False
    return not any(phrase in low for phrase in _MISSING_EVIDENCE)


def optimize_instruction(document: dict, packets: list[EvidencePacket], generate,
                         n: int = 6, seed: int = 0,
                         method: str = "sobol+refine") -> dict:
    """Rank rendered instructions by the mean section score they produce.

    `document` is a validated prompt-factorization document and `generate` is a
    callable (instruction, packet) -> section text. The evidence handed to the
    writer is identical across candidates, so a difference in score is a
    difference in how the instruction expresses fixed evidence.

    The default design is phi_p-refined rather than plain Sobol. Over binary
    factors a Sobol design can place two factors on identical level patterns, and
    for this document at eight runs it confounds two pairs completely, which
    leaves their effects unattributable. The refinement breaks that coincidence.
    """
    from aiv._vendor.prompt_factorization import factorization as F

    built = F.build_document_population(document, n=n, seed=seed, method=method)
    results = []
    arms = [("baseline", built["baseline_prompt"], None)]
    arms += [(f"candidate_{i}", cand["system_prompt"], row)
             for i, (cand, row) in enumerate(zip(built["population"], built["rows"]))]
    for name, instruction, row in arms:
        per_section = []
        for packet in packets:
            per_section.append(score_section(generate(instruction, packet), packet))
        mean = {k: sum(s[k] for s in per_section) / len(per_section)
                for k in per_section[0]}
        results.append({"name": name, "factors": row, "instruction": instruction,
                        "sections": per_section, "mean": mean})
    results.sort(key=lambda r: -r["mean"]["overall"])
    return {"ranked": results, "active_factors": built["active_factors"],
            "requested": built["requested"], "returned": built["returned"]}
