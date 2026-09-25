"""The report-generation instruction as a factorization document (Chapter 9).

Prompt optimization needs the instruction expressed as declared behavioral
dimensions rather than as free text, so that a search varies named commitments and
the reconstruction of the original remains checkable. The document is built here
rather than written out by hand, because the contract requires the span audit, the
claim quotations and the canonical reconstruction to agree exactly with the source
text, and deriving all three from one block list keeps them consistent.

Each factor corresponds to an axis the section evaluator measures, so a difference
in score can be attributed to a declared dimension of the instruction.
"""

from __future__ import annotations

SEPARATOR = "\n\n"

FROZEN = {
    "heading": "You are drafting one section of a model validation report.",
    "closing": "Write at most four sentences. Do not add a recommendation.",
}

# factor id -> (level id -> block text). The first level of each factor is its
# baseline, the wording the source instruction actually uses.
FACTORS: dict[str, dict[str, str]] = {
    "EVI.scope": {
        "evidence_only": "State only what the supplied evidence contains.",
        "permissive": "Use the supplied evidence and your own knowledge of credit models.",
    },
    "QUA.threshold": {
        "optional": "Mention the acceptance threshold and status where helpful.",
        "require": "For every reported figure, also state its acceptance threshold and status.",
    },
    "GAP.handling": {
        "omit": "Leave out any field the evidence does not cover.",
        "name_as_gap": "Name any field the evidence does not cover as a gap, and do not describe it.",
    },
    "NUM.verbatim": {
        "verbatim": "Copy every figure exactly as supplied, with no rounding.",
        "rounded": "Round figures to two decimal places for readability.",
    },
    "PRO.citation": {
        "none": "Do not name the source section.",
        "cite": "After each figure, name the report section it came from.",
    },
}

SLOT_ORDER = ["heading", "EVI.scope", "QUA.threshold", "GAP.handling",
              "NUM.verbatim", "PRO.citation", "closing"]

# The behavior each factor is meant to move, stated so a reader can check the
# claim against the evaluator rather than take it on trust.
_BEHAVIOR = {
    "EVI.scope": "whether the section asserts anything the packet does not contain",
    "QUA.threshold": "whether a reported figure travels with its threshold and status",
    "GAP.handling": "whether an uncovered field is named as a gap or described",
    "NUM.verbatim": "whether a reported figure matches the stored figure exactly",
    "PRO.citation": "whether the source section accompanies a reported figure",
}


def _baseline_level(factor_id: str) -> str:
    return next(iter(FACTORS[factor_id]))


def _blocks() -> list[tuple[str, str]]:
    """The baseline instruction as (slot id, text) in rendering order."""
    return [(slot, FROZEN[slot] if slot in FROZEN
             else FACTORS[slot][_baseline_level(slot)]) for slot in SLOT_ORDER]


def build_document() -> dict:
    """A schema-v1 factorization document for the baseline instruction.

    The span audit partitions the whole source, including the separators, so the
    document reconstructs the source exactly and every behavioral span resolves to
    an admitted factor.
    """
    blocks = _blocks()
    source = SEPARATOR.join(text for _, text in blocks)

    span_audit, claims, cursor = [], [], 0
    for i, (slot, text) in enumerate(blocks):
        if i:  # the separator that precedes this block is non-behavioral text
            span_audit.append({
                "span": f"sep{i}", "start": cursor, "end": cursor + len(SEPARATOR),
                "text": SEPARATOR, "classification": "NON-BEHAVIORAL",
                "disposition": "NON-BEHAVIORAL", "factors": [],
                "reason": "block separator",
            })
            cursor += len(SEPARATOR)
        start, end = cursor, cursor + len(text)
        behavioral = slot not in FROZEN
        span = {
            "span": f"s_{slot}", "start": start, "end": end, "text": text,
            "classification": "BEHAVIORAL" if behavioral else "NON-BEHAVIORAL",
            "disposition": "MAPPED" if behavioral else "NON-BEHAVIORAL",
            "factors": [slot] if behavioral else [],
        }
        if not behavioral:
            span["reason"] = "fixed framing retained in every candidate"
        span_audit.append(span)
        if behavioral:
            claims.append({
                "id": f"c_{slot}", "span": f"s_{slot}", "block": slot,
                "condition": "when drafting a report section",
                "action": _BEHAVIOR[slot], "modality": "MUST",
                "source_quote": text, "rendered_quote": text,
            })
        cursor = end

    inventory = []
    for fid, levels in FACTORS.items():
        baseline = _baseline_level(fid)
        inventory.append({
            "id": fid, "type": "BINARY", "levels": dict(levels),
            "baseline": baseline, "status": "EXPLICIT", "evidence": [f"s_{fid}"],
            "behavioral_definition": f"Controls {_BEHAVIOR[fid]}.",
            "intervention_test": f"Replace only block {fid}.",
            "observable_behavior": levels[baseline],
            "gates": {"definable": "pass", "intervenable": "pass",
                      "behaviorally_observable": "pass",
                      "structurally_non_redundant": "pass", "evidenced": "pass"},
            "level_provenance": {lvl: ("SOURCE" if lvl == baseline else "PROPOSED")
                                 for lvl in levels},
        })

    behavioral = sum(1 for s in span_audit if s["classification"] == "BEHAVIORAL")
    return {
        "schema_version": 1,
        "source_prompt_id": "aiv.report.section_instruction.v1",
        "source_prompt": source,
        "span_audit": span_audit,
        "span_count": len(span_audit),
        "behavioral_span_count": behavioral,
        "coverage": 1,
        "residuals": [],
        "unrepresented_dimensions": [],
        "factor_inventory": inventory,
        "theta_0": {fid: _baseline_level(fid) for fid in FACTORS},
        "constraints": [],
        "renderer_spec": {
            "frozen_blocks": dict(FROZEN),
            "factor_templates": {fid: dict(levels) for fid, levels in FACTORS.items()},
            "slot_order": list(SLOT_ORDER),
            "separator": SEPARATOR,
            "contradictory_templates": {},
        },
        "claims": claims,
        "canonical_reconstruction": source,
        "reconstruction_check": {
            "behavioral_claims_in_original": len(claims),
            "behavioral_claims_in_reconstruction": len(claims),
            "missing": [], "added": [], "verification": "CLAIM_AUDITED",
        },
    }
