"""Versioned prompt documents, provenance checks, and deterministic rendering.

Structural/quotation checks establish an auditable reconstruction contract, not
semantic equivalence of arbitrary prose or empirical model behavior.
"""
from __future__ import annotations

from copy import deepcopy
import math
import warnings

from . import factor_doe as doe


SCHEMA_VERSION = 1


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _text(value, label, *, empty=False):
    _require(isinstance(value, str) and (empty or bool(value)), f"{label} must be text")


def _unique(items, label):
    _require(isinstance(items, list) and all(isinstance(x, str) for x in items), f"{label} must be a list of IDs")
    _require(len(items) == len(set(items)), f"{label} contains duplicate IDs")
    return set(items)


def _contradictory(value):
    return isinstance(value, dict) and value.get("state") == "CONTRADICTORY"


def _baseline_value(value):
    if isinstance(value, dict) and value.get("state") == doe.NOT_APPLICABLE:
        return doe.NOT_APPLICABLE
    return value


def _renderer_parts(doc, factors):
    spec = doc.get("renderer_spec")
    _require(isinstance(spec, dict), "renderer_spec is required")
    frozen = spec.get("frozen_blocks")
    templates = spec.get("factor_templates")
    _require(isinstance(frozen, dict), "frozen_blocks must map IDs to literal text")
    _require(isinstance(templates, dict), "factor_templates must map every factor to templates")
    ids = {f.name for f in factors}
    _require(set(templates) == ids and not (set(frozen) & ids), "renderer must cover every factor with distinct frozen block IDs")
    for key, value in frozen.items():
        _text(key, "frozen block ID")
        _text(value, f"frozen block {key}", empty=True)
    order = spec.get("slot_order")
    _require(_unique(order, "slot_order") == ids | set(frozen), "slot_order must list every factor and frozen block exactly once")
    separator = spec.get("separator", "\n\n")
    _text(separator, "separator", empty=True)
    by_inventory = {f["id"]: f for f in doc["factor_inventory"]}
    for f in factors:
        template = templates[f.name]
        if f.kind == "continuous":
            _text(template, f"template for {f.name}")
            _require("{value}" in template, f"{f.name}: continuous template needs {{value}}")
        else:
            _require(isinstance(template, dict) and set(template) == set(f.levels), f"{f.name}: template keys must equal level IDs")
            for value in template.values():
                _text(value, f"template for {f.name}", empty=True)
        payload = by_inventory[f.name].get("verbatim_payload")
        if payload is not None:
            _text(payload, f"{f.name}: verbatim_payload")
            _require(payload in doc["source_prompt"], f"{f.name}: immutable payload is absent from source")
            levels = by_inventory[f.name].get("payload_levels")
            _require(f.kind == "discrete" and isinstance(levels, list) and bool(levels), f"{f.name}: immutable payload requires payload_levels")
            _require(all(v in f.levels and payload in template[v] for v in levels), f"{f.name}: immutable payload missing from a declared payload level")
    for constraint in doc.get("constraints", []):
        if constraint["type"].upper() == "ORDERS":
            before, after = doe._order_pair(constraint, {f.name: f for f in factors})
            _require(order.index(before) < order.index(after), f"slot_order violates ORDERS: {before} precedes {after}")
    return spec, frozen, templates, order, separator


def _render_function(doc, factors):
    spec, frozen, templates, order, separator = _renderer_parts(doc, factors)
    baseline = {k: _baseline_value(v) for k, v in doc["theta_0"].items()}
    feasibility, activations, _ = doe.compile_constraints(doc.get("constraints", []), factors)
    unresolved = {k for k, v in baseline.items() if _contradictory(v)}
    contradiction_text = spec.get("contradictory_templates", {})
    _require(isinstance(contradiction_text, dict) and set(contradiction_text) == unresolved,
             "contradictory_templates must cover exactly the unresolved baseline factors")
    for key in unresolved:
        _text(contradiction_text[key], f"contradictory template {key}")

    def render(overrides=None):
        overrides = dict(overrides or {})
        _require(not (set(overrides) - set(baseline)), "renderer overrides contain unknown factors")
        row = {**baseline, **overrides}
        remaining = {k for k, v in row.items() if _contradictory(v)}
        if remaining:
            # An unresolved source can be preserved, but cannot be an experiment.
            _require(not overrides, "resolve all contradictory baseline factors before rendering interventions")
            _require(not activations and not feasibility,
                     "unresolved baseline with feasibility/activation constraints requires an explicit reconstruction outside this renderer")
        else:
            normalized = doe._normalize_row(row, factors, activations)
            _require(normalized is not None, "renderer row has invalid active levels or applicability")
            _require(all(p(normalized) for p in feasibility), "renderer row violates feasibility constraints")
            row = normalized
        blocks = dict(frozen)
        for f in factors:
            value = row[f.name]
            if f.name in remaining:
                blocks[f.name] = contradiction_text[f.name]
            elif value == doe.NOT_APPLICABLE:
                blocks[f.name] = ""
            elif f.kind == "continuous":
                blocks[f.name] = templates[f.name].replace("{value}", format(value, ".17g"))
            else:
                blocks[f.name] = templates[f.name][value]
        return {key: blocks[key] for key in order}

    return render, separator


def _validate_full(doc, factors):
    _require(type(doc.get("schema_version")) is int and doc["schema_version"] == SCHEMA_VERSION,
             "schema_version must be 1 (minimal inventories should use encode_factors)")
    _text(doc.get("source_prompt_id"), "source_prompt_id")
    source = doc.get("source_prompt")
    _text(source, "source_prompt", empty=True)
    spans = doc.get("span_audit")
    _require(isinstance(spans, list), "span_audit must be a list")
    ids = {f.name for f in factors}
    by_span, behavioral, mapped, cursor = {}, set(), set(), 0
    for span in spans:
        _require(isinstance(span, dict), "span must be an object")
        sid = span.get("span")
        _text(sid, "span ID")
        _require(sid not in by_span, f"duplicate span {sid}")
        start, end = span.get("start"), span.get("end")
        _require(type(start) is int and type(end) is int and start == cursor and start < end <= len(source),
                 f"{sid}: spans must partition source_prompt using character offsets")
        _require(span.get("text") == source[start:end], f"{sid}: source text does not match offsets")
        cursor = end
        classification, disposition = span.get("classification"), span.get("disposition")
        support = _unique(span.get("factors", []), f"{sid} factors")
        _require(support <= ids, f"{sid}: mapping references unknown factors")
        if classification == "BEHAVIORAL":
            behavioral.add(sid)
            _require(disposition in ("MAPPED", "RESIDUAL"), f"{sid}: invalid behavioral disposition")
            if disposition == "MAPPED":
                _require(bool(support), f"{sid}: mapped span requires factor IDs")
                mapped.add(sid)
            else:
                _require(not support, f"{sid}: residual cannot also map to factors")
        else:
            _require(classification in ("NON-BEHAVIORAL", "META") and disposition == classification and not support,
                     f"{sid}: invalid non-behavioral disposition")
            _text(span.get("reason"), f"{sid} reason")
        by_span[sid] = span
    _require(cursor == len(source), "span_audit does not cover the complete source_prompt")
    for key, count in (("span_count", len(spans)), ("behavioral_span_count", len(behavioral))):
        _require(type(doc.get(key)) is int and doc[key] == count, f"{key} does not match span_audit")
    computed = len(mapped) / len(behavioral) if behavioral else 1.0
    _require(doc.get("coverage") == computed, f"coverage disagrees with span_audit: computed {computed}")
    _require(computed == 1, "span coverage < 1: incomplete factorization")
    baseline = doc.get("theta_0")
    _require(isinstance(baseline, dict) and set(baseline) == ids, "theta_0 must assign every factor exactly once")
    required_gates = {"definable", "intervenable", "behaviorally_observable", "structurally_non_redundant", "evidenced"}
    for raw, f in zip(doc["factor_inventory"], factors):
        evidence = _unique(raw.get("evidence"), f"{f.name} evidence")
        _require(bool(evidence) and evidence <= mapped, f"{f.name}: evidence must reference mapped behavioral spans")
        _require(evidence == {sid for sid in mapped if f.name in by_span[sid]["factors"]}, f"{f.name}: evidence and span mappings disagree")
        _require(raw.get("status") in ("EXPLICIT", "IMPLIED", "SUPPRESSED", "CONTRADICTORY"), f"{f.name}: invalid evidence status")
        for field in ("behavioral_definition", "intervention_test", "observable_behavior"):
            _text(raw.get(field), f"{f.name} {field}")
        gates = raw.get("gates")
        _require(isinstance(gates, dict) and required_gates <= set(gates) and all(gates[k] == "pass" for k in required_gates), f"{f.name}: all admission gates must pass")
        if f.ptype in ("ORDINAL", "NESTED_ORDINAL"):
            field = "nesting_justification" if f.ptype == "NESTED_ORDINAL" else "ordering_justification"
            _text(raw.get(field), f"{f.name} {field}")
        if f.kind == "continuous":
            _text(raw.get("units"), f"{f.name} units")
        else:
            provenance = raw.get("level_provenance")
            _require(isinstance(provenance, dict) and set(provenance) == set(f.levels) and
                     all(v in ("SOURCE", "PROPOSED") for v in provenance.values()),
                     f"{f.name}: level_provenance must label each level SOURCE or PROPOSED")
        if raw["status"] == "IMPLIED":
            _text(raw.get("inference_note"), f"{f.name} inference_note")
        value = _baseline_value(baseline[f.name])
        _require("baseline" in raw and raw["baseline"] == baseline[f.name], f"{f.name}: baseline disagrees with theta_0")
        if raw["status"] == "CONTRADICTORY":
            _require(_contradictory(value), f"{f.name}: contradictory status requires an unresolved baseline")
            candidates = value.get("candidate_values")
            _require(isinstance(candidates, list) and len(candidates) >= 2 and len({str(v) for v in candidates}) == len(candidates), f"{f.name}: contradiction requires distinct candidate values")
            _require(all(doe._normalize_row({f.name: v}, [f]) is not None for v in candidates), f"{f.name}: invalid contradiction candidate")
            support = _unique(value.get("spans"), f"{f.name} contradiction spans")
            _require(len(support) >= 2 and support <= evidence, f"{f.name}: contradiction spans must be evidence")
        elif value != doe.NOT_APPLICABLE:
            _require(doe._normalize_row({f.name: value}, [f]) is not None, f"{f.name}: invalid baseline value")
        if f.kind == "discrete" and value != doe.NOT_APPLICABLE:
            source_values = value["candidate_values"] if _contradictory(value) else [value]
            for source_value in source_values:
                canonical_value = doe._normalize_row({f.name: source_value}, [f])[f.name]
                _require(raw["level_provenance"][canonical_value] == "SOURCE",
                         f"{f.name}: baseline commitments must have SOURCE level provenance")
    render, separator = _render_function(doc, factors)
    blocks = render()
    canonical = separator.join(blocks.values())
    _require(doc.get("canonical_reconstruction") == canonical, "canonical_reconstruction does not equal deterministic baseline rendering")
    _, activations, _ = doe.compile_constraints(doc.get("constraints", []), factors)
    normalized_baseline = None if any(_contradictory(v) for v in baseline.values()) else doe._normalize_row(
        {k: _baseline_value(v) for k, v in baseline.items()}, factors, activations)
    inactive_baseline = {k for k, v in (normalized_baseline or {}).items() if v == doe.NOT_APPLICABLE}
    templates = doc["renderer_spec"]["factor_templates"]
    claims = doc.get("claims")
    _require(isinstance(claims, list), "claims must be a list")
    claim_ids, claimed_spans, claimed_factors = set(), set(), set()
    for claim in claims:
        _require(isinstance(claim, dict), "claim must be an object")
        cid, sid, block = claim.get("id"), claim.get("span"), claim.get("block")
        _text(cid, "claim ID")
        _require(cid not in claim_ids, f"duplicate claim {cid}")
        claim_ids.add(cid)
        _require(isinstance(sid, str) and sid in mapped and isinstance(block, str) and block in by_span[sid]["factors"], f"{cid}: claim must link a mapped source span to its factor block")
        for field in ("condition", "action", "source_quote"):
            _text(claim.get(field), f"{cid} {field}")
        _require(claim.get("modality") in ("MUST", "SHOULD", "MAY", "MUST NOT"), f"{cid}: invalid modality")
        _require(claim["source_quote"] in by_span[sid]["text"], f"{cid}: source quote missing")
        if block in inactive_baseline:
            _require(claim.get("inactive") is True and claim.get("rendered_quote") is None,
                     f"{cid}: inactive baseline claim must be marked inactive with rendered_quote: null")
            # The dormant commitment remains auditable in its stored level template,
            # although it has no active baseline rendering under this parent state.
            level = claim.get("template_level")
            quote = claim.get("template_quote")
            _text(quote, f"{cid} template_quote")
            template = templates[block]
            if isinstance(template, dict):
                _require(isinstance(level, (str, int, float, bool)) and level in template,
                         f"{cid}: inactive claim requires a valid template_level")
                template = template[level]
            _require(quote in template, f"{cid}: inactive claim quote missing from template")
        else:
            _require(not claim.get("inactive", False), f"{cid}: active claim cannot be marked inactive")
            _text(claim.get("rendered_quote"), f"{cid} rendered_quote")
            _require(claim["rendered_quote"] in blocks[block], f"{cid}: rendered quote missing")
        claimed_spans.add(sid)
        claimed_factors.add(block)
    _require(claimed_spans == mapped and claimed_factors == ids, "claims must cover every mapped span and factor")
    check = doc.get("reconstruction_check")
    _require(isinstance(check, dict) and check.get("missing") == [] and check.get("added") == [], "reconstruction_check must record no missing or added claims")
    _require(check.get("verification") == "CLAIM_AUDITED", "reconstruction verification must be CLAIM_AUDITED; this is not empirical equivalence")
    for key in ("behavioral_claims_in_original", "behavioral_claims_in_reconstruction"):
        _require(type(check.get(key)) is int and check[key] == len(claims), f"{key} does not match claims")
    unrepresented = doc.get("unrepresented_dimensions", [])
    _require(isinstance(unrepresented, list), "unrepresented_dimensions must be a list")
    for dimension in unrepresented:
        _require(isinstance(dimension, dict) and dimension.get("prompt_status") == "SILENT" and dimension.get("runtime_default") == "UNKNOWN" and "baseline" not in dimension,
                 "unrepresented dimensions must remain SILENT/UNKNOWN without baselines")
        _text(dimension.get("dimension"), "unrepresented dimension ID")
        _require(dimension.get("dimension") not in ids, "unrepresented dimension cannot be an admitted factor")


def validate_document(doc: dict, *, strict: bool = True) -> list[str]:
    """Validate schema v1; legacy documents can obtain non-strict diagnostics.

    No-behavior source documents may have empty inventories and full audit
    coverage; generating a population from them is a separate invalid operation.
    """
    _require(isinstance(doc, dict), "factorization must be an object")
    inv = doc.get("factor_inventory")
    if inv == [] and type(doc.get("schema_version")) is int and doc["schema_version"] == 1:
        factors = []
    else:
        factors = doe.encode_factors(inv)
    issues = []
    coverage = doc.get("coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or not 0 <= coverage <= 1:
        issues.append("coverage must be a finite number in [0, 1]")
    elif coverage < 1:
        issues.append(f"span coverage {coverage} < 1: incomplete factorization")
    if not isinstance(doc.get("residuals", []), list):
        issues.append("residuals must be a list")
    if doc.get("residuals"):
        issues.append("residual spans remain unfactorized")
    try:
        doe.compile_constraints(doc.get("constraints", []), factors)
        _validate_full(doc, factors)
    except (ValueError, TypeError, KeyError) as exc:
        issues.append(str(exc))
    if issues:
        if strict:
            raise ValueError("invalid factorization: " + "; ".join(issues))
        for issue in issues:
            warnings.warn(f"factorization: {issue}", stacklevel=2)
    return issues


def load_document(path_or_text):
    """Load YAML with duplicate-key detection and validate the complete contract."""
    doc = doe._load_yaml(path_or_text)
    validate_document(doc)
    return doc


def document_renderer(doc: dict):
    """Return a baseline/override renderer after validating the full document.

    Returned blocks preserve slot order. Use render_prompt for the assembled text.
    The document is copied so later caller edits cannot change renderer behavior.
    """
    doc = deepcopy(doc)
    validate_document(doc)
    factors = doe.encode_factors(doc["factor_inventory"]) if doc["factor_inventory"] else []
    return _render_function(doc, factors)[0]


def render_prompt(doc: dict, overrides: dict | None = None) -> str:
    render = document_renderer(doc)
    return doc["renderer_spec"].get("separator", "\n\n").join(render(overrides).values())


def build_document_population(doc: dict, n: int = 32, *, active_factors=None,
                              fixed: dict | None = None, method="sobol", seed=0, **design_kw):
    """Design active factors while preserving all fixed commitments.

    Returns candidates plus rows/diagnostics and both baseline controls. Unresolved
    factors must be varied or explicitly fixed to a valid value for experiments.
    """
    doc = deepcopy(doc)
    validate_document(doc)
    inv = doc["factor_inventory"]
    _require(bool(inv), "cannot design over a document with no behavioral factors")
    ids = {f["id"] for f in inv}
    active = ids if active_factors is None else _unique(list(active_factors), "active_factors")
    _require(active <= ids, "active_factors references unknown factors")
    fixed = dict(fixed or {})
    _require(set(fixed) <= ids and not (set(fixed) & active), "fixed assignments must reference declared, non-active factors")
    assignments = {k: _baseline_value(v) for k, v in doc["theta_0"].items() if k not in active}
    assignments.update(fixed)
    _require(not any(_contradictory(v) for v in assignments.values()), "resolve or activate contradictory baseline factors before design")
    result = doe.generate_design(inv, doc.get("constraints", []), n, method=method,
                                 seed=seed, fixed=assignments, **design_kw)
    render = document_renderer(doc)
    separator = doc["renderer_spec"].get("separator", "\n\n")
    # Full rows resolve contradictory baselines; every candidate contains one
    # assembled prompt, preventing adapters from accidentally dropping framing.
    result["population"] = [{"system_prompt": separator.join(render(row).values())} for row in result["rows"]]
    result["source_prompt"] = doc["source_prompt"]
    result["baseline_prompt"] = separator.join(render().values())
    result["active_factors"] = sorted(active)
    result["fixed"] = assignments
    return result
