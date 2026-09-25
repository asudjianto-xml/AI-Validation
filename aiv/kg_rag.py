#!/usr/bin/env python
"""KG-grounded RAG over the credit-model report.

Query path: geometric KG retrieval (no language model) -> grounded facts with
their source provenance -> those facts and source spans are served to an LLM as
context -> the LLM answers from correct evidence. This is retrieval-augmented
generation whose context is precise and provenanced rather than fuzzy top-k
chunks.

Retrieval is a thin caller of the knowlytix store API (the same API the gms MCP
server wraps); the answer is produced by the Claude CLI, run airtight (no tools,
no MCP) so the only thing that varies between the two arms is the context.

Run in a Python environment with knowlytix installed, and with the `claude` CLI on
PATH for the answering step:
  python -m aiv.kg_rag
GMS_STORES_DIR and GMS_STORE select the store. The directory defaults to the
checkout's stores/ when it exists and to ./stores otherwise; the store defaults
to "model_dev_report".
"""
from __future__ import annotations

import json
import os
import subprocess

from knowlytix.knowledge.mcp_tools import ActiveStore

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_STORES = os.path.join(_REPO, "stores")
STORES_DIR = os.environ.get("GMS_STORES_DIR") or (
    _DEFAULT_STORES if os.path.isdir(_DEFAULT_STORES) else os.path.abspath("stores"))
STORE = os.environ.get("GMS_STORE", "model_dev_report")


def load_tools():
    active = ActiveStore(STORES_DIR)
    if not active.use(STORE):
        raise SystemExit(f"store {STORE!r} not found under {STORES_DIR}")
    return active.tools, os.path.join(STORES_DIR, STORE)


def load_provenance(store_dir: str) -> dict:
    """Map each stored triple to its source span(s) from provenance.json."""
    prov = json.load(open(os.path.join(store_dir, "provenance.json")))
    idx = {}
    for e in prov.get("triples", []):
        h, r, t = e["triple"]
        idx[(h, r, t)] = [ev.get("span", "") for ev in e.get("evidence", [])]
    return idx


def grounded_facts(question: str, tools):
    """Geometric parse binds the question to entities; expand each bound head to
    all of its asserted facts, so the context is complete for that entity."""
    r = tools.rag_retrieve(question)
    heads = [b[0] for b in r.get("bound_triples", [])]
    facts, seen = [], set()
    for h in dict.fromkeys(heads):                     # preserve order, dedup
        for rel, t in tools.lookup(h).get("as_head", []):
            key = (h, rel, t)
            if key not in seen:
                seen.add(key)
                facts.append(key)
    return r, facts


def grounded_facts_compiled(question: str, tools):
    """Same expansion as grounded_facts, but bind the question with the store's
    fine-tuned query compiler (rag_retrieve_compiled) instead of the geometric
    parser. Returns ([], []) when the compiler abstains."""
    r = tools.rag_retrieve_compiled(question)
    if not isinstance(r, dict) or r.get("error") or r.get("decision") != "accept":
        return r, []
    heads = [b[0] for b in r.get("bound_triples", [])]
    facts, seen = [], set()
    for h in dict.fromkeys(heads):
        for rel, t in tools.lookup(h).get("as_head", []):
            key = (h, rel, t)
            if key not in seen:
                seen.add(key)
                facts.append(key)
    return r, facts


def build_context(facts, prov) -> str:
    lines = []
    for (h, r, t) in facts:
        spans = prov.get((h, r, t), [])
        src = f"   [source: {spans[0]}]" if spans and spans[0] else ""
        lines.append(f"- {h} {r} {t}{src}")
    return "\n".join(lines) if lines else "(no grounded facts retrieved)"


def ask_llm(prompt: str, timeout: int = 180) -> str:
    """Claude CLI, airtight: no tools, no MCP servers."""
    cmd = ["claude", "-p", prompt, "--strict-mcp-config",
           "--mcp-config", '{"mcpServers":{}}', "--allowedTools", ""]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return out.stdout.strip() or f"[no output; stderr: {out.stderr.strip()[:200]}]"


def answer_grounded(question: str, tools, prov):
    r, facts = grounded_facts(question, tools)
    ctx = build_context(facts, prov)
    prompt = (
        "Answer the question using ONLY the grounded facts below, each shown with "
        "its source from the model report. If the facts do not answer it, say you "
        "cannot answer from the given facts.\n\n"
        f"GROUNDED FACTS (retrieved from a knowledge graph over the report):\n{ctx}\n\n"
        f"QUESTION: {question}\nANSWER:"
    )
    return ask_llm(prompt), ctx, r.get("decision")


def answer_plain(question: str, report_text: str):
    """Baseline: the whole report in context, no KG grounding."""
    prompt = (
        "Answer the question using the model report below.\n\n"
        f"REPORT:\n{report_text}\n\nQUESTION: {question}\nANSWER:"
    )
    return ask_llm(prompt)


QUESTIONS = [
    "What is the AUC value and did it meet its threshold?",
    "What is the monotonicity constraint on dti?",
    "Which feature has the highest permutation importance (pfi_auc_drop)?",
]


def main():
    tools, sdir = load_tools()
    prov = load_provenance(sdir)
    report = open(os.path.join(sdir, "documents", "combined.md")).read()
    for q in QUESTIONS:
        print("=" * 88)
        print("Q:", q)
        g, ctx, dec = answer_grounded(q, tools, prov)
        print(f"\n[KG retrieval decision: {dec}]")
        print("[KG-grounded context served to LLM]")
        print(ctx)
        print("\n[Grounded answer]")
        print(g)
        print("\n[Plain LLM, full report in context]")
        print(answer_plain(q, report))


if __name__ == "__main__":
    main()
