"""Paraphrase benchmark for the GENERALIZATION push — disjoint train/test templates.

The field's open metric where Yaz (and all GRACE-class lookup editors) lose is
*generalization*: an edit made in one phrasing should hold under other phrasings.
We build a ZsRE-style split:
  - TRAIN templates (8): the model trains on these.
  - HELD-OUT TEST templates (5): DISJOINT, never seen in training. The edit-transfer
    test probes only these — so any transfer is real generalization, not memorization.

All templates END at the answer (causal LM: the capital is the next token after the prompt),
so routing supervision lands on the same answer position regardless of phrasing.

Country->capital pairs are taken from the existing facts_50.jsonl (50 facts), deduped.

Outputs:
  data/facts_para_train.jsonl       — 50 facts x 8 train templates (text=prefix+capital+".")
  data/probes_para_indist.jsonl     — reliability probes, TRAIN template #0
  data/probes_para_heldout.jsonl    — generalization probes, the 5 TEST templates (250 rows)
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "facts_50.jsonl"

# All prefixes end with a space; text = prefix + capital + "."
TRAIN_TEMPLATES = [
    "The capital of {C} is ",
    "{C}'s capital is ",
    "The capital city of {C} is ",
    "Capital of {C}: ",
    "In {C}, the capital is ",
    "{C} has its capital at ",
    "The country {C} has its capital, which is ",
    "Q: What is the capital of {C}? A: ",
]
# DISJOINT held-out phrasings — never trained on.
TEST_TEMPLATES = [
    "{C} — capital: ",
    "The seat of government of {C} is located in ",
    "If you visit {C}, the capital you arrive in is ",
    "The administrative capital of {C} is ",
    "Name the capital of {C}: ",
]


def pairs():
    seen, out = set(), []
    for l in SRC.read_text().splitlines():
        if not l:
            continue
        r = json.loads(l)
        if r["country"] in seen:
            continue
        seen.add(r["country"])
        out.append((r["country"], r["capital"]))
    return out


def main():
    ps = pairs()
    # training facts: 8 phrasings per fact, tagged with template_id (0..7)
    train_rows = []
    for c, cap in ps:
        for tid, tmpl in enumerate(TRAIN_TEMPLATES):
            train_rows.append({"country": c, "capital": cap, "template_id": tid,
                               "text": tmpl.format(C=c) + cap + "."})
    (ROOT / "data" / "facts_para_train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n")

    # reliability probes: in-distribution (train template #0)
    indist = [{"country": c, "capital": cap,
               "prompt": TRAIN_TEMPLATES[0].format(C=c), "expected_first_byte": cap[0]}
              for c, cap in ps]
    (ROOT / "data" / "probes_para_indist.jsonl").write_text(
        "\n".join(json.dumps(r) for r in indist) + "\n")

    # generalization probes: held-out templates (one row per country x test-template)
    held = []
    for c, cap in ps:
        for tid, tmpl in enumerate(TEST_TEMPLATES):
            held.append({"country": c, "capital": cap, "test_template_id": tid,
                         "prompt": tmpl.format(C=c), "expected_first_byte": cap[0]})
    (ROOT / "data" / "probes_para_heldout.jsonl").write_text(
        "\n".join(json.dumps(r) for r in held) + "\n")

    print(f"facts: {len(ps)}  train_rows: {len(train_rows)} ({len(TRAIN_TEMPLATES)} tmpl/fact)")
    print(f"indist probes: {len(indist)}  heldout probes: {len(held)} "
          f"({len(TEST_TEMPLATES)} tmpl/fact)")
    print("train/test templates are DISJOINT:",
          set(TRAIN_TEMPLATES).isdisjoint(set(TEST_TEMPLATES)))


if __name__ == "__main__":
    main()
