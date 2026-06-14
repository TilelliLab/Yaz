"""F0 feasibility gate: can Engram semantic embeddings route held-out phrasings
to the correct country-atom?

Mechanism under test (the keystone): each country's atom key = mean Engram embedding
of its 8 TRAIN-template prompt prefixes (frozen). A held-out phrasing is routed to the
nearest centroid by cosine. This is EXACTLY the routing the `semantic-frozen` model will
use at eval time, so this script is a tight upper-bound predictor of edit-generalization:

    edit-gen  ~=  held-out routing accuracy  x  decoder reliability

Decision gate:
  held-out routing acc >= 0.5  -> GO (build + train both arms)
  held-out routing acc <  0.5  -> NO-GO (semantic re-keying can't lift edit-gen)

Leak-free by construction: centroids use ONLY train templates; held-out templates never
enter any key. MiniLM is frozen (never fine-tuned on these facts).
"""
from __future__ import annotations
import os
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("YAZ_EMBEDDER_PATH", ""))
sys.path.insert(0, str(ROOT))

from scripts.gen_paraphrase_data import TRAIN_TEMPLATES, TEST_TEMPLATES, pairs  # noqa: E402
from engram import Embedder  # noqa: E402


def main():
    ps = pairs()  # [(country, capital), ...] 50 facts
    countries = [c for c, _ in ps]
    cidx = {c: i for i, c in enumerate(countries)}
    n = len(countries)

    emb = Embedder(prefer="auto")
    print(f"Engram mode={emb.mode} dim={emb.dim}  countries={n}")
    assert emb.mode == "st", f"need semantic embeddings, got mode={emb.mode}"

    # Routing keys are PROMPT PREFIXES (country filled in, no capital) -- the exact
    # string the model routes on at generation time.
    train_prompts, train_lbl = [], []
    for c, _ in ps:
        for tmpl in TRAIN_TEMPLATES:
            train_prompts.append(tmpl.format(C=c))
            train_lbl.append(cidx[c])
    held_prompts, held_lbl = [], []
    for c, _ in ps:
        for tmpl in TEST_TEMPLATES:
            held_prompts.append(tmpl.format(C=c))
            held_lbl.append(cidx[c])

    train_vec = emb.encode(train_prompts)            # (50*8, 384) L2-normed
    held_vec = emb.encode(held_prompts)              # (50*5, 384)
    train_lbl = np.array(train_lbl)
    held_lbl = np.array(held_lbl)

    # Per-country centroid over its 8 train prefixes; renormalize for cosine.
    cent = np.zeros((n, train_vec.shape[1]), dtype=np.float32)
    for i in range(n):
        m = train_vec[train_lbl == i].mean(0)
        cent[i] = m / (np.linalg.norm(m) + 1e-8)

    def route(vecs):
        return (vecs @ cent.T).argmax(1)  # nearest centroid by cosine (vecs are unit-norm)

    # Sanity: train routing accuracy (should be very high).
    tr_pred = route(train_vec)
    tr_acc = float((tr_pred == train_lbl).mean())

    # Headline: held-out routing accuracy.
    ho_pred = route(held_vec)
    ho_acc = float((ho_pred == held_lbl).mean())

    # Per held-out template breakdown (which phrasings are hard).
    per_tmpl = []
    k = len(TEST_TEMPLATES)
    ho_pred_g = ho_pred.reshape(n, k)
    ho_lbl_g = held_lbl.reshape(n, k)
    for tid in range(k):
        acc = float((ho_pred_g[:, tid] == ho_lbl_g[:, tid]).mean())
        per_tmpl.append({"test_template_id": tid, "template": TEST_TEMPLATES[tid], "acc": acc})

    # Which countries fail any held-out phrasing (confusion sample).
    fails = []
    for ci in range(n):
        for tid in range(k):
            if ho_pred_g[ci, tid] != ci:
                fails.append({"country": countries[ci],
                              "phrasing": TEST_TEMPLATES[tid].format(C=countries[ci]),
                              "routed_to": countries[int(ho_pred_g[ci, tid])]})

    report = {
        "engram_mode": emb.mode, "dim": int(emb.dim), "n_countries": n,
        "train_routing_acc": tr_acc, "n_train_prompts": len(train_prompts),
        "heldout_routing_acc": ho_acc, "n_heldout_prompts": len(held_prompts),
        "per_template": per_tmpl,
        "n_heldout_fails": len(fails), "fail_examples": fails[:25],
        "gate": "GO" if ho_acc >= 0.5 else "NO-GO", "gate_threshold": 0.5,
    }
    out = ROOT / "results" / "f0_routing_gate.json"
    out.write_text(json.dumps(report, indent=2))

    print("\n=== F0 ROUTING GATE ===")
    print(f"train  routing acc : {tr_acc:.3f}  ({len(train_prompts)} prompts)")
    print(f"HELD-OUT routing acc: {ho_acc:.3f}  ({len(held_prompts)} prompts)  <-- headline")
    print("per held-out template:")
    for p in per_tmpl:
        print(f"  t{p['test_template_id']} acc={p['acc']:.3f}  {p['template']!r}")
    print(f"held-out fails: {len(fails)}/{len(held_prompts)}")
    for f in fails[:10]:
        print(f"  {f['phrasing']!r} -> {f['routed_to']}")
    print(f"\nGATE: {report['gate']}  (threshold {report['gate_threshold']}, "
          f"held-out acc {ho_acc:.3f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
