"""GRACE-style baseline head-to-head.

The yaz-gen thesis: a SUPERVISED-ATOM edit generalizes to unseen phrasings where a
GRACE/SERAC-style SURFACE-ACTIVATION codebook (paraphrase-bound) cannot. This script
implements a faithful GRACE editor on the SAME backbone (yaz_gen_semantic_v2.pt), the
SAME 8 edits, the SAME held-out split, scored on the SAME first-byte metric as Yaz's
headline 0.675 — same backbone, edits, split, and metric for both.

GRACE (Hartvigsen et al. 2023): a discrete key->value codebook inserted at a hidden layer.
  key   = backbone hidden activation (ln_final, last position) of the edit prompt
  value = the new capital's first byte (what the edit should emit)
  defer = if a query's nearest key is within radius epsilon -> emit value; else fall back
          to the unedited model. Paraphrase-bound because held-out phrasings land far from
          the stored keys in activation space.

To avoid a strawman, we give GRACE TWO settings —
  grace_1key : one key per edit = the single in-dist edit request (standard GRACE).
  grace_8key : EIGHT keys per edit = all 8 train phrasings of the edited country
               (a maximally generous codebook; the most paraphrase coverage it can get
               from the same data Yaz's router was built on).
Distance reported under both L2 and cosine. For each we sweep epsilon and report
edit-gen at the LARGEST epsilon that keeps locality at 0 side-effects (Yaz's locality),
plus the full curve so it is not cherry-picked.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
HELDOUT = ROOT / "data" / "probes_para_heldout.jsonl"
INDIST = ROOT / "data" / "probes_para_indist.jsonl"
UPDATE_PAIRS = [["France", "Peru"], ["Japan", "Germany"], ["Egypt", "Iran"],
                ["Germany", "Spain"], ["Brazil", "Bulgaria"], ["Italy", "Norway"],
                ["Poland", "Greece"], ["Canada", "Cuba"]]


def ids(s):
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def hidden(model, prompt):
    """Backbone hidden state (ln_final output) at the LAST position — GRACE's key space."""
    x = ids(prompt); B, T = x.shape
    pos = torch.arange(T)
    h = model.tok_embed(x) + model.pos_embed(pos)[None]
    for blk in model.blocks:
        h = blk(h)
    h = model.ln_final(h)
    return h[0, -1].numpy().astype(np.float32)


def dists(q, K, metric):
    if metric == "l2":
        return np.linalg.norm(K - q[None], axis=1)
    qn = q / (np.linalg.norm(q) + 1e-8)
    Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
    return 1.0 - (Kn @ qn)                      # cosine distance


def main():
    torch.set_num_threads(1)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    capof = {}
    for l in (ROOT / "data" / "facts_para_train.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); capof.setdefault(r["country"], r["capital"])

    edited = {tgt for tgt, _ in UPDATE_PAIRS}
    held = [json.loads(l) for l in HELDOUT.read_text().splitlines() if l]
    indist = [json.loads(l) for l in INDIST.read_text().splitlines() if l]

    # Held-out queries of EDITED countries (the edit-gen test set: 8 x 5 = 40).
    eg_q = [(p["country"], p["prompt"], capof[dict(UPDATE_PAIRS)[p["country"]]][0])
            for p in held if p["country"] in edited]
    # Locality domain: in-dist probes of NON-edited countries (must NOT fire).
    loc_q = [(p["country"], p["prompt"]) for p in indist if p["country"] not in edited]
    Heg = np.stack([hidden(model, p) for _, p, _ in eg_q])
    Hloc = np.stack([hidden(model, p) for _, p in loc_q])

    def build_keys(n_key):
        keys, kc, kv = [], [], []           # key vec, owner country, value first-byte
        for tgt, src in UPDATE_PAIRS:
            tmpls = TRAIN_TEMPLATES[:n_key]
            for t in tmpls:
                keys.append(hidden(model, t.format(C=tgt)))
                kc.append(tgt); kv.append(capof[src][0])
        return np.stack(keys), kc, kv

    report = {}
    for n_key, tag in [(1, "grace_1key"), (8, "grace_8key")]:
        K, kc, kv = build_keys(n_key)
        for metric in ("l2", "cosine"):
            # nearest-key distance + which key, for every eg and loc query
            Deg = np.stack([dists(q, K, metric) for q in Heg])      # (40, |K|)
            Dloc = np.stack([dists(q, K, metric) for q in Hloc])    # (|loc|, |K|)
            nn_eg = Deg.argmin(1); nn_eg_d = Deg.min(1)
            nn_loc_d = Dloc.min(1)
            # candidate epsilons = sorted unique nearest-distances
            cand = sorted(set(np.round(np.concatenate([nn_eg_d, nn_loc_d]), 5)))
            curve = []
            for eps in cand:
                # fires on eg query if within eps; correct if the fired key's value matches target byte
                fire = nn_eg_d <= eps
                correct = sum(1 for i in range(len(eg_q))
                              if fire[i] and kv[nn_eg[i]] == eg_q[i][2])
                side = int((nn_loc_d <= eps).sum())                 # non-edited prompts that fire
                curve.append({"eps": float(eps), "edit_gen": correct / len(eg_q),
                              "fires": int(fire.sum()), "side_effects": side})
            # largest eps with 0 side-effects
            zero = [c for c in curve if c["side_effects"] == 0]
            best0 = max(zero, key=lambda c: c["eps"]) if zero else {"eps": 0.0, "edit_gen": 0.0,
                                                                     "fires": 0, "side_effects": 0}
            # best edit-gen at any eps keeping side-effects <= Yaz's locality (0)
            report[f"{tag}_{metric}"] = {
                "edit_gen_at_0_sideeffects": best0["edit_gen"],
                "eps_at_0_sideeffects": best0["eps"],
                "fires_at_0_sideeffects": best0["fires"],
                "max_edit_gen_any_eps": max(c["edit_gen"] for c in curve),
                "side_effects_at_max_edit_gen": next(c["side_effects"] for c in curve
                                                     if c["edit_gen"] == max(cc["edit_gen"] for cc in curve)),
                "curve": curve,
            }
            print(f"[{tag}/{metric}] edit-gen @0-side-effects = {best0['edit_gen']:.3f} "
                  f"(eps={best0['eps']:.4f}, fires {best0['fires']}/40) | "
                  f"max edit-gen any-eps = {report[f'{tag}_{metric}']['max_edit_gen_any_eps']:.3f} "
                  f"(@ {report[f'{tag}_{metric}']['side_effects_at_max_edit_gen']} side-effects)", flush=True)

    report["yaz_semantic_v2_frozen_editgen"] = 0.675
    report["note"] = ("Yaz (Engram-routed supervised atom) = 0.675 first-byte edit-gen at 0 "
                      "side-effects on the same 40 held-out probes. Compare to GRACE rows above.")
    out = ROOT / "results" / "t2_grace_headtohead.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nYAZ (supervised atom): edit-gen 0.675 @ 0 side-effects (same split)")
    print(f"wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
