"""U4 — fair head-to-head vs a GRACE-style activation-key lookup editor (same split).

Prior analysis places this approach in the GRACE/SERAC/PENME family. This MEASURES
the structural claim instead of asserting it: on the SAME trained backbone, SAME 8 edits, SAME 5
held-out phrasings, how does a faithful GRACE-style editor compare on edit-generalization?

GRACE = discrete key->value cache at a layer with a deferral radius eps. We key on the post-ln_final
hidden at the answer position (the residual the unembed sees):
  - edit (tgt<-src): key = h_last(in-dist edit prompt for tgt); stored answer = src capital first byte.
  - inference on a probe: h = h_last(probe); nearest stored key by L2; if dist < eps -> emit stored
    answer byte, else the model's own first byte.
Sweep eps. edit-gen = first-byte hits on held-out phrasings of the 8 edited targets. locality =
spurious deferrals on the OTHER 49 countries' held-out probes (an edit firing where it shouldn't).
FAIR comparison point: max GRACE edit-gen at locality == 0 (ours: edit-gen 0.675 first-byte, locality 0).

This is FIRST-BYTE (same resolution our 0.675 headline uses). EVAL-ONLY on yaz_gen_semantic_v2.pt.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES, TEST_TEMPLATES, pairs

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
EDITS = [("France","Peru"),("Japan","Germany"),("Egypt","Iran"),("Germany","Spain"),
         ("Brazil","Bulgaria"),("Italy","Norway"),("Poland","Greece"),("Canada","Cuba")]
EDIT_PROMPT = TRAIN_TEMPLATES[0]   # "The capital of {C} is " — the in-dist edit phrasing


def ids(s): return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


def make_h_last(model):
    @torch.no_grad()
    def h_last(prompt):
        x = ids(prompt); T = x.shape[1]
        pos = torch.arange(T)
        h = model.tok_embed(x) + model.pos_embed(pos)[None]
        for blk in model.blocks: h = blk(h)
        h = model.ln_final(h)
        return h[0, -1].clone()                 # (d_model,)
    return h_last


@torch.no_grad()
def model_first_byte(model, prompt, n=4):
    p = ids(prompt); out = p; mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        out = torch.cat([out, model(ctx)[:, -1].argmax(-1, keepdim=True)], dim=1)
    g = bytes(out[0, len(p[0]):].tolist()).decode("latin-1", "ignore").lstrip()
    return g[:1]


def main():
    ps = pairs(); order = [c for c, _ in ps]; cap = {c: cp for c, cp in ps}
    ck = torch.load(CKPT, map_location="cpu")
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    h_last = make_h_last(model)

    # GRACE store: one key per edit (keyed on the edit prompt), answer = new capital first byte.
    keys = []; answers = []
    edited = set()
    for tgt, src in EDITS:
        keys.append(h_last(EDIT_PROMPT.format(C=tgt)))
        answers.append(cap[src][0])
        edited.add(tgt)
    K = torch.stack(keys)                          # (8, d)

    def nearest(h):
        d = (K - h).norm(dim=-1); i = int(d.argmin()); return i, float(d[i])

    # held-out probes for edited targets (the edit-gen test) and for others (locality test)
    edit_probes = []      # (tgt, src, prompt, h, new_first)
    for tgt, src in EDITS:
        for t in TEST_TEMPLATES:
            pr = t.format(C=tgt)
            edit_probes.append((tgt, src, pr, h_last(pr), cap[src][0]))
    other_probes = []     # (country, prompt, h) for the 49 non-edited countries
    for c in order:
        if c in edited: continue
        for t in TEST_TEMPLATES:
            pr = t.format(C=c)
            other_probes.append((c, pr, h_last(pr)))

    # precompute model's own first byte on edit probes (the non-deferred fallback)
    own = {pr: model_first_byte(model, pr) for (_, _, pr, _, _) in edit_probes}

    # distance diagnostics
    ed_d = np.array([nearest(h)[1] for (_, _, _, h, _) in edit_probes])
    ot_d = np.array([nearest(h)[1] for (_, _, h) in other_probes])
    print(f"nearest-key L2 distance:")
    print(f"  edit-target held-out phrasings: mean {ed_d.mean():.2f}  min {ed_d.min():.2f}  max {ed_d.max():.2f}")
    print(f"  other-country held-out probes  : mean {ot_d.mean():.2f}  min {ot_d.min():.2f}  max {ot_d.max():.2f}")

    print(f"\n{'eps':>6} | {'edit-gen(fb)':>12} | {'locality(spurious/245)':>22} | note")
    best_at_loc0 = (0.0, None)
    for eps in [0,2,4,6,8,10,12,14,16,18,20,25,30,40,60,1e9]:
        hits = tot = 0
        for (tgt, src, pr, h, nf) in edit_probes:
            i, d = nearest(h)
            pred = answers[i] if d < eps else own[pr]
            # GRACE only counts as a transferred EDIT if it defers to the RIGHT edit's answer
            hits += int(pred == nf); tot += 1
        spurious = sum(1 for (_, _, h) in other_probes if nearest(h)[1] < eps)
        eg = hits/tot
        if spurious == 0 and eg > best_at_loc0[0]: best_at_loc0 = (eg, eps)
        note = "<- locality clean" if spurious == 0 else ""
        es = "inf" if eps > 1e8 else f"{eps:.0f}"
        print(f"{es:>6} | {eg:>12.3f} | {spurious:>22d} | {note}")

    print(f"\n=== FAIR COMPARISON (first-byte, locality==0) ===")
    print(f"  OURS  (Yaz+Engram semantic): edit-gen 0.675, locality 0 side-effects")
    print(f"  GRACE (activation-key)     : edit-gen {best_at_loc0[0]:.3f} at eps={best_at_loc0[1]} (best clean)")


if __name__ == "__main__":
    main()
