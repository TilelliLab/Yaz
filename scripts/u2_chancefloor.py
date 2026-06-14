"""U2 — is first-byte edit-gen 0.675 meaningfully above CHANCE for the first-byte metric?

Capitals have only 18 distinct first bytes / 50 (B=10). So "first byte == new-capital first
byte" has real chance mass. NULL: instead of swapping in the TRUE source column, swap in a
RANDOM other country's W_dec column and score first-byte hits against THAT random source's
capital, on the same 8 targets x 5 held-out phrasings. Average over many random draws = the
chance floor of "an arbitrary column swap lands the right first byte on held-out phrasings."

Also reports a pure label-chance number: mean over capitals of P(a random capital shares the
first byte) = expected first-byte hit if the emitted byte were a uniformly random real capital.

EVAL-ONLY on yaz_gen_semantic_v2.pt. Engram-frozen routing.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("YAZ_EMBEDDER_PATH", ""))
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from yaz.semantic_router import SemanticRouter
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES, TEST_TEMPLATES, pairs

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
TARGETS = ["France","Japan","Egypt","Germany","Brazil","Italy","Poland","Canada"]
N_DRAWS = 30


def ids(s): return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def first_byte(model, prompt, atom, n=4):
    p = ids(prompt); out = p; ra = torch.tensor([int(atom)]); mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        out = torch.cat([out, model(ctx, route_atom=ra)[:, -1].argmax(-1, keepdim=True)], dim=1)
    g = bytes(out[0, len(p[0]):].tolist()).decode("latin-1", "ignore").lstrip()
    return g[:1]


def main():
    ps = pairs(); order = [c for c, _ in ps]; cidx = {c: i for i, c in enumerate(order)}
    cap = {c: cp for c, cp in ps}
    ck = torch.load(CKPT, map_location="cpu")
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids(); router.train_head()
    W = model.fact_layer.W_dec.weight

    # label-chance: mean over capitals of P(another capital shares first byte)
    caps = list(cap.values())
    lab = np.mean([ (sum(1 for d in caps if d[0]==c[0])-1)/(len(caps)-1) for c in caps ])

    rng = np.random.default_rng(2026)
    Wbak = W.detach().clone()
    draw_rates = []
    for _ in range(N_DRAWS):
        hits = tot = 0
        for tgt in TARGETS:
            src = order[rng.integers(0, len(order))]
            while src == tgt: src = order[rng.integers(0, len(order))]
            with torch.no_grad():
                W[:, cidx[tgt]] = Wbak[:, cidx[src]]      # mutate one column in place
            exp = cap[src][0]
            for t in TEST_TEMPLATES:
                pr = t.format(C=tgt)
                hits += int(first_byte(model, pr, router.route_frozen(pr)) == exp); tot += 1
            with torch.no_grad():
                W[:, cidx[tgt]] = Wbak[:, cidx[tgt]]      # restore
        draw_rates.append(hits/tot)
    dr = np.array(draw_rates)
    print(f"label-chance (uniform real capital): {lab:.3f}")
    print(f"random-source-swap first-byte edit-gen over {N_DRAWS} draws (8x5=40 each):")
    print(f"  mean {dr.mean():.3f}  sd {dr.std():.3f}  min {dr.min():.3f}  max {dr.max():.3f}")
    print(f"  95th pct {np.percentile(dr,95):.3f}")
    print(f"\nTRUE-source first-byte edit-gen = 0.675 (from the win run).")
    z = (0.675 - dr.mean())/ (dr.std() if dr.std()>0 else 1e-9)
    print(f"  => 0.675 is {z:.1f} sd above the random-swap floor "
          f"({(0.675 > np.percentile(dr,95))=})")


if __name__ == "__main__":
    main()
