"""U2 — does the edit-generalization win survive a FULL-WORD bar (not just first byte)?

The headline edit-gen 0.675 is FIRST-BYTE only, and capitals have just 18 distinct first
bytes across 50 (B alone = 10). The in-dist edit dumps show trail 1/4-1/6: the model emits the
right first byte then gibberish (France->'Larancam' for Lima). So first-byte massively flatters
the result. This re-scores the SAME 8 edits on the SAME 5 held-out phrasings under three bars:
  - first_byte  : generated[0] == new_capital[0]      (the published metric)
  - full_word   : generated starts with the WHOLE new capital
  - prefix>=N    : generated shares the first N bytes of new capital (N=min(len,4))

EVAL-ONLY on checkpoints/yaz_gen_semantic_v2.pt (no retrain). Engram-frozen routing, same as win.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("YAZ_EMBEDDER_PATH", ""))
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from yaz.semantic_router import SemanticRouter
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES, TEST_TEMPLATES, pairs

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
EDITS = [("France","Peru"),("Japan","Germany"),("Egypt","Iran"),("Germany","Spain"),
         ("Brazil","Bulgaria"),("Italy","Norway"),("Poland","Greece"),("Canada","Cuba")]


def ids(s): return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def gen(model, prompt, atom, n=14):
    p = ids(prompt); out = p; ra = torch.tensor([int(atom)]); mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        lo = model(ctx, route_atom=ra)
        out = torch.cat([out, lo[:, -1].argmax(-1, keepdim=True)], dim=1)
    return bytes(out[0, len(p[0]):].tolist()).decode("latin-1", "ignore").lstrip()


def main():
    ps = pairs(); order = [c for c, _ in ps]; cidx = {c: i for i, c in enumerate(order)}
    cap = {c: cp for c, cp in ps}
    ck = torch.load(CKPT, map_location="cpu")
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids(); router.train_head()
    W = model.fact_layer.W_dec.weight

    agg = {"first_byte": [0, 0], "prefix4": [0, 0], "full_word": [0, 0]}
    print(f"{'edit':16s} {'new cap':10s} | per held-out phrasing (5):  fb / pfx4 / full")
    for tgt, src in EDITS:
        m = YazLM(cfg); m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()}); m.eval()
        with torch.no_grad():
            m.fact_layer.W_dec.weight[:, cidx[tgt]] = W[:, cidx[src]].clone()
        new_cap = cap[src]; pn = min(len(new_cap), 4)
        fb = p4 = fw = 0; outs = []
        for t in TEST_TEMPLATES:
            prompt = t.format(C=tgt)
            g = gen(m, prompt, router.route_frozen(prompt))
            fb += int(g[:1] == new_cap[:1])
            p4 += int(g[:pn] == new_cap[:pn])
            fw += int(g.startswith(new_cap))
            outs.append(g[:len(new_cap)+2])
        agg["first_byte"][0] += fb; agg["first_byte"][1] += 5
        agg["prefix4"][0] += p4; agg["prefix4"][1] += 5
        agg["full_word"][0] += fw; agg["full_word"][1] += 5
        print(f"{tgt+'<-'+src:16s} {new_cap:10s} | {fb}/5 / {p4}/5 / {fw}/5   e.g. {outs[:3]}")

    print("\n=== held-out EDIT-GEN under each bar (40 = 8 edits x 5 phrasings) ===")
    for k in ("first_byte", "prefix4", "full_word"):
        h, n = agg[k]; print(f"  {k:10s}: {h/n:.3f}  ({h}/{n})")
    print("\nPublished headline (first_byte) = 0.675. Full_word is the honest transfer rate.")


if __name__ == "__main__":
    main()
