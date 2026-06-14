"""U5 — are the chance-floor + full-word findings SEED-STABLE? Re-run the two decisive controls
(full-word edit-gen, random-swap first-byte floor) on each seed checkpoint, so the higher first-byte
edit-gen of the new seeds (0.825-0.85) is interpreted correctly: edit-gen tracks base-gen, with NO
margin over a random swap, and full-word stays ~0. Run: python3 scripts/u5_seed_controls.py <ckpt>...
EVAL-ONLY.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("YAZ_EMBEDDER_PATH", "")); sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from yaz.semantic_router import SemanticRouter
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES, TEST_TEMPLATES, pairs

EDITS = [("France","Peru"),("Japan","Germany"),("Egypt","Iran"),("Germany","Spain"),
         ("Brazil","Bulgaria"),("Italy","Norway"),("Poland","Greece"),("Canada","Cuba")]
N_DRAWS = 20


def ids(s): return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def gen(model, prompt, atom, n=12):
    p = ids(prompt); out = p; ra = torch.tensor([int(atom)]); mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        out = torch.cat([out, model(ctx, route_atom=ra)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return bytes(out[0, len(p[0]):].tolist()).decode("latin-1", "ignore").lstrip()


def run(ckpt):
    ps = pairs(); order = [c for c, _ in ps]; cidx = {c: i for i, c in enumerate(order)}; cap = {c: cp for c, cp in ps}
    ck = torch.load(ckpt, map_location="cpu")
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids(); router.train_head()
    W = model.fact_layer.W_dec.weight; Wbak = W.detach().clone()

    # true edit: first-byte + full-word on held-out
    fb = fw = tot = 0
    for tgt, src in EDITS:
        with torch.no_grad(): W[:, cidx[tgt]] = Wbak[:, cidx[src]]
        new = cap[src]
        for t in TEST_TEMPLATES:
            pr = t.format(C=tgt); g = gen(model, pr, router.route_frozen(pr))
            fb += int(g[:1] == new[:1]); fw += int(g.startswith(new)); tot += 1
        with torch.no_grad(): W[:, cidx[tgt]] = Wbak[:, cidx[tgt]]
    true_fb, true_fw = fb/tot, fw/tot

    # random-swap floor (first-byte)
    rng = np.random.default_rng(2026); rates = []
    for _ in range(N_DRAWS):
        h = n = 0
        for tgt, _src in EDITS:
            s = order[rng.integers(0, len(order))]
            while s == tgt: s = order[rng.integers(0, len(order))]
            with torch.no_grad(): W[:, cidx[tgt]] = Wbak[:, cidx[s]]
            exp = cap[s][0]
            for t in TEST_TEMPLATES:
                pr = t.format(C=tgt); h += int(gen(model, pr, router.route_frozen(pr), n=4)[:1] == exp); n += 1
            with torch.no_grad(): W[:, cidx[tgt]] = Wbak[:, cidx[tgt]]
        rates.append(h/n)
    r = np.array(rates)
    return true_fb, true_fw, r.mean(), r.std()


def main():
    print(f"{'checkpoint':32s} {'true-fb':>8} {'rand-fb':>14} {'margin':>8} {'full-word':>10}")
    for c in sys.argv[1:]:
        tfb, tfw, rm, rs = run(c)
        print(f"{Path(c).name:32s} {tfb:>8.3f} {rm:>7.3f}±{rs:<5.3f} {tfb-rm:>+8.3f} {tfw:>10.3f}")


if __name__ == "__main__":
    main()
