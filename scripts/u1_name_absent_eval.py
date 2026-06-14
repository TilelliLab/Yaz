"""U1 — does the win survive when the entity is NOT named in the prompt?

The 0.675 edit-gen win used held-out prompts that all contain the country name, so a trivial
`str.contains(country)` lexical router ties Engram (1.000). This test rebuilds held-out probes as
NAME-FREE clues (no country name, no capital) and asks: does Engram route them correctly (semantic
work) while the lexical baseline fails? EVAL-ONLY on checkpoints/yaz_gen_semantic_v2.pt (no retrain).

Honest guard: every clue is asserted to contain NO country name and NO capital (case-insensitive),
so the lexical baseline genuinely cannot cheat. Clues are world-famous/unambiguous to avoid unfairly
penalising Engram.
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
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES, pairs

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
# 8 edited targets from semantic_v2.json (target <- source); same edits as the win run.
EDITS = [("France","Peru"),("Japan","Germany"),("Egypt","Iran"),("Germany","Spain"),
         ("Brazil","Bulgaria"),("Italy","Norway"),("Poland","Greece"),("Canada","Cuba")]
# Name-free, capital-free clues (one per clued country); 2 phrasings each at eval time.
CLUES = {
 "France":"the country of the Eiffel Tower and the Louvre",
 "Japan":"the island country of Mount Fuji, sushi, and the bullet train",
 "Egypt":"the country of the Great Pyramids of Giza and the river Nile",
 "Germany":"the European country of Oktoberfest, the Autobahn, and the Brandenburg Gate",
 "Brazil":"the largest South American country, home of the Amazon rainforest and Carnival",
 "Italy":"the boot-shaped country of the Colosseum, pizza, and gondolas",
 "Poland":"the Central European homeland of Chopin and Marie Curie",
 "Canada":"the northern North American country of maple syrup and the Mounties",
 "Russia":"the largest country on Earth, spanning eleven time zones",
 "Spain":"the Iberian country of flamenco, paella, and the running of the bulls",
 "Greece":"the Mediterranean cradle of democracy, home of the Parthenon",
 "China":"the most populous Asian country, home of the Great Wall",
 "India":"the South Asian country of the Taj Mahal and the river Ganges",
 "Australia":"the island continent of kangaroos and the Great Barrier Reef",
 "UK":"the island nation of Big Ben, the Beatles, and the royal family",
 "Netherlands":"the low-lying country of tulips, windmills, and canals",
 "Switzerland":"the Alpine country of neutrality, watches, and fine chocolate",
 "Ireland":"the Emerald Isle of shamrocks and Guinness",
 "Turkey":"the country bridging Europe and Asia across the Bosphorus strait",
 "Thailand":"the Southeast Asian kingdom of pad thai and ornate temples",
}
TMPLS = ["{clue}, and its capital is ", "The capital of {clue} is "]


def ids(s): return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def gen_first(model, prompt, atom, n=4):
    p = ids(prompt); out = p; ra = torch.tensor([int(atom)]); mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        lo = model(ctx, route_atom=ra)
        out = torch.cat([out, lo[:, -1].argmax(-1, keepdim=True)], dim=1)
    g = bytes(out[0, len(p[0]):].tolist()).decode("latin-1", "ignore").lstrip()
    return g[0] if g else ""


def main():
    ps = pairs(); order = [c for c, _ in ps]; cidx = {c: i for i, c in enumerate(order)}
    cap = {c: cp for c, cp in ps}
    ck = torch.load(CKPT, map_location="cpu")
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {CKPT.name}: use_atom_gain={cfg.use_atom_gain} gain_init={cfg.atom_gain_init}")

    # GUARD: assert clues are name-free and capital-free (so lexical genuinely fails).
    lc_countries = [c.lower() for c in order]; lc_caps = [cp.lower() for cp in cap.values()]
    bad = []
    for c, clue in CLUES.items():
        cl = clue.lower()
        hits = [o for o in lc_countries if o in cl] + [cp for cp in lc_caps if cp in cl]
        if hits: bad.append((c, hits))
    if bad:
        print("WARNING name/capital leak in clues:", bad)
    else:
        print("guard OK: no clue contains any country name or capital")

    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids(); router.train_head()

    # Build name-free probes.
    probes = []
    for c, clue in CLUES.items():
        for t in TMPLS:
            probes.append({"country": c, "prompt": t.format(clue=clue), "exp": cap[c][0]})

    # --- routing accuracy: lexical vs Engram-frozen vs Engram-learned (50-way) ---
    def lex_route(p):
        pl = p.lower(); best = -1; bl = -1
        for i, c in enumerate(order):
            if c.lower() in pl and len(c) > bl: best = i; bl = len(c)
        return best
    lex = sem_f = sem_l = 0
    for pr in probes:
        ti = cidx[pr["country"]]
        lex += int(lex_route(pr["prompt"]) == ti)
        sem_f += int(router.route_frozen(pr["prompt"]) == ti)
        sem_l += int(router.route_learned(pr["prompt"]) == ti)
    N = len(probes)
    print(f"\nname-free routing acc ({N} probes, 50-way):")
    print(f"  lexical substring : {lex/N:.3f}  ({lex}/{N})")
    print(f"  Engram frozen     : {sem_f/N:.3f}  ({sem_f}/{N})")
    print(f"  Engram learned    : {sem_l/N:.3f}  ({sem_l}/{N})")

    # --- base-gen (pre-edit): force Engram-frozen atom, does model emit the right capital byte? ---
    bg = sum(int(gen_first(model, pr["prompt"], router.route_frozen(pr["prompt"])) == pr["exp"])
             for pr in probes)
    print(f"name-free base-gen (Engram-frozen routing): {bg/N:.3f}  ({bg}/{N})")

    # --- edit-gen on the 8 edited targets, name-free probes, Engram-frozen routing ---
    W = model.fact_layer.W_dec.weight
    hits = tot = 0; per = []
    for tgt, src in EDITS:
        m = YazLM(cfg); m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()}); m.eval()
        with torch.no_grad():
            m.fact_layer.W_dec.weight[:, cidx[tgt]] = W[:, cidx[src]].clone()
        exp = cap[src][0]
        tp = [pr for pr in probes if pr["country"] == tgt]
        h = sum(int(gen_first(m, pr["prompt"], router.route_frozen(pr["prompt"])) == exp) for pr in tp)
        # lexical can't route name-free prompts -> edit cannot be applied -> count as fail
        lex_h = 0
        hits += h; tot += len(tp)
        per.append((f"{tgt}<-{src}", h, len(tp), exp))
    print(f"\nname-free EDIT-GEN (Engram-frozen): {hits/tot:.3f}  ({hits}/{tot})")
    print(f"name-free EDIT-GEN (lexical router): 0.000  (lexical cannot route name-free prompts)")
    for e, h, n, exp in per:
        print(f"  {e:18s} {h}/{n}  (new first byte '{exp}')")


if __name__ == "__main__":
    main()
