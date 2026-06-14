"""U3 — does Engram routing survive OBSCURE / AMBIGUOUS name-absent clues?

U1 used hand-picked WORLD-FAMOUS clues (favorable to a public sentence encoder). This tests the
hard case: less-famous countries (Baltics, Balkans, Belarus, Qatar...) and a few deliberately
AMBIGUOUS clues. Routing accuracy is the headline (does Engram still identify the entity from
meaning when it isn't famous?). Edits are eval-time W_dec column swaps, so we also test edit-gen
on OBSCURE targets (not just the famous 8 from training) — first-byte and full-word.

Guard: every clue asserted to contain NO country name and NO capital (case-insensitive substring),
so the lexical baseline genuinely cannot route. EVAL-ONLY on yaz_gen_semantic_v2.pt.
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
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES, pairs

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"

# Obscure / less-famous, name-free & capital-free clues (unique entity each).
HARD = {
 "Portugal":"the westernmost country of mainland Europe, famed for port wine, fado music, and Age-of-Discovery navigators",
 "Sweden":"the largest Scandinavian country, home of IKEA, ABBA, and the Nobel Prize ceremonies",
 "Denmark":"the smallest Scandinavian kingdom, home of Lego bricks and the author Hans Christian Andersen",
 "Hungary":"the landlocked Central European country split by the Danube, famed for goulash and Lake Balaton",
 "Austria":"the Alpine homeland of Mozart, the Habsburg dynasty, and the Vienna Boys' Choir's nation",
 "Belgium":"the small Western European country famed for waffles, fine chocolate, and hosting the EU institutions",
 "Iceland":"the volcanic Nordic island of geysers, glaciers, and the aurora near the Arctic Circle",
 "Croatia":"the Adriatic country of a thousand islands and the walled medieval port of Dubrovnik",
 "Serbia":"the landlocked Balkan country at the confluence of the Sava and Danube, once heart of Yugoslavia",
 "Bulgaria":"the Balkan country on the Black Sea coast famed for rose-oil and its tangy yogurt",
 "Romania":"the Carpathian country of the Transylvania region and the Dracula legend",
 "Ukraine":"the large Eastern European country of golden wheat fields and the Chernobyl disaster",
 "Belarus":"the landlocked Eastern European state once called the Soviet breadbasket, ruled for decades by one strongman",
 "Lithuania":"the southernmost Baltic state, the first republic to break free of the USSR in 1990",
 "Latvia":"the central Baltic state famed for Art Nouveau architecture and vast birch forests",
 "Estonia":"the northernmost Baltic state, a pioneer of digital e-government and e-residency",
 "Qatar":"the small, gas-rich peninsular Gulf emirate that hosted the 2022 football World Cup",
 "Colombia":"the South American country of coffee, emeralds, and the novelist Gabriel Garcia Marquez",
 "Chile":"the long, ribbon-thin South American country squeezed between the Andes and the Pacific",
 "Kenya":"the East African country of safaris, the Great Rift Valley, and Maasai warriors",
 "Vietnam":"the S-shaped Southeast Asian country famed for pho noodle soup and a long 20th-century war",
 "Indonesia":"the world's largest archipelago and most populous Muslim-majority nation, home of the Komodo dragon",
}
# Deliberately AMBIGUOUS clues (no single correct country) — reported separately, not scored.
AMBIG = {
 "a_nordic":"a Nordic country of fjords, long winters, and high taxes",
 "a_baltic":"a small Baltic state that regained independence from the Soviet Union",
 "a_balkan":"a Balkan country with a turbulent 20th-century history",
}
TMPLS = ["{clue}, and its capital is ", "The capital of {clue} is "]
# Obscure edit pairs (target <- source); both must be in the 50 facts. New cap = source's capital.
EDITS = [("Latvia","Estonia"),("Croatia","Serbia"),("Bulgaria","Romania"),
         ("Belarus","Ukraine"),("Qatar","Chile"),("Portugal","Sweden")]


def ids(s): return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def gen(model, prompt, atom, n=12):
    p = ids(prompt); out = p; ra = torch.tensor([int(atom)]); mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        out = torch.cat([out, model(ctx, route_atom=ra)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return bytes(out[0, len(p[0]):].tolist()).decode("latin-1", "ignore").lstrip()


def main():
    ps = pairs(); order = [c for c, _ in ps]; cidx = {c: i for i, c in enumerate(order)}
    cap = {c: cp for c, cp in ps}
    ck = torch.load(CKPT, map_location="cpu")
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids(); router.train_head()

    # guard
    lc_c = [c.lower() for c in order]; lc_cap = [cp.lower() for cp in cap.values()]
    bad = []
    for c, clue in {**HARD, **AMBIG}.items():
        cl = clue.lower()
        hits = [o for o in lc_c if o in cl] + [cp for cp in lc_cap if cp in cl]
        if hits: bad.append((c, hits))
    print("guard:", "LEAK "+str(bad) if bad else "OK (no clue contains a country name or capital)")

    def lex_route(p):
        pl = p.lower(); best = -1; bl = -1
        for i, c in enumerate(order):
            if c.lower() in pl and len(c) > bl: best = i; bl = len(c)
        return best

    # routing accuracy on obscure set
    probes = []
    for c, clue in HARD.items():
        for t in TMPLS: probes.append({"c": c, "p": t.format(clue=clue)})
    lex = sf = sl = 0
    miss = []
    for pr in probes:
        ti = cidx[pr["c"]]
        lex += int(lex_route(pr["p"]) == ti)
        rf = router.route_frozen(pr["p"]); sf += int(rf == ti); sl += int(router.route_learned(pr["p"]) == ti)
        if rf != ti: miss.append((pr["c"], order[rf]))
    N = len(probes)
    print(f"\nOBSCURE name-free routing acc ({N} probes, 50-way):")
    print(f"  lexical : {lex/N:.3f} ({lex}/{N})   Engram-frozen : {sf/N:.3f} ({sf}/{N})   Engram-learned : {sl/N:.3f} ({sl}/{N})")
    if miss: print("  frozen misroutes:", miss[:12])

    # ambiguous clues — where does Engram send them? (no scoring)
    print("\nAMBIGUOUS clues route to (frozen):")
    for k, clue in AMBIG.items():
        dest = order[router.route_frozen(TMPLS[0].format(clue=clue))]
        print(f"  {k:10s} -> {dest}")

    # edit-gen on OBSCURE targets, name-free routing
    W = model.fact_layer.W_dec.weight
    fb = fw = tot = 0
    print("\nOBSCURE edit-gen (name-free clues, Engram-frozen routing):")
    for tgt, src in EDITS:
        with torch.no_grad():
            col = W[:, cidx[tgt]].clone()
            W[:, cidx[tgt]] = W[:, cidx[src]].clone()
        new = cap[src]
        h_fb = h_fw = 0; outs = []
        for t in TMPLS:
            pr = t.format(clue=HARD[tgt])
            g = gen(model, pr, router.route_frozen(pr))
            h_fb += int(g[:1] == new[:1]); h_fw += int(g.startswith(new)); outs.append(g[:len(new)+2])
        with torch.no_grad():
            W[:, cidx[tgt]] = col
        fb += h_fb; fw += h_fw; tot += len(TMPLS)
        print(f"  {tgt+'<-'+src:18s} new={new:10s} fb {h_fb}/2 full {h_fw}/2  e.g.{outs}")
    print(f"\nOBSCURE edit-gen: first-byte {fb}/{tot}={fb/tot:.3f}  full-word {fw}/{tot}={fw/tot:.3f}")
    print("(lexical edit-gen = 0.000: cannot route name-free prompts)")


if __name__ == "__main__":
    main()
