"""T4 — HARDER name-absent routing + edit-gen (stresses U1's 'famous clues' caveat).

U1 used world-famous, unambiguous clues (favorable to Engram). This rebuilds the name-free
set with (a) less-famous countries (Baltics, Balkans, Nordics, Gulf) and (b) deliberately
more OBLIQUE / ambiguous clues, then re-measures:
  - routing accuracy (50-way) for Engram frozen/learned vs the lexical substring baseline
  - name-free edit-gen on the same 8 edited targets, but with harder oblique clues.
EVAL-ONLY on checkpoints/yaz_gen_semantic_v2.pt. Guard: every clue is asserted to contain
NO country name and NO capital (case-insensitive) so lexical genuinely cannot cheat.
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
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
EDITS = [("France","Peru"),("Japan","Germany"),("Egypt","Iran"),("Germany","Spain"),
         ("Brazil","Bulgaria"),("Italy","Norway"),("Poland","Greece"),("Canada","Cuba")]

# Harder, oblique, name+capital-free clues for LESS-famous countries (and oblique re-clues
# of the 8 edited targets). No "Eiffel Tower"-level giveaways.
CLUES = {
 # edited targets — deliberately more oblique than U1
 "France":"the country whose old regime fell in a 1789 revolution and which gave the world a famous statue to New York",
 "Japan":"the archipelago nation that surrendered in 1945 and now builds much of the world's cameras and game consoles",
 "Egypt":"the ancient civilisation ruled by pharaohs whose people built monuments along a great north-flowing river",
 "Germany":"the country reunified in 1990 after a wall dividing its former eastern and western halves came down",
 "Brazil":"the only Portuguese-speaking nation in its continent, five-time football world champions",
 "Italy":"the peninsula nation that was the heart of an ancient empire and later the Renaissance",
 "Poland":"the nation invaded in September 1939 to start a world war, later led out of communism by a shipyard union",
 "Canada":"the bilingual federation that is the world's second-largest country by area, sharing the longest land border",
 # less-famous / harder countries
 "Estonia":"the smallest and northernmost of three Baltic states, a digital-government pioneer",
 "Latvia":"the central of the three Baltic states, known for its art-nouveau capital architecture",
 "Lithuania":"the southernmost Baltic state, once part of a vast medieval grand duchy",
 "Croatia":"the Adriatic nation shaped like a crescent, famed for its Dalmatian coastline",
 "Serbia":"the landlocked western-Balkan nation at the confluence of the Sava and Danube",
 "Romania":"the Carpathian nation of Transylvania and the Dracula legend",
 "Bulgaria":"the eastern-Balkan nation on the Black Sea, known for rose-oil and yoghurt",
 "Belarus":"the landlocked eastern-European nation often called Europe's last dictatorship",
 "Ukraine":"the large eastern-European nation invaded by its neighbour in 2022, the breadbasket of the continent",
 "Iceland":"the volcanic north-Atlantic island nation of geysers and the oldest surviving parliament",
 "Finland":"the Nordic nation of a thousand lakes that consistently ranks happiest on Earth",
 "Hungary":"the central-European nation of a Magyar people and a famous goulash",
 "Austria":"the Alpine nation that was once the seat of a Habsburg empire and Mozart's birthplace",
 "Qatar":"the small Gulf peninsula state that hosted the 2022 football world cup",
 "Vietnam":"the elongated Southeast-Asian nation of a long war with America and the Mekong delta",
 "Philippines":"the Southeast-Asian archipelago of over seven thousand islands, formerly a Spanish then American colony",
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
    return bytes(out[0, len(p[0]):].tolist()).decode("latin-1", "ignore").lstrip()[:1]


def main():
    torch.set_num_threads(1)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    c2i = ck["country_to_target_atom"]; order = list(c2i.keys())
    capof = {}
    for l in (ROOT / "data" / "facts_para_train.jsonl").read_text().splitlines():
        if l:
            import json as _j; r = _j.loads(l); capof.setdefault(r["country"], r["capital"])

    # guard: no country name / capital leaks into any clue
    low_names = [c.lower() for c in order] + [capof[c].lower() for c in order]
    for c, clue in CLUES.items():
        cl = clue.lower()
        bad = [t for t in low_names if t and t in cl]
        assert not bad, f"LEAK in {c}: clue contains {bad}"

    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids(); router.train_head()

    # ---- routing accuracy (50-way) on harder clues ----
    probes = [(c, t.format(clue=clue)) for c, clue in CLUES.items() for t in TMPLS]
    res = {"n_routing": len(probes)}
    for tag, route in [("frozen", router.route_frozen), ("learned", router.route_learned)]:
        hit = sum(int(order[route(p)] == c) for c, p in probes)
        res[f"routing_{tag}"] = hit / len(probes)
    # lexical substring baseline (cannot match — names are absent)
    def lexical(p):
        pl = p.lower()
        for c in order:
            if c.lower() in pl: return c
        return None
    res["routing_lexical"] = sum(int(lexical(p) == c) for c, p in probes) / len(probes)

    # ---- name-free edit-gen on the 8 edited targets with OBLIQUE clues ----
    eg_probes = [(tgt, src, t.format(clue=CLUES[tgt]))
                 for tgt, src in EDITS for t in TMPLS]
    for tag, route in [("frozen", router.route_frozen)]:
        hit = bn = 0
        for tgt, src, prompt in eg_probes:
            m = YazLM(cfg); m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()}); m.eval()
            with torch.no_grad():
                m.fact_layer.W_dec.weight[:, c2i[tgt]] = m.fact_layer.W_dec.weight[:, c2i[src]].clone()
            atom = route(prompt)
            exp = capof[src][0]
            hit += int(gen_first(m, prompt, atom) == exp)
            # base-gen (pre-edit, Engram-routed) for context
            ab = route(prompt)
            bn += int(gen_first(model, prompt, c2i[tgt]) == capof[tgt][0])
        res["name_free_editgen_frozen"] = hit / len(eg_probes)
        res["name_free_basegen_frozen"] = bn / len(eg_probes)
        res["n_editgen"] = len(eg_probes)

    print(f"[T4 harder name-absent] routing 50-way (n={res['n_routing']}): "
          f"Engram-frozen {res['routing_frozen']:.3f} / learned {res['routing_learned']:.3f} "
          f"| lexical {res['routing_lexical']:.3f}")
    print(f"[T4] name-free edit-gen (oblique clues, 8 edits x2, n={res['n_editgen']}): "
          f"Engram-frozen {res['name_free_editgen_frozen']:.3f} "
          f"(base-gen {res['name_free_basegen_frozen']:.3f})")
    import json
    out = ROOT / "results" / "t4_hard_nameabsent.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
