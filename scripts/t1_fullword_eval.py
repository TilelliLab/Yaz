"""T1 — FULL-WORD edit-generalization (removes the 'first-byte only' caveat).

Eval-only on checkpoints/yaz_gen_semantic_v2.pt. For each of the 8 edits, do the same
W_dec column-swap as the CRUD code, then on each HELD-OUT phrasing of the edited country
generate enough bytes and check whether the full NEW capital word is produced (all bytes),
not just its first byte. Compares first-byte vs full-word rates on the identical split.

Leak-free: routers built from TRAIN templates only (same as training); held-out templates
never enter fitting; MiniLM frozen. Same 8 edits, same held-out split as the v2 headline.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES
from yaz.semantic_router import SemanticRouter

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
HELDOUT = ROOT / "data" / "probes_para_heldout.jsonl"
UPDATE_PAIRS = [["France", "Peru"], ["Japan", "Germany"], ["Egypt", "Iran"],
                ["Germany", "Spain"], ["Brazil", "Bulgaria"], ["Italy", "Norway"],
                ["Poland", "Greece"], ["Canada", "Cuba"]]


def ids(s):
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def gen_routed(model, prompt, atom, n):
    ra = torch.tensor([int(atom)], dtype=torch.long)
    out = ids(prompt); mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        lo = model(ctx, route_atom=ra)
        nxt = lo[:, -1].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
    return bytes(out[0, out.shape[1]-n:].tolist()).decode("latin-1", "ignore")


def main():
    torch.set_num_threads(1)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    c2i = ck["country_to_target_atom"]; order = list(c2i.keys())
    capof = {}
    for l in (ROOT / "data" / "facts_para_train.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); capof.setdefault(r["country"], r["capital"])

    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids(); router.train_head()

    held = [json.loads(l) for l in HELDOUT.read_text().splitlines() if l]
    ho_by = {}
    for p in held:
        ho_by.setdefault(p["country"], []).append(p["prompt"])

    results = {}
    for tag, route in [("frozen", router.route_frozen), ("learned", router.route_learned)]:
        fb_hits = fw_hits = n = 0; per = []
        for tgt, src in UPDATE_PAIRS:
            new_cap = capof[src]                       # the capital the edit installs
            m = YazLM(cfg); m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()}); m.eval()
            with torch.no_grad():
                a_t = c2i[tgt]; a_s = c2i[src]
                m.fact_layer.W_dec.weight[:, a_t] = m.fact_layer.W_dec.weight[:, a_s].clone()
            gen_len = len(new_cap) + 4
            for prompt in ho_by[tgt]:
                atom = route(prompt)                   # semantic routing of held-out phrasing
                g = gen_routed(m, prompt, atom, gen_len).lstrip()
                fb = int(g[:1] == new_cap[:1])
                fw = int(g.startswith(new_cap))
                fb_hits += fb; fw_hits += fw; n += 1
                per.append({"edit": f"{tgt}<-{src}", "new_capital": new_cap,
                            "gen": g.strip(), "first_byte": fb, "full_word": fw})
        results[tag] = {"n": n, "first_byte_rate": fb_hits / n, "full_word_rate": fw_hits / n,
                        "first_byte_hits": fb_hits, "full_word_hits": fw_hits, "per": per}
        print(f"[{tag}] held-out edit-gen  first-byte {fb_hits}/{n}={fb_hits/n:.3f}  "
              f"FULL-WORD {fw_hits}/{n}={fw_hits/n:.3f}", flush=True)

    out = ROOT / "results" / "t1_fullword_editgen.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
