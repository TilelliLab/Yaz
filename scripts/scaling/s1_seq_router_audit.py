"""P1 — Sequential-edit router-scaling + collision audit (the honest core).

Tests Yaz's ONLY non-tautological failure mode under accumulating edits. Yaz edits are disjoint
columns selected by a FROZEN external router, so an accumulated edit can clobber an earlier one
ONLY if two facts ROUTE to the same atom (collision) or a query MIS-ROUTES. This script applies
edits sequentially to ONE mutating model (no per-edit clone) and, at each prefix k, measures:
  - retained_indist   : of the k edits so far, frac whose in-dist prompt still gives the new first byte
  - retained_heldout  : of the k edits so far, mean first-byte hit over their 5 held-out phrasings (frozen route)
  - locality_collateral : # of a FIXED never-edited holdout whose generation changed vs pre-edit
  - bpc / bpc_delta_pct : backbone drift (structurally ~0 for Yaz; reported to prove it)
  - misroute_heldout    : frac of edited countries' held-out phrasings the frozen router sends to the WRONG atom
  - n_route_collisions  : # of distinct atoms that >1 edited country's held-out phrasings route to (true clobber risk)

FLAT (Yaz hypothesis) = retained_* stays within +/-eps of its k=1 value, collateral small, bpc flat.
Honest caveat baked into the verdict: Yaz flatness is largely STRUCTURAL (disjoint slots) — this
audit's job is to show whether the ROUTER (the only thing that can break it) holds as k grows.
Eval-only on checkpoints/yaz_gen_semantic_v2.pt. Namespaced; touches no shared file.
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("YAZ_EMBEDDER_PATH", ""))
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from yaz.semantic_router import SemanticRouter
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
TS_VALID = (Path(os.environ.get("YAZ_TINYSTORIES_DIR", "data/tinystories")) / "TinyStories-valid.txt")
N_HOLDOUT = 10          # never-edited countries reserved for the locality domain
EVAL_KS = None          # set after we know N (log-spaced)
SEED = 2026


def ids(s):
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def gen_routed(model, prompt, atom, n=8):
    ra = torch.tensor([int(atom)], dtype=torch.long)
    out = ids(prompt); plen = out.shape[1]; mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        out = torch.cat([out, model(ctx, route_atom=ra)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return bytes(out[0, plen:].tolist()).decode("latin-1", "ignore")


@torch.no_grad()
def first_byte_routed(model, prompt, atom):
    return gen_routed(model, prompt, atom, n=4).lstrip()[:1]


@torch.no_grad()
def bpc_quick(model, n=40_000):
    ts = TS_VALID.read_bytes()[:n]; S = model.cfg.max_seq_len
    starts = list(range(0, len(ts)-S-1, S))[:(len(ts)//S)]; tl = 0.0; tot = 0
    for i in range(0, len(starts), 16):
        bs = starts[i:i+16]
        arr = np.stack([np.frombuffer(ts[s:s+S+1], dtype=np.uint8) for s in bs])
        x = torch.from_numpy(arr).long()
        lo = model(x[:, :-1])
        tl += float(torch.nn.functional.cross_entropy(
            lo.reshape(-1, model.cfg.vocab_size), x[:, 1:].reshape(-1), reduction="sum"))
        tot += x[:, 1:].numel()
    return tl / tot / 0.6931471805599453


def main():
    torch.set_num_threads(1)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    c2i = ck["country_to_target_atom"]; order = list(c2i.keys())
    capof = {}; indist = {}
    for l in (ROOT / "data" / "facts_para_train.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); capof.setdefault(r["country"], r["capital"])
    for l in (ROOT / "data" / "probes_para_indist.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); indist.setdefault(r["country"], r["prompt"])
    held = {}
    for l in (ROOT / "data" / "probes_para_heldout.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); held.setdefault(r["country"], []).append(r["prompt"])

    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids()

    # deterministic edit list: hold out the last N_HOLDOUT countries (never edited);
    # edit each remaining country -> a fixed pseudo-random OTHER country's capital.
    rng = np.random.default_rng(SEED)
    perm = list(order); rng.shuffle(perm)
    holdout = perm[-N_HOLDOUT:]                       # never edited (locality domain)
    edit_countries = perm[:-N_HOLDOUT]
    # SNAPSHOT every source column BEFORE any mutation, so each edit installs the INTENDED
    # value (capof[src]) and does not copy an already-overwritten live column. This removes
    # the source-reuse confound and isolates the true accumulation question.
    W0 = model.fact_layer.W_dec.weight.detach().clone()
    edits = []                                        # (tgt, src, new_cap, src_col_snapshot)
    for tgt in edit_countries:
        src = order[int(rng.integers(0, len(order)))]
        while src == tgt:
            src = order[int(rng.integers(0, len(order)))]
        edits.append((tgt, src, capof[src], W0[:, c2i[src]].clone()))
    N = len(edits)
    eval_ks = sorted(set([1, 2, 3, 5, 8, 12, 18, 25, 32, N]))
    eval_ks = [k for k in eval_ks if k <= N]

    # pre-edit baseline generations for the never-edited holdout (locality reference)
    base_hold = {c: gen_routed(model, indist[c], c2i[c]) for c in holdout}
    base_bpc = bpc_quick(model)

    # static router diagnostics on edited countries' held-out phrasings
    misroute = 0; tot_hp = 0; routed_atom_of_hp = {}
    for tgt, _, _, _ in edits:
        for hp in held[tgt]:
            a = router.route_frozen(hp); tot_hp += 1
            misroute += int(order[a] != tgt)
            routed_atom_of_hp.setdefault(a, set()).add(tgt)
    n_collisions = sum(1 for a, cs in routed_atom_of_hp.items() if len(cs) > 1)

    rows = []
    for k in range(1, N + 1):
        tgt, src, new_cap, src_col = edits[k - 1]
        with torch.no_grad():                          # apply edit k to the SAME model (accumulate)
            model.fact_layer.W_dec.weight[:, c2i[tgt]] = src_col          # intended snapshot value
        if k not in eval_ks:
            continue
        # retained accuracy over ALL edits applied so far (1..k), re-evaluated now
        ri = rh = rh_n = 0
        for (et, es, ecap, _) in edits[:k]:
            ri += int(first_byte_routed(model, indist[et], c2i[et]) == ecap[0])
            for hp in held[et]:
                a = router.route_frozen(hp)             # frozen route the held-out phrasing
                rh += int(first_byte_routed(model, hp, a) == ecap[0]); rh_n += 1
        # locality on never-edited holdout
        coll = sum(1 for c in holdout if gen_routed(model, indist[c], c2i[c]) != base_hold[c])
        bpc = bpc_quick(model)
        row = {"k": k, "retained_indist": ri / k, "retained_heldout": rh / rh_n,
               "locality_collateral": coll, "locality_domain": N_HOLDOUT,
               "bpc": bpc, "bpc_delta_pct": (bpc - base_bpc) / base_bpc * 100}
        rows.append(row)
        print(f"  k={k:3d}/{N}  retained in-dist {row['retained_indist']:.3f}  held-out {row['retained_heldout']:.3f}  "
              f"collateral {coll}/{N_HOLDOUT}  bpc {bpc:.3f} ({row['bpc_delta_pct']:+.2f}%)", flush=True)

    out = {"n_edits": N, "n_holdout": N_HOLDOUT, "eval_ks": eval_ks, "seed": SEED,
           "static_router": {"misroute_heldout_rate": misroute / tot_hp, "n_heldout_probes": tot_hp,
                             "n_route_collisions": n_collisions},
           "curve": rows}
    # verdict
    r1, rN = rows[0], rows[-1]
    flat_in = abs(rN["retained_indist"] - r1["retained_indist"]) <= 0.05
    flat_ho = abs(rN["retained_heldout"] - r1["retained_heldout"]) <= 0.05
    max_coll = max(r["locality_collateral"] for r in rows)
    max_bpc = max(abs(r["bpc_delta_pct"]) for r in rows)
    out["verdict"] = {
        "retained_indist_flat": bool(flat_in), "retained_heldout_flat": bool(flat_ho),
        "max_collateral": max_coll, "max_bpc_delta_pct": max_bpc,
        "router_holds": bool(out["static_router"]["misroute_heldout_rate"] < 0.25 and n_collisions == 0),
    }
    (ROOT / "results" / "scaling" / "s1_seq_router_audit.json").write_text(json.dumps(out, indent=2))
    print(f"\n[P1] N={N} edits. retained in-dist {r1['retained_indist']:.3f}->{rN['retained_indist']:.3f} "
          f"(flat={flat_in}); held-out {r1['retained_heldout']:.3f}->{rN['retained_heldout']:.3f} (flat={flat_ho}); "
          f"max collateral {max_coll}/{N_HOLDOUT}; max bpc {max_bpc:.2f}%")
    print(f"[P1] static router: misroute held-out {out['static_router']['misroute_heldout_rate']:.3f}, "
          f"collisions {n_collisions} -> router_holds={out['verdict']['router_holds']}")
    print("wrote results/scaling/s1_seq_router_audit.json")


if __name__ == "__main__":
    main()
