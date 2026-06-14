"""P2 — Honest accumulating tournament at MATCHED LOCALITY.

Same ordered edit list / backbone / first-byte metric as P1. Three arms, edits applied
SEQUENTIALLY to one accumulating state per arm:
  - YAZ   : W_dec column = snapshot of source column (disjoint slots, Engram-routed eval).
  - GRACE : growing key->value codebook on FROZEN base activations; deferral radius eps chosen
            per step as the LARGEST eps with 0 collateral on the never-edited holdout (matched
            locality to Yaz); a held-out query is "retained" iff its nearest key is within eps AND
            that key belongs to the right country (emits the right value). Route-to-base otherwise.
  - ROME  : closed-form rank-1 boost to the SHARED unembed per edit, accumulated (weight-editing).

Headline = retained HELD-OUT first-byte accuracy vs N (the paraphrase-reach question on which Yaz
and GRACE — both non-collapsing memories — actually differ), plus bpc drift (the ROME collapse signal).
Eval-only on checkpoints/yaz_gen_semantic_v2.pt. Namespaced; touches no shared file.
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("YAZ_EMBEDDER_PATH", ""))
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from yaz.semantic_router import SemanticRouter
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
TS_VALID = (Path(os.environ.get("YAZ_TINYSTORIES_DIR", "data/tinystories")) / "TinyStories-valid.txt")
N_HOLDOUT = 10; SEED = 2026


def ids(s):
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def hidden(model, prompt):
    x = ids(prompt); T = x.shape[1]; pos = torch.arange(T)
    h = model.tok_embed(x) + model.pos_embed(pos)[None]
    for blk in model.blocks: h = blk(h)
    return model.ln_final(h)[0, -1].clone()


@torch.no_grad()
def x_full_last(model, prompt):
    """The residual the unembed actually sees at the last position (for the ROME solve)."""
    x = ids(prompt); T = x.shape[1]; pos = torch.arange(T)
    h = model.tok_embed(x) + model.pos_embed(pos)[None]
    for blk in model.blocks: h = blk(h)
    h = model.ln_final(h)
    fc, _ = model.fact_layer(h)
    return (h + model.cfg.fact_gain * fc)[0, -1].clone()


@torch.no_grad()
def yaz_first_byte(model, prompt, atom):
    ra = torch.tensor([int(atom)]); out = ids(prompt); plen = out.shape[1]; mc = model.cfg.max_seq_len
    for _ in range(4):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        out = torch.cat([out, model(ctx, route_atom=ra)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return bytes(out[0, plen:].tolist()).decode("latin-1", "ignore").lstrip()[:1]


@torch.no_grad()
def plain_first_byte(model, prompt):
    out = ids(prompt); plen = out.shape[1]; mc = model.cfg.max_seq_len
    for _ in range(4):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        out = torch.cat([out, model(ctx)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return bytes(out[0, plen:].tolist()).decode("latin-1", "ignore").lstrip()[:1]


@torch.no_grad()
def bpc_quick(model, n=40_000):
    ts = TS_VALID.read_bytes()[:n]; S = model.cfg.max_seq_len
    starts = list(range(0, len(ts)-S-1, S))[:(len(ts)//S)]; tl = 0.0; tot = 0
    for i in range(0, len(starts), 16):
        bs = starts[i:i+16]
        arr = np.stack([np.frombuffer(ts[s:s+S+1], dtype=np.uint8) for s in bs])
        x = torch.from_numpy(arr).long(); lo = model(x[:, :-1])
        tl += float(F.cross_entropy(lo.reshape(-1, model.cfg.vocab_size), x[:, 1:].reshape(-1), reduction="sum"))
        tot += x[:, 1:].numel()
    return tl / tot / 0.6931471805599453


def clone_model(model, cfg):
    m = YazLM(cfg); m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()}); m.eval(); return m


def main():
    torch.set_num_threads(1)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = YazConfig(**ck["cfg"]); base = YazLM(cfg); base.load_state_dict(ck["model"]); base.eval()
    c2i = ck["country_to_target_atom"]; order = list(c2i.keys())
    capof = {}; indist = {}; held = {}
    for l in (ROOT / "data" / "facts_para_train.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); capof.setdefault(r["country"], r["capital"])
    for l in (ROOT / "data" / "probes_para_indist.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); indist.setdefault(r["country"], r["prompt"])
    for l in (ROOT / "data" / "probes_para_heldout.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); held.setdefault(r["country"], []).append(r["prompt"])

    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids()

    rng = np.random.default_rng(SEED); perm = list(order); rng.shuffle(perm)
    holdout = perm[-N_HOLDOUT:]; edit_countries = perm[:-N_HOLDOUT]
    W0 = base.fact_layer.W_dec.weight.detach().clone()
    edits = []
    for tgt in edit_countries:
        src = order[int(rng.integers(0, len(order)))]
        while src == tgt: src = order[int(rng.integers(0, len(order)))]
        edits.append({"tgt": tgt, "src": src, "new": capof[src], "col": W0[:, c2i[src]].clone()})
    N = len(edits); eval_ks = [k for k in [1, 2, 3, 5, 8, 12, 18, 25, 32, N] if k <= N]

    # ---- precompute FROZEN-base activations for GRACE (keys = in-dist prompts; queries) ----
    key_h = {c: hidden(base, indist[c]) for c in order}            # candidate keys (in-dist)
    hq_held = {c: [hidden(base, p) for p in held[c]] for c in edit_countries}  # held-out queries
    hq_hold = {c: hidden(base, indist[c]) for c in holdout}        # locality queries
    def l2(a, b): return float((a - b).norm())

    # ---- arms: mutating Yaz model + mutating ROME unembed ----
    yaz = clone_model(base, cfg)
    rome = clone_model(base, cfg); base_bpc = bpc_quick(base)

    rows = []
    for k in range(1, N + 1):
        e = edits[k - 1]
        with torch.no_grad():
            yaz.fact_layer.W_dec.weight[:, c2i[e["tgt"]]] = e["col"]    # YAZ edit
            # ROME-lite: rank-1 boost so new first byte wins at this prompt's residual (shared unembed)
            xs = x_full_last(rome, indist[e["tgt"]]); b = ord(e["new"][0])
            logits = rome.unembed.weight @ xs
            margin = float(logits.max() - logits[b]) + 2.0
            rome.unembed.weight[b] += (margin) * xs / float(xs @ xs)
        if k not in eval_ks:
            continue
        applied = edits[:k]
        # GRACE eps: largest with 0 collateral on the never-edited holdout (matched locality)
        keys = [(c2i[a["tgt"]], key_h[a["tgt"]], a["new"][0], a["tgt"]) for a in applied]
        hold_nn = [min(l2(hq_hold[c], kh) for _, kh, _, _ in keys) for c in holdout]
        eps = min(hold_nn) - 1e-6                                    # 0 holdout fires
        # retained held-out per arm
        yaz_h = yaz_n = grace_h = rome_h = 0
        for a in applied:
            for p, hq in zip(held[a["tgt"]], hq_held[a["tgt"]]):
                yaz_n += 1
                ra = router.route_frozen(p)
                yaz_h += int(yaz_first_byte(yaz, p, ra) == a["new"][0])
                # GRACE: nearest key within eps, and is it the right country's key?
                dists = [(l2(hq, kh), kc, kv, kcn) for (kc, kh, kv, kcn) in keys]
                dmin, _, kv, kcn = min(dists, key=lambda z: z[0])
                grace_h += int(dmin <= eps and kcn == a["tgt"])      # fires & right value
                rome_h += int(plain_first_byte(rome, p) == a["new"][0])
        coll_yaz = sum(1 for c in holdout if plain_first_byte(yaz, indist[c]) != plain_first_byte(base, indist[c]))
        rome_bpc = bpc_quick(rome)
        row = {"k": k, "eps": eps,
               "yaz_retained_heldout": yaz_h / yaz_n,
               "grace_retained_heldout": grace_h / yaz_n,
               "rome_retained_heldout": rome_h / yaz_n,
               "yaz_collateral_holdout": coll_yaz,
               "rome_bpc_delta_pct": (rome_bpc - base_bpc) / base_bpc * 100}
        rows.append(row)
        print(f"  k={k:3d}/{N}  held-out retained  YAZ {row['yaz_retained_heldout']:.3f}  "
              f"GRACE {row['grace_retained_heldout']:.3f}  ROME {row['rome_retained_heldout']:.3f}  "
              f"| yaz-collateral {coll_yaz}  ROME-bpc {row['rome_bpc_delta_pct']:+.1f}%", flush=True)

    out = {"n_edits": N, "n_holdout": N_HOLDOUT, "seed": SEED, "curve": rows,
           "final": rows[-1], "matched_locality": "GRACE eps set to 0 holdout-collateral each step"}
    (ROOT / "results" / "scaling" / "s2_tournament.json").write_text(json.dumps(out, indent=2))
    f = rows[-1]
    print(f"\n[P2] N={N}. FINAL held-out retained: YAZ {f['yaz_retained_heldout']:.3f}  "
          f"GRACE {f['grace_retained_heldout']:.3f}  ROME {f['rome_retained_heldout']:.3f}  "
          f"(ROME bpc {f['rome_bpc_delta_pct']:+.1f}% = weight-edit collapse; YAZ collateral {f['yaz_collateral_holdout']})")
    print("wrote results/scaling/s2_tournament.json")


if __name__ == "__main__":
    main()
