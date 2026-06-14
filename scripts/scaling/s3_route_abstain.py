"""P3 — Routing-confidence ABSTENTION (the only under-occupied leg per the novelty gate).

Every published editor treats low routing confidence as "use the base model" (route-to-base);
NONE refuse. Here we expose the Engram centroid MARGIN (top1 - top2 cosine similarity) and use it
as a selective-prediction signal: answer/apply-edit when margin >= t, else ABSTAIN. We measure the
risk-coverage tradeoff (AURC, lower=better) on a query set spanning easy (named) and hard (oblique,
name-free) clues, where routing is sometimes wrong (T4 showed ~0.50 routing on oblique clues).

Three confidence signals ranked head-to-head (same queries, same correctness labels):
  - engram_margin : top1-top2 centroid cosine similarity  (the proposed, novel-leg signal)
  - grace_distance: -(L2 to nearest in-dist activation key) (the codebook family's only signal)
  - random        : shuffled scores (the no-information floor)
Plus the no-abstain point (answer everything) and the oracle (perfect ranking) as bounds.

Claim under test: engram_margin gives a materially lower AURC than grace_distance and no-abstain,
i.e. refusing low-confidence routes removes the wrong ones. Eval-only; namespaced.
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
from scripts.t4_hard_nameabsent_eval import CLUES as HARD_CLUES, TMPLS as HARD_TMPLS

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"


def ids(s):
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def hidden(model, prompt):
    x = ids(prompt)
    if x.shape[1] > model.cfg.max_seq_len:          # clip long oblique clues to last context window
        x = x[:, -model.cfg.max_seq_len:]
    T = x.shape[1]; pos = torch.arange(T)
    h = model.tok_embed(x) + model.pos_embed(pos)[None]
    for blk in model.blocks: h = blk(h)
    return model.ln_final(h)[0, -1].numpy().astype(np.float32)


def aurc(scores, errors):
    """Area under risk-coverage curve. Sort by confidence desc, sweep coverage, risk=cum error rate."""
    idx = np.argsort(-scores); err = np.asarray(errors)[idx]
    n = len(err); risks = np.cumsum(err) / np.arange(1, n + 1)
    return float(risks.mean())                       # mean risk over all coverage levels in (0,1]


def main():
    torch.set_num_threads(1)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    c2i = ck["country_to_target_atom"]; order = list(c2i.keys())
    indist = {}
    for l in (ROOT / "data" / "probes_para_indist.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); indist.setdefault(r["country"], r["prompt"])
    named_held = []
    for l in (ROOT / "data" / "probes_para_heldout.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); named_held.append((r["country"], r["prompt"]))

    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids()
    C = router.centroids                                          # (n, dim) unit-norm
    key_h = np.stack([hidden(model, indist[c]) for c in order])   # GRACE in-dist activation keys

    # query set: EASY (named held-out) + HARD (oblique, name-free clues from T4) -> mix of right/wrong routes
    queries = []                                                  # (country, prompt, difficulty)
    for c, p in named_held:
        queries.append((c, p, "easy"))
    for c, clue in HARD_CLUES.items():
        if c in order:
            for t in HARD_TMPLS:
                queries.append((c, t.format(clue=clue), "hard"))

    rows = []
    for (c, p, diff) in queries:
        v = router.embed(p)                                       # unit-norm Engram embedding
        sims = C @ v; o = np.argsort(-sims)
        route = int(o[0]); margin = float(sims[o[0]] - sims[o[1]])
        correct = int(order[route] == c)                          # routing correct?
        hq = hidden(model, p)
        gdist = float(np.linalg.norm(key_h - hq[None], axis=1).min())   # nearest in-dist key (GRACE)
        rows.append({"country": c, "diff": diff, "route_correct": correct,
                     "engram_margin": margin, "grace_negdist": -gdist})

    err = [1 - r["route_correct"] for r in rows]
    base_err = float(np.mean(err))
    em = np.array([r["engram_margin"] for r in rows])
    gd = np.array([r["grace_negdist"] for r in rows])
    rng = np.random.default_rng(0); rnd = rng.permutation(em.astype(float))

    res = {
        "n_queries": len(rows), "n_easy": sum(r["diff"] == "easy" for r in rows),
        "n_hard": sum(r["diff"] == "hard" for r in rows),
        "no_abstain_error": base_err,                            # answer everything
        "aurc_engram_margin": aurc(em, err),                     # proposed signal (lower=better)
        "aurc_grace_distance": aurc(gd, err),                    # codebook-family signal
        "aurc_random": aurc(rnd, err),                           # no-information floor
        "aurc_oracle": aurc(np.array([1 - e for e in err], dtype=float) + 1e-9 * np.arange(len(err)), err),
    }
    # coverage @ which engram-margin keeps answered-error <= 0.05 (a useful operating point)
    idx = np.argsort(-em); cum_err = np.cumsum(np.array(err)[idx]) / np.arange(1, len(err) + 1)
    ok = np.where(cum_err <= 0.05)[0]
    res["engram_coverage_at_5pct_risk"] = float((ok[-1] + 1) / len(err)) if len(ok) else 0.0
    # same for grace
    idxg = np.argsort(-gd); cum_errg = np.cumsum(np.array(err)[idxg]) / np.arange(1, len(err) + 1)
    okg = np.where(cum_errg <= 0.05)[0]
    res["grace_coverage_at_5pct_risk"] = float((okg[-1] + 1) / len(err)) if len(okg) else 0.0

    # HARD-subset only (the honest test: does margin separate right/wrong among oblique clues,
    # not just trivially-separable easy named queries?)
    hmask = np.array([r["diff"] == "hard" for r in rows])
    h_err = list(np.array(err)[hmask]); h_em = em[hmask]; h_gd = gd[hmask]
    h_rnd = rng.permutation(h_em.astype(float))
    res["hard_only"] = {
        "n": int(hmask.sum()), "no_abstain_error": float(np.mean(h_err)),
        "aurc_engram_margin": aurc(h_em, h_err), "aurc_grace_distance": aurc(h_gd, h_err),
        "aurc_random": aurc(h_rnd, h_err),
        "aurc_oracle": aurc(np.array([1 - e for e in h_err], dtype=float) + 1e-9 * np.arange(len(h_err)), h_err),
    }
    (ROOT / "results" / "scaling" / "s3_route_abstain.json").write_text(json.dumps(res, indent=2))
    h = res["hard_only"]
    print(f"[P3] HARD-ONLY (n={h['n']}, base err {h['no_abstain_error']:.3f}): AURC engram-margin "
          f"{h['aurc_engram_margin']:.3f} | grace-distance {h['aurc_grace_distance']:.3f} | "
          f"random {h['aurc_random']:.3f} | oracle {h['aurc_oracle']:.3f}")
    print(f"[P3] queries={res['n_queries']} (easy {res['n_easy']} / hard {res['n_hard']}), "
          f"no-abstain routing error {base_err:.3f}")
    print(f"[P3] AURC (lower=better):  engram-margin {res['aurc_engram_margin']:.3f}  |  "
          f"grace-distance {res['aurc_grace_distance']:.3f}  |  random {res['aurc_random']:.3f}  |  "
          f"oracle {res['aurc_oracle']:.3f}")
    print(f"[P3] coverage @ <=5% answered-risk:  engram-margin {res['engram_coverage_at_5pct_risk']:.3f}  |  "
          f"grace-distance {res['grace_coverage_at_5pct_risk']:.3f}")
    print("wrote results/scaling/s3_route_abstain.json")


if __name__ == "__main__":
    main()
