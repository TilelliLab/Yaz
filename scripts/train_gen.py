"""Phase 8 — config-driven train->eval->CRUD, adds answer-position BACKBONE-SUPPRESSION
(Exp 28). A copy of 49_phase7.py with ONE backward-compatible knob; defaults reproduce
Phase 7 exactly.

Always uses the Exp24 'both' recipe (LR floor eta_min=5e-5 + dec_weight ramp 0.1->0.5).
Config (json, argv[1]) overrides: name, facts_file, probes_file, aux_weight,
full_span_sup, update_pairs, delete_targets, plus the NEW Phase-8 keys:

  supp_masked (bool, default false): when true, REPLACE the all-position atoms-only CE
    (`ce_atoms`, a mis-specified target since one atom cannot predict generic text) with
    an ANSWER-POSITION-masked suppression CE. unembed is linear, so a backbone-suppressed
    logit at factor beta is exactly  supp = beta*logits + (1-beta)*ao  (beta=1 -> full
    backbone, beta=0 -> atoms-only). Supervising supp at answer positions forces the atom
    to carry the fact -> raises atoms-only accuracy -> a W_dec column-swap controls more
    of the answer logit. aux_weight remains the multiplier on this term.
  supp_beta (float, default 0.0): the backbone-suppression factor beta above.

Usage: python3 scripts/55_phase8.py configs/phase8_exp28_mask0.json
"""
from __future__ import annotations
import os
import json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from yaz.model import greedy_generate

CFG = json.loads(Path(sys.argv[1]).read_text())
NAME = CFG["name"]
FACTS = ROOT / CFG["facts_file"]
PROBES = ROOT / CFG["probes_file"]
AUX_WEIGHT = float(CFG.get("aux_weight", 1.0))
FULL_SPAN = bool(CFG.get("full_span_sup", False))
SUPP_MASKED = bool(CFG.get("supp_masked", False))   # Phase 8: mask atoms-only CE to answer positions
SUPP_BETA = float(CFG.get("supp_beta", 0.0))        # Phase 8: backbone-suppression factor beta
UPDATE_PAIRS = [tuple(p) for p in CFG["update_pairs"]]
DELETE_TARGETS = CFG.get("delete_targets", [])
# GENERALIZATION knob: keep only the first N_PHRASINGS train templates per fact.
# n_phrasings=1 -> single-phrasing (current Yaz, surface-bound like GRACE); 8 -> all.
N_PHRASINGS = int(CFG.get("n_phrasings", 8))
HELDOUT = ROOT / CFG.get("heldout_probes_file", "data/probes_para_heldout.jsonl")
# ROUTER: "surface" (legacy learned argmax(W_enc·h)) or "semantic" (force the atom chosen
# by a frozen Engram embedding). The semantic run trains the model ONCE with forced
# true-country routing, then evaluates under BOTH semantic routers (frozen-centroid and
# learned-linear), writing one results file per router.
ROUTER = CFG.get("router", "surface")
SEMANTIC = (ROUTER == "semantic")
# Corrected semantic run: learnable per-atom gain so the forced atom dominates the
# residual (fix for the activation=1.0 backbone-co-memorization regression).
USE_ATOM_GAIN = bool(CFG.get("use_atom_gain", False))
ATOM_GAIN_INIT = float(CFG.get("atom_gain_init", 1.0))

CKPT = ROOT / "checkpoints" / f"yaz_gen_{NAME}.pt"
OUT = ROOT / "results" / f"gen_{NAME}.json"

CORPUS = (Path(os.environ.get("YAZ_TINYSTORIES_DIR", "data/tinystories")) / "TinyStories-train.txt")
TS_VALID = (Path(os.environ.get("YAZ_TINYSTORIES_DIR", "data/tinystories")) / "TinyStories-valid.txt")
CORPUS_BYTES = 3 * 1024 * 1024
SEQ = 128; BATCH = 32
STEPS = int(os.environ.get("YAZ_STEPS", "6000"))
LR = 5e-4; FACT_RATE = 0.85
SEED = int(CFG.get("seed", os.environ.get("YAZ_SEED", "2026")))  # config-driven for multi-seed robustness; default reproduces the win
LB_WEIGHT = 0.05; ORTHO_WEIGHT = 0.01; SUP_WEIGHT = 0.5
DEC_WEIGHT = 0.1; DEC_WEIGHT_MAX = 0.5; RAMP_START = 0.6
ETA_MIN = LR * 0.1                # both-recipe LR floor
RESURRECT_EVERY = 200; RESURRECT_THRESHOLD = 0.10; EMA_BETA = 0.95
D_MODEL = 128; D_DICT = 512
SIDE_ALIVE = 8; SIDE_PARTIAL = 20


def dec_w_at(step):
    frac = min(1.0, max(0.0, (step - RAMP_START*STEPS) / ((1-RAMP_START)*STEPS)))
    return DEC_WEIGHT + (DEC_WEIGHT_MAX - DEC_WEIGHT) * frac


def rows():
    rs = [json.loads(l) for l in FACTS.read_text().splitlines() if l]
    # generalization knob: keep only the first N_PHRASINGS templates per fact.
    return [r for r in rs if r.get("template_id", 0) < N_PHRASINGS]


def c2idx(rs):
    s = {}
    for r in rs:
        if r["country"] not in s: s[r["country"]] = len(s)
    return s


def c2capfirst(rs):
    s = {}
    for r in rs:
        if r["country"] not in s: s[r["country"]] = r["capital"][0]
    return s


def batch_stories(buf, rng):
    n = len(buf) - SEQ - 1
    idx = rng.integers(0, n, size=BATCH)
    arr = np.zeros((BATCH, SEQ+1), dtype=np.uint8)
    for i, j in enumerate(idx):
        arr[i] = np.frombuffer(buf[j:j+SEQ+1], dtype=np.uint8)
    return torch.from_numpy(arr).long()


def batch_facts(rs, c2i, rng):
    arr = np.zeros((BATCH, SEQ+1), dtype=np.uint8)
    countries = np.zeros(BATCH, dtype=np.int64)
    amask = np.zeros((BATCH, SEQ), dtype=bool)
    for i in range(BATCH):
        r = rs[rng.integers(0, len(rs))]
        text = r["text"]; capital = r["capital"]
        countries[i] = c2i[r["country"]]
        cap_idx = text.rindex(capital)            # robust: answer at end
        line_b = (text + "\n").encode("utf-8"); L = len(line_b)
        chunk = (line_b * max(1, (SEQ+1)//L + 1))[:SEQ+1]
        arr[i, :len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
        n_tiles = (SEQ + L - 1)//L + 1
        span = len(capital) if FULL_SPAN else 1     # mark whole capital span or just byte0
        for tile in range(n_tiles):
            for d in range(span):
                ans = tile*L + cap_idx - 1 + d
                if 0 <= ans < SEQ:
                    amask[i, ans] = True
    return (torch.from_numpy(arr).long(), torch.from_numpy(countries).long(),
            torch.from_numpy(amask))


def dec_spec_loss(model, c2atom, capfirst):
    W = model.fact_layer.W_dec.weight; U = model.unembed.weight
    cols = W[:, c2atom]; tg = U[capfirst]
    cn = cols / cols.norm(dim=0, keepdim=True).clamp_min(1e-6)
    tn = tg / tg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (1.0 - (cn.t()*tn).sum(dim=-1)).mean()


def ids(s):
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


# ROUTE_FN: prompt_str -> atom_id (int) for semantic arms; None = surface (learned argmax).
# Set in main() once per router; all eval helpers consult it via _atom_for so the CRUD
# code path is unchanged across arms.
ROUTE_FN = None


def _atom_for(prompt):
    return None if ROUTE_FN is None else ROUTE_FN(prompt)


def _ra(atom):
    return None if atom is None else torch.tensor([int(atom)], dtype=torch.long)


def train(rs, c2i, c2cap):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    buf = CORPUS.read_bytes()[:CORPUS_BYTES]
    order = list(c2i.keys())
    c2atom = torch.tensor([c2i[c] for c in order], dtype=torch.long)
    capfirst = torch.tensor([ord(c2cap[c]) for c in order], dtype=torch.long)
    cfg = YazConfig(d_model=D_MODEL, d_dict=D_DICT,
                    use_atom_gain=USE_ATOM_GAIN, atom_gain_init=ATOM_GAIN_INIT)
    model = YazLM(cfg)
    print(f"[{NAME}] params {model.count_params():,} aux={AUX_WEIGHT} full_span={FULL_SPAN} "
          f"supp_masked={SUPP_MASKED} supp_beta={SUPP_BETA}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=ETA_MIN)
    ema = torch.zeros(cfg.d_dict); losses = []; t0 = time.time()
    for step in range(STEPS):
        is_fact = rng.random() < FACT_RATE
        if is_fact:
            x_ids, cidx, amask = batch_facts(rs, c2i, rng)
            x, y = x_ids[:, :-1], x_ids[:, 1:]
            B, T = x.shape
            m = amask[:, :T]
            if SEMANTIC:
                # SEMANTIC RE-KEYING: force the true country atom at answer positions
                # (train uses train labels — not a leak). Atom selection is decoupled
                # from W_enc; ce_sup (surface-router supervision) is dropped.
                logits, fz, dense, ao = model(x, return_dense=True, return_atoms_only=True,
                                              route_atom=cidx, route_pos=m)
            else:
                logits, fz, dense, ao = model(x, return_dense=True, return_atoms_only=True)
            ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            if SUPP_MASKED:
                # Phase 8: answer-position backbone-suppression. supp = beta*full + (1-beta)*atoms_only.
                # Supervise ONLY at answer positions so the atom must carry the fact byte.
                supp = SUPP_BETA * logits + (1.0 - SUPP_BETA) * ao
                ce_atoms = (F.cross_entropy(supp[m], y[m]) if int(m.sum()) > 0
                            else torch.tensor(0.0))
            else:
                # Phase 7 behaviour: all-position atoms-only CE (beta=0, unmasked).
                ce_atoms = F.cross_entropy(ao.reshape(-1, cfg.vocab_size), y.reshape(-1))
            if SEMANTIC:
                # Atom is forced by the semantic embedding, so the surface router
                # (W_enc) gets no fact supervision and we skip the extra forward.
                ce_sup = torch.tensor(0.0)
            else:
                pos = torch.arange(T)
                h = model.tok_embed(x) + model.pos_embed(pos)[None]
                for blk in model.blocks: h = blk(h)
                h = model.ln_final(h)
                el = model.fact_layer.encode_logits(h)
                tgt = cidx.unsqueeze(1).expand(B, T)
                ce_sup = F.cross_entropy(el[m], tgt[m].reshape(-1)) if int(m.sum()) > 0 else torch.tensor(0.0)
        else:
            x_ids = batch_stories(buf, rng)
            x, y = x_ids[:, :-1], x_ids[:, 1:]
            logits, fz, dense = model(x, return_dense=True)
            ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            ce_atoms = ce_sup = torch.tensor(0.0)
        soft = F.softmax(dense, dim=-1); pa = soft.mean(dim=(0,1))
        lb = (math.log(cfg.d_dict) - (-(pa*(pa+1e-8).log()).sum())) / math.log(cfg.d_dict)
        ortho = model.fact_layer.orthogonality_loss()
        ds = dec_spec_loss(model, c2atom, capfirst)
        dw = dec_w_at(step)
        loss = (ce + (AUX_WEIGHT*ce_atoms if is_fact else 0.0) + (SUP_WEIGHT*ce_sup if is_fact else 0.0)
                + dw*ds + LB_WEIGHT*lb + ORTHO_WEIGHT*ortho)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            ema.mul_(EMA_BETA).add_((fz != 0).float().mean(dim=(0,1)) * (1-EMA_BETA))
        if step > 0 and step % RESURRECT_EVERY == 0:
            mr = ema.mean().clamp_min(1e-8); dead = torch.nonzero(ema < RESURRECT_THRESHOLD*mr).flatten()
            if dead.numel() > 0:
                with torch.no_grad():
                    pos = torch.arange(x.size(1)); h = model.tok_embed(x) + model.pos_embed(pos)[None]
                    for blk in model.blocks: h = blk(h)
                    model.fact_layer.resurrect(dead, model.ln_final(h).reshape(-1, cfg.d_model))
                ema[dead] = mr.item()
        losses.append({"dec": float(ds.item()), "ce": float(ce.item()),
                       "ce_sup": float(ce_sup.item()) if is_fact else None})
        if step % 400 == 0 or step == STEPS-1:
            cs = [l["ce_sup"] for l in losses[-100:] if l["ce_sup"] is not None]
            print(f"  [{NAME}] {step:5d}/{STEPS} ce={np.mean([l['ce'] for l in losses[-100:]]):.3f} "
                  f"ce_sup={np.mean(cs) if cs else float('nan'):.3f} dec={ds.item():.3f} dw={dw:.2f} "
                  f"lr={sched.get_last_lr()[0]:.1e} t={time.time()-t0:.0f}s", flush=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "country_to_target_atom": c2i, "country_to_capital_first": c2cap}, CKPT)
    final_dec = float(np.mean([l["dec"] for l in losses[-50:]]))
    return model, final_dec, round(time.time()-t0, 1)


@torch.no_grad()
def eval_model(model, probes, c2target):
    correct = ao_c = hits = 0; per = []
    for p in probes:
        atom = _atom_for(p["prompt"]); ra = _ra(atom)
        gen = gen_after(model, p["prompt"], n=8)
        fr = next((c for c in gen if c != " "), "")
        correct += int(fr == p["expected_first_byte"])
        _, fz = model(ids(p["prompt"]), return_fact_z=True, route_atom=ra); atom = int(fz[0, -1].argmax())
        _, ao = model(ids(p["prompt"]), return_atoms_only=True, route_atom=ra)
        ao1 = chr(int(ao[0, -1].argmax())); ao_c += int(ao1 == p["expected_first_byte"])
        tgt = c2target.get(p["country"], -1); hits += int(atom == tgt)
        per.append({"country": p["country"], "capital": p["capital"],
                    "expected_first": p["expected_first_byte"], "generated": gen.strip(),
                    "fact_atom_id": atom, "target_atom": tgt})
    n = len(probes)
    return {"correct": correct, "n": n, "first_byte_accuracy": correct/n,
            "atoms_only_first_byte_accuracy": ao_c/n,
            "supervised_target_atom_hit_rate": hits/n,
            "n_collision_atoms": len([a for a, c in Counter(p["fact_atom_id"] for p in per).items() if c > 1]),
            "per_probe": per}


def bpc_quick(model, n=40_000):
    ts = TS_VALID.read_bytes()[:n]; S = model.cfg.max_seq_len
    starts = list(range(0, len(ts)-S-1, S))[:(len(ts)//S)]; tl = 0.0; tot = 0
    with torch.no_grad():
        for i in range(0, len(starts), 16):
            bs = starts[i:i+16]
            arr = np.stack([np.frombuffer(ts[s:s+S+1], dtype=np.uint8) for s in bs])
            x = torch.from_numpy(arr).long()
            lo = model(x[:, :-1])
            tl += float(F.cross_entropy(lo.reshape(-1, model.cfg.vocab_size), x[:, 1:].reshape(-1), reduction="sum")); tot += x[:, 1:].numel()
    return tl/tot/0.6931471805599453


def clone(model):
    nm = YazLM(model.cfg); nm.load_state_dict({k: v.clone() for k, v in model.state_dict().items()}); nm.eval(); return nm


@torch.no_grad()
def gen_after(model, prompt, n=8):
    atom = _atom_for(prompt)
    p = ids(prompt)
    if atom is None:
        out = greedy_generate(model, p, n_new=n)
    else:
        ra = _ra(atom); out = p; mc = model.cfg.max_seq_len; model.eval()
        for _ in range(n):
            ctx = out if out.shape[1] <= mc else out[:, -mc:]
            lo = model(ctx, route_atom=ra)          # force semantic atom at last position
            nxt = lo[:, -1].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
    return bytes(out[0, len(p[0]):].tolist()).decode("latin-1", "ignore")


@torch.no_grad()
def top5(model, prompt):
    ra = _ra(_atom_for(prompt))
    last = model(ids(prompt), route_atom=ra)[0, -1]; v, ix = last.topk(5)
    return [(chr(int(i)), float(val)) for val, i in zip(v, ix)]


@torch.no_grad()
def gen_first(model, prompt):
    g = gen_after(model, prompt, n=4).lstrip()
    return g[0] if g else ""


def crud(model, base, heldout_by_country):
    per = base["per_probe"]
    prompt_of = {json.loads(l)["country"]: json.loads(l)["prompt"]
                 for l in PROBES.read_text().splitlines() if l}
    probes = [{"country": r["country"], "capital": r["capital"],
               "prompt": prompt_of[r["country"]],
               "expected_first_byte": r["expected_first"]} for r in per]
    addr = {r["country"]: r["fact_atom_id"] for r in per}
    capof = {r["country"]: r["capital"] for r in per}
    avail = set(addr)
    edits = []; verdicts = []
    for tgt, src in UPDATE_PAIRS:
        if tgt not in avail or src not in avail:
            print(f"SKIP {tgt}<-{src}"); continue
        m = clone(model)
        b_bpc = bpc_quick(m); b_gen = {p["country"]: gen_after(m, p["prompt"]) for p in probes}
        with torch.no_grad():
            m.fact_layer.W_dec.weight[:, addr[tgt]] = m.fact_layer.W_dec.weight[:, addr[src]].clone()
        a_bpc = bpc_quick(m); a_gen = {p["country"]: gen_after(m, p["prompt"]) for p in probes}
        prm = next(p["prompt"] for p in probes if p["country"] == tgt)
        t5 = top5(m, prm); new_cap = capof[src]; exp = new_cap[0]
        chars = [c for c, _ in t5]
        rank = chars.index(exp)+1 if exp in chars else None
        side = sum(1 for c in avail if c != tgt and b_gen[c] != a_gen[c])
        bpcd = (a_bpc - b_bpc)/b_bpc*100
        # multi-byte: how many leading bytes of new_cap does the post-edit free-run produce?
        af = a_gen[tgt].lstrip(); trail = 0
        for k in range(min(len(new_cap), len(af))):
            if af[k] == new_cap[k]: trail += 1
            else: break
        verdict = ("ALIVE" if rank == 1 and side <= SIDE_ALIVE and abs(bpcd) <= 5
                   else "PARTIAL" if rank is not None and rank <= 5 and side <= SIDE_PARTIAL and abs(bpcd) <= 15
                   else "DEAD")
        verdicts.append(verdict)
        # GENERALIZATION: does the edit (column swap) transfer to UNSEEN phrasings?
        # Probe the edited country with each held-out test template; count first-byte hits.
        ho = heldout_by_country.get(tgt, [])
        gen_hits = sum(1 for pr in ho if gen_first(m, pr) == exp)
        gen_n = len(ho)
        gen_rate = gen_hits / gen_n if gen_n else None
        print(f"  [{NAME}] UPDATE {tgt}<-{src}: '{b_gen[tgt].strip()}'->'{a_gen[tgt].strip()}' "
              f"rank({exp})={rank} side={side} trail={trail}/{len(new_cap)} {verdict} "
              f"| GEN held-out {gen_hits}/{gen_n}")
        edits.append({"edit": f"{tgt}<-{src}", "rank": rank, "side_effects": side,
                      "n_others": len(avail)-1, "new_capital": new_cap, "trail_bytes": trail,
                      "target_after": a_gen[tgt].strip(), "top5": t5, "bpc_delta_pct": bpcd,
                      "verdict": verdict, "gen_heldout_hits": gen_hits, "gen_heldout_n": gen_n,
                      "gen_rate": gen_rate})
    dels = []
    for tgt in DELETE_TARGETS:
        if tgt not in avail: continue
        m = clone(model); b = {p["country"]: gen_after(m, p["prompt"]) for p in probes}
        with torch.no_grad():
            a = addr[tgt]
            m.fact_layer.W_dec.weight[:, a] = 0.0; m.fact_layer.W_enc.weight[a, :] = 0.0; m.fact_layer.W_enc.bias[a] = 0.0
        af = {p["country"]: gen_after(m, p["prompt"]) for p in probes}
        side = sum(1 for c in avail if c != tgt and b[c] != af[c])
        dels.append({"delete": tgt, "first_changed": b[tgt][:1] != af[tgt][:1], "side_effects": side})
        print(f"  [{NAME}] DELETE {tgt}: '{b[tgt].strip()}'->'{af[tgt].strip()}' side={side}")
    alive = sum(v == "ALIVE" for v in verdicts); pp = sum(v in ("ALIVE", "PARTIAL") for v in verdicts)
    agg = "ALIVE" if alive >= 4 else ("PARTIAL" if pp >= 3 else "DEAD")
    g_hits = sum(e["gen_heldout_hits"] for e in edits)
    g_n = sum(e["gen_heldout_n"] for e in edits)
    return {"edits": edits, "deletes": dels, "individual_verdicts": verdicts,
            "n_alive": alive, "n_partial_or_better": pp, "aggregate_verdict": agg,
            "max_trail": max((e["trail_bytes"] for e in edits), default=0),
            "gen_heldout_hits": g_hits, "gen_heldout_n": g_n,
            "gen_heldout_rate": (g_hits / g_n if g_n else None)}


@torch.no_grad()
def heldout_base_accuracy(model, held):
    """Pre-edit: can the model answer HELD-OUT phrasings at all? (reachability prerequisite)"""
    ok = 0
    for p in held:
        ok += int(gen_first(model, p["prompt"]) == p["expected_first_byte"])
    return ok, len(held)


def evaluate_and_write(model, c2i, final_dec, wall, router_tag, out_path):
    """Run the full eval+CRUD under the currently-installed ROUTE_FN and write a result
    file. Used once for surface, twice (frozen/learned) for the semantic model."""
    probes = [json.loads(l) for l in PROBES.read_text().splitlines() if l]
    ev = eval_model(model, probes, c2i)
    ev["tinystories_bpc"] = bpc_quick(model)
    held = [json.loads(l) for l in HELDOUT.read_text().splitlines() if l]
    heldout_by_country = {}
    for p in held:
        heldout_by_country.setdefault(p["country"], []).append(p["prompt"])
    ho_ok, ho_n = heldout_base_accuracy(model, held)
    print(f"[{NAME}/{router_tag}] reliability(in-dist) {ev['correct']}/{ev['n']}={ev['first_byte_accuracy']:.3f} "
          f"| base-generalization(held-out, pre-edit) {ho_ok}/{ho_n}={ho_ok/ho_n:.3f} "
          f"atoms-only={ev['atoms_only_first_byte_accuracy']:.3f} target-hit={ev['supervised_target_atom_hit_rate']:.3f} "
          f"bpc={ev['tinystories_bpc']:.3f}", flush=True)
    cr = crud(model, ev, heldout_by_country)
    out = {"name": NAME, "router": router_tag, "n_phrasings": N_PHRASINGS, "aux_weight": AUX_WEIGHT,
           "full_span_sup": FULL_SPAN, "supp_masked": SUPP_MASKED, "supp_beta": SUPP_BETA,
           "params": model.count_params(), "steps": STEPS, "final_dec_spec": final_dec,
           "wall_sec": wall, "reliability_indist": ev["first_byte_accuracy"],
           "base_generalization_heldout": ho_ok / ho_n,
           "eval": {k: ev[k] for k in ev if k != "per_probe"}, "crud": cr}
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"=== [{NAME}/{router_tag}] EDIT in-dist ALIVE={cr['n_alive']}/{len(cr['individual_verdicts'])} "
          f"| EDIT-GENERALIZATION held-out {cr['gen_heldout_hits']}/{cr['gen_heldout_n']}="
          f"{cr['gen_heldout_rate']:.3f} | base-gen {ho_ok/ho_n:.3f} | reliability {ev['first_byte_accuracy']:.3f} "
          f"final_dec={final_dec:.3f} | wrote {Path(out_path).name}", flush=True)
    return out


def main():
    global ROUTE_FN
    torch.set_num_threads(int(os.environ.get("YAZ_THREADS", "1")))
    rs = rows(); c2i = c2idx(rs); c2cap = c2capfirst(rs)
    print(f"[{NAME}] router={ROUTER} N_PHRASINGS={N_PHRASINGS} -> {len(rs)} train rows "
          f"over {len(c2i)} facts", flush=True)
    model, final_dec, wall = train(rs, c2i, c2cap)

    if not SEMANTIC:
        ROUTE_FN = None
        evaluate_and_write(model, c2i, final_dec, wall, "surface", OUT)
        return

    # SEMANTIC: build both routers over frozen Engram embeddings, eval the one model twice.
    from yaz.semantic_router import SemanticRouter
    from scripts.gen_paraphrase_data import TRAIN_TEMPLATES
    order = list(c2i.keys())                      # country i -> fact-atom id i
    router = SemanticRouter(order, TRAIN_TEMPLATES)
    router.build_centroids()
    head_tr_acc = router.train_head()
    print(f"[{NAME}] semantic routers built (learned-head train acc {head_tr_acc:.3f})", flush=True)
    for tag, fn in [("frozen", router.route_frozen), ("learned", router.route_learned)]:
        ROUTE_FN = fn
        out_path = ROOT / "results" / f"gen_{NAME}_{tag}.json"
        evaluate_and_write(model, c2i, final_dec, wall, tag, out_path)
    ROUTE_FN = None


if __name__ == "__main__":
    main()
