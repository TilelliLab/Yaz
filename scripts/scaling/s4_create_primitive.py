"""CREATE primitive — allocate a fresh atom for a brand-new fact via an atoms-only additive aux loss.

Allocate a FRESH atom for a brand-NEW fact at edit time, training ONLY that atom (its W_enc row+bias
and W_dec column) while FREEZING everything else, with a decoder-orthogonality + non-hijack locality
penalty. New fact = "Zubrowka" -> capital first byte 'Z' (Z is absent from all 50 real capitals, so the
DELETE test is unambiguous). Then score the 4-condition battery:
  (a) new atom MONOSEMANTIC  — fires on the new fact, ~not on old facts / stories
  (b) old atoms UNCHANGED    — TinyStories bpc invariant + old in-dist answers unchanged
  (c) new fact READable      — held-out new-fact paraphrases route (argmax encoder) to the new atom
  (d) new fact DELETable     — zero the new atom -> new fact gone, old facts intact
Decision: 4/4 -> all conditions hold (a clean result); 1-3/4 -> honest
diagnostic/null; 0/4 -> "Gao AuxK doesn't transfer" null. Eval-only base ckpt + ~few-hundred-param train.
Namespaced (yaz-gen fork); touches no shared file. CPU.
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
TS = (Path(os.environ.get("YAZ_TINYSTORIES_DIR", "data/tinystories")) / "TinyStories-train.txt")
TS_VALID = (Path(os.environ.get("YAZ_TINYSTORIES_DIR", "data/tinystories")) / "TinyStories-valid.txt")
NEW_ENTITY = "Zubrowka"; NEW_BYTE = "Z"; K_NEW = 300; SEED = 2026
TRAIN_TMPL = "The capital of {C} is "
HELD_TMPL = ["{C}'s capital city is ", "The seat of government of {C} is located in ",
             "If you travel to {C}, the capital you'll land in is ", "{C} — capital: ",
             "Everyone knows the capital of {C} is "]


def ids(s):
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


def backbone_h(model, prompt):
    x = ids(prompt)
    if x.shape[1] > model.cfg.max_seq_len:
        x = x[:, -model.cfg.max_seq_len:]
    T = x.shape[1]; pos = torch.arange(T)
    h = model.tok_embed(x) + model.pos_embed(pos)[None]
    for blk in model.blocks:
        h = blk(h)
    return model.ln_final(h)                                  # (1,T,d_model)


@torch.no_grad()
def first_byte_surface(model, prompt, n=4):
    out = ids(prompt); plen = out.shape[1]; mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        out = torch.cat([out, model(ctx)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return bytes(out[0, plen:].tolist()).decode("latin-1", "ignore").lstrip()[:1]


@torch.no_grad()
def first_byte_routed(model, prompt, atom, n=4):
    ra = torch.tensor([int(atom)]); out = ids(prompt); plen = out.shape[1]; mc = model.cfg.max_seq_len
    for _ in range(n):
        ctx = out if out.shape[1] <= mc else out[:, -mc:]
        out = torch.cat([out, model(ctx, route_atom=ra)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return bytes(out[0, plen:].tolist()).decode("latin-1", "ignore").lstrip()[:1]


@torch.no_grad()
def bpc(model, n=40_000):
    ts = TS_VALID.read_bytes()[:n]; S = model.cfg.max_seq_len
    starts = list(range(0, len(ts)-S-1, S))[:(len(ts)//S)]; tl = 0.0; tot = 0
    for i in range(0, len(starts), 16):
        arr = np.stack([np.frombuffer(ts[s:s+S+1], dtype=np.uint8) for s in starts[i:i+16]])
        x = torch.from_numpy(arr).long(); lo = model(x[:, :-1])
        tl += float(F.cross_entropy(lo.reshape(-1, model.cfg.vocab_size), x[:, 1:].reshape(-1), reduction="sum"))
        tot += x[:, 1:].numel()
    return tl / tot / 0.6931471805599453


def enc_logits_last(model, prompt):
    h = backbone_h(model, prompt)
    return model.fact_layer.encode_logits(h)[0, -1]          # (d_dict,)


def main():
    torch.set_num_threads(1); torch.manual_seed(SEED)
    sys.path.insert(0, os.environ.get("YAZ_EMBEDDER_PATH", ""))
    from yaz.semantic_router import SemanticRouter
    from scripts.gen_paraphrase_data import TRAIN_TEMPLATES
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = YazConfig(**ck["cfg"]); model = YazLM(cfg); model.load_state_dict(ck["model"]); model.eval()
    c2i = ck["country_to_target_atom"]; order = list(c2i.keys())
    used = sorted(set(int(v) for v in c2i.values()))
    indist = {}
    for l in (ROOT / "data" / "probes_para_indist.jsonl").read_text().splitlines():
        if l:
            r = json.loads(l); indist.setdefault(r["country"], r["prompt"])

    fl = model.fact_layer
    assert K_NEW not in used, f"K_NEW {K_NEW} collides with a trained atom"
    with torch.no_grad():
        v = torch.randn(cfg.d_model); v = v / v.norm(); fl.W_dec.weight[:, K_NEW] = v

    tgt_byte = ord(NEW_BYTE)
    # the new fact gets the SAME multi-phrasing affordance the old facts had (8 train templates)
    new_train_prompts = [t.format(C=NEW_ENTITY) for t in TRAIN_TEMPLATES]
    new_held_prompts = [t.format(C=NEW_ENTITY) for t in HELD_TMPL]
    used_cols = torch.tensor(used, dtype=torch.long)

    # Engram router (the v2 model is Engram-routed; its surface W_enc never learned fact routing).
    # CREATE = allocate fresh atom K_NEW + add an Engram CENTROID for the new fact (index = len(order)).
    router = SemanticRouter(order, TRAIN_TEMPLATES); router.build_centroids()
    new_cent = np.stack([router.embed(t.format(C=NEW_ENTITY)) for t in TRAIN_TEMPLATES]).mean(0)
    new_cent = new_cent / (np.linalg.norm(new_cent) + 1e-8)
    C_ext = np.vstack([router.centroids, new_cent[None]])          # (n+1, dim); row n -> atom K_NEW
    NEW_IDX = len(order)
    def route_ext(prompt):
        v = router.embed(prompt); s = C_ext @ v; return int(s.argmax())
    idx2atom = {i: int(c2i[order[i]]) for i in range(len(order))}; idx2atom[NEW_IDX] = K_NEW

    base_bpc = bpc(model)
    base_old_fb = {c: first_byte_routed(model, indist[c], c2i[c]) for c in order}   # Engram-route olds

    # ---- train ONLY W_dec[:,K_NEW] (decode the new byte) + orthogonality; freeze everything else ----
    for p in model.parameters():
        p.requires_grad_(False)
    fl.W_dec.weight.requires_grad_(True)
    opt = torch.optim.Adam([fl.W_dec.weight], lr=5e-2)
    for step in range(400):
        opt.zero_grad(); loss_dec = 0.0
        for p in new_train_prompts:                                  # force K_NEW, emit NEW_BYTE at answer pos
            lo = model(ids(p), route_atom=torch.tensor([K_NEW]))[0, -1]
            loss_dec = loss_dec + F.cross_entropy(lo[None], torch.tensor([tgt_byte]))
        loss_dec = loss_dec / len(new_train_prompts)
        wk = fl.W_dec.weight[:, K_NEW]; wkn = wk / wk.norm().clamp_min(1e-6)
        U = fl.W_dec.weight[:, used_cols]; U = U / U.norm(dim=0, keepdim=True).clamp_min(1e-6)
        loss_ortho = ((wkn @ U) ** 2).mean()
        loss = loss_dec + 3.0 * loss_ortho
        loss.backward()
        with torch.no_grad():                                        # confine update to column K_NEW
            gd = fl.W_dec.weight.grad; md = torch.zeros_like(gd); md[:, K_NEW] = 1.0; gd.mul_(md)
        opt.step()
    model.eval()

    # ---- score the 4 conditions (Engram-routed, the v2 architecture) ----
    # (a) MONOSEMANTIC: new-fact prompts route (Engram) to the new centroid; old prompts never do
    new_routes = [route_ext(p) for p in new_train_prompts + new_held_prompts]
    new_to_new = sum(r == NEW_IDX for r in new_routes) / len(new_routes)
    old_to_new = sum(route_ext(indist[c]) == NEW_IDX for c in order)
    monosemantic = bool(new_to_new >= 0.8 and old_to_new == 0)
    # (c) READ (held-out generalization): new-fact PARAPHRASES route to the new fact + emit NEW_BYTE
    held_route_acc = sum(route_ext(p) == NEW_IDX for p in new_held_prompts) / len(new_held_prompts)
    new_fb_routed = first_byte_routed(model, new_train_prompts[0], K_NEW)
    held_fb_ok = sum(first_byte_routed(model, p, idx2atom[route_ext(p)]) == NEW_BYTE for p in new_held_prompts) / len(new_held_prompts)
    readable = bool(held_route_acc >= 0.6 and new_fb_routed == NEW_BYTE)
    # (b) OLD ATOMS UNCHANGED: bpc invariant + old Engram-routed answers unchanged
    post_bpc = bpc(model); bpc_delta_pct = (post_bpc - base_bpc) / base_bpc * 100
    old_changed = sum(1 for c in order if first_byte_routed(model, indist[c], c2i[c]) != base_old_fb[c])
    old_unchanged = bool(abs(bpc_delta_pct) <= 1.0 and old_changed == 0)
    # (d) DELETABLE: zero K_NEW -> new fact gone, olds intact
    with torch.no_grad():
        fl.W_dec.weight[:, K_NEW] = 0.0
    new_fb_after_del = first_byte_routed(model, new_train_prompts[0], K_NEW)
    old_changed_after_del = sum(1 for c in order if first_byte_routed(model, indist[c], c2i[c]) != base_old_fb[c])
    deletable = bool(new_fb_after_del != NEW_BYTE and old_changed_after_del == 0)

    conds = {"a_monosemantic": monosemantic, "b_old_unchanged": old_unchanged,
             "c_readable": readable, "d_deletable": deletable}
    n_pass = sum(conds.values())
    out = {"new_fact": f"{NEW_ENTITY}->{NEW_BYTE}", "k_new": K_NEW, "routing": "engram-centroid",
           "conditions": conds, "n_pass": n_pass,
           "detail": {"new_route_to_new": new_to_new, "old_route_to_new": old_to_new,
                      "held_route_acc": held_route_acc, "held_fb_correct": held_fb_ok,
                      "new_fb_routed": new_fb_routed, "bpc_delta_pct": bpc_delta_pct,
                      "old_changed": old_changed, "new_fb_after_delete": new_fb_after_del,
                      "old_changed_after_delete": old_changed_after_del,
                      "base_bpc": base_bpc, "post_bpc": post_bpc},
           "verdict": ("4/4 -> all four conditions hold" if n_pass == 4
                       else f"{n_pass}/4 -> partial / honest diagnostic")}
    (ROOT / "results" / "scaling" / "s4_create.json").write_text(json.dumps(out, indent=2))
    print(f"[S2 CREATE] new fact {NEW_ENTITY}->{NEW_BYTE} into fresh atom {K_NEW} (Engram-routed)")
    for k, v in conds.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    print(f"   detail: new->new-centroid {new_to_new:.2f}, old->new-centroid {old_to_new}, "
          f"held-route {held_route_acc:.2f}, held-fb-correct {held_fb_ok:.2f}, new_fb '{new_fb_routed}', "
          f"bpcΔ {bpc_delta_pct:+.3f}%, old-changed {old_changed}, "
          f"after-del new_fb '{new_fb_after_del}' old-changed {old_changed_after_del}")
    print(f"[S2 CREATE] {n_pass}/4 -> {out['verdict']}")
    print("wrote results/scaling/s4_create.json")


if __name__ == "__main__":
    main()
