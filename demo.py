"""Yaz+Engram demo — an editable knowledge model that KNOWS WHEN IT DOESN'T KNOW.

Free-text prompt -> route it to a fact-atom via a FROZEN Engram (MiniLM) embedding ->
if the routing CONFIDENCE MARGIN (top1-top2 centroid cosine) is high enough, ANSWER
(emit the routed fact); else ABSTAIN ("I'm not confident which fact you mean") and show
the top-2 candidates. Facts are live-editable (--edit) and deletable (--delete) with no
retraining, and the confidence signal is unchanged by edits.

This packages the routing-confidence abstention result (scripts/scaling/s3_route_abstain.py:
AURC 0.004 vs oracle 0.003; on hard name-free clues 0.194 vs GRACE-distance 0.436). Honest
scope: 807K byte-LM, country->capital, first-byte routing, CPU. A real, novel feature
(no published editor refuses on low routing confidence) — but not a unique advantage (the
pieces are copyable). Self-contained: imports the router/model but edits no shared file.

Usage:
  python demo.py --demo                         # scripted transcript
  python demo.py --prompt "the country of the Eiffel Tower"
  python demo.py --prompt "The capital of France is " --edit France=Lima
  python demo.py --prompt "What is the deal with quantum gravity"   # -> ABSTAIN
"""
from __future__ import annotations
import os
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
_emb_path = os.environ.get("YAZ_EMBEDDER_PATH", "")
if _emb_path:                       # only add a real path (avoid inserting "" == cwd)
    sys.path.insert(0, _emb_path)
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM
from yaz.semantic_router import SemanticRouter
from scripts.gen_paraphrase_data import TRAIN_TEMPLATES

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"
DEFAULT_THRESHOLD = 0.08          # margin below this -> abstain (tunable; see s3 risk-coverage)


def ids(s):
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long).unsqueeze(0)


class YazDemo:
    def __init__(self, threshold=DEFAULT_THRESHOLD):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        self.cfg = YazConfig(**ck["cfg"])
        self.model = YazLM(self.cfg); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.c2i = ck["country_to_target_atom"]; self.order = list(self.c2i.keys())
        self.threshold = threshold
        # capitals (full word) for resolving --edit RHS and for nicer output
        self.capof = {}
        for l in (ROOT / "data" / "facts_para_train.jsonl").read_text().splitlines():
            if l:
                import json; r = json.loads(l); self.capof.setdefault(r["country"], r["capital"])
        self.router = SemanticRouter(self.order, TRAIN_TEMPLATES); self.router.build_centroids()
        self.C = self.router.centroids                       # (n, dim) unit-norm

    # ---- routing with confidence ----
    def route(self, prompt):
        v = self.router.embed(prompt)
        sims = self.C @ v; o = np.argsort(-sims)
        return int(o[0]), float(sims[o[0]] - sims[o[1]]), [(self.order[int(i)], float(sims[i])) for i in o[:3]]

    @torch.no_grad()
    def _gen(self, prompt, atom, n=10):
        ra = torch.tensor([int(atom)]); out = ids(prompt); plen = out.shape[1]; mc = self.cfg.max_seq_len
        for _ in range(n):
            ctx = out if out.shape[1] <= mc else out[:, -mc:]
            out = torch.cat([out, self.model(ctx, route_atom=ra)[:, -1].argmax(-1, keepdim=True)], dim=1)
        return bytes(out[0, plen:].tolist()).decode("latin-1", "ignore").strip()

    # ---- CRUD (live, no retrain) ----
    def _resolve_source(self, rhs):
        """RHS of --edit may be a country name or a capital; return source country."""
        if rhs in self.c2i:
            return rhs
        for c, cap in self.capof.items():
            if cap.lower() == rhs.lower():
                return c
        return None

    def edit(self, tgt, rhs):
        src = self._resolve_source(rhs)
        if tgt not in self.c2i or src is None:
            print(f"  [edit skipped: unknown '{tgt}' or '{rhs}']"); return
        with torch.no_grad():
            self.model.fact_layer.W_dec.weight[:, self.c2i[tgt]] = \
                self.model.fact_layer.W_dec.weight[:, self.c2i[src]].clone()
        print(f"  ✎ EDIT applied: {tgt}'s capital -> {self.capof.get(src, src)} (copied {src}'s atom)")

    def delete(self, tgt):
        if tgt not in self.c2i:
            print(f"  [delete skipped: unknown '{tgt}']"); return
        a = self.c2i[tgt]
        with torch.no_grad():
            self.model.fact_layer.W_dec.weight[:, a] = 0.0
            self.model.fact_layer.W_enc.weight[a, :] = 0.0; self.model.fact_layer.W_enc.bias[a] = 0.0
        print(f"  🗑 DELETE applied: {tgt}'s atom zeroed")

    # ---- the answer-or-abstain decision ----
    def ask(self, prompt):
        atom, margin, top3 = self.route(prompt)
        country = self.order[atom]
        if margin < self.threshold:
            cands = ", ".join(f"{c}" for c, _ in top3[:2])
            print(f'  Q: "{prompt}"')
            print(f"  → ABSTAIN (margin {margin:.3f} < {self.threshold}). "
                  f"I'm not confident which fact you mean — {cands}?")
            return
        gen = self._gen(prompt, atom)
        fb = gen.lstrip()[:1]
        cap = self.capof.get(country, "?")
        print(f'  Q: "{prompt}"')
        print(f"  → ANSWER (routed: {country}, margin {margin:.3f}): first byte '{fb}'  "
              f"[raw gen: {gen!r}]")


def run_demo(d):
    print("\n=== Yaz+Engram: an editor that knows when it doesn't know ===\n")
    print("[1] Confident, in-scope question:")
    d.ask("The capital of France is ")
    print("\n[2] Name-free clue (semantic routing does the entity-ID):")
    d.ask("the country of the Eiffel Tower and the Louvre, its capital is ")
    print("\n[3] Live edit (no retrain), then re-ask a PARAPHRASE — edit transfers:")
    d.edit("France", "Lima")
    d.ask("the country of the Eiffel Tower and the Louvre, its capital is ")
    print("\n[4] Out-of-scope / unknown question — the model REFUSES instead of confabulating:")
    d.ask("What is the deal with quantum gravity and black holes")
    d.ask("Tell me about the best pizza topping")
    print("\n(Confidence = Engram top1-top2 centroid margin. The feature on show is ABSTENTION + edit;")
    print(" the answer is judged on the FIRST BYTE — this 807K model is a first-byte editor, so the")
    print(" multi-byte gen is garbled by design (full-word transfer ~0.05, measured). Evidence the")
    print(" confidence signal is real: s3 AURC 0.004 vs oracle 0.003; hard name-free 0.194 vs GRACE 0.436.")
    print(" A real, novel feature — no published editor refuses on low routing confidence — but a STEP,")
    print(" the pieces are individually published.)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--edit", type=str, default=None, help="Country=CapitalOrCountry, e.g. France=Lima")
    ap.add_argument("--delete", type=str, default=None, help="Country to delete, e.g. France")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    d = YazDemo(threshold=args.threshold)
    if args.demo:
        run_demo(d); return
    if args.delete:
        d.delete(args.delete)
    if args.edit:
        tgt, rhs = args.edit.split("=", 1); d.edit(tgt.strip(), rhs.strip())
    if args.prompt:
        d.ask(args.prompt)
    elif not (args.edit or args.delete):
        ap.print_help()


if __name__ == "__main__":
    main()
