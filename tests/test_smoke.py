"""Smoke test — asserts headline claims deterministically, no external deps (no embedder/corpus).

Loads the shipped checkpoint and checks, via forced atom routing:
  1. in-distribution reliability == 1.000 (every fact's prompt emits its capital's first byte)
  2. delete-locality: zeroing one fact's atom removes it and leaves the others' first bytes unchanged
Run: pytest -q
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yaz import YazConfig, YazLM

CKPT = ROOT / "checkpoints" / "yaz_gen_semantic_v2.pt"


def _load():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = YazConfig(**ck["cfg"]); m = YazLM(cfg); m.load_state_dict(ck["model"]); m.eval()
    return m, ck["country_to_target_atom"], ck["country_to_capital_first"]


def _first_byte(model, prompt, atom, n=4):
    ra = torch.tensor([int(atom)])
    out = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long).unsqueeze(0)
    plen = out.shape[1]; mc = model.cfg.max_seq_len
    with torch.no_grad():
        for _ in range(n):
            ctx = out if out.shape[1] <= mc else out[:, -mc:]
            out = torch.cat([out, model(ctx, route_atom=ra)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return bytes(out[0, plen:].tolist()).decode("latin-1", "ignore").lstrip()[:1]


def test_indist_reliability_is_perfect():
    m, c2i, capfirst = _load()
    hits = sum(_first_byte(m, f"The capital of {c} is ", c2i[c]) == capfirst[c] for c in c2i)
    assert hits == len(c2i), f"reliability {hits}/{len(c2i)} != 1.000"


def test_delete_is_local():
    m, c2i, capfirst = _load()
    tgt = "France"
    before = {c: _first_byte(m, f"The capital of {c} is ", c2i[c]) for c in c2i}
    with torch.no_grad():
        a = c2i[tgt]
        m.fact_layer.W_dec.weight[:, a] = 0.0
        m.fact_layer.W_enc.weight[a, :] = 0.0; m.fact_layer.W_enc.bias[a] = 0.0
    after = {c: _first_byte(m, f"The capital of {c} is ", c2i[c]) for c in c2i}
    assert after[tgt] != before[tgt], "delete did not change the target fact"
    others_changed = [c for c in c2i if c != tgt and after[c] != before[c]]
    assert others_changed == [], f"delete leaked to others: {others_changed}"
